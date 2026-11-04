"""Offline backend: the same interface as servers.Servers, with every model on this machine (laptop or Pi 5).

hearing : Moonshine (sherpa-onnx, ONNX Runtime)              ~0.1-0.4 s per turn on a Pi 5
planner : Qwen2.5-Instruct GGUF in llama.cpp's llama-server   started and stopped by FDA, 127.0.0.1 only
voice   : Piper (default, ~30x real time on the laptop) or Kokoro (nicer, ~3x: too slow for a Pi)

The planner prompt is laid out so its long fixed part (rules + examples) stays in llama.cpp's prompt cache and each
turn only reads the new words (brain.plan_messages(static_prefix=True)). No API key, no network.
`python -m fda download` fetches the models (about 1.4 GB with the defaults) into offline_models/.
"""
from __future__ import annotations

import asyncio
import atexit
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
import soxr

from . import brain
from .config import ROOT
from .servers import Servers

log = logging.getLogger("fda.local")
DIR = ROOT / "offline_models"

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR = {   # name -> folder in the sherpa-onnx release (and in offline_models/)
    "moonshine-base": "sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
    "parakeet": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",            # NVIDIA Parakeet TDT 0.6B: top English accuracy
    "parakeet-small": "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8",   # 110M: fast, for the Pi
}
TTS = {
    "kokoro": "kokoro-en-v0_19",                    # full size: 3x faster than int8 on a CPU
    "piper": "vits-piper-en_US-ryan-medium",
}
KOKORO_VOICES = ["af", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael",
                 "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
LLM = {   # name -> (Hugging Face repo, file)
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    # app builder (create_app): code models, used only by the tools MCP server
    "qwen2.5-coder-1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-coder-3b": ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
}
LLAMA_BUILD = "b11776"    # llama.cpp release of 2026-10-21


def _llama_asset() -> str:
    m = platform.machine().lower()
    if sys.platform == "win32":
        return "win-cpu-arm64.zip" if m in ("arm64", "aarch64") else "win-cpu-x64.zip"
    if sys.platform == "darwin":
        return "macos-arm64.tar.gz" if m == "arm64" else "macos-x64.tar.gz"
    return "ubuntu-arm64.tar.gz" if m in ("aarch64", "arm64") else "ubuntu-x64.tar.gz"


def _need(cfg) -> list[tuple[str, str, Path]]:
    """(what, url, where it ends up) for the models this config uses."""
    for key, table, name in (("offline_asr", ASR, cfg.offline_asr), ("offline_tts", TTS, cfg.offline_tts),
                             ("offline_llm", LLM, cfg.offline_llm)):
        if name not in table:
            raise SystemExit(f"{key} = {name!r}: choose one of {', '.join(table)}")
    asr, tts = ASR[cfg.offline_asr], TTS[cfg.offline_tts]
    repo, gguf = LLM[cfg.offline_llm]
    asset = _llama_asset()
    return [
        (f"hearing ({cfg.offline_asr})", f"{_SHERPA}/asr-models/{asr}.tar.bz2", DIR / asr),
        (f"voice ({cfg.offline_tts})", f"{_SHERPA}/tts-models/{tts}.tar.bz2", DIR / tts),
        (f"planner ({cfg.offline_llm})", f"https://huggingface.co/{repo}/resolve/main/{gguf}", DIR / gguf),
        (f"llama.cpp {LLAMA_BUILD}", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
                                     f"llama-{LLAMA_BUILD}-bin-{asset}", DIR / f"llama.cpp-{LLAMA_BUILD}"),
    ]


def missing(cfg) -> list[str]:
    out = [what for what, _, dest in _need(cfg) if not dest.exists()]
    if f"llama.cpp {LLAMA_BUILD}" in out and shutil.which("llama-server"):
        out.remove(f"llama.cpp {LLAMA_BUILD}")      # e.g. built from source on the Pi
    return out


def download(cfg):
    DIR.mkdir(exist_ok=True)
    for what, url, dest in _need(cfg):
        if dest.exists():
            print(f"  ok        {what}")
            continue
        arc = DIR / "_dl" / url.rsplit("/", 1)[1]
        if not arc.exists():
            arc.parent.mkdir(exist_ok=True)
            print(f"  download  {what}: {url}")
            _fetch(url, arc)
        print(f"  unpack    {what}")
        if arc.suffix == ".gguf":
            arc.replace(dest)
        elif arc.name.endswith(".zip"):
            with zipfile.ZipFile(arc) as z:
                z.extractall(dest)
            arc.unlink()
        else:
            target = dest if "llama" in arc.name else DIR
            with tarfile.open(arc) as t:
                t.extractall(target, filter="data")
            arc.unlink()
        if not dest.exists():
            raise SystemExit(f"unpacking {arc.name} did not create {dest}")
    if sys.platform != "win32":
        for exe in (DIR / f"llama.cpp-{LLAMA_BUILD}").rglob("llama-server"):
            exe.chmod(0o755)
    print(f"offline models ready in {DIR}")


def _fetch(url: str, dest: Path):
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fda-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                print(f"            {done / 1e6:7.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else ""), flush=True)
    part.replace(dest)


def _llama_server_exe() -> str:
    for exe in sorted((DIR / f"llama.cpp-{LLAMA_BUILD}").rglob("llama-server*")):
        if exe.name in ("llama-server", "llama-server.exe"):
            return str(exe)
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("llama-server not found: run  python -m fda download")


def _port_busy(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _threads(cfg) -> tuple[int, int]:
    """(planner threads, threads for hearing / voice)."""
    n = os.cpu_count() or 4
    llm = cfg.offline_threads or (max(1, n - 1) if n <= 4 else min(8, n // 2))   # Pi: 3 of 4 cores
    return llm, (2 if n <= 4 else 4)


def serve_judge(model: str = "qwen2.5-3b", port: int = 8092):
    """A free, local, OpenAI-compatible judge for fd-benchmark (no account, no tokens). Runs until Ctrl+C."""
    if model not in LLM:
        raise SystemExit(f"--model: choose one of {', '.join(LLM)}")
    gguf = DIR / LLM[model][1]
    if not gguf.exists():
        raise SystemExit(f"{gguf.name} is not downloaded: put  offline_llm = \"{model}\"  in fda.toml once and run "
                         "python -m fda download")
    if _port_busy(port):
        raise SystemExit(f"port {port} is in use (judge already running?), or pass --port")
    n = os.cpu_count() or 4
    cmd = [_llama_server_exe(), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-c", "8192",
           "-np", "1", "-t", str(max(1, n - 1 if n <= 4 else min(12, n // 2))), "--no-webui", "--alias", model]
    url = f"http://127.0.0.1:{port}/v1"
    print(f"local judge: {model} on {url}  (Ctrl+C to stop)\n"
          f"score a finished run with it (in fd-benchmark, fdbench's own python):\n"
          f"  .\\.venv\\Scripts\\python -m fdbench score runs\\<student>\\<run> --judge-provider custom "
          f"--judge-url {url} --judge-model {model}\n", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(5)


class LocalServers(Servers):
    kind = "offline"
    static_prefix = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.chat_url = f"http://127.0.0.1:{cfg.llm_port}"
        self.chat_model = cfg.offline_llm
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        self.web = httpx.AsyncClient(timeout=10.0)
        self.proc: subprocess.Popen | None = None
        self.asr = self.tts = None
        self.sid = 0
        self.ok = {"asr": False, "tts": False, "llm": False}
        self._asr_lock, self._tts_lock = threading.Lock(), threading.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._log = None

    # ---------------- start / stop ----------------
    async def start(self):
        async with self._start_lock:
            if self._started:
                return
            self._started = True
            gone = missing(self.cfg)
            if gone:
                log.error("offline models missing (%s): run  python -m fda download", ", ".join(gone))
                return
            t0 = time.perf_counter()
            self._start_llm()
            await asyncio.gather(asyncio.to_thread(self._load_asr), asyncio.to_thread(self._load_tts),
                                 self._wait_llm())
            log.info("offline models up in %.1f s: %s", time.perf_counter() - t0, self.ok)

    def _start_llm(self):
        repo, gguf = LLM[self.cfg.offline_llm]
        threads, _ = _threads(self.cfg)
        port = self.cfg.llm_port
        if _port_busy(port):        # another FDA (or test) already uses it: never share a planner
            port = _free_port()
            log.warning("port %d is in use, the planner uses port %d instead", self.cfg.llm_port, port)
        self.chat_url = f"http://127.0.0.1:{port}"
        cmd = [_llama_server_exe(), "-m", str(DIR / gguf), "--host", "127.0.0.1", "--port", str(port),
               "-c", "4096", "-np", "1", "-t", str(threads), "--no-webui"]
        self._log = open(DIR / "llama-server.log", "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     creationflags=flags)
        atexit.register(self._kill)
        log.info("llama-server pid %s: %s, %d threads", self.proc.pid, gguf, threads)

    async def _wait_llm(self, timeout: float = 180):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.proc.poll() is not None:
                log.error("llama-server exited with code %s, see %s", self.proc.returncode, DIR / "llama-server.log")
                return
            try:
                r = await self.http.get(f"{self.chat_url}/health", timeout=2)
                if r.status_code == 200:
                    self.ok["llm"] = True
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
        log.error("llama-server did not start within %.0f s", timeout)

    def _load_asr(self):
        import sherpa_onnx
        d = DIR / ASR[self.cfg.offline_asr]
        threads = _threads(self.cfg)[1]
        try:
            if self.cfg.offline_asr == "parakeet":
                self.asr = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(next(d.glob("encoder*.onnx"))), decoder=str(next(d.glob("decoder*.onnx"))),
                    joiner=str(next(d.glob("joiner*.onnx"))), tokens=str(d / "tokens.txt"),
                    model_type="nemo_transducer", num_threads=threads)
                self.ok["asr"] = True
                return
            if self.cfg.offline_asr == "parakeet-small":
                self.asr = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=str(next(d.glob("model*.onnx"))), tokens=str(d / "tokens.txt"), num_threads=threads)
                self.ok["asr"] = True
                return
            self.asr = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=str(d / "encoder_model.ort"), decoder=str(d / "decoder_model_merged.ort"),
                tokens=str(d / "tokens.txt"), num_threads=_threads(self.cfg)[1])
            self.ok["asr"] = True
        except Exception as e:
            log.error("hearing model failed to load: %s", e)

    def _load_tts(self):
        import sherpa_onnx
        name = self.cfg.offline_tts
        d = DIR / TTS[name]
        try:
            if name == "kokoro":
                model = sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(next(d.glob("model*.onnx"))), voices=str(d / "voices.bin"), tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data")))
                voice = self.cfg.offline_voice
                if voice not in KOKORO_VOICES:
                    log.warning("offline_voice %r is not a Kokoro voice, using am_michael", voice)
                    voice = "am_michael"
                self.sid = KOKORO_VOICES.index(voice)
            else:
                onnx = next(d.glob("*.onnx"))
                model = sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(onnx), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")))
                self.sid = 0
            model.num_threads = _threads(self.cfg)[1]
            self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1))
            self.ok["tts"] = True
        except Exception as e:
            log.error("voice model failed to load: %s", e)

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log:
            self._log.close()
            self._log = None

    async def close(self):
        self._kill()
        await self.http.aclose()
        await self.web.aclose()

    async def check(self) -> dict:
        await self.start()
        return {"asr": self.ok["asr"] and self.ok["llm"], "tts": self.ok["tts"]}

    # ---------------- hearing ----------------
    @staticmethod
    def audio_part(wav: bytes) -> dict:
        return {"type": "input_wav", "wav": wav}

    async def hear(self, messages: list, schema: dict | None = None) -> str:
        wav = next(p["wav"] for m in messages if isinstance(m["content"], list)
                   for p in m["content"] if p.get("type") == "input_wav")
        return await asyncio.to_thread(self.transcribe, *_read_wav(wav))

    def transcribe(self, samples: np.ndarray, rate: int) -> str:
        if self.asr is None:
            raise RuntimeError("offline hearing model is not loaded")
        with self._asr_lock:
            st = self.asr.create_stream()
            st.accept_waveform(rate, samples)
            self.asr.decode_stream(st)
            text = st.result.text.strip()
        return text if any(c.isalnum() for c in text) else ""

    # ---------------- voice ----------------
    async def speak_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        if self.tts is None:
            raise RuntimeError("offline voice model is not loaded")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def on_chunk(samples, progress):
            loop.call_soon_threadsafe(q.put_nowait, np.array(samples, dtype=np.float32))
            return 0 if stop.is_set() else 1          # 0 stops generating (the phrase was cancelled)

        def run():
            try:
                with self._tts_lock:
                    if not stop.is_set():
                        self.tts.generate(text, sid=self.sid, speed=self.cfg.offline_speed, callback=on_chunk)
            except Exception as e:
                log.warning("offline voice failed for %r: %s", text, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        rate = self.tts.sample_rate
        rs = soxr.ResampleStream(rate, self.cfg.tts_rate, 1, dtype="float32") if rate != self.cfg.tts_rate else None
        loop.run_in_executor(None, run)
        try:
            while (x := await q.get()) is not None:
                y = rs.resample_chunk(x) if rs else x
                if len(y):
                    yield y
            if rs:
                tail = rs.resample_chunk(np.zeros(0, np.float32), last=True)
                if len(tail):
                    yield tail
        finally:
            stop.set()

    # ---------------- warm-up ----------------
    async def warm_up(self):
        """Load everything and put the planner's fixed prompt into llama.cpp's cache, so the first turn is fast."""
        await self.start()
        try:
            t = time.perf_counter()
            if self.ok["asr"]:
                await asyncio.to_thread(self.transcribe, np.zeros(8000, np.float32), 16000)
            if self.ok["tts"]:
                async for _ in self.speak_stream("OK."):
                    pass
            if self.ok["llm"]:
                msgs = brain.plan_messages(self.cfg, {}, [], "hello", static_prefix=True)
                async for _ in self.chat_stream(msgs, brain.plan_schema(self.cfg.planner_say_first), max_tokens=1):
                    pass
            log.info("offline warm-up %.1f s", time.perf_counter() - t)
        except Exception as e:
            log.warning("offline warm-up failed: %s", e)


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit WAV, got {8 * width}-bit")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), rate


async def bench(cfg, s: LocalServers):
    """Speed of each offline part on this machine (python -m fda check)."""
    print(f"machine: {platform.machine()}, {os.cpu_count()} CPU threads, planner threads {_threads(cfg)[0]}")
    t = time.perf_counter(); await s.warm_up()
    print(f"load + warm-up        : {time.perf_counter() - t:5.1f} s")
    if s.ok["asr"]:
        test = next((DIR / ASR[cfg.offline_asr]).glob("test_wavs/*.wav"), None)
        if test:
            x, rate = _read_wav(test.read_bytes())
            t = time.perf_counter(); text = await asyncio.to_thread(s.transcribe, x, rate)
            print(f"hearing ({len(x) / rate:.1f} s audio) : {(time.perf_counter() - t) * 1000:5.0f} ms  {text[:60]!r}")
    if s.ok["llm"]:
        room = {"lights": {"power": "on", "brightness": 80, "color": "warm white"}, "thermostat_c": 24}
        for heard in ("make the lights blue and set a timer for two minutes", "tell me a joke"):
            msgs = brain.plan_messages(cfg, room, [], heard, static_prefix=True)
            t = time.perf_counter(); first_say = None; out = ""
            async for ev, v in s.plan(msgs, brain.plan_schema(cfg.planner_say_first)):
                if ev == "say" and first_say is None:
                    first_say = time.perf_counter() - t
                if ev == "done":
                    out = v
            total = time.perf_counter() - t
            print(f"planner first word    : {(first_say or total) * 1000:5.0f} ms, whole answer {total * 1000:5.0f} ms"
                  f"  ({heard[:28]}...) -> say={out.get('say', '')[:50]!r} do={[a.get('tool') for a in out.get('do', [])]}")
    if s.ok["tts"]:
        phrase = "Lights are blue, and your two minute timer is running."
        t = time.perf_counter(); first = None; n = 0
        async for pcm in s.speak_stream(phrase):
            first = first or time.perf_counter() - t
            n += len(pcm)
        gen = time.perf_counter() - t
        print(f"voice first audio     : {first * 1000:5.0f} ms, {n / cfg.tts_rate:.1f} s of speech made in {gen:.1f} s"
              f" ({n / cfg.tts_rate / gen:.1f}x real time)")
