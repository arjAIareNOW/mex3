"""Full-duplex audio core.

    speaker <- Mixer (voice | music | fx channels, smooth gain ramps) --+--> reference
    mic ------------------------------------------------------------------+--> AEC (WebRTC AEC3) -> 16 kHz clean audio

The device callback only copies blocks; a worker thread resamples mic + reference to 16 kHz (identical resamplers, so
they stay aligned) and runs AEC in 10 ms frames. Everything the speaker plays goes through the mixer, so the AEC
reference is always exactly what was played (voice, music, alarm tones, at the current volume).
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

log = logging.getLogger("fda.audio")
SR = 16000          # processing rate (VAD, AEC, ASR)
FRAME = 160         # 10 ms at 16 kHz


class Channel:
    """A queue of float32 audio at the device rate with a gain that ramps (no clicks)."""

    def __init__(self, name: str, rate: int, gain: float = 1.0):
        self.name, self.rate = name, rate
        self.chunks: collections.deque[np.ndarray] = collections.deque()
        self.gain = self.target = gain
        self.step = 0.0
        self.played = 0            # samples rendered since the last clear (for "what has been said")
        self.queued = 0
        self.loop: np.ndarray | None = None   # looping source (music, ringing)
        self.loop_pos = 0
        self.clear_when_silent = False
        self.lock = threading.Lock()

    def ramp(self, target: float, ms: float):
        with self.lock:
            self.target = float(target)
            n = max(1, int(self.rate * ms / 1000))
            self.step = (self.target - self.gain) / n

    def push(self, x: np.ndarray):
        with self.lock:
            self.chunks.append(np.asarray(x, np.float32))
            self.queued += len(x)

    def set_loop(self, x: np.ndarray | None):
        with self.lock:
            self.loop, self.loop_pos = (None if x is None else np.asarray(x, np.float32)), 0

    def clear(self):
        with self.lock:
            self.chunks.clear(); self.queued = 0; self.played = 0
            self.clear_when_silent = False

    def fade_out_and_clear(self, ms: float):
        """Fade to silence, then drop everything queued (barge-in)."""
        self.ramp(0.0, ms)
        with self.lock:
            self.clear_when_silent = True

    @property
    def busy(self) -> bool:
        return bool(self.chunks) or self.loop is not None

    def render(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self.lock:
            filled = 0
            if self.loop is not None and len(self.loop):
                while filled < n:
                    k = min(n - filled, len(self.loop) - self.loop_pos)
                    out[filled:filled + k] = self.loop[self.loop_pos:self.loop_pos + k]
                    filled += k; self.loop_pos = (self.loop_pos + k) % len(self.loop)
            else:
                while filled < n and self.chunks:
                    c = self.chunks[0]; k = min(n - filled, len(c))
                    out[filled:filled + k] = c[:k]; filled += k
                    if k == len(c):
                        self.chunks.popleft()
                    else:
                        self.chunks[0] = c[k:]
                self.played += filled; self.queued -= filled
            # gain ramp
            if self.step != 0.0:
                g = self.gain + self.step * np.arange(1, n + 1, dtype=np.float32)
                g = np.minimum(g, self.target) if self.step > 0 else np.maximum(g, self.target)
                self.gain = float(g[-1])
                if self.gain == self.target:
                    self.step = 0.0
                out *= g
            else:
                out *= self.gain
            if self.clear_when_silent and self.gain <= 1e-4:
                self.chunks.clear(); self.queued = 0; self.clear_when_silent = False
                self.gain = self.target = 1.0; self.step = 0.0
        return out


class Mixer:
    def __init__(self, rate: int):
        self.rate = rate
        self.voice = Channel("voice", rate)
        self.music = Channel("music", rate, gain=0.6)
        self.fx = Channel("fx", rate, gain=0.8)
        self.volume = 0.8           # master volume (VOLUME_UP / VOLUME_DOWN)

    def render(self, n: int) -> np.ndarray:
        y = (self.voice.render(n) + self.music.render(n) + self.fx.render(n)) * self.volume
        return np.tanh(y) if np.abs(y).max(initial=0) > 0.95 else y   # soft limit


class AudioEngine:
    """Owns the device (or a simulated one), the mixer and the AEC thread.

    on_audio(clean16k, ref16k) is called from the worker thread for every block: the echo-cancelled mic and the
    reference (exactly what was played) for the same samples, both float32 at 16 kHz.
    """

    def __init__(self, cfg, on_audio: Callable[[np.ndarray], None]):
        self.cfg = cfg
        self.on_audio = on_audio
        self.rate = cfg.device_rate
        self.mixer = Mixer(self.rate)
        self.q: queue.Queue = queue.Queue(maxsize=500)
        self.delay_ms = cfg.aec_delay_ms if cfg.aec_delay_ms is not None else 40
        self._stop = threading.Event()
        self.stream = None
        self.apm = None
        if cfg.aec:
            from livekit import rtc
            self._rtc = rtc
            self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=cfg.noise_suppression,
                                                 high_pass_filter=True, auto_gain_control=False)
        self.rec = None
        if cfg.record_dir:
            d = Path(cfg.record_dir); d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.rec = {k: wave.open(str(d / f"{stamp}-{k}.wav"), "wb") for k in ("mic", "ref", "clean")}
            for w in self.rec.values():
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            log.info("recording to %s/%s-*.wav", d, stamp)
        self.dropped = 0

    # ---- device side ----
    def callback(self, mic: np.ndarray, n: int) -> np.ndarray:
        """Called by the (real or simulated) device for every block: returns what to play."""
        out = self.mixer.render(n)
        try:
            self.q.put_nowait((mic.copy(), out.copy()))
        except queue.Full:
            self.dropped += 1
        return out

    def _start_worker(self):
        self._worker = threading.Thread(target=self._process, name="aec", daemon=True)
        self._worker.start()

    def start(self):
        if getattr(self.cfg, "_simulated", False):
            self._start_worker()
            return
        import sounddevice as sd
        dev = (_find_device(self.cfg.input_device, "input"), _find_device(self.cfg.output_device, "output"))
        try:
            sd.query_devices(dev[0], "input"); sd.query_devices(dev[1], "output")
        except Exception:
            raise SystemExit("No microphone/speaker found. Plug them in (python -m fda devices lists what is "
                             "available), or try without hardware: python -m fda --sim")
        rate = self.rate
        try:
            sd.check_input_settings(device=dev[0], samplerate=rate, channels=1, dtype="float32")
            sd.check_output_settings(device=dev[1], samplerate=rate, channels=1, dtype="float32")
        except Exception:
            rate = int(sd.query_devices(dev[1], "output")["default_samplerate"])
            log.warning("device rate %s not supported, using %s", self.rate, rate)
            self.rate = self.mixer.rate = self.mixer.voice.rate = self.mixer.music.rate = self.mixer.fx.rate = rate
        out_ch = 1
        try:
            sd.check_output_settings(device=dev[1], samplerate=rate, channels=1, dtype="float32")
        except Exception:
            out_ch = 2

        def cb(indata, outdata, frames, time_info, status):
            if status:
                log.debug("audio status: %s", status)
            y = self.callback(indata[:, 0], frames)
            outdata[:] = y[:, None] if out_ch == 1 else np.repeat(y[:, None], out_ch, axis=1)

        self._start_worker()   # after the rate is final
        self.stream = sd.Stream(device=dev, samplerate=rate, blocksize=int(rate * self.cfg.block_ms / 1000),
                                channels=(1, out_ch), dtype="float32", latency="low", callback=cb)
        self.stream.start()
        if self.cfg.aec_delay_ms is None:
            lin, lout = self.stream.latency
            self.delay_ms = int((lin + lout) * 1000)
        log.info("audio: in=%s out=%s rate=%s latency=%s -> AEC delay %d ms", dev[0], dev[1], rate,
                 self.stream.latency, self.delay_ms)

    def stop(self):
        self._stop.set()
        if self.stream is not None:
            self.stream.stop(); self.stream.close()
        if self.rec:
            for w in self.rec.values():
                w.close()

    # ---- worker: resample + AEC ----
    def _process(self):
        rs_mic = soxr.ResampleStream(self.rate, SR, 1, dtype="float32")
        rs_ref = soxr.ResampleStream(self.rate, SR, 1, dtype="float32")
        mic_buf = np.zeros(0, np.float32); ref_buf = np.zeros(0, np.float32)
        while not self._stop.is_set():
            try:
                mic, ref = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.rate != SR:
                mic = rs_mic.resample_chunk(mic); ref = rs_ref.resample_chunk(ref)
            mic_buf = np.concatenate([mic_buf, mic]); ref_buf = np.concatenate([ref_buf, ref])
            n = (min(len(mic_buf), len(ref_buf)) // FRAME) * FRAME
            if n == 0:
                continue
            m, r = mic_buf[:n], ref_buf[:n]
            mic_buf, ref_buf = mic_buf[n:], ref_buf[n:]
            clean = self._aec(m, r) if self.apm is not None else m
            if self.rec:
                for k, x in (("mic", m), ("ref", r), ("clean", clean)):
                    self.rec[k].writeframes(_to_i16(x).tobytes())
            try:
                self.on_audio(clean, r)
            except Exception:
                log.exception("on_audio failed")

    def _aec(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        rtc = self._rtc
        out = np.empty_like(mic)
        mi, ri = _to_i16(mic), _to_i16(ref)
        self.apm.set_stream_delay_ms(int(self.delay_ms))
        for i in range(0, len(mic), FRAME):
            rf = rtc.AudioFrame(ri[i:i + FRAME].tobytes(), SR, 1, FRAME)
            self.apm.process_reverse_stream(rf)
            mf = rtc.AudioFrame(mi[i:i + FRAME].tobytes(), SR, 1, FRAME)
            self.apm.process_stream(mf)
            out[i:i + FRAME] = np.frombuffer(mf.data, np.int16).astype(np.float32) / 32768.0
        return out


def _to_i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _find_device(spec, kind: str):
    if spec is None or isinstance(spec, int):
        return spec
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if spec.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0:
            return i
    raise SystemExit(f"no {kind} device matching {spec!r}; run: python -m fda devices")


def level_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)) if len(x) else -120.0


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    return x if src == dst else soxr.resample(x, src, dst).astype(np.float32)


def wav_bytes(x16k: np.ndarray, rate: int = SR) -> bytes:
    import io
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(_to_i16(x16k).tobytes())
    return b.getvalue()


def read_wav(path: str | Path, rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2 ** 31
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"unsupported WAV sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, sr, rate)
