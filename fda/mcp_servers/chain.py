"""Speech engines in a chain: `asr_engine = "groq,dgx3,moonshine"` means try Whisper on Groq first, then the school
DGX3, then Moonshine on this machine. An engine that fails (no network, VPN off, rate limit) is skipped for
`retry_s` seconds, so a dead service does not add a timeout to every turn. The chain always ends with an offline
engine, so FDA keeps listening and talking without internet.

ASR engines: moonshine (moonshine-base), moonshine-tiny, parakeet, parakeet-small (offline); dgx3; groq (Whisper
large-v3-turbo); openai (any OpenAI-compatible /audio/transcriptions: ASR_BASE_URL, ASR_API_KEY, ASR_MODEL).
TTS engines: piper, kokoro (offline); dgx3; deepgram (DEEPGRAM_API_KEY).
"""
from __future__ import annotations

import copy
import io
import logging
import os
import re
import time
import wave

import numpy as np

log = logging.getLogger("fda.mcp.chain")
LOCAL_ASR = {"moonshine": "moonshine-base", "moonshine-base": "moonshine-base", "moonshine-tiny": "moonshine-tiny",
             "parakeet": "parakeet", "parakeet-small": "parakeet-small"}
LOCAL_TTS = {"piper", "kokoro"}
CLOUD_ASR = {"groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "whisper-large-v3-turbo"),
             "openai": (None, "ASR_API_KEY", None)}


def split(spec: str) -> list[str]:
    return [x.strip().lower() for x in str(spec).split(",") if x.strip()]


def _wav(x: np.ndarray, rate: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


class Backend:
    """One engine. call() returns a result, or None when this engine could not be used (then try the next)."""
    retry_s = 30.0

    def __init__(self, name: str):
        self.name, self.down_until, self.fails = name, 0.0, 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.down_until

    def failed(self, what: Exception | str):
        self.fails += 1
        self.down_until = time.monotonic() + self.retry_s
        log.warning("%s failed (%s); skipped for %.0f s", self.name, str(what)[:120], self.retry_s)

    def ok(self):
        if self.fails:
            log.info("%s is back", self.name)
        self.fails, self.down_until = 0, 0.0


# ---------------- speech to text ----------------
class LocalASR(Backend):
    def __init__(self, cfg, name: str):
        super().__init__(name)
        from ..local import ASR, DIR, LocalServers
        c = copy.copy(cfg)
        c.offline_asr = LOCAL_ASR[name]
        if not (DIR / ASR[c.offline_asr]).exists():
            raise RuntimeError(f"model {c.offline_asr} is not downloaded (python -m fda download)")
        self.local = LocalServers(c)
        self.local._load_asr()
        if self.local.asr is None:
            raise RuntimeError(f"{name} failed to load")
        self.model = c.offline_asr

    def __call__(self, x, rate, language):
        return self.local.transcribe(x, rate)        # offline: never "down"


class DGX3ASR(Backend):
    def __init__(self, cfg):
        super().__init__("dgx3")
        from .dgx3 import DGX3
        self.dgx, self.model = DGX3(cfg, retry_s=self.retry_s), cfg.asr_model
        if not self.dgx.key:
            raise RuntimeError("QWEN_API_KEY is not set")
        if self.dgx.check():
            self.dgx.down = False
        else:
            log.warning("DGX3 is not reachable now (school network / VPN?); using the next engine until it is")

    @property
    def available(self):
        return self.dgx.available

    def __call__(self, x, rate, language):
        return self.dgx.transcribe(x, rate)           # None = could not be used (its own breaker)


class CloudASR(Backend):
    def __init__(self, name: str, hint: str = ""):
        super().__init__(name)
        self.hint = hint.strip()          # Whisper's prompt: names it should spell right (the agent's, places)
        self._hint_words = set(re.findall(r"[a-z0-9']+", self.hint.lower()))
        import httpx
        base, key_env, model = CLOUD_ASR[name]
        self.base = (base or os.environ.get("ASR_BASE_URL", "")).rstrip("/")
        self.key = os.environ.get(key_env, "")
        self.model = os.environ.get("ASR_MODEL") or model
        if not self.base or not self.key or not self.model:
            raise RuntimeError(f"set {key_env}" + (", ASR_BASE_URL, ASR_MODEL" if not base else "") + " in .env")
        self.http = httpx.Client(timeout=httpx.Timeout(8.0, connect=1.5),
                                 headers={"Authorization": f"Bearer {self.key}"})

    def __call__(self, x, rate, language):
        try:
            r = self.http.post(f"{self.base}/audio/transcriptions",
                               files={"file": ("turn.wav", _wav(x, rate), "audio/wav")},
                               data={"model": self.model, "language": language, "response_format": "json",
                                     "temperature": "0", **({"prompt": self.hint} if self.hint else {})})
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            text = (r.json().get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            self.failed(e)
            return None
        self.ok()
        words = set(re.findall(r"[a-z0-9']+", text.lower()))
        if self._hint_words and len(words) >= 2 and words <= self._hint_words:
            return ""                     # Whisper said its prompt back on noise (just the name alone still counts)
        return text if any(c.isalnum() for c in text) else ""


def asr_backend(cfg, name: str) -> Backend:
    if name in LOCAL_ASR:
        return LocalASR(cfg, name)
    if name == "dgx3":
        return DGX3ASR(cfg)
    if name in CLOUD_ASR:
        hint = ", ".join(x for x in (getattr(cfg, "name", ""), getattr(cfg, "location", ""),
                                     getattr(cfg, "asr_hint", "")) if x)
        return CloudASR(name, hint=hint + "." if hint else "")
    raise RuntimeError(f"unknown asr engine {name!r}: {', '.join([*LOCAL_ASR, 'dgx3', *CLOUD_ASR])}")


# ---------------- text to speech ----------------
class LocalTTS(Backend):
    def __init__(self, cfg, name: str):
        super().__init__(name)
        from ..local import DIR, TTS, LocalServers
        c = copy.copy(cfg)
        c.offline_tts = name
        if not (DIR / TTS[name]).exists():
            raise RuntimeError(f"voice {name} is not downloaded (set offline_tts = \"{name}\"; python -m fda download)")
        self.local = LocalServers(c)
        self.local._load_tts()
        if self.local.tts is None:
            raise RuntimeError(f"{name} failed to load")
        self.rate = self.local.tts.sample_rate
        self.speed = cfg.offline_speed

    def __call__(self, text, voice, speed):
        a = self.local.tts.generate(text, sid=self.local.sid, speed=speed or self.speed)
        return np.asarray(a.samples, np.float32), self.rate


class DGX3TTS(Backend):
    def __init__(self, cfg):
        super().__init__("dgx3")
        from .dgx3 import DGX3
        self.dgx, self.rate = DGX3(cfg, retry_s=self.retry_s), cfg.tts_rate
        if not self.dgx.key:
            raise RuntimeError("QWEN_API_KEY is not set")
        if self.dgx.check():
            self.dgx.down = False
        else:
            log.warning("DGX3 is not reachable now (school network / VPN?); using the next voice until it is")

    @property
    def available(self):
        return self.dgx.available

    def __call__(self, text, voice, speed):
        x = self.dgx.speak(text, voice)
        return None if x is None else (x, self.rate)


class DeepgramTTS(Backend):
    def __init__(self):
        super().__init__("deepgram")
        import httpx
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError("set DEEPGRAM_API_KEY in .env")
        self.http = httpx.Client(timeout=httpx.Timeout(10.0, connect=1.5), headers={"Authorization": f"Token {key}"})
        self.rate = 24000

    def __call__(self, text, voice, speed):
        try:
            r = self.http.post("https://api.deepgram.com/v1/speak",
                               params={"model": os.environ.get("DEEPGRAM_VOICE", "aura-2-orion-en"),
                                       "encoding": "linear16", "sample_rate": self.rate, "container": "none"},
                               json={"text": text})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            self.failed(e)
            return None
        self.ok()
        return np.frombuffer(r.content, "<i2").astype(np.float32) / 32768.0, self.rate


def tts_backend(cfg, name: str) -> Backend:
    if name in LOCAL_TTS:
        return LocalTTS(cfg, name)
    if name == "dgx3":
        return DGX3TTS(cfg)
    if name == "deepgram":
        return DeepgramTTS()
    raise RuntimeError(f"unknown tts engine {name!r}: piper, kokoro, dgx3, deepgram")


# ---------------- the chains ----------------
class Chain:
    def __init__(self, cfg, spec: str, make, offline: str, local_names, kind: str):
        names = split(spec)
        if not any(n in local_names for n in names):
            names.append(offline)                         # always end offline: FDA keeps working without internet
        self.backends: list[Backend] = []
        for n in names:
            try:
                self.backends.append(make(cfg, n))
            except Exception as e:  # noqa: BLE001
                log.warning("%s engine %s skipped: %s", kind, n, e)
        if not self.backends:
            raise SystemExit(f"no {kind} engine could start (asked for: {', '.join(names)})")
        self.last = ""
        log.info("%s chain: %s", kind, " -> ".join(b.name for b in self.backends))

    def info(self) -> dict:
        return {"chain": [{"engine": b.name, "up": b.available, **({"model": b.model} if hasattr(b, "model") else {})}
                          for b in self.backends], "last_used": self.last}


class ASRChain(Chain):
    def __init__(self, cfg, spec: str):
        super().__init__(cfg, spec, asr_backend, cfg.offline_asr if cfg.offline_asr in LOCAL_ASR else "moonshine",
                         LOCAL_ASR, "asr")

    def transcribe(self, x: np.ndarray, rate: int, language: str = "en") -> str:
        for b in self.backends:
            if not b.available:
                continue
            text = b(x, rate, language)
            if text is not None:
                self.last = b.name
                if language == "en" and text:
                    letters = [c for c in text if c.isalpha()]
                    if letters and sum(c.isascii() for c in letters) < 0.6 * len(letters):
                        return ""             # a mumble written as Chinese etc.: not words for an English agent
                    if any(w in text.lower().split() for w in ("não", "é", "você")):
                        return ""
                return text
        self.last = "none"
        return ""


class TTSChain(Chain):
    def __init__(self, cfg, spec: str):
        super().__init__(cfg, spec, tts_backend, cfg.offline_tts if cfg.offline_tts in LOCAL_TTS else "piper",
                         LOCAL_TTS, "tts")
        self.rate = self.backends[0].rate

    def synthesize(self, text: str, voice: str = "", speed: float = 0.0) -> tuple[np.ndarray, int]:
        for b in self.backends:
            if not b.available:
                continue
            out = b(text, voice, speed)
            if out is not None:
                self.last = b.name
                return out
        self.last = "none"
        return np.zeros(0, np.float32), self.rate
