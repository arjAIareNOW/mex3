"""The school DGX3 server (Qwen2-Audio-7B for speech to text, Qwen3-TTS-1.7B for the voice), used by the asr and
tts MCP servers as engine "dgx3". Needs QWEN_API_KEY in .env and the school network or VPN.

When the DGX3 does not answer (VPN off, server down), the MCP servers fall back to Moonshine / Piper on this
machine. A failure "opens" the breaker: the DGX3 is tried again only after `retry_s`, so a dropped VPN does not
add a timeout to every turn.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
import wave

import httpx
import numpy as np

log = logging.getLogger("fda.mcp.dgx3")

ASR_PROMPT = "Transcribe the speech exactly, in the language spoken. Output only the words, nothing else."
# what Qwen2-Audio says when there are no words (noise, a cough): treat as nothing heard
_NO_SPEECH = re.compile(r"^\W*(no speech|there is no|the audio (is|contains|does)|silence|\[?(noise|music|inaudible)"
                        r"|i('m| am) sorry|sorry,? i)", re.I)


_PREAMBLE = re.compile(r"^\s*(?:the\s+)?(?:speech|audio|speaker|original content|transcription|transcript|text)"
                       r"[^:'\"“]{0,60}?(?:is|says?|reads?|as follows|:)\s*[:,]?\s*", re.I)


def clean_transcript(text: str) -> str:
    """Qwen2-Audio sometimes answers "The speech is transcribed as follows: 'turn the lights off.'"."""
    t = text.strip()
    m = re.match(r"^[^'\"“]{0,90}?[:,]\s*['\"“](.+?)['\"”]\s*\.?\s*$", t, re.S)
    if m:
        t = m.group(1)
    else:
        t = _PREAMBLE.sub("", t, count=1) if _PREAMBLE.match(t) else t
    return t.strip().strip("'\"“”").strip()


def wav16(x: np.ndarray, rate: int) -> bytes:
    if rate != 16000 and len(x):
        x = np.interp(np.arange(0, len(x), rate / 16000), np.arange(len(x)), x)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


class DGX3:
    def __init__(self, cfg, retry_s: float = 30.0):
        self.cfg = cfg
        self.key = os.environ.get("QWEN_API_KEY", "")
        self.http = httpx.Client(headers={"Authorization": f"Bearer {self.key}"},
                                 timeout=httpx.Timeout(10.0, connect=0.8),
                                 limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300))
        self.retry_s = retry_s
        self.down_until = 0.0
        self.fails = 0

    @property
    def available(self) -> bool:
        return bool(self.key) and time.monotonic() >= self.down_until

    def _failed(self, what: str, e: Exception):
        self.fails += 1
        self.down_until = time.monotonic() + self.retry_s
        log.warning("DGX3 %s failed (%s: %s); using the local engine for %.0f s",
                    what, type(e).__name__, str(e)[:120], self.retry_s)

    def _ok(self):
        if self.fails:
            log.info("DGX3 is back")
        self.fails, self.down_until = 0, 0.0

    def check(self) -> bool:
        try:
            ok = all(self.http.get(f"{u}/v1/models", timeout=3).status_code == 200
                     for u in (self.cfg.asr_url, self.cfg.tts_url))
        except Exception:  # noqa: BLE001
            ok = False
        return ok

    def transcribe(self, x: np.ndarray, rate: int) -> str | None:
        """Text, "" for no words, or None if the DGX3 could not be used (then use the local engine)."""
        if not self.available:
            return None
        body = {"model": self.cfg.asr_model, "temperature": 0.0, "max_tokens": 160,
                "messages": [{"role": "user", "content": [
                    {"type": "audio_url",
                     "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(wav16(x, rate)).decode()}},
                    {"type": "text", "text": ASR_PROMPT}]}]}
        try:
            r = self.http.post(f"{self.cfg.asr_url}/v1/chat/completions", json=body)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            self._failed("speech to text", e)
            return None
        self._ok()
        text = clean_transcript(text)
        if not any(c.isalnum() for c in text) or _NO_SPEECH.match(text):
            return ""
        return text

    def speak(self, text: str, voice: str = "") -> np.ndarray | None:
        """24 kHz float32 audio, or None if the DGX3 could not be used."""
        if not self.available:
            return None
        body = {"model": self.cfg.tts_model, "input": text, "voice": voice or self.cfg.voice,
                "language": self.cfg.language, "response_format": "pcm", "stream": True, "stream_format": "audio",
                "sample_rate": self.cfg.tts_rate}
        if self.cfg.voice_style:
            body["instructions"] = self.cfg.voice_style
        try:
            buf = bytearray()
            with self.http.stream("POST", f"{self.cfg.tts_url}/v1/audio/speech", json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:200]!r}")
                for chunk in r.iter_bytes():
                    buf += chunk
        except Exception as e:  # noqa: BLE001
            self._failed("voice", e)
            return None
        self._ok()
        n = len(buf) // 2 * 2
        return np.frombuffer(bytes(buf[:n]), "<i2").astype(np.float32) / 32768.0
