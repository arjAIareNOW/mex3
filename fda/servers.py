"""Clients for the two DGX3 servers (vLLM 0.30, OpenAI-compatible), streaming and cancellable.

Qwen2-Audio (port 8000): the user's audio in, a JSON turn out (schema-constrained). The fields are parsed while the
JSON is still streaming, so FDA can react to "heard" and start speaking "say" before the response is complete.
Qwen3-TTS (port 8001): /v1/audio/speech with stream=true, stream_format=audio -> raw 24 kHz PCM16 chunks.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import AsyncIterator

import httpx
import numpy as np

log = logging.getLogger("fda.servers")


class Servers:
    kind = "server"
    static_prefix = False     # planner prompt layout (see brain.plan_messages)

    def __init__(self, cfg):
        self.cfg = cfg
        self.h = {"Authorization": f"Bearer {cfg.api_key}"}
        self.chat_url, self.chat_model = cfg.asr_url, cfg.asr_model
        self.http = httpx.AsyncClient(headers=self.h, timeout=httpx.Timeout(30.0, connect=5.0),
                                      limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=300))
        self.web = httpx.AsyncClient(timeout=10.0)   # internet (weather): never carries the DGX3 key

    async def close(self):
        await self.http.aclose()
        await self.web.aclose()

    async def check(self) -> dict:
        out = {}
        for name, url in (("asr", self.cfg.asr_url), ("tts", self.cfg.tts_url)):
            try:
                r = await self.http.get(f"{url}/v1/models", timeout=5)
                out[name] = r.status_code == 200
            except Exception:
                out[name] = False
        return out

    # ---------------- Qwen2-Audio ----------------
    async def chat_stream(self, messages: list, schema: dict | None = None, max_tokens: int = 300,
                          temperature: float = 0.2) -> AsyncIterator[str]:
        body = {"model": self.chat_model, "messages": messages, "stream": True,
                "max_tokens": max_tokens, "temperature": temperature}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "turn", "schema": schema}}
        async with self.http.stream("POST", f"{self.chat_url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"planner HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                c = d["choices"][0]["delta"].get("content") if d.get("choices") else None
                if c:
                    yield c

    @staticmethod
    def audio_part(wav: bytes) -> dict:
        return {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(wav).decode()}}

    async def hear(self, messages: list, schema: dict) -> str:
        buf = ""
        async for piece in self.chat_stream(messages, schema, max_tokens=120, temperature=0.0):
            buf += piece
            v, complete = partial_string(buf, "heard")
            if complete:          # stop reading as soon as the transcript string closes
                return v.strip()
        return (partial_string(buf, "heard")[0] or "").strip()

    async def plan(self, messages: list, schema: dict) -> AsyncIterator[tuple[str, object]]:
        """Planner JSON {for_me, do, say} streamed. Yields ("for_me", bool), ("do", list), ("say", growing str),
        ("say_end", str) as soon as the reply string closes (before "do" in say-first order), ("done", dict)."""
        buf, sent_for_me, sent_do, said, say_closed = "", False, False, "", False
        t0 = time.perf_counter()
        async for piece in self.chat_stream(messages, schema, max_tokens=250, temperature=0.0):
            buf += piece
            if not sent_for_me:
                m = re.search(r'"for_me"\s*:\s*(true|false)', buf)
                if m:
                    sent_for_me = True
                    yield "for_me", m.group(1) == "true"
            if sent_for_me and not sent_do and buf.find('"do"') != -1 and buf.find('"do"') < max(buf.find('"say"'), 0):
                m = re.search(r'"say"\s*:', buf)
                if m:   # do-first order: the "do" array is complete once the "say" key appears
                    sent_do = True
                    try:
                        yield "do", json.loads(buf[:m.start()].rstrip().rstrip(",") + "}").get("do", [])
                    except json.JSONDecodeError:
                        yield "do", []
            if sent_for_me and not say_closed:
                v, complete = partial_string(buf, "say")
                if v is not None and len(v) > len(said):
                    said = v
                    yield "say", said
                if complete:
                    say_closed = True
                    yield "say_end", said
        try:
            obj = json.loads(buf)
        except json.JSONDecodeError:
            log.warning("unparseable planner JSON: %r", buf[:400])
            obj = {"for_me": False, "say": said, "do": []}
        obj["_ms"] = (time.perf_counter() - t0) * 1000
        if not sent_do:          # say-first order: "do" is known at the end
            yield "do", obj.get("do", [])
        yield "done", obj

    # ---------------- Qwen3-TTS ----------------
    async def speak_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        body = {"model": self.cfg.tts_model, "input": text, "voice": self.cfg.voice, "language": self.cfg.language,
                "response_format": "pcm", "stream": True, "stream_format": "audio", "sample_rate": self.cfg.tts_rate}
        if self.cfg.voice_style:
            body["instructions"] = self.cfg.voice_style
        carry = b""
        async with self.http.stream("POST", f"{self.cfg.tts_url}/v1/audio/speech", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Qwen3-TTS HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for chunk in r.aiter_bytes():
                chunk = carry + chunk
                n = len(chunk) // 2 * 2
                carry = chunk[n:]
                if n:
                    yield np.frombuffer(chunk[:n], np.int16).astype(np.float32) / 32768.0

    async def warm_up(self):
        """One tiny request to each server so the first real turn is not slow."""
        try:
            async for _ in self.chat_stream([{"role": "user", "content": "Say OK."}], max_tokens=2):
                pass
            async for _ in self.speak_stream("OK."):
                pass
        except Exception as e:
            log.warning("warm-up failed: %s", e)


def dgx_reachable(cfg, timeout: float = 1.5) -> bool:
    """Both DGX3 ports accept a connection (fast check, no request)."""
    import socket
    from urllib.parse import urlparse
    for url in (cfg.asr_url, cfg.tts_url):
        u = urlparse(url)
        try:
            socket.create_connection((u.hostname, u.port or 80), timeout=timeout).close()
        except OSError:
            return False
    return True


def make_servers(cfg):
    """DGX3 or the offline models on this machine, from cfg.backend (server | offline | auto)."""
    use_server = cfg.backend == "server" or (cfg.backend == "auto" and cfg.api_key and dgx_reachable(cfg))
    if use_server:
        log.info("backend: DGX3 servers")
        return Servers(cfg)
    if cfg.backend == "auto":
        log.warning("backend: DGX3 %s, using the offline models",
                    "has no key (QWEN_API_KEY)" if not cfg.api_key else "is not reachable")
    from .local import LocalServers
    return LocalServers(cfg)


_ESC_TAIL = re.compile(r'\\(u[0-9a-fA-F]{0,3})?$')


def partial_string(buf: str, key: str) -> tuple[str | None, bool]:
    """Value of a JSON string field from a possibly incomplete JSON text: (value so far, is_complete)."""
    m = re.search(r'"%s"\s*:\s*"' % re.escape(key), buf)
    if not m:
        return None, False
    i = m.end(); j = i
    while j < len(buf):
        c = buf[j]
        if c == "\\":
            j += 2; continue
        if c == '"':
            return _unescape(buf[i:j]), True
        j += 1
    raw = _ESC_TAIL.sub("", buf[i:])
    return _unescape(raw), False


def _unescape(raw: str) -> str:
    try:
        return json.loads('"' + raw + '"')
    except json.JSONDecodeError:
        return raw.replace('\\"', '"').replace("\\n", " ")
