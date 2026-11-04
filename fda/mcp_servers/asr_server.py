"""MCP server: speech to text. Runs on the Pi (or laptop); the agent calls its `transcribe` tool.

    python -m fda.mcp_servers.asr_server                     # HTTP on 127.0.0.1:8101/mcp (default)
    python -m fda.mcp_servers.asr_server --transport stdio   # as a subprocess of the agent

Engines (setting `asr_engine` in fda.toml, or --engine), one or several in order, e.g. "groq,dgx3,moonshine":
  moonshine       Moonshine v2 base on this machine: free, offline, fastest                     (default)
  moonshine-tiny  smaller and faster, less accurate
  parakeet        NVIDIA Parakeet TDT 0.6B v2: top open English accuracy, offline (laptop)
  parakeet-small  Parakeet TDT-CTC 110M: offline, fast enough for the Pi
  dgx3            Qwen2-Audio-7B on the school DGX3 (QWEN_API_KEY, school network/VPN)
  groq            Whisper large-v3-turbo on Groq (GROQ_API_KEY; free tier)
  openai          any OpenAI-compatible /audio/transcriptions (ASR_BASE_URL, ASR_API_KEY, ASR_MODEL)
An engine that fails is skipped for 30 s; the chain always ends with an offline one (offline_asr).
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
import wave

import numpy as np

from mcp.server.mcpserver import MCPServer

from ..config import Config
from .chain import ASRChain

log = logging.getLogger("fda.mcp.asr")


def decode_audio(audio_b64: str, sample_rate: int) -> np.ndarray:
    """base64 of raw int16 mono PCM, or of a WAV file. Returns float32 at its own rate."""
    raw = base64.b64decode(audio_b64)
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as w:
            sample_rate = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                x = x.reshape(-1, w.getnchannels()).mean(1)
        return x, sample_rate
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0, sample_rate


def _wav(x: np.ndarray, rate: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


def build(cfg: Config, engine: str) -> MCPServer:
    eng = ASRChain(cfg, engine)
    mcp = MCPServer("fda-asr", instructions="Speech to text for the FDA voice agent.")

    @mcp.tool()
    def transcribe(audio_b64: str, sample_rate: int = 16000, language: str = "en") -> str:
        """Transcribe one spoken turn. audio_b64: base64 of 16-bit mono PCM (or a WAV file). Returns the text."""
        t = time.perf_counter()
        x, rate = decode_audio(audio_b64, sample_rate)
        text = eng.transcribe(x, rate, language)
        log.info("%s %.2f s audio -> %r in %.0f ms", getattr(eng, "last", engine), len(x) / rate, text,
                 (time.perf_counter() - t) * 1000)
        return text

    @mcp.tool()
    def asr_info() -> dict:
        """The speech-to-text engines in order, whether each is up now, and which one answered last."""
        return {"engine": engine, **eng.info()}

    return mcp


def main():
    ap = argparse.ArgumentParser(description="FDA ASR MCP server")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--transport", default="streamable-http", choices=["streamable-http", "stdio"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    args = ap.parse_args()
    cfg = Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp = build(cfg, args.engine or cfg.asr_engine)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
