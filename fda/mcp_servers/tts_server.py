"""MCP server: text to speech. Runs on the Pi (or laptop); the agent calls its `synthesize` tool once per phrase.

    python -m fda.mcp_servers.tts_server                     # HTTP on 127.0.0.1:8102/mcp (default)
    python -m fda.mcp_servers.tts_server --transport stdio

Engines (setting `tts_engine` in fda.toml, or --engine), one or several in order, e.g. "dgx3,kokoro,piper":
  piper     Piper "ryan" on this machine: free, offline, ~30x real time, fine on the Pi            (default)
  kokoro    Kokoro-82M on this machine: much more natural, ~3x real time on a laptop (slow on a Pi)
  dgx3      Qwen3-TTS-1.7B on the school DGX3 (QWEN_API_KEY, school network/VPN; voice = `voice`)
  deepgram  Deepgram Aura (free sign-up credit; DEEPGRAM_API_KEY, DEEPGRAM_VOICE e.g. aura-2-orion-en)
An engine that fails is skipped for 30 s; the chain always ends with an offline one (offline_tts).

`synthesize` returns {"sample_rate": int, "audio_b64": base64 of 16-bit mono PCM}.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import time

import numpy as np

from mcp.server.mcpserver import MCPServer

from ..config import Config
from .chain import TTSChain

log = logging.getLogger("fda.mcp.tts")


def build(cfg: Config, engine: str) -> MCPServer:
    eng = TTSChain(cfg, engine)
    mcp = MCPServer("fda-tts", instructions="Text to speech for the FDA voice agent.")

    @mcp.tool()
    def synthesize(text: str, voice: str = "", speed: float = 0.0) -> dict:
        """Speak one phrase. Returns {"sample_rate", "audio_b64"} (base64 of 16-bit mono PCM)."""
        t = time.perf_counter()
        x, rate = eng.synthesize(text, voice, speed) if text.strip() else (np.zeros(0, np.float32), eng.rate)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        log.info("%s %r -> %.2f s audio in %.0f ms", getattr(eng, "last", engine), text[:60], len(x) / rate,
                 (time.perf_counter() - t) * 1000)
        return {"sample_rate": rate, "audio_b64": base64.b64encode(pcm).decode()}

    @mcp.tool()
    def tts_info() -> dict:
        """The voice engines in order, whether each is up now, and which one spoke last."""
        return {"engine": engine, "sample_rate": eng.rate, **eng.info()}

    return mcp


def main():
    ap = argparse.ArgumentParser(description="FDA TTS MCP server")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--transport", default="streamable-http", choices=["streamable-http", "stdio"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8102)
    args = ap.parse_args()
    cfg = Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp = build(cfg, args.engine or cfg.tts_engine)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
