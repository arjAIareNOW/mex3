"""python -m fda.pc          FDA on Pipecat with MCP servers (mic + speaker, display on http://localhost:8765)
python -m fda.pc --sim    no hardware: simulated room (the bot hears its own echo, like a real speaker)

Starts the asr / tts / tools MCP servers (separate processes, see fda/pc/launch.py) unless they already run,
the local LLM if llm_engine = "local", the display, and the Pipecat pipeline (fda/pc/agent.py).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from loguru import logger

from ..config import Config


def setup_logging(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss.SSS} {name} {message}",
               filter=lambda r: not r["name"].startswith("pipecat.transports") or r["level"].no >= 30)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "aiohttp.access", "mcp", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run(cfg, sim: bool = False):
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.transports.base_transport import TransportParams

    from ..ui import UI
    from .agent import Agent
    from .launch import start_mcp_servers
    from .transport import FdaAudioTransport

    urls = await asyncio.to_thread(start_mcp_servers, cfg)
    ui = UI(cfg)
    await ui.start()
    out_rate = 24000
    transport = FdaAudioTransport(cfg, TransportParams(audio_in_enabled=True, audio_out_enabled=True,
                                                       audio_in_sample_rate=16000, audio_out_sample_rate=out_rate),
                                  sim=sim)
    agent = Agent(cfg, transport, transport, urls, ui=ui, out_rate=out_rate)
    transport.on_level = lambda db: ui.send({"type": "level", "db": round(db, 1)})
    worker = await agent.build()
    ui.agent = agent           # taps on the screen -> agent.room.execute
    ui.send({"type": "health", "asr": True, "tts": True})
    ui.send({"type": "status", "status": "listening"})
    runner = WorkerRunner(handle_sigint=not sys.platform.startswith("win"))
    if cfg.say_ready:
        await worker.queue_frames([__import__("pipecat.frames.frames", fromlist=["TTSSpeakFrame"]).TTSSpeakFrame(
            f"Hi, I am {cfg.name}. What should we build today?")])
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await agent.close()
        await ui.stop()


def main():
    ap = argparse.ArgumentParser(prog="fda.pc", description="FDA on Pipecat + MCP")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.no_browser:
        cfg.open_browser = False
    setup_logging(cfg.log_level.upper())
    try:
        asyncio.run(run(cfg, sim=args.sim))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
