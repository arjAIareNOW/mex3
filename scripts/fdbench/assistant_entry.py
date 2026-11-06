"""fd-benchmark entry for FDA on Pipecat + MCP (fda/pc).

The launcher hands us its Pipecat transport (16 kHz audio to/from fdbench over a socket); the agent pipeline is
built exactly as in normal use (fda/pc/agent.py), with that transport in place of FDA's mic/speaker engine.
The asr / tts / tools MCP servers are started as separate processes if they are not running yet.

Check:   ..\\.venv\\Scripts\\python launch_assistant.py --check        (must print READY)
Zero tokens: llm_engine = "local", asr_engine = "moonshine", tts_engine = "piper" (the defaults).
The classic (non-Pipecat) FDA can still be benchmarked with  python -m fda bench  (fda/benchlink.py).
"""
from __future__ import annotations

import asyncio
import os
import platform
import sys
from pathlib import Path

PROJECT_MARKER = "fda/pc/agent.py"
# touched by:  --reset-cmd "python tools/fda_reset.py"  (a fresh session between trials)
RESET_FILE = Path(__file__).resolve().parent / "runs" / ".fda_reset"
# Elkit on the Pi, fdbench on the laptop: the reset comes over HTTP instead
#   --reset-cmd "python tools/fda_reset.py http://<pi>:8792/reset"
RESET_PORT = int(os.environ.get("FDA_RESET_PORT", "8792"))


def device_info() -> str:
    """For the latency report: what this runs on (RPi model, RAM, OS)."""
    model = ""
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            model = Path(p).read_text(errors="ignore").strip("\x00\n ")
            break
        except OSError:
            pass
    ram = ""
    try:
        kb = int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemTotal")))
        ram = f"{kb / 1024 / 1024:.0f} GB RAM"
    except (OSError, StopIteration, ValueError):
        pass
    return ", ".join(x for x in (model or platform.node(), ram, f"{platform.system()} {platform.release()}",
                                 f"Python {platform.python_version()}") if x)


def audio_rates() -> tuple[int, int]:
    return 16000, 16000


async def build(transport, log_dir: Path):
    from fda.config import Config
    from fda.pc.__main__ import setup_logging
    from fda.pc.agent import Agent
    from fda.pc.launch import start_mcp_servers
    from fda.pc.transport import NullMixerTransport
    from pipecat.pipeline.runner import WorkerRunner

    cfg = Config.load()
    cfg.open_browser = False
    cfg.say_ready = False
    cfg.state_file = ""          # test clips ("set an alarm for 6:30") must never reach the real saved alarms / lists
    setup_logging(cfg.log_level.upper())
    from loguru import logger
    logger.info(f"device: {device_info()}")
    urls = await asyncio.to_thread(start_mcp_servers, cfg)
    agent = Agent(cfg, transport, NullMixerTransport(cfg), urls, log_dir=Path(log_dir), out_rate=16000)
    worker = await agent.build()
    runner = WorkerRunner(handle_sigint=not sys.platform.startswith("win"))
    print("FDA (Pipecat + MCP) ready", flush=True)

    async def watch_reset():
        # each trial is its own conversation (as in the papers): fdbench runs  --reset-cmd  between trials, which
        # touches RESET_FILE; then the history, views, video, timers and music start fresh
        last = RESET_FILE.stat().st_mtime if RESET_FILE.exists() else 0.0
        while True:
            await asyncio.sleep(0.1)
            m = RESET_FILE.stat().st_mtime if RESET_FILE.exists() else 0.0
            if m != last:
                last = m
                await agent.fresh_session()

    async def reset_server():
        # the same reset over HTTP, for fdbench on another machine (it cannot touch our RESET_FILE)
        from aiohttp import web

        async def reset(_req):
            await agent.fresh_session()
            return web.Response(text="ok")
        app = web.Application()
        app.router.add_get("/reset", reset)
        app.router.add_post("/reset", reset)
        r = web.AppRunner(app)
        await r.setup()
        try:
            await web.TCPSite(r, "0.0.0.0", RESET_PORT).start()
            print(f"reset: http://<this machine>:{RESET_PORT}/reset", flush=True)
        except OSError as e:
            print(f"reset over HTTP not available ({e})", flush=True)
        return r

    async def run():
        reset_task = asyncio.create_task(watch_reset())
        web_runner = await reset_server()
        try:
            await runner.add_workers(worker)
            await runner.run()
        finally:
            reset_task.cancel()
            await web_runner.cleanup()
            await agent.close()

    async def cancel():
        await worker.cancel()

    return run(), cancel
