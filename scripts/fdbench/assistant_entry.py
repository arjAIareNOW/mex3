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
import sys
from pathlib import Path

PROJECT_MARKER = "fda/pc/agent.py"
# touched by:  --reset-cmd "python tools/fda_reset.py"  (a fresh session between trials)
RESET_FILE = Path(__file__).resolve().parent / "runs" / ".fda_reset"


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

    async def run():
        reset_task = asyncio.create_task(watch_reset())
        try:
            await runner.add_workers(worker)
            await runner.run()
        finally:
            reset_task.cancel()
            await agent.close()

    async def cancel():
        await worker.cancel()

    return run(), cancel
