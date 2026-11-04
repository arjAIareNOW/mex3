"""End-to-end test of FDA on Pipecat + MCP in the simulated room (FDA's echo is played back into the 'mic').

    .venv\\Scripts\\python tests\\pc_sim.py                  # all scenarios
    .venv\\Scripts\\python tests\\pc_sim.py --only weather   # one

Each scenario speaks user lines (Piper voice through the tts MCP server, as the 'user') at set times and checks
what the agent did: answered, stayed silent, kept talking over a backchannel, stopped for an interruption, called
tools, built an app.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fda.config import Config  # noqa: E402

SCENARIOS = {
    "chat": [(1.0, "Hi there, how are you doing today?")],
    "weather": [(1.0, "What's the weather like right now?")],
    "lights": [(1.0, "Please turn the lights off.")],
    "not_for_me": [(1.0, "Honey, did you remember to buy the milk on your way home?")],
    "backchannel": [(1.0, "Tell me a short story about a brave little robot."), ("after_bot+2.0", "uh huh")],
    "interrupt": [(1.0, "Tell me a long story about the history of the moon landing."),
                  ("after_bot+2.0", "Wait, stop. What time is it?")],
    "app": [(1.0, "Can you make me a simple tip calculator app?")],
    "news": [(1.0, "What are the top news headlines today?")],
}


async def tts_audio(conn, text: str, rate: int) -> np.ndarray:
    r = await conn.call("synthesize", {"text": text})
    x = np.frombuffer(base64.b64decode(r["audio_b64"]), "<i2").astype(np.float32) / 32768.0
    return soxr.resample(x, r["sample_rate"], rate).astype(np.float32) * 0.8


async def run_scenario(name, lines, cfg, urls, wait_s=25.0):
    from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, FunctionCallInProgressFrame,
                                       InterruptionFrame, TranscriptionFrame, TTSTextFrame)
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.transports.base_transport import TransportParams

    from fda.pc.agent import Agent
    from fda.pc.services import McpConnection
    from fda.pc.transport import FdaAudioTransport

    log = []
    t0 = time.monotonic()

    class Obs(BaseObserver):
        async def on_push_frame(self, data):
            f = data.frame
            t = round(time.monotonic() - t0, 2)
            if isinstance(f, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame)):
                if not log or log[-1][1] != type(f).__name__:
                    log.append((t, type(f).__name__, ""))
            elif isinstance(f, TranscriptionFrame) and type(data.source).__name__ == "McpSTTService":
                log.append((t, "heard", f.text))
            elif isinstance(f, TTSTextFrame) and type(data.source).__name__ == "McpTTSService":
                log.append((t, "said", f.text))
            elif isinstance(f, FunctionCallInProgressFrame) and type(data.source).__name__ == "OpenAILLMService":
                log.append((t, "tool", f"{f.function_name}({f.arguments})"))

    transport = FdaAudioTransport(cfg, TransportParams(audio_in_enabled=True, audio_out_enabled=True,
                                                       audio_in_sample_rate=16000, audio_out_sample_rate=24000),
                                  sim=True)
    agent = Agent(cfg, transport, transport, urls, out_rate=24000)
    worker = await agent.build()
    worker.add_observer(Obs())
    runner = WorkerRunner(handle_sigint=False)
    run = asyncio.create_task(await_(runner, worker))
    tts = McpConnection(urls["tts"], "tts-user")
    eng = transport.engine
    await asyncio.sleep(1.5)
    t0 = time.monotonic()
    for when, text in lines:
        audio = await tts_audio(tts, text, eng.rate)
        if isinstance(when, str):            # "after_bot+2.0": seconds after the bot starts talking
            delay = float(when.split("+")[1])
            t_end = time.monotonic() + 30
            while not transport.bot_speaking and time.monotonic() < t_end:
                await asyncio.sleep(0.05)
            await asyncio.sleep(delay)
            eng.say_at(0.0, audio)
        else:
            await asyncio.sleep(max(0, when - (time.monotonic() - t0)))
            eng.say_at(0.0, audio)
        log.append((round(time.monotonic() - t0, 2), "user", text))
    t_end = time.monotonic() + wait_s
    while time.monotonic() < t_end:
        await asyncio.sleep(0.25)
        if name == "app" and any(k == "app_ready" for k in (e["kind"] for e in agent.events)):
            await asyncio.sleep(4)
            break
    await worker.cancel()
    await asyncio.wait_for(run, 15)
    await agent.close()
    await tts.close()
    return log, agent


async def await_(runner, worker):
    await runner.add_workers(worker)
    await runner.run()


def judge(name, log, agent) -> tuple[bool, str]:
    said = [x for x in log if x[1] == "said"]
    tools = [x[2] for x in log if x[1] == "tool"]
    kinds = [x[1] for x in log]
    if name == "not_for_me":
        return not said, "stayed silent" if not said else "answered speech meant for someone else"
    if name in ("chat",):
        return bool(said), "answered" if said else "no answer"
    if name == "weather":
        return any("get_weather" in t for t in tools) and bool(said), f"tools={tools}"
    if name == "lights":
        return any(t.startswith("lights") for t in tools) and agent.room.state["lights"]["power"] == "off", \
            f"tools={tools} lights={agent.room.state['lights']['power']}"
    if name == "backchannel":
        bc = [i for i, x in enumerate(log) if x[1] == "user" and x[2] == "uh huh"]
        stopped = bc and any(x[1] == "InterruptionFrame" for x in log[bc[0]:])
        more = bc and any(x[1] == "said" for x in log[bc[0]:])
        return bool(bc) and not stopped and bool(more), f"interrupted={bool(stopped)} kept_talking={bool(more)}"
    if name == "interrupt":
        u = [i for i, x in enumerate(log) if x[1] == "user" and x[2].startswith("Wait")]
        if not u:
            return False, "interruption not played"
        t_u = log[u[0]][0]
        stop = next((x[0] for x in log[u[0]:] if x[1] in ("InterruptionFrame", "BotStoppedSpeakingFrame")), None)
        answer = any(x[1] == "said" for x in log[u[0]:] if stop and x[0] > stop)
        return stop is not None and answer, f"stopped after {round(stop - t_u, 2) if stop else None} s, answered={answer}"
    if name == "news":
        return any("get_news" in t for t in tools) and bool(said), f"tools={tools}"
    if name == "app":
        ok = any(e["kind"] == "app_ready" for e in agent.events)
        return any("create_app" in t for t in tools) and ok, f"tools={tools} built={ok}"
    return False, "?"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--app-wait", type=float, default=240)
    args = ap.parse_args()
    from fda.pc.__main__ import setup_logging
    from fda.pc.launch import start_mcp_servers
    cfg = Config.load()
    cfg.open_browser = False
    cfg.say_ready = False
    setup_logging("INFO")
    urls = await asyncio.to_thread(start_mcp_servers, cfg)
    results = []
    for name, lines in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        log, agent = await run_scenario(name, lines, cfg, urls, wait_s=args.app_wait if name == "app" else 22)
        ok, why = judge(name, log, agent)
        results.append((name, ok, why))
        print(f"\n=== {name}: {'PASS' if ok else 'FAIL'}  ({why})")
        for x in log:
            print(f"   {x[0]:6.2f}  {x[1]:<24} {x[2][:110]}")
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {n:<12} {w}" for n, ok, w in results))
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
