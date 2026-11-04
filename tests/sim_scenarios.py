"""End-to-end test in a simulated room (no hardware): FDA's own voice echoes back into the 'mic' (delay, reverb,
noise) and a second voice plays the user (vivian on DGX3; offline: Kokoro af_bella). Runs in real time.

    python tests/sim_scenarios.py [--record DIR] [--only NAME] [--offline | --server]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fda.agent import Agent               # noqa: E402
from fda.audio import resample            # noqa: E402
from fda.config import Config             # noqa: E402
from fda.servers import Servers, make_servers  # noqa: E402
from fda.sim import SimAudioEngine        # noqa: E402

USER_VOICE = "vivian"


class Harness:
    def __init__(self, agent: Agent, user_audio: dict):
        self.a, self.ua = agent, user_audio
        self.results = []

    @property
    def sim(self) -> SimAudioEngine:
        return self.a.audio

    def say(self, text: str, delay: float = 0.0) -> float:
        """User says text after delay; returns wall time the speech ends."""
        x = self.ua[text]
        self.sim.say_at(delay, x)
        return time.time() + delay + len(x) / self.sim.rate

    async def wait_event(self, kinds, since: int, timeout=15.0):
        kinds = {kinds} if isinstance(kinds, str) else set(kinds)
        t0 = time.time()
        while time.time() - t0 < timeout:
            for e in self.a.events[since:]:
                if e["kind"] in kinds:
                    return e
            await asyncio.sleep(0.02)
        return None

    async def wait_quiet(self, timeout=30.0):
        t0 = time.time()
        await asyncio.sleep(0.3)
        while time.time() - t0 < timeout:
            if not self.a.speaker.active and not self.a.in_turn and self.a.status == "listening":
                await asyncio.sleep(0.6)
                if not self.a.speaker.active and not self.a.in_turn:
                    return
            await asyncio.sleep(0.05)

    async def first_audio_after(self, t_end: float, timeout=12.0):
        """ms from the end of the user's speech until FDA's voice starts (None if it never does)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            fa = self.a.speaker.first_audio_at
            if fa and fa > t_end - 0.3:
                return int((fa - t_end) * 1000)
            await asyncio.sleep(0.01)
        return None

    def check(self, name, ok, **info):
        self.results.append({"scenario": name, "ok": bool(ok), **info})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=str)}", flush=True)


async def scenario_lights_fast(h: Harness):
    n = len(h.a.events)
    t_end = h.say("Turn off the lights.")
    lat = await h.first_audio_after(t_end)
    fp = await h.wait_event("fast_path", n)
    await h.wait_quiet()
    h.check("fast path: lights off", h.a.room.state["lights"]["power"] == "off" and fp is not None,
            first_audio_ms=lat, heard=(await h.wait_event("heard", n))["text"])


async def scenario_planner(h: Harness):
    n = len(h.a.events)
    t_end = h.say("Can you make the lights blue and set a timer for two minutes?")
    lat = await h.first_audio_after(t_end)
    await h.wait_quiet()
    L = h.a.room.state["lights"]; timers = h.a.room.state["timers"]
    h.check("planner: lights blue + 2 min timer", L["color"] == "blue" and any(t["seconds"] == 120 for t in timers),
            first_audio_ms=lat, lights=L, timers=[t["seconds"] for t in timers],
            said=h.a.history[-1]["content"] if h.a.history else "")


async def scenario_bargein(h: Harness):
    n = len(h.a.events)
    t_end = h.say("Tell me a long story about a brave little dragon.")
    lat = await h.first_audio_after(t_end, timeout=15)
    # let FDA talk for 3 s: nothing should trigger from its own echo
    await asyncio.sleep(3.0)
    false_turns = [e for e in h.a.events[n:] if e["kind"] in ("bargein_duck", "speculative_start") and e["t"] > t_end + 0.8]
    n2 = len(h.a.events)
    t_int = time.time()
    h.say("Stop. What time is it?")
    duck = await h.wait_event("bargein_duck", n2, timeout=5)
    stop = await h.wait_event("fda_stopped", n2, timeout=6)
    duck_ms = int((duck["t"] - t_int) * 1000) if duck else None
    stop_ms = int((stop["t_fade_start"] - t_int) * 1000) if stop else None
    await h.wait_quiet()
    hist = [m for m in h.a.history if m["role"] == "assistant"]
    interrupted = any("[interrupted]" in m["content"] for m in hist[-3:])
    answered = any("AM" in m["content"] or "PM" in m["content"] for m in hist[-2:])
    h.check("barge-in: stop mid-story, then answer", stop is not None and interrupted and answered,
            first_audio_ms=lat, echo_false_triggers=len(false_turns), duck_after_ms=duck_ms, fade_after_ms=stop_ms,
            story_heard=[m["content"][:80] for m in hist[-2:]])


async def scenario_backchannel(h: Harness):
    n = len(h.a.events)
    t_end = h.say("Tell me a story about the moon.")
    await h.first_audio_after(t_end, timeout=15)
    await asyncio.sleep(1.5)
    n2 = len(h.a.events)
    h.say("Uh huh.")
    bc = await h.wait_event(["backchannel", "bargein_stop", "resume_unduck", "resume_rest", "blip_ignored"], n2, timeout=6)
    kept_talking = h.a.speaker.active
    await h.wait_quiet(timeout=40)
    # pass: FDA kept talking and was never stopped (not noticing a short "uh huh" at all is fine too)
    stopped = any(e["kind"] in ("bargein_stop", "fda_stopped") for e in h.a.events[n2:])
    h.check("backchannel: 'uh huh' does not stop FDA", kept_talking and not stopped and (bc is None or bc["kind"] != "bargein_stop"),
            event=bc and bc["kind"], kept_talking=kept_talking)


async def scenario_not_for_me(h: Harness):
    n = len(h.a.events)
    h.say("Honey, did you remember to lock the front door when you came in?")
    fm = await h.wait_event(["for_me", "fast_path"], n, timeout=10)
    await h.wait_quiet()
    h.check("not for me: talking to someone else", fm is not None and fm["kind"] == "for_me" and fm["value"] is False,
            event=fm)


async def scenario_search(h: Harness):
    n = len(h.a.events)
    t_end = h.say("Who is the current president of France?")
    lat = await h.first_audio_after(t_end)
    await h.wait_quiet(timeout=40)
    msg = h.a.history[-1]["content"] if h.a.history else ""
    h.check("search agent: web question", "macron" in msg.lower(), first_audio_ms=lat, said=msg)


SCENARIOS = {"lights": scenario_lights_fast, "planner": scenario_planner, "bargein": scenario_bargein,
             "backchannel": scenario_backchannel, "notforme": scenario_not_for_me, "search": scenario_search}
LINES = ["Turn off the lights.", "Can you make the lights blue and set a timer for two minutes?",
         "Tell me a long story about a brave little dragon.", "Stop. What time is it?",
         "Tell me a story about the moon.", "Uh huh.",
         "Honey, did you remember to lock the front door when you came in?", "Who is the current president of France?"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--harsh", action="store_true", help="mic next to the speaker: loud, distorted echo the AEC cannot fully remove")
    ap.add_argument("--no-selfvoice", action="store_true", help="turn off FDA's own-voice checks (to compare)")
    args = ap.parse_args()
    cfg = Config.load()
    cfg.open_browser = False
    cfg.say_ready = True
    if args.no_selfvoice:
        cfg.selfvoice = cfg.strip_own_words = False
    if args.offline:
        cfg.backend = "offline"
    elif args.server:
        cfg.backend = "server"
    if args.record:
        cfg.record_dir = args.record
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # user lines spoken by a different voice
    ucfg = Config.load(); ucfg.voice = USER_VOICE; ucfg.backend = cfg.backend
    s = make_servers(ucfg)
    if s.kind == "offline":            # voice only (the agent starts its own planner)
        ucfg.offline_tts, ucfg.offline_voice = "kokoro", "af_bella"
        from fda.local import LocalServers
        await s.close(); s = LocalServers(ucfg)
        await asyncio.to_thread(s._load_tts)
    print("backend:", s.kind, flush=True)
    ua = {}
    for line in LINES:
        pcm = np.concatenate([c async for c in s.speak_stream(line)])
        ua[line] = resample(pcm, 24000, cfg.device_rate) * 0.7
    await s.close()

    room = dict(echo_gain=0.6, echo_delay_ms=5.0, drive=2.0) if args.harsh else {}
    agent = Agent(cfg, ui=None, audio_factory=lambda c, cb: SimAudioEngine(c, cb, **room))
    stop = asyncio.Event()
    run = asyncio.create_task(agent.run(stop))
    h = Harness(agent, ua)
    await asyncio.sleep(1.0)
    await h.wait_quiet()
    for name, fn in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        print(f"\n=== {name}", flush=True)
        await fn(h)
        await asyncio.sleep(1.0)
    stop.set()
    await run
    ok = sum(r["ok"] for r in h.results)
    if agent.own is not None:
        print(f"\nown voice: {agent.own_windows} windows recognised as FDA itself, "
              f"learned AEC leak {agent.own.leak_db:.1f} dB")
    print(f"\n{ok}/{len(h.results)} scenarios passed")
    Path("tests/last_sim_results.json").write_text(json.dumps(h.results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
