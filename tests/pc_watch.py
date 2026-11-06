"""Live view: Elkit in the simulated room while a scripted person talks to it. Watch the display, hear both voices.

    ..\\.venv\\Scripts\\python tests\\pc_watch.py                    # every scene, in order
    ..\\.venv\\Scripts\\python tests\\pc_watch.py --list             # the scenes
    ..\\.venv\\Scripts\\python tests\\pc_watch.py --only room pick   # some of them
    --no-sound      captions and display only        --exit      close when done (default: stay open)

The person's voice is Kokoro (offline, "am_adam"), so it sounds different from Elkit. It goes into Elkit's
simulated microphone exactly like pc_sim.py: Elkit hears it through ears -> brain -> voice like a real conversation
(Groq tokens are used as usual). The display opens in the browser with captions: what the person said, what Elkit
answered. Each scene ends with a quick automatic check; the transcript is saved to tests/last_watch.txt.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import faulthandler
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fda.config import Config  # noqa: E402

SAY, SAY_ONLY, CUT_IN, QUIET, APP, PAUSE, RING = "say", "say_only", "cut_in", "quiet", "app", "pause", "ring"

# (name, title, steps). Scenes after "game" use the game it builds; "edits" uses the website.
SCENES = [
    ("hello", "Small talk", [(SAY, "Hi Elkit, how are you today?")]),
    ("room", "The room by voice", [
        (SAY, "Elkit, turn the lights blue."),
        (SAY, "Add milk and eggs to my shopping list."),
        (SAY, "What's on my shopping list?")]),
    ("alarm", "Asking something while a timer rings", [
        (SAY, "Set a timer for ten seconds."),
        (RING, 30),
        (PAUSE, 1.5),
        (SAY, "What time is it?")]),
    ("side_talk", "Talking to someone else: Elkit stays quiet", [
        (QUIET, "John, can you pass the salt?"),
        (QUIET, "Carlo, dinner is ready.")]),
    ("interrupt", "Cutting in while Elkit talks", [
        (SAY_ONLY, "Tell me a long story about the first trip to the moon."),
        (CUT_IN, "Wait, stop. What time is it?", 3.0)]),
    ("game", "Building a game", [
        (SAY, "Elkit, make me a game where a bird flaps between pipes."),
        (APP, 300)]),
    ("pick", "The pick pop-up", [
        (SAY, "Show me other birds."),
        (PAUSE, 2.0),
        (SAY, "Number two.")]),
    ("change_mind", "Changing my mind: green, Elkit takes it, then black instead", [
        (SAY_ONLY, "Make the bird green."),               # then stop: Elkit takes it and starts to answer
        (CUT_IN, "Wait, no. I want it to be black instead.", 0.4)]),
    ("change_mind_pipes", "Changing my mind on the pipes: red, Elkit takes it, then blue instead", [
        (SAY_ONLY, "Make the pipes red."),
        (CUT_IN, "Wait, no. I want them to be blue instead.", 0.4)]),
    ("mind_timer", "Changing my mind on a timer: 5 minutes, then 10", [
        (SAY_ONLY, "Set a timer for 5 minutes."),
        (CUT_IN, "Wait, no, make it 10 minutes.", 0.4)]),
    ("mind_lights", "Changing my mind on the lights: blue, then red", [
        (SAY_ONLY, "Turn the lights blue."),
        (CUT_IN, "Actually, red instead.", 0.4)]),
    ("mind_build", "Changing my mind on a build: a snake game, then a racing game", [
        (SAY_ONLY, "Make me a snake game."),
        (CUT_IN, "Wait, no, a racing game instead.", 1.0),
        (APP, 300)]),
    ("background", "A library background", [
        (SAY, "Change the sky to the night sky.")]),
    ("website", "A website, colours first", [
        (SAY, "Make me a website for my coffee shop called Bean There."),
        (PAUSE, 2.0),
        (SAY, "The dark one."),
        (APP, 300)]),
    ("edits", "Two changes while it is still working", [
        (SAY, "Make the title bigger."),
        (SAY, "And make the buttons round."),
        (APP, 300), (APP, 300)]),
    ("home_move", "A home request from another tab: Elkit says it moves", [
        (SAY, "Add bread to my shopping list."),
        (PAUSE, 2.0),
        (SAY, "Go back to the studio.")]),
    ("office", "Slides open in the Virtual Office", [
        (SAY, "Make me a short slide presentation about my coffee shop Bean There."),
        (APP, 300)]),
]


class Person:
    """The scripted person's voice: Kokoro, offline, not Elkit's voice."""

    def __init__(self, cfg, voice: str = "am_adam"):
        from fda.local import LocalServers
        c = copy.copy(cfg)
        c.offline_tts, c.offline_voice = "kokoro", voice
        self.local = LocalServers(c)
        self.local._load_tts()
        if self.local.tts is None:
            raise SystemExit("The Kokoro voice is missing (offline_models/kokoro-en-v0_19).")
        self.rate = self.local.tts.sample_rate

    def say(self, text: str, rate: int) -> np.ndarray:
        a = self.local.tts.generate(text, sid=self.local.sid, speed=1.0)
        x = np.asarray(a.samples, np.float32)
        return soxr.resample(x, self.rate, rate).astype(np.float32) * 0.8


class Watch:
    def __init__(self, agent, transport, ui, person):
        self.agent, self.tr, self.ui, self.person = agent, transport, ui, person
        self.eng = transport.engine
        self.log: list[tuple[float, str, str]] = []
        self.sent: list[dict] = []                      # every display message (for the checks)
        self.cap = {"scene": "", "you": "", "bot": ""}
        self.reply: list[str] = []
        self.t0 = time.monotonic()
        send = ui.send

        def tap(m):
            self.sent.append(m)
            send(m)
        ui.send = tap

    def t(self) -> float:
        return round(time.monotonic() - self.t0, 1)

    def caption(self, **kw):
        self.cap.update(kw)
        self.ui.send({"type": "demo", **self.cap})

    def heard_bot(self, text: str):
        self.reply.append(text.strip())
        self.caption(bot=" ".join(x for x in self.reply if x))

    async def speak(self, text: str):
        audio = await asyncio.to_thread(self.person.say, text, self.eng.rate)
        self.reply = []
        self.caption(you=text, bot="")
        self.eng.say_at(0.0, audio)
        self.log.append((self.t(), "you", text))
        print(f"{self.t():7.1f}  YOU    {text}")
        await asyncio.sleep(len(audio) / self.eng.rate)

    async def wait_reply(self, start_s: float = 25, quiet_s: float = 2.0, max_s: float = 120) -> bool:
        """Until Elkit has answered and then been quiet for a moment."""
        t0 = time.monotonic()
        while not self.tr.bot_speaking and time.monotonic() - t0 < start_s:
            await asyncio.sleep(0.05)
        if not self.tr.bot_speaking:
            return False
        last = time.monotonic()
        while time.monotonic() - t0 < max_s:
            await asyncio.sleep(0.1)
            if self.tr.bot_speaking:
                last = time.monotonic()
            elif time.monotonic() - last > quiet_s:
                return True
        return True

    async def wait_quiet(self, quiet_s: float = 3.0, max_s: float = 60):
        t0 = last = time.monotonic()
        while time.monotonic() - t0 < max_s:
            if self.tr.bot_speaking or self.agent.room.state.get("ringing"):
                last = time.monotonic()
            elif time.monotonic() - last > quiet_s:
                return
            await asyncio.sleep(0.1)

    def apps_done(self) -> int:
        return sum(1 for e in self.agent.events if e["kind"] in ("app_ready", "app_failed"))

    async def run_scene(self, i: int, n: int, name: str, title: str, steps) -> dict:
        print(f"\n=== {i}/{n} {name}: {title}")
        await self.wait_quiet()                         # a person lets Elkit finish before starting something new
        self.caption(scene=f"Scene {i}/{n} · {title}", you="", bot="")
        start, apps0, waited = len(self.log), self.apps_done(), 0
        for st in steps:
            kind = st[0]
            if kind == SAY:
                await self.speak(st[1])
                if not await self.wait_reply():
                    print(f"{self.t():7.1f}  (no answer)")
            elif kind == SAY_ONLY:
                await self.speak(st[1])
            elif kind == CUT_IN:
                t_end = time.monotonic() + 30
                while not self.tr.bot_speaking and time.monotonic() < t_end:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(st[2])
                await self.speak(st[1])
                await self.wait_reply()
            elif kind == QUIET:
                await self.speak(st[1])
                await asyncio.sleep(8)
            elif kind == APP:
                waited += 1
                t_end = time.monotonic() + st[1]
                self.caption(bot="(building ...)" if not self.reply else self.cap["bot"])
                while self.apps_done() < apps0 + waited and time.monotonic() < t_end:
                    await asyncio.sleep(0.5)
                e = next((e for e in reversed(self.agent.events) if e["kind"].startswith("app_")), {})
                print(f"{self.t():7.1f}  APP    {e.get('kind', 'timeout')} {e.get('title') or ''} "
                      f"{e.get('seconds') or ''}")
                await self.wait_reply(start_s=15)
            elif kind == PAUSE:
                await asyncio.sleep(st[1])
            elif kind == RING:                          # until the timer / alarm rings
                t_end = time.monotonic() + st[1]
                while not self.agent.room.state.get("ringing") and time.monotonic() < t_end:
                    await asyncio.sleep(0.2)
                self.log.append((self.t(), "ringing", str(self.agent.room.state.get("ringing"))))
                print(f"{self.t():7.1f}  RING   {self.agent.room.state.get('ringing')}")
        await asyncio.sleep(1.0)
        return {"log": self.log[start:], "apps": self.apps_done() - apps0}


def check(name: str, r: dict, w: Watch) -> tuple[bool, str]:
    """A quick automatic look at each scene (the real check is watching it)."""
    log, room = r["log"], w.agent.room.state
    said = [x for x in log if x[1] == "said"]
    tools = [x[2] for x in log if x[1] == "tool"]
    msgs = w.sent
    if name == "hello":
        return bool(said), "answered" if said else "no answer"
    if name == "room":
        ok = room["lights"]["color"] in ("blue", "#0000ff") and any("milk" in i.lower() for i in
                                                                    room.get("lists", {}).get("shopping", []))
        return ok, f"lights={room['lights']['power']}/{room['lights']['color']} timers={len(room['timers'])} " \
                   f"shopping={room.get('lists', {}).get('shopping')}"
    if name == "alarm":
        rang = any(x[1] == "ringing" and x[2] != "None" for x in log)
        u = next((i for i, x in enumerate(log) if x[1] == "you" and x[2].startswith("What time")), None)
        heard = u is not None and any(x[1] == "heard" for x in log[u:])
        answered = u is not None and any(x[1] == "said" for x in log[u:])
        return rang and heard and answered and not room.get("ringing"), \
            f"rang={rang} heard_during_ring={heard} answered={answered} still_ringing={bool(room.get('ringing'))}"
    if name == "side_talk":
        return not said, "stayed quiet" if not said else f"answered: {said[0][2][:60]}"
    if name == "interrupt":
        u = next((i for i, x in enumerate(log) if x[1] == "you" and x[2].startswith("Wait")), None)
        after = log[u:] if u is not None else []
        stopped = any(x[1] in ("InterruptionFrame", "BotStoppedSpeakingFrame") for x in after)
        answered = any(x[1] == "said" for x in after)
        before = log[:u] if u is not None else log
        story = [x[2] for x in before if x[1] == "said"]
        told = bool(story) and not re.search(r"sorry,? but i (?:can.?t|cannot)", " ".join(story), re.I)
        return told and stopped and answered, f"story_told={told} stopped={stopped} answered={answered}"
    if name == "mind_timer":
        secs = [t["seconds"] for t in room.get("timers", [])]
        told = any(re.search(r"not 5 minutes", x[2], re.I) for x in said)
        return secs == [600] and told, f"timers={secs} said 'not 5 minutes'={told}"
    if name == "mind_lights":
        told = any(re.search(r"not blue", x[2], re.I) for x in said)
        return room["lights"]["color"] == "red" and told, f"lights={room['lights']['color']} said 'not blue'={told}"
    if name == "mind_build":
        built = [e.get("title", "") for e in w.agent.events if e["kind"] == "app_ready"]
        racing = any(re.search(r"rac|car|road|drive", t, re.I) for t in built)
        snake = any(re.search(r"snake", t, re.I) for t in built)
        told = any(re.search(r"not (?:a )?snake|instead of (?:a )?snake|racing", x[2], re.I) for x in said)
        return racing and not snake and told, f"built={built} said racing / not snake={told}"
    if name in ("game", "website"):
        ready = any(e["kind"] == "app_ready" for e in w.agent.events)
        pop = any(m.get("type") == "choices" and m.get("options") for m in msgs) if name == "website" else True
        return ready and pop, f"built={ready}" + (f" colours offered={pop}" if name == "website" else "")
    if name == "open_game":
        return bool(w.agent.app_keys), f"game on screen, its settings: {w.agent.app_keys}"
    if name == "pick":
        shown = [m for m in msgs if m.get("type") == "choices" and m.get("options")]
        changed = [m for m in msgs if m.get("type") == "app_set"]
        return bool(shown) and bool(changed), f"pop-up={bool(shown)} changed={[m['patch'] for m in changed][-1:]}"
    if name in ("change_mind", "change_mind_pipes"):
        c1, c2, c2_re = ("green", "black", r"black|#000(?:000)?\b") if name == "change_mind" else \
            ("red", "blue", r"blue|#00f\b|#0000ff")
        sets = [m["patch"] for m in msgs if m.get("type") == "app_set"]
        last = sets[-1] if sets else {}
        heard_1 = any(x[1] == "heard" and re.search(c1, x[2], re.I) and not re.search(c2, x[2], re.I) for x in log)
        done_2 = any(re.search(c2_re, str(v), re.I) for v in last.values())
        told = any(re.search(rf"not {c1}|instead of {c1}", x[2], re.I) for x in said)
        never_1 = not any(re.search(rf"\b{c1}\b", str(v), re.I) for p in sets for v in p.values())
        return heard_1 and done_2 and told and never_1, (
            f"{c1} taken on its own={heard_1} then {c2}={done_2} {c1} never shown={never_1} "
            f"said 'not {c1}'={told} last change={last}")
    if name == "background":
        sets = [m["patch"] for m in msgs if m.get("type") == "app_set"]
        return any("sky-night" in str(v) for p in sets for v in p.values()), f"changes={sets[-1:]}"
    if name == "edits":
        return r["apps"] >= 2, f"edits finished={r['apps']}"
    if name == "home_move":
        told = any(x[2].startswith("Over to the Virtual Home tab") for x in said)
        bread = any("bread" in i.lower() for i in room.get("lists", {}).get("shopping", []))
        back = w.agent.cur_tab == "design"
        return told and bread and back, f"said_move={told} bread_on_list={bread} back_on_interface={back}"
    if name == "office":
        office = any(v.get("kindword") in ("slides", "doc", "sheet") for m in msgs if m.get("type") == "views"
                     for v in m.get("views", []))
        shown = any(m.get("type") == "tab" and m.get("tab") == "office" for m in msgs)
        return office and shown, f"office_item={office} office_tab_shown={shown}"
    return True, ""


async def main():
    # Elkit's words can hold any character ("three‑stage" has U+2011); Windows' default cp1252 console/file
    # encoding crashed print() on it mid-run (the "freeze" at the interrupt scene)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--exit", action="store_true", help="close when the scenes are done")
    ap.add_argument("--step", action="store_true",
                    help="pause before each scene until tests/watch_next.txt exists (one scene at a time)")
    ap.add_argument("--voice", default="am_adam", help="the person's Kokoro voice")
    args = ap.parse_args()
    if args.list:
        for n, t, s in SCENES:
            print(f"{n:12} {t}  ({len(s)} steps)")
        return
    scenes = [s for s in SCENES if not args.only or s[0] in args.only]
    # a run that starts after the game scene (a fresh Elkit, no game on screen) opens the game from before first,
    # by voice, or "make the bird black" has no bird to change
    game_scenes = {"pick", "change_mind", "change_mind_pipes", "background", "interrupt_game"}
    apps_dir = Path(__file__).resolve().parents[1] / "apps"
    if args.only and "game" not in args.only and game_scenes & set(args.only) and (apps_dir / "flappy-bird.html").exists():
        scenes.insert(0, ("open_game", "Opening the game from before", [(SAY, "Open the Flappy Bird game.")]))

    from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, FunctionCallInProgressFrame,
                                       InterruptionFrame, TranscriptionFrame, TTSTextFrame)
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.transports.base_transport import TransportParams

    from fda.pc.__main__ import setup_logging
    from fda.pc.agent import Agent
    from fda.pc.launch import start_mcp_servers
    from fda.pc.transport import FdaAudioTransport
    from fda.ui import UI

    cfg = Config.load()
    cfg.say_ready = False
    cfg.open_browser = not args.no_browser
    # a fresh room every run: your real alarms, reminders and lists (room_state.json) are not touched or used
    cfg.state_file = str(Path(tempfile.mkdtemp(prefix="elkit-watch-")) / "room_state.json")
    setup_logging("INFO")                            # everything to logs/agent.log ...
    from loguru import logger as _lg
    from fda.config import ROOT
    _lg.remove()                                     # ... but only warnings on this console (the scenes print here)
    _lg.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} {name} {message}",
            filter=lambda r: not r["name"].startswith("pipecat.transports"))
    _lg.add(ROOT / "logs" / "agent.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} {name} {message}",
            filter=lambda r: not r["name"].startswith("pipecat.transports") or r["level"].no >= 30,
            rotation="20 MB", retention=3, encoding="utf-8", enqueue=True)
    print("Starting Elkit (servers, display, voice) ...")
    person = await asyncio.to_thread(Person, cfg, args.voice)
    urls = await asyncio.to_thread(start_mcp_servers, cfg)
    ui = UI(cfg)
    await ui.start()
    transport = FdaAudioTransport(cfg, TransportParams(audio_in_enabled=True, audio_out_enabled=True,
                                                       audio_in_sample_rate=16000, audio_out_sample_rate=24000),
                                  sim=True, monitor=not args.no_sound)
    agent = Agent(cfg, transport, transport, urls, ui=ui, out_rate=24000)
    transport.on_level = lambda db: ui.send({"type": "level", "db": round(db, 1)})
    worker = await agent.build()
    ui.agent = agent
    w = Watch(agent, transport, ui, person)

    seen: set[int] = set()

    class Obs(BaseObserver):
        async def on_push_frame(self, data):
            f, src = data.frame, type(data.source).__name__
            if f.id in seen:                            # a frame is seen once per pipeline stage: count it once
                return
            seen.add(f.id)
            if isinstance(f, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame)):
                if not w.log or w.log[-1][1] != type(f).__name__:
                    w.log.append((w.t(), type(f).__name__, ""))
            elif isinstance(f, TranscriptionFrame) and src == "McpSTTService":
                w.log.append((w.t(), "heard", f.text))
                print(f"{w.t():7.1f}  heard  {f.text}")
            elif isinstance(f, TTSTextFrame) and src == "McpTTSService":
                w.log.append((w.t(), "said", f.text))
                w.heard_bot(f.text)
            elif isinstance(f, FunctionCallInProgressFrame):
                w.log.append((w.t(), "tool", f"{f.function_name}({f.arguments})"))
                print(f"{w.t():7.1f}  tool   {f.function_name} {str(f.arguments)[:90]}")

    worker.add_observer(Obs())
    runner = WorkerRunner(handle_sigint=False)

    async def go():
        await runner.add_workers(worker)
        await runner.run()

    run = asyncio.create_task(go())
    ui.send({"type": "status", "status": "listening"})
    w.caption(scene="Live view · starting", you="", bot="")
    print(f"Display: http://localhost:{cfg.ui_port}/  (sound: {'on' if not args.no_sound else 'off'})")
    await asyncio.sleep(5)                           # the browser connects; the pipeline settles

    results = []
    nxt = Path(__file__).with_name("watch_next.txt")
    nxt.unlink(missing_ok=True)
    for i, (name, title, steps) in enumerate(scenes, 1):
        if args.step:                                  # one scene at a time: wait for "next"
            w.caption(scene=f"Paused · next: scene {i}/{len(scenes)} · {title}", you="", bot="")
            print(f"PAUSED before {i}/{len(scenes)} {name}", flush=True)
            while not nxt.exists():
                await asyncio.sleep(0.3)
            nxt.unlink(missing_ok=True)
        # a stuck scene must not stop the run: after 7 minutes, print where everything is and go on
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(420, exit=False)
        start = len(w.log)
        try:
            r = await asyncio.wait_for(w.run_scene(i, len(scenes), name, title, steps), timeout=480)
        except asyncio.TimeoutError:
            print(f"{w.t():7.1f}  (scene {name} stuck: skipped)")
            r = {"log": w.log[start:], "apps": 0}
        except asyncio.CancelledError:
            # something cancelled the run itself (seen at the interrupt scene): show who, then carry on
            me = asyncio.current_task()
            if me is not None:
                me.uncancel()
            print(f"{w.t():7.1f}  (scene {name}: the run was CANCELLED from outside)", flush=True)
            traceback.print_exc()
            r = {"log": w.log[start:], "apps": 0}
        except Exception:  # noqa: BLE001
            print(f"{w.t():7.1f}  (scene {name} ERROR)", flush=True)
            traceback.print_exc()
            r = {"log": w.log[start:], "apps": 0}
        said = " ".join(x[2] for x in r["log"] if x[1] == "said")
        if said:
            print(f"         ELKIT  {said[:300]}")
        try:
            ok, why = check(name, r, w)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"check failed: {e!r}"
        results.append((name, ok, why))
        print(f"   -> {'OK   ' if ok else 'CHECK'} {why}")
    faulthandler.cancel_dump_traceback_later()
    summary = "\n".join(f"{'OK   ' if ok else 'CHECK'} {n:12} {why}" for n, ok, why in results)
    print("\n" + summary + f"\n{sum(ok for _, ok, _ in results)}/{len(results)} scenes OK")
    w.caption(scene=f"Done · {sum(ok for _, ok, _ in results)}/{len(results)} scenes OK", you="", bot="")
    out = Path(__file__).with_name("last_watch.txt")
    out.write_text("\n".join(f"{t:7.1f}  {k:<24} {x}" for t, k, x in w.log) + "\n\n" + summary + "\n", encoding="utf-8")
    print(f"Transcript: {out}")
    if not args.exit:
        print("Elkit stays on so you can look around the display (tabs, apps). Ctrl+C closes it.")
        try:
            await run
        except asyncio.CancelledError:
            pass
    await worker.cancel()
    await asyncio.wait_for(run, 15)
    await agent.close()
    await ui.stop()


async def _main_logged():
    """main(), but whatever ends it is printed AT ONCE (asyncio.run can then hang closing the pipeline's tasks,
    which hid the reason before)."""
    try:
        await main()
    except BaseException as e:  # noqa: BLE001
        print(f"*** the run ended: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        if isinstance(e, asyncio.CancelledError):
            for t in asyncio.all_tasks():
                if "cancel" in repr(t).lower() or t is asyncio.current_task():
                    print("   task:", repr(t)[:200], flush=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(_main_logged())
    except KeyboardInterrupt:
        pass
