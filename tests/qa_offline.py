"""Offline QA for Elkit: no LLM, no internet, no tokens, no mic, no servers. Runs in a few seconds.

    ..\\.venv\\Scripts\\python tests\\qa_offline.py              # every check
    ..\\.venv\\Scripts\\python tests\\qa_offline.py -k voice      # only sections whose name has "voice"
    ..\\.venv\\Scripts\\python tests\\qa_offline.py --replay      # also: your real logs through today's code

What it checks (everything that does not need the model):
    voice       what is said aloud: links, emoji, odd Unicode, pesos, "(no further response)", filler
    thinking    leaked reasoning ("The user wants ...") and dot garbage are caught, real replies are not
    stop        bare "stop", "continue", video commands, the agent's name
    rules       backchannels, side talk, fast-path room commands, evidence for a tool
    room        reminders (add / edit / delete), alarms (set / cancel), timers, lights, thermostat
    replies     the reply filter: (silent), thinking aloud, garbage, repeats, resuming after a cut-in
    tabs        Interface sub-tabs by voice: names, switching, closing, room / interface
    video       video by voice: pause, louder, volume 30, skip, close, photos next
    app         hands-free app use: go to contact, scroll, press space, click send
    tweak       live changes by voice: flip the bird, make it blue
    history     the conversation sent to the LLM: leaked reasoning and broken tool calls removed
    tools       every tool has a schema, required fields exist, the display's JavaScript parses

--replay reads logs/agent.log (what was heard) and logs/mcp-tts.log (what was said) and shows what today's code
does with each line: which heard lines never reach the LLM (stop, ignored, direct action) and which spoken lines
would now be cut or changed. Nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()                       # quiet: only the report

PASS, FAIL = [], []
SECTION = [""]


def check(name: str, got, want):
    ok = got == want
    (PASS if ok else FAIL).append((SECTION[0], name, got, want))
    if not ok:
        print(f"  FAIL  {name}\n        got:  {got!r}\n        want: {want!r}")


def section(name: str):
    SECTION[0] = name
    print(f"\n== {name}")


# ---------------------------------------------------------------- voice
def t_voice():
    from fda.pc.services import speakable
    cases = [
        ("Hello there.", "Hello there."),
        ("**Bold** and _italic_ and `code`.", "Bold and italic and code."),
        ("See [Cebu Pacific](https://cebupacific.com) for more.", "See Cebu Pacific for more."),
        ("The link is https://example.com/x?y=1 on the screen.", "The link is on the screen."),
        ("Cheapest is ₱2,869 on AirAsia.", "Cheapest is 2,869 pesos on AirAsia."),
        ("It costs ₱ 1,200.50.", "It costs 1,200.50 pesos."),
        ("It's 30°C today.", "It's 30 degrees C today."),
        ("Great 🎉 done 👍", "Great done"),
        ("Fact【3†source】 here.", "Fact here."),
        ("(silent)", ""),
        ("[silent]", ""),
        ("(no further response)", ""),
        ("Reminder set. (no further response)", "Reminder set."),
        ("[pause]", ""),
        ("(laughs) Good one.", "Good one."),
        ("It is 32 degrees (feels like 36).", "It is 32 degrees (feels like 36)."),
        ("Done. Let me know if you need anything else.", "Done."),
        ("Done. Feel free to ask if you want more changes.", "Done."),
        ("Anything else you need?", ""),
        ("Done. Anything else?", "Done."),
        ("Sure. What would you like next?", "Sure."),
        ("Sure, what should the reminder say?", "Sure, what should the reminder say?"),
        ("......", ""),
        ("Wait........ okay.", "Wait okay."),
        ("- first\n- second", "first second"),
        ("Two hundred and ten", "Two hundred and ten"),
        ("Non‑stop flights", "Non-stop flights"),
        ("Manila → Davao", "Manila to Davao"),
        ("User wants the weather. It's sunny.", "It's sunny."),
        ("We We Okay The user wants Articuno bird.", ""),
        ("... Weird……… … …… It's …", ""),
        ("...…... All set, pipes are blue.", "All set, pipes are blue."),
        ("Is there anything else I can help with?", ""),
    ]
    for text, want in cases:
        check(f"speak {text[:40]!r}", speakable(text), want)


# ---------------------------------------------------------------- thinking
def t_thinking():
    from fda import rules
    leaks = ['The user wants to set reminders: they said "drink water".', "User wants to set up voice command?",
             "We We Okay The user wants Articuno bird.", "We need to call edit_app with the change.",
             "Let's respond briefly.", "Probably answer: list options.", "So the user is asking about the weather.",
             "Okay, we should check the time.", "The user asked for a timer.", "We have to use web_search.",
             "Need to answer briefly.", "Answer: the bird is blue.", "Let's call tweak_app.",
             "The person said hi.", "We must comply.", "Maybe the user wants a parrot."]
    real = ["The bird can be a sparrow, a parrot, or an eagle. Which one do you like?",
            "We can make the bird blue if you want.", "The user interface tab is open.",
            "Let's go with the parrot then.", "It's 3 pm in Davao.", "Okay, done.",
            "The cheapest flight to Davao is 2,869 pesos on Cebu Pacific. The link is on the screen.",
            "Now playing lo-fi beats.", "So, the lights are on.", "Maybe a parrot would look nice.",
            "They say it will rain tomorrow in Manila.", "We could try a snake game next.",
            "Sure, I can check the weather.", "Thus far it's sunny.", "Well done!", "The users love this game."]
    for s in leaks:
        check(f"leak caught: {s[:45]!r}", rules.is_thinking(s), True)
    for s in real:
        check(f"real reply kept: {s[:45]!r}", rules.is_thinking(s), False)
    garbage = ["... … --- … ... … … … ... … The", "………", "... ... ... We", ". . . . ."]
    fine = ["Done.", "...…... All set pipes are blue.", "1. 2. 3.", "Okay … sure.", "It's 2,869 pesos."]
    for s in garbage:
        check(f"garbage caught: {s!r}", rules.is_garbage(s), True)
    for s in fine:
        check(f"not garbage: {s!r}", rules.is_garbage(s), False)
    check("could be thinking: 'The'", rules.could_be_thinking("The"), True)
    check("could be thinking: 'It's 3'", rules.could_be_thinking("It's 3"), False)
    check("could be thinking: long", rules.could_be_thinking("The cheapest flight to Davao is 2,869 pesos"), False)


# ---------------------------------------------------------------- stop / continue / media
def t_stop():
    from fda.pc.processors import CONTINUE, MEDIA_CMD, called_by_name, is_bare_stop, strip_name
    for s in ["stop", "Stop.", "okay stop", "wait", "hold on", "Elkit, stop", "stop stop", "that's enough",
              "never mind", "be quiet", "okay, wait"]:
        check(f"bare stop: {s!r}", is_bare_stop(s, "Elkit"), True)
    for s in ["stop the music", "stop the timer", "can you stop the video and play another one",
              "don't stop", "what's the weather", "Stop making the bird blue and make it red"]:
        check(f"not bare stop: {s!r}", is_bare_stop(s, "Elkit"), False)
    for s in ["continue", "go on", "okay, continue", "you were saying", "keep going please", "carry on"]:
        check(f"continue: {s!r}", bool(CONTINUE.match(s)), True)
    for s in ["continue the video", "go on youtube", "what's next"]:
        check(f"not continue: {s!r}", bool(CONTINUE.match(s)), False)
    for s in ["pause the video", "turn it up", "volume 30 percent", "skip ahead 10 seconds", "close the video",
              "mute it", "next photo", "restart it from the beginning"]:
        check(f"media command: {s!r}", bool(MEDIA_CMD.search(s)), True)
    for s in ["what a nice song", "I love this", "haha"]:
        check(f"not media command: {s!r}", bool(MEDIA_CMD.search(s)), False)
    for s in ["Elkit, pause", "El kit pause", "elk it, louder", "Hey Elkit"]:
        check(f"called by name: {s!r}", called_by_name(s, "Elkit"), True)
    check("not called by name", called_by_name("pause it", "Elkit"), False)
    check("strip name", strip_name("Elkit, louder", "Elkit"), "louder")


# ---------------------------------------------------------------- rules
def t_rules():
    from fda import rules
    for s in ["uh huh", "yeah", "okay", "mm hmm", "right"]:
        check(f"backchannel: {s!r}", rules.is_backchannel(s), True)
    for s in ["yeah turn off the lights", "okay what's the time", "no"]:
        check(f"not backchannel: {s!r}", rules.is_backchannel(s), False)
    for s in ["Honey, did you buy the milk?", "Mom, where are my keys?", "talk to you later, bye bye"]:
        check(f"side talk: {s!r}", rules.addressed_to_other(s, "Elkit"), True)
    for s in ["Elkit, what's the weather?", "Turn on the lights", "Describe, a cat", "Okay, set a timer"]:
        check(f"for me: {s!r}", rules.addressed_to_other(s, "Elkit"), False)
    room = {"lights": {"power": "off", "brightness": 80, "color": "warm white"}, "thermostat_c": 24, "timers": [],
            "music": {"state": "stopped"}, "volume": 50, "reminders": [], "alarms": []}
    def tools(s):
        m = rules.match(s, room)
        return [a.get("tool") for a in m[0]] if m else None
    check("fast: turn on the lights", tools("turn on the lights"), ["lights"])
    check("fast: lights off please", tools("lights off please"), ["lights"])
    check("fast: set a timer for 5 minutes", tools("set a timer for 5 minutes"), ["timer"])
    check("fast: not a command in a story", tools("and then she said turn off the lights in the hallway"), None)
    check("evidence: lights needs light words", rules.plausible({"tool": "lights", "power": "off"},
                                                               "can you make an app"), False)
    check("evidence: delete reminder", rules.plausible({"tool": "delete_reminder", "which": "1"},
                                                      "scratch the first reminder"), True)
    check("evidence: cancel alarm", rules.plausible({"tool": "cancel_alarm"}, "cancel my alarm"), True)
    check("evidence: music play needs music words", rules.plausible({"tool": "music", "action": "play"},
                                                                    "how are you"), False)


# ---------------------------------------------------------------- room
class _Fx:
    def push(self, *_):
        pass


class _Music:
    loop = None

    def play(self, *a, **k):
        pass

    def stop(self, *a, **k):
        pass


class FakeMixer:
    volume, rate = 0.5, 24000
    fx, music = _Fx(), _Music()


def make_room():
    from fda.room import Room

    class Cfg:
        music_dir = str(ROOT / "nonexistent-music")
    changes = []

    async def ring(_):
        pass
    room = Room(Cfg(), FakeMixer(), changes.append, ring)
    return room, changes


def t_room():
    room, changes = make_room()
    run = lambda a: asyncio.run(room.execute(a))   # noqa: E731
    texts = lambda: [r["text"] + ("@" + r["time"] if r["time"] else "") for r in room.state["reminders"]]  # noqa
    for t, h in [("drink water", "8am"), ("call mother", "18:00"), ("exercise", "")]:
        run({"tool": "add_reminder", "text": t, "time": h})
    check("reminders added", texts(), ["drink water@08:00", "call mother@18:00", "exercise"])
    check("display told", len(changes) >= 3, True)
    run({"tool": "edit_reminder", "which": "call mother", "text": "call father"})
    check("edit text", texts(), ["drink water@08:00", "call father@18:00", "exercise"])
    run({"tool": "edit_reminder", "which": "last", "time": "7pm"})
    check("edit time of last", texts()[-1], "exercise@19:00")
    r = run({"tool": "edit_reminder", "which": "dentist", "text": "x"})
    check("edit unknown: not ok", r["ok"], False)
    run({"tool": "delete_reminder", "which": "number one"})
    check("delete number one", texts(), ["call father@18:00", "exercise@19:00"])
    run({"tool": "delete_reminder", "which": "the exercise reminder"})
    check("delete by words", texts(), ["call father@18:00"])
    r = run({"tool": "delete_reminder", "which": "the dentist"})
    check("delete unknown: not ok, lists them", (r["ok"], "call father" in r["note"]), (False, True))
    run({"tool": "add_reminder", "text": "buy rice"})
    run({"tool": "delete_reminder", "which": "all"})
    check("delete all", texts(), [])
    r = run({"tool": "delete_reminder", "which": "1"})
    check("delete with none: says so", r.get("speak"), "You don't have any reminders.")
    r = run({"tool": "list_reminders"})
    check("list none", r.get("speak"), "You don't have any reminders.")
    run({"tool": "add_reminder", "text": "stretch", "time": "15:00"})
    check("list one", run({"tool": "list_reminders"}).get("speak"), "You have one reminder: stretch at 3 PM.")

    al = lambda: [a["time"] for a in room.state["alarms"]]  # noqa: E731
    run({"tool": "alarm", "time": "6:30 am"})
    check("alarm set", al(), ["06:30"])
    check("cancel only alarm", run({"tool": "cancel_alarm"})["ok"], True)
    check("alarms empty", al(), [])
    for t in ("06:30", "7pm", "21:00"):
        run({"tool": "alarm", "time": t})
    r = run({"tool": "cancel_alarm"})
    check("cancel without time when 3 set: asks which", r["ok"], False)
    run({"tool": "cancel_alarm", "time": "7"})
    check("cancel '7' finds 7 pm", al(), ["06:30", "21:00"])
    run({"tool": "cancel_alarm", "time": "6:30"})
    check("cancel by time", al(), ["21:00"])
    run({"tool": "cancel_alarm", "time": "all"})
    check("cancel all", al(), [])
    check("cancel with none: says so", run({"tool": "cancel_alarm"}).get("speak"), "You don't have any alarms set.")
    check("bad alarm time", run({"tool": "alarm", "time": "banana"})["ok"], False)

    run({"tool": "timer", "seconds": 300, "label": "tea"})
    check("timer started", len(room.state["timers"]), 1)
    run({"tool": "cancel_timer"})
    check("timer cancelled", room.state["timers"], [])
    run({"tool": "lights", "power": "on", "color": "Cool White"})
    check("light colour stored lower case", room.state["lights"]["color"], "cool white")
    run({"tool": "lights", "brightness": 0})
    check("brightness 0 turns off", room.state["lights"]["power"], "off")
    check("thermostat 18 ok", run({"tool": "thermostat", "celsius": 18})["ok"], True)
    check("thermostat 50 refused", run({"tool": "thermostat", "celsius": 50})["ok"], False)
    check("thermostat kept at 18", room.state["thermostat_c"], 18.0)
    check("unknown tool", run({"tool": "teleport"})["ok"], False)


# ---------------------------------------------------------------- replies (NotForMe)
class FakeIO:
    def __init__(self):
        self.last_heard, self.turn, self.bot_speaking, self.media_playing = "hi", 1, False, False


def make_nfm():
    import pipecat.processors.frame_processor as fp
    from fda.pc.processors import NotForMe
    fp.FrameProcessor.process_frame = lambda self, f, d: asyncio.sleep(0)    # no pipeline needed
    n = NotForMe(io=FakeIO())
    n.out = []

    async def push(f, d=None):
        n.out.append(f)
    n.push_frame = push
    return n


async def _reply(n, chunks, calls=False):
    from pipecat.frames.frames import (FunctionCallsStartedFrame, LLMFullResponseEndFrame,
                                       LLMFullResponseStartFrame, LLMTextFrame)
    from pipecat.processors.frame_processor import FrameDirection as D
    await n.process_frame(LLMFullResponseStartFrame(), D.DOWNSTREAM)
    for c in chunks:
        await n.process_frame(LLMTextFrame(c), D.DOWNSTREAM)
    if calls:
        await n.process_frame(FunctionCallsStartedFrame(function_calls=[]), D.DOWNSTREAM)
    await n.process_frame(LLMFullResponseEndFrame(), D.DOWNSTREAM)


def _said(n):
    from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
    return "".join(f.text for f in n.out if isinstance(f, LLMTextFrame)) + \
        "".join(f" [again: {f.text}]" for f in n.out if isinstance(f, TTSSpeakFrame))


def _asked_again(n):
    from pipecat.frames.frames import LLMMessagesAppendFrame
    return any(isinstance(f, LLMMessagesAppendFrame) for f in n.out)


def t_replies():
    from pipecat.frames.frames import InterruptionFrame
    from pipecat.processors.frame_processor import FrameDirection as D

    async def go():
        n = make_nfm(); await _reply(n, ["(sil", "ent)"])
        check("(silent) not spoken", _said(n), "")
        n = make_nfm(); await _reply(n, ["The", " bird", " can", " be", " a", " parrot", " or", " a", " robin."])
        check("normal reply spoken", _said(n), "The bird can be a parrot or a robin.")
        n = make_nfm(); await _reply(n, ["Okay."])
        check("short reply spoken", _said(n), "Okay.")
        n = make_nfm(); await _reply(n, ["We", " We", " Okay", " The", " user", " wants", " Articuno", " bird."])
        check("thinking aloud dropped", (_said(n), _asked_again(n)), ("", True))
        n = make_nfm(); await _reply(n, ["We need to call", " edit_app now."], calls=True)
        check("thinking + tool call: dropped, not asked again", (_said(n), _asked_again(n)), ("", False))
        n = make_nfm(); await _reply(n, ["...", " …", " ---", " …", " The", " …", " …", " …"])
        check("dot garbage dropped", (_said(n), _asked_again(n)), ("", True))
        n = make_nfm(); await _reply(n, ["Sure, a parrot works.", " The user wants a parrot.", " Done."])
        check("thinking mid-reply: rest dropped", _said(n), "Sure, a parrot works.")
        n = make_nfm(); await _reply(n, ["Done, it's blue now."]); n.io.turn = 1
        await _reply(n, ["Done, it's blue now."])
        check("same sentence twice in a turn: once", _said(n), "Done, it's blue now.")

        async def interrupted(real_turn: bool, far: bool = True):
            n = make_nfm()
            from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
            await n.process_frame(LLMFullResponseStartFrame(), D.DOWNSTREAM)
            for c in ["Why did the scarecrow get promoted?", " Because he was outstanding in his field."]:
                await n.process_frame(LLMTextFrame(c), D.DOWNSTREAM)
            n.spoken, n.last_sentence = "Why did the scarecrow get promoted? Because he", "Because he"
            await n.process_frame(InterruptionFrame(), D.DOWNSTREAM)
            n.out.clear()
            if real_turn:
                n.io.turn += 1
            await n._maybe_resume(from_ignored=far)
            return _said(n)
        check("cut in, new request, then noise: no repeat", await interrupted(True), "")
        check("cut in by noise only: finishes the sentence", await interrupted(False),
              " [again: Because he was outstanding in his field.]")
        n = make_nfm()
        n.paused, n.paused_said, n.paused_done, n.paused_t = "rest of it", "first part", True, time.monotonic()
        check("'continue' gets the paused rest", n.take_paused(), ("rest of it", "first part", True))
        check("'continue' only once", n.take_paused(), None)
    asyncio.run(go())


# ---------------------------------------------------------------- agent (tabs, video, app, tweak) without servers
class FakeUI:
    def __init__(self):
        self.sent, self.clients, self.agent = [], [1], None

    def send(self, msg):
        self.sent.append(msg)
        if msg.get("type") == "app_nav" and self.agent is not None:      # the display answers at once
            reply = {"type": "app_result", "id": msg["id"], "ok": msg.get("target") != "nowhere",
                     "did": f"{msg['action']} {msg.get('target', '')}".strip()}
            asyncio.get_running_loop().call_soon(self.agent.on_app_result, reply)

    def last(self, kind):
        return next((m for m in reversed(self.sent) if m.get("type") == kind), None)


class FakeTools:
    def __init__(self):
        self.calls = []

    async def call(self, tool, args, timeout=0):
        self.calls.append((tool, args))
        if tool == "tweak_app":
            k, v = next(iter(args["changes"].items()))
            patch = {k: "🐦|flip" if v == "flip" else v}
            return {"ok": True, "url": "/apps/flappy-bird.html", "patch": patch, "changed": {k: patch[k]}}
        return {"ok": True}


def make_agent():
    from pipecat.processors.aggregators.llm_context import LLMContext
    from fda.pc.agent import Agent

    class Cfg:
        name = "Elkit"
    ui = FakeUI()
    a = Agent(Cfg(), None, FakeIO(), {}, ui=ui)
    ui.agent = a
    a.context = LLMContext()
    a.room, _ = make_room()
    a.tools_conn = FakeTools()
    return a, ui


def t_tabs():
    async def go():
        a, ui = make_agent()
        a._open_app_view("/apps/flappy-bird.html", "Flappy Bird")
        a._open_view("video", {"video_id": "abc", "title": "Lo-fi beats to study", "state": "playing"},
                     ["play lofi beats", "Lo-fi beats to study"], kindword="video")
        a._open_app_view("/apps/portfolio.html", "Portfolio site")
        a._open_view("links", {"items": [], "title": "Flights to Davao"}, ["cheapest flight to Davao"],
                     kindword="links")
        labels = [v["label"] for v in a.views]
        print("        tabs:", labels)
        check("4 tabs", len(labels), 4)
        check("tab names all different", len({l.split()[0] for l in labels}), 4)
        active = lambda: a._active()["label"] if a._active() else None   # noqa: E731
        game = next(v for v in a.views if "flappy" in v.get("url", ""))["label"]
        site = next(v for v in a.views if "portfolio" in v.get("url", ""))["label"]
        vid = next(v for v in a.views if v["kind"] == "video")["label"]
        cases = [(f"go to the {game.split()[0]} tab", game), ("switch to the video", vid),
                 (f"Elkit, open the {site.split()[0].lower()} tab", site), ("next tab", None),
                 ("go back to the flappy bird game", game), (f"{vid.split()[0]} tab", vid)]
        for said, want in cases:
            before = active()
            handled = await a._view_direct(said)
            if want is None:
                check(f"tabs: {said!r} moves on", (handled, active() != before), (True, True))
            else:
                check(f"tabs: {said!r}", (handled, active()), (True, want))
        check("tabs: go to the room", (await a._view_direct("go to the room"), ui.last("tab")["tab"]), (True, "room"))
        check("tabs: show the interface", (await a._view_direct("show the interface"), ui.last("tab")["tab"]),
              (True, "design"))
        check("tabs: not a tab ('go to contact' is for the app)", await a._view_direct("go to contact"), False)
        n = len(a.views)
        await a._view_direct("go to the davao tab")
        check("tabs: close this tab", (await a._view_direct("close this tab"), len(a.views)), (True, n - 1))
        check("tabs: the right one closed", any(v["kind"] == "links" for v in a.views), False)
        check("tabs: close the video tab", await a._view_direct("close the video tab"), True)
        check("tabs: video gone", any(v["kind"] == "video" for v in a.views), False)
        for i in range(10):
            a._open_view("links", {"items": [], "title": f"search {i}"}, [f"search number {i} words{i}"],
                         kindword="links")
        check("tabs: at most 8 kept", len(a.views), 8)
        check("tabs: names still unique", len({v["label"] for v in a.views}), 8)
    asyncio.run(go())


def t_video():
    async def go():
        a, ui = make_agent()
        a._open_view("video", {"video_id": "abc", "title": "Cats", "state": "playing"}, ["cats video"],
                     kindword="video")
        a.io.media_playing = True
        cases = [("pause the video", "pause"), ("Elkit, turn it up", "louder"), ("turn it down", "quieter"),
                 ("set the volume to 30 percent", "volume"), ("skip ahead 10 seconds", "forward"),
                 ("mute it", "mute"), ("unmute it", "unmute"), ("play it from the beginning", "restart"),
                 ("resume the video", "resume")]
        for said, want in cases:
            ui.sent.clear()
            handled = await a._media_direct(said)
            m = ui.last("video_cmd")
            check(f"video: {said!r}", (handled, m and m["action"]), (True, want))
        ui.sent.clear(); await a._media_direct("set the volume to 30 percent")
        check("video: volume level 30", ui.last("video_cmd").get("level"), 30)
        ui.sent.clear(); await a._media_direct("skip ahead 10 seconds")
        check("video: skip 10 s", ui.last("video_cmd").get("seconds"), 10)
        check("video: 'turn the lights up' is not for the video", await a._media_direct("turn the lights up"), False)
        a.io.media_playing = True
        await a._intercept("stop")
        check("video: bare 'stop' pauses it", ui.last("video_cmd")["action"], "pause")
        check("video: close it", (await a._media_direct("close the video"), a.video), (True, None))
        a._open_view("photos", {"photos": [{"title": "a"}, {"title": "b"}], "i": 0, "title": "a"}, ["bird photos"],
                     kindword="photos")
        await a._media_direct("next photo")
        check("photos: next", a.photo["i"], 1)
        await a._media_direct("show the previous picture")
        check("photos: previous", a.photo["i"], 0)
    asyncio.run(go())


def t_app():
    async def go():
        a, ui = make_agent()
        check("app: nothing open -> not handled", await a._nav_direct("go to contact"), False)
        a._open_app_view("/apps/portfolio.html", "Portfolio")
        cases = [("go to the contact tab", {"action": "go_to", "target": "contact"}),
                 ("Elkit, scroll down", {"action": "scroll", "target": "down"}),
                 ("scroll to the top", {"action": "scroll", "target": "top"}),
                 ("press space", {"action": "press", "target": "space"}),
                 ("click the send button", {"action": "click", "target": "send"}),
                 ("open the projects page", {"action": "go_to", "target": "projects"}),
                 ("go back", {"action": "back"})]
        for said, want in cases:
            ui.sent.clear()
            handled = await a._nav_direct(said)
            m = ui.last("app_nav") or {}
            got = {k: m[k] for k in ("action", "target") if k in m and (k != "target" or m[k])}
            check(f"app: {said!r}", (handled, got), (True, want))
        check("app: 'go to the room' is not inside the app", await a._nav_direct("go to the room"), False)
        check("app: target not found -> the LLM gets it", await a._nav_direct("click nowhere"), False)
    asyncio.run(go())


def t_tweak():
    async def go():
        a, ui = make_agent()
        a._open_app_view("/apps/flappy-bird.html", "Flappy Bird")
        a.app_keys = ["bird", "pipe", "sky", "ground", "text", "speed"]
        for said, key, value in [("flip the bird", "bird", "flip"), ("make the bird face right", "bird", "flip"),
                                 ("the bird is facing the wrong way", "bird", "flip"),
                                 ("make the pipes blue", "pipe", "blue"), ("change the sky to night", "sky", "night")]:
            a.tools_conn.calls.clear()
            handled = await a._direct_tweak(said)
            got = a.tools_conn.calls[-1][1]["changes"] if a.tools_conn.calls else None
            check(f"tweak: {said!r}", (handled, got), (True, {key: value}))
            if value == "flip":
                check(f"tweak: {said!r} tells the display", "|flip" in json.dumps(ui.last("app_set") or {},
                                                                                  ensure_ascii=False), True)
        check("tweak: a question is not a tweak", await a._direct_tweak("what options are there for the bird"), False)
    asyncio.run(go())


# ---------------------------------------------------------------- history sent to the LLM
def t_history():
    from fda.pc.fallback_llm import clean_messages
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "make the bird articuno"},
            {"role": "assistant", "content": "We We Okay The user wants Articuno bird."},
            {"role": "assistant", "content": "Done, it's Articuno now."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": ""}}]},
            {"role": "tool", "tool_call_id": "x", "content": "{}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "y", "function": {"name": "get_time"}}]},
            {"role": "tool", "tool_call_id": "y", "content": "{}"},
            {"role": "assistant", "content": "..."}]
    out = clean_messages(msgs)
    check("history: leaked reasoning removed", any("user wants" in str(m.get("content")) for m in out), False)
    check("history: real reply kept", any(m.get("content") == "Done, it's Articuno now." for m in out), True)
    check("history: nameless tool call and its result removed", any(m.get("tool_call_id") == "x" for m in out), False)
    check("history: good tool call kept", any(m.get("tool_call_id") == "y" for m in out), True)
    check("history: empty '...' turn removed", out[-1].get("role"), "tool")


# ---------------------------------------------------------------- tools / display
def t_tools():
    from fda import brain
    from fda.pc import agent as ag
    from fda.mcp_servers import live
    for name in ag.ROOM_TOOLS:
        check(f"room tool {name} has a schema", name in brain.TOOLS, True)
        for req in brain.REQUIRED.get(name, []):
            check(f"room tool {name}: required {req} exists", req in brain.TOOLS.get(name, {}), True)
    for name in ag.SLOW_TOOLS:
        check(f"slow tool {name} is given to the agent", name in ag.AGENT_MCP_TOOLS, True)
    check("show_photo not given to the agent", "show_photo" in ag.AGENT_MCP_TOOLS, False)
    check("flip: is_flipped", live.is_flipped("🐦|flip"), True)
    check("flip: plain not flipped", live.is_flipped("🐦"), False)
    v, _ = live.resolve_value("flip", "🐦", None)
    check("flip: flip toggles on", v, "🐦|flip")
    v, _ = live.resolve_value("flip", "🐦|flip", None)
    check("flip: flip again toggles off", v, "🐦")
    check("match_key: pipes -> pipe", live.match_key("pipes", ["bird", "pipe", "sky"]), "pipe")

    from fda.pc.quick import QuickAnswers
    q = QuickAnswers(ROOT / "quick_answers.txt", name="Elkit")
    hit = q.match("what is your name")
    check("quick answer: 'what is your name'", bool(hit and "Elkit" in hit[0]), True)
    check("quick answer: unrelated goes to the LLM", q.match("what's the cheapest flight to davao"), None)

    node = shutil.which("node")
    if not node:
        print("        (node not found: JavaScript syntax not checked)")
        return
    html = (ROOT / "fda" / "web" / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    with tempfile.TemporaryDirectory() as d:
        for i, js in enumerate(scripts + [(ROOT / "fda" / "mcp_servers" / "live.js").read_text(encoding="utf-8")]):
            p = Path(d) / f"s{i}.js"
            p.write_text(js, encoding="utf-8")
            r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
            what = "live.js" if i == len(scripts) else f"index.html script {i + 1}"
            check(f"JavaScript parses: {what}", r.returncode == 0 or r.stderr[-300:], True)


# ---------------------------------------------------------------- replay real logs
def replay():
    from fda.pc.processors import is_bare_stop
    from fda.pc.services import speakable
    from fda import rules
    print("\n== replay: what was heard (logs/agent.log)")
    heard = []
    p = ROOT / "logs" / "agent.log"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"fda\.pc\.services heard \([^)]*, (-?\d+) dB\): (.*)$", line)
            if m:
                heard.append((int(m.group(1)), m.group(2)))
    counts = {}
    for db, text in heard[-400:]:
        how = ("stop (no reply)" if is_bare_stop(text, "Elkit") else
               "side talk, ignored" if rules.addressed_to_other(text, "Elkit") else
               "backchannel (ignored while talking)" if rules.is_backchannel(text) else
               "unfinished, waits for more" if rules.sounds_unfinished(text) else "to the LLM")
        counts[how] = counts.get(how, 0) + 1
        if how != "to the LLM":
            print(f"  {how:<36} {text[:70]}")
    print("  totals:", counts or "no heard lines in the log")

    print("\n== replay: what was said (logs/mcp-tts.log), lines today's voice filter changes")
    p = ROOT / "logs" / "mcp-tts.log"
    said = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"fda\.mcp\.tts \w+ (['\"])(.*)\1 -> ", line)
            if m:
                said.append(m.group(2))
    changed = 0
    for s in said:
        new = speakable(s)
        if new != s.strip():
            changed += 1
            print(f"  {s[:60]!r}\n     -> {new[:60]!r}")
    print(f"  {changed} of {len(said)} spoken lines would now be cut or changed")
    reps = sum(1 for a, b in zip(said, said[1:]) if a == b and len(a) > 12)
    print(f"  {reps} times the same sentence was said twice in a row in that log (before the repeat fix)")


SECTIONS = {"voice": t_voice, "thinking": t_thinking, "stop": t_stop, "rules": t_rules, "room": t_room,
            "replies": t_replies, "tabs": t_tabs, "video": t_video, "app": t_app, "tweak": t_tweak,
            "history": t_history, "tools": t_tools}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="only sections whose name contains this")
    ap.add_argument("--replay", action="store_true", help="also run the real logs through today's code")
    args = ap.parse_args()
    t0 = time.perf_counter()
    for name, fn in SECTIONS.items():
        if args.k and args.k not in name:
            continue
        section(name)
        try:
            fn()
        except Exception:  # noqa: BLE001
            FAIL.append((name, "crashed", traceback.format_exc(), ""))
            print("  CRASH\n" + traceback.format_exc())
        n_ok = sum(1 for s, *_ in PASS if s == name)
        n_bad = sum(1 for s, *_ in FAIL if s == name)
        print(f"  {n_ok} passed" + (f", {n_bad} FAILED" if n_bad else ""))
    if args.replay:
        replay()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed in {time.perf_counter() - t0:.1f} s (no LLM, no tokens)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
