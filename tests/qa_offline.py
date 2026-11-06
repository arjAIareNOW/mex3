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
import base64
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
    for s in ["Elkit, what's the weather?", "Turn on the lights", "Describe, a cat", "Okay, set a timer",
              "Articuno, the bird.", "Exercise, three study.", "Number one, drink water.", "Pikachu, yellow one."]:
        check(f"for me: {s!r}", rules.addressed_to_other(s, "Elkit"), False)
    for s in ["John, pass the salt.", "Ana, can you help me?", "Carlo, dinner is ready."]:
        check(f"side talk: {s!r}", rules.addressed_to_other(s, "Elkit"), True)
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


# ---------------------------------------------------------------- echo (its own voice through the mic)
def t_echo():
    from fda import rules
    said = ("Sure, why did the scarecrow get promoted? Because he was outstanding in his field. "
            "Do you want the blue one or the red one?")
    for heard in ["because he was outstanding in his field", "why did the scarecrow get promoted",
                  "Because he was outstanding in his fields.", "do you want the blue one or the red",
                  "the scarecrow get promoted because he was"]:
        check(f"echo: {heard!r}", rules.is_echo(heard, said), True)
    for heard in ["the blue one", "red one please", "no, tell me another joke", "blue", "yes",
                  "why did you say that", "make the bird blue and the pipes red"]:
        check(f"not echo: {heard!r}", rules.is_echo(heard, said), False)
    check("echo: nothing said lately", rules.is_echo("because he was outstanding in his field", ""), False)

    import pipecat.processors.frame_processor as fp
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection as D
    from fda.pc.processors import DuplexGuard, HeardTap
    fp.FrameProcessor.process_frame = lambda self, f, d: asyncio.sleep(0)

    async def go():
        io = FakeIO()
        io.bot_speaking = True
        heard_turns, ignored = [], []
        tap = HeardTap(io, None, name="Elkit", on_heard=lambda: heard_turns.append(1),
                       on_ignored=lambda: ignored.append(1), intercept=None, echo_text=lambda: said)
        out = []

        async def push(f, d=None):
            out.append(f)
        tap.push_frame = push
        await tap.process_frame(TranscriptionFrame("Because he was outstanding in his field.", "u", "t"),
                                D.DOWNSTREAM)
        check("echo while talking: no turn, no resume", (len(out), heard_turns, ignored), (0, [], []))
        io.bot_speaking, io.t_bot_stopped = False, time.monotonic() - 1.0
        await tap.process_frame(TranscriptionFrame("do you want the blue one or the red one", "u", "t"),
                                D.DOWNSTREAM)
        check("echo 1 s after it stopped: no turn", len(out), 0)
        await tap.process_frame(TranscriptionFrame("the blue one", "u", "t"), D.DOWNSTREAM)
        check("a short answer right after: is a turn", (len(out), len(heard_turns)), (1, 1))
        io.t_bot_stopped = time.monotonic() - 30
        await tap.process_frame(TranscriptionFrame("because he was outstanding in his field", "u", "t"),
                                D.DOWNSTREAM)
        check("same words long after it spoke: a turn (you quoting it)", len(out), 2)

        class Cfg:
            name = "Elkit"
        g = DuplexGuard(FakeIO(), Cfg())
        g.bot_text = lambda: "Do you want the blue one or the red one?"     # the sentence playing now
        g.echo_text = lambda: said                                           # and what came before it
        check("overlap: late echo of the previous sentence does not stop it",
              g.not_an_interruption("because he was outstanding in his field", b""), "echo")
        check("overlap: a real interruption still stops it",
              g.not_an_interruption("no wait tell me the weather instead", b""), "")

        # its own echo heard as "Not exactly." while ducked: full voice again as soon as that is decided
        class DIO(FakeIO):
            user_speaking = True
            ducks = []

            def duck(self, on):
                self.ducks.append(on)

        class Asr:
            async def call(self, *_a, **_k):
                return "Not exactly."
        dio = DIO()
        dio.bot_speaking = True
        g2 = DuplexGuard(dio, Cfg())
        g2.asr, g2.bot_text, g2.echo_text = Asr(), (lambda: "It's 9:51 PM."), (lambda: "")
        g2._set_duck(True)
        stopped = await g2._peek(b"\0" * 6400)
        check("overlap 'too short': voice unducked at once, not stopped", (stopped, g2.ducked, dio.ducks),
              (False, False, [True, False]))
        # fd-benchmark: stop sooner. 2 words count once the overlap is long enough; backchannels still never stop it
        long_audio, short_audio = b"\0" * int(1.3 * 32000), b"\0" * 6400
        check("overlap: 2 words after ~0.7 s of talking over it stop it ('turn left')",
              g.not_an_interruption("turn left", long_audio), "")
        check("overlap: 2 words in a short bit do not ('not exactly' = its own echo)",
              g.not_an_interruption("not exactly", short_audio), "too short")
        check("overlap: a long 'yeah okay' is still a backchannel", g.not_an_interruption("yeah okay", long_audio),
              "backchannel")
        check("guard: peeks from 0.5 s, fallback stop at 2 s", (DuplexGuard(FakeIO(), Cfg()).peek_s[0],
                                                               DuplexGuard(FakeIO(), Cfg()).sustained_s), (0.5, 2.0))
    asyncio.run(go())


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

        async def interrupted(real_turn: bool, far: bool = True, counted_first: bool = False):
            n = make_nfm()
            from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
            await n.process_frame(LLMFullResponseStartFrame(), D.DOWNSTREAM)
            for c in ["Why did the scarecrow get promoted?", " Because he was outstanding in his field."]:
                await n.process_frame(LLMTextFrame(c), D.DOWNSTREAM)
            n.spoken, n.last_sentence = "Why did the scarecrow get promoted? Because he", "Because he"
            if counted_first:                 # the real order: the request is counted, then its interruption comes
                n.io.turn += 1
                n.io.t_turn = time.monotonic()
            await n.process_frame(InterruptionFrame(), D.DOWNSTREAM)
            n.out.clear()
            if real_turn and not counted_first:
                n.io.turn += 1
            await n._maybe_resume(from_ignored=far)
            return _said(n)
        check("cut in, new request, then noise: no repeat", await interrupted(True), "")
        check("request counted just before its interruption, then noise: no repeat",
              await interrupted(True, counted_first=True), "")
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

        # fd-benchmark 2026-11-06: a clip asked for a music video; with no display nothing plays, yet 12 turns
        # were ignored ("video playing, not called by name"). Only a connected display plays videos.
        import pipecat.processors.frame_processor as fp
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection as D
        from fda.pc.processors import HeardTap
        fp.FrameProcessor.process_frame = lambda self, f, d: asyncio.sleep(0)
        io = FakeIO()
        io.media_playing = True
        for clients, want in (([], 1), ([1], 0)):
            ui2 = FakeUI()
            ui2.clients = clients
            tap = HeardTap(io, ui2, name="Elkit", intercept=None, echo_text=lambda: "")
            out = []

            async def push(f, d=None):
                out.append(f)
            tap.push_frame = push
            await tap.process_frame(TranscriptionFrame("Add budget review to my calendar next Monday.", "u", "t"),
                                    D.DOWNSTREAM)
            check(f"video flag with {'a display' if clients else 'no display'}: "
                  f"{'ignored unless called by name' if clients else 'still answers'}", len(out), want)
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
        for said, key, value in [("flip the bird", "bird", "flip"), ("make the bird face right", "bird", "face right"),
                                 ("the bird is facing the wrong way", "bird", "flip"),
                                 ("make the pipes blue", "pipe", "blue"), ("change the sky to night", "sky", "night")]:
            a.tools_conn.calls.clear()
            handled = await a._direct_tweak(said)
            await a._flush_held()                # a colour is held a moment ("Green, okay."): made at once here
            got = a.tools_conn.calls[-1][1]["changes"] if a.tools_conn.calls else None
            check(f"tweak: {said!r}", (handled, got), (True, {key: value}))
            if value == "flip":          # (the fake server here does not resolve "face right"; t_difficulty does)
                check(f"tweak: {said!r} tells the display", "|flip" in json.dumps(ui.last("app_set") or {},
                                                                                  ensure_ascii=False), True)
        check("tweak: a question is not a tweak", await a._direct_tweak("what options are there for the bird"), False)
    asyncio.run(go())


# ---------------------------------------------------------------- room commands without the LLM
class FakeWorker:
    def __init__(self):
        self.said = []

    async def queue_frames(self, frames):
        self.said += [getattr(f, "text", "") for f in frames]


def t_fastroom():
    async def go():
        a, ui = make_agent()
        a.worker = FakeWorker()
        st = a.room.state
        cases = [("Can you turn on the lights?", lambda: st["lights"]["power"] == "on"),
                 ("Elkit, lights off please", lambda: st["lights"]["power"] == "off"),
                 ("make the lights blue", lambda: st["lights"]["color"] == "blue"),
                 ("dim the lights to 30 percent", lambda: st["lights"]["brightness"] == 30),
                 ("set a timer for 5 minutes", lambda: st["timers"] and st["timers"][-1]["seconds"] == 300),
                 ("set the thermostat to 22 degrees", lambda: st["thermostat_c"] == 22.0),
                 ("volume 30", lambda: st["volume"] == 30),
                 ("what are my reminders", lambda: True)]
        for said, ok in cases:
            n = len(a.worker.said)
            handled = await a._room_direct(said)
            check(f"room fast: {said!r}", (handled, bool(ok())), (True, True))
            check(f"room fast: {said!r} answered", len(a.worker.said) > n or said.startswith("volume"), True)
        n = len(a.context.get_messages())
        await a._room_direct("turn on the lights")
        check("room fast: the LLM's history knows it", len(a.context.get_messages()), n + 2)
        a.worker.said.clear()
        await a._room_direct("what time is it")
        check("room fast: true time, no LLM", bool(a.worker.said and a.worker.said[0].startswith("It's ")), True)
        for said in ["make the bird blue", "turn on the lights in the hallway and then tell me a story",
                     "what's the weather in Davao", "can you make the temperature around 18 degrees",
                     "stop"]:
            check(f"room fast: not taken: {said!r}", await a._room_direct(said), False)
        check("room fast: set thermostat 50 refused", (await a._room_direct("set the thermostat to 50 degrees"),
                                                       st["thermostat_c"]), (False, 22.0))
    asyncio.run(go())


# ---------------------------------------------------------------- DGX3 off the VPN: speech never waits
def t_dgx3():
    import os
    import socket
    from fda.mcp_servers.dgx3 import DGX3

    class Cfg:
        asr_url = tts_url = "http://127.0.0.1:9"        # nothing listens there
        tts_model = asr_model = "x"; voice = "ryan"; language = "English"; tts_rate = 24000; voice_style = ""
    os.environ.setdefault("QWEN_API_KEY", "test")
    d = DGX3(Cfg(), retry_s=0.2)
    time.sleep(0.3)
    check("dgx3 unreachable: marked down by the background check", d.available, False)
    t = time.perf_counter()
    r = d.speak("hello")
    check("dgx3 down: speak gives up at once (no timeout)", (r is None, time.perf_counter() - t < 0.05), (True, True))
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(5)
    port = srv.getsockname()[1]
    d.cfg.asr_url = d.cfg.tts_url = f"http://127.0.0.1:{port}"     # the VPN comes back
    for _ in range(40):
        if d.available:
            break
        time.sleep(0.1)
    check("dgx3 back: found by the background check", d.available, True)
    srv.close()


def t_polish():
    import asyncio
    import base64
    import numpy as np
    from fda.mcp_servers.chain import CloudASR
    from fda.pc import processors as P
    from fda.pc.services import McpTTSService, _EXTRA_OK

    # Whisper hint: its prompt said back on noise is dropped; the name alone, or real speech, is kept
    asr = CloudASR.__new__(CloudASR)
    asr.hint = "Elkit, Manila."
    asr._hint_words = {"elkit", "manila"}
    asr.failed = asr.ok = lambda *a: None

    class R:
        status_code = 200
        def __init__(self, t): self.t = t
        def json(self): return {"text": self.t}

    class H:
        def __init__(self, t): self.t = t
        def post(self, *a, **k): return R(self.t)
    asr.base, asr.model = "x", "x"
    for said, want in [("Elkit, Manila.", ""), ("Manila Elkit", ""), ("Elkit?", "Elkit?"),
                       ("Elkit, turn on the lights.", "Elkit, turn on the lights."), ("...", "")]:
        asr.http = H(said)
        check(f"whisper hint: {said!r}", asr(np.zeros(1600, np.float32), 16000, "en"), want)

    # a second "All set." after the real confirmation is not said; a first one, or a real sentence, is
    for t, want in [("All set.", True), ("Done.", True), ("You're all set!", True), ("Okay.", True),
                    ("Lights are on.", False), ("Done, the bird is blue.", False), ("Sure, here it is.", False)]:
        check(f"extra ok pattern: {t!r}", bool(_EXTRA_OK.match(t)), want)

    class Conn:
        def __init__(self): self.said = []
        async def call(self, tool, args, timeout=0):
            self.said.append(args["text"])
            return {"audio_b64": base64.b64encode(bytes(960)).decode(), "sample_rate": 24000}
    conn = Conn()
    tts = McpTTSService(conn)

    async def say(text, ctx):
        try:
            async for _ in tts.run_tts(text, ctx):
                pass
        except Exception:  # noqa: BLE001  (no running pipeline here: only what was sent to the voice matters)
            pass
    for text, ctx in [("Lights are on.", "a"), ("All set.", "a"), ("All set.", "b"), ("Done.", "c"),
                      ("Bird is blue.", "c"), ("Okay.", "c")]:
        asyncio.run(say(text, ctx))
    check("second confirmation not said; first one is", conn.said,
          ["Lights are on.", "All set.", "Done.", "Bird is blue."])

    # turn timing: each step measured once per turn, logged at the first audio
    c = P.TurnClock()
    seq = [("speech_end", 0.0), ("heard", 0.5), ("sent", 0.9), ("first_word", 1.3), ("first_word", 1.5),
           ("first_audio", 1.8), ("first_audio", 2.5)]
    real = time.monotonic
    try:
        for name, at in seq:
            time.monotonic = lambda at=at: 100.0 + at
            c.mark(name)
    finally:
        time.monotonic = real
    check("turn timing steps", c.last, {"asr": 500, "turn wait": 400, "llm": 400, "tts": 500, "total": 1800})
    c2 = P.TurnClock()
    c2.mark("first_word"); c2.mark("first_audio")
    check("turn timing: nothing logged without a turn sent", c2.last, {})


def t_assets():
    """Elkit's own art library: files present, voice tweaks find it, tints and direction kept, app writer told."""
    import xml.etree.ElementTree as ET
    from fda.mcp_servers import live as L
    from fda.mcp_servers import tools_server as T
    man = json.loads((L.ASSETS / "manifest.json").read_text(encoding="utf-8"))["items"]
    missing = [i["path"] for i in man if not (L.ASSETS.parent / i["path"].lstrip("/")).exists()]
    check("library: every manifest entry has its file", missing, [])
    bad = []
    for i in man:
        try:
            ET.parse(L.ASSETS.parent / i["path"].lstrip("/"))
        except ET.ParseError:
            bad.append(i["name"])
    check("library: every SVG is valid XML", bad, [])
    check("library: has sprites, tiles, backgrounds, icons",
          sorted({i["kind"] for i in man}), ["background", "icon", "sprite", "tile"])
    b = "/assets/sprites/bird.svg"
    rv = lambda v, cur: L.resolve_value(v, cur, web=False)[0]
    check("voice: thing with no emoji -> library picture, still facing left like the emoji (offline)",
          rv("spaceship", "🐦"), "/assets/sprites/spaceship.svg|flip")
    check("voice: app drawing library art stays library art", rv("fish", b), "/assets/sprites/fish.svg")
    check("voice: plural / article", rv("the robots", b), "/assets/sprites/robot.svg")
    check("voice: colour tints a library picture", rv("black", b), b + "@black")
    check("voice: re-tint replaces the tint", rv("red", b + "@black"), b + "@red")
    check("voice: mirrored emoji -> library art not mirrored (both face right)", rv("spaceship", "🐦|flip"),
          "/assets/sprites/spaceship.svg")
    check("voice: library art -> emoji gets mirrored (keeps facing right)", rv("dragon", b), "🐉|flip")
    check("voice: emoji stays emoji when the app uses emoji", rv("bird", "🙂"), "🐦")
    check("voice: two-word background", rv("night sky", "#000"), "/assets/backgrounds/sky-night.svg")
    rule = T.art_rule()
    check("app writer is told the library and its names",
          ("/assets/sprites/NAME.svg" in rule, " spaceship," in rule, "sky-night" in rule, rule in T.APP_SYSTEM),
          (True, True, True, True))
    node = shutil.which("node")
    if node:
        js = Path(__file__).resolve().parent / "live_draw_check.js"
        if js.exists():
            r = subprocess.run([node, str(js), str(L.HERE / "live.js")], capture_output=True, text=True, timeout=20)
            check("FDA.draw: pictures, tints, flips, tiles (node)", (r.returncode, r.stdout.count("FAIL")), (0, 0))


def t_choices():
    """The pick pop-up: 4 numbered pictures / colour schemes on the display, picked by number, name, colour or tap."""
    from fda.pc import choices as C
    from fda.pc.agent import Agent
    birds = C.art_choice(C.group_of("birds")[1], "/assets/sprites/bird.svg")
    check("other birds: 4, not the one showing", [o["name"] for o in birds], ["parrot", "owl", "penguin", "duck"])
    check("group from the picture showing", C.group_of("", "/assets/sprites/owl.svg@red|flip")[0], "bird")
    check("group by a word for it", (C.group_of("background")[0], C.group_of("the player")[0], C.group_of("cars")[0]),
          ("sky", "hero", "car"))
    check("no group: None", C.group_of("spreadsheet"), None)
    picks = {"number two": 1, "the second one": 1, "two": 1, "number one": 0, "last one": 3, "I like number 4": 3,
             "the owl": 1, "I'll take the penguin": 2, "okay three": 2, "duck please": 3}
    check("picks by number and name", {t: C.parse_pick(t, birds) for t in picks}, picks)
    check("not a pick: a request, a long sentence",
          [C.parse_pick(t, birds) for t in ("make the bird bigger", "what is the weather like in Manila today then")],
          [None, None])
    cars = C.art_choice(C.group_of("car")[1], "/assets/sprites/truck.svg")
    check("longest name wins ('race car' is not 'car')", (C.parse_pick("race car", cars), C.parse_pick("the car", cars)),
          (1, 0))
    pals = C.palette_choice()
    check("colour schemes: 4", [p["name"] for p in pals], ["ocean", "sunset", "forest", "midnight"])
    check("colour schemes: by colour / dark / name",
          [C.parse_pick(t, pals) for t in ("the blue one", "the green one", "the dark one", "sunset")], [0, 2, 3, 1])
    check("more / close words", (C.is_more("show me more"), C.is_more("other ones"), C.is_close("none of them"),
                                 C.is_close("never mind"), C.is_close("the owl")), (True, True, True, True, False))
    pop = C.popup("Pick a bird", birds)
    check("display message", (pop["type"], [o["n"] for o in pop["options"]], pop["options"][0]["label"],
                              pop["options"][0]["img"]), ("choices", [1, 2, 3, 4], "Parrot", "/assets/sprites/parrot.svg"))

    # the agent side, with stand-ins for the display, the tools server and the speech
    class UI:
        def __init__(self):
            self.sent = []
        def send(self, m):
            self.sent.append(m)

    class Tools:
        def __init__(self):
            self.calls = []
        async def call(self, name, args, timeout=0):
            self.calls.append((name, args))
            if name == "app_settings":
                return {"ok": True, "settings": "bird=/assets/sprites/bird.svg, pipe=#2e8b57, speed=3"}
            if name == "tweak_app":
                return {"ok": True, "url": "/apps/sky.html", "patch": dict(args["changes"]), "changed": "x"}
            return {"ok": True, "building": True}

    class Ctx:
        def __init__(self):
            self.msgs = []
        def add_message(self, m):
            self.msgs.append(m)

    class Cfg:
        name = "Elkit"

    def agent():
        a = Agent.__new__(Agent)
        a.cfg, a.ui, a.tools_conn, a.context = Cfg(), UI(), Tools(), Ctx()
        a.choice, a.picked, a.views, a.tasks, a.events = None, {}, [], [], []
        a.app_url, a.app_keys = "/apps/sky.html", ["bird", "pipe", "speed"]
        a.said = []

        class Worker:
            def __init__(self):
                self.frames = []
            async def queue_frames(self, fs):
                self.frames += fs
        a.worker = Worker()

        async def announce(t):
            a.said.append(t)
        a._announce = announce
        return a

    async def go():
        a = agent()
        r = await a._offer("bird")
        check("agent: offer for the app on screen", (r["ok"], a.choice["key"], a.choice["for_new"],
                                                     [o["name"] for o in a.choice["options"]]),
              (True, "bird", False, ["parrot", "owl", "penguin", "duck"]))
        check("agent: pop-up sent, Interface tab shown", [m["type"] for m in a.ui.sent], ["tab", "choices"])
        check("agent: 'what's the weather' is not a pick", await a._choice_direct("what's the weather"), False)
        check("agent: 'number two' picked at once", await a._choice_direct("Number two."), True)
        check("agent: the app's bird became the owl", a.tools_conn.calls[-1],
              ("tweak_app", {"changes": {"bird": "/assets/sprites/owl.svg"}}))
        check("agent: pop-up closed, said so", (a.choice, {"type": "choices", "close": True} in a.ui.sent, a.said[-1]),
              (None, True, "Done, the bird is the owl now."))
        check("agent: live change pushed to the app", any(m.get("type") == "app_set" for m in a.ui.sent), True)
        check("agent: the brain's history knows", "picked: bird = owl" in a.context.msgs[-2]["content"], True)

        a = agent()
        await a._offer("colour scheme", for_new=True)
        await a._choice_direct("show me more")
        check("agent: 'show me more' = the next four schemes", [o["name"] for o in a.choice["options"]],
              ["candy", "coffee", "mono", "neon"])
        a.said.clear()
        await a._choice_direct("the dark one")
        check("agent: scheme for a site not built yet: kept for the build, nothing edited",
              ("neon" in a.picked.get("colours", ""), [c[0] for c in a.tools_conn.calls]), (True, []))
        f = a.worker.frames[-1] if a.worker.frames else None
        check("agent: ... and the brain carries on with the build (not a dead end)",
              (type(f).__name__, getattr(f, "run_llm", None), "#ff2d75" in str(getattr(f, "messages", ""))),
              ("LLMMessagesAppendFrame", True, True))
        check("agent: nothing said over it (the brain answers)", a.said, [])

        a = agent()
        a.app_keys = []
        await a._offer("colour scheme")
        await a._choice_direct("ocean")
        a = agent()
        await a._offer("colour scheme")
        await a._choice_direct("the green one")
        check("agent: scheme for the site on screen: an edit with the colours",
              (a.tools_conn.calls[-1][0], "#2e7d4f" in a.tools_conn.calls[-1][1]["change"]), ("edit_app", True))

        a = agent()
        await a._offer("bird")
        check("agent: 'never mind' closes", (await a._choice_direct("never mind"), a.choice, a.said[-1]),
                                            (True, None, "Okay."))
        await a._offer("bird")
        check("agent: 'you pick' closes and leaves it to the brain", (await a._choice_direct("you pick"), a.choice),
              (False, None))
        await a._offer("bird")
        a.on_choice({"n": 3})
        await asyncio.gather(*a.tasks)
        check("agent: a tap on card 3", (a.tools_conn.calls[-1][1], a.choice),
              ({"changes": {"bird": "/assets/sprites/penguin.svg"}}, None))
        r = await a._offer("spreadsheet")
        check("agent: nothing in the library: say ideas instead", (r["ok"], a.choice), (False, None))
        await a._offer("bird")
        a.choice["t"] -= 400
        check("agent: an old pop-up goes away by itself", (await a._choice_direct("number one"), a.choice), (False, None))

    asyncio.run(go())
    from fda.mcp_servers import live as L
    check("a library path as a tweak value passes through (no photo search)",
          L.resolve_value("/assets/sprites/owl.svg", "/assets/sprites/bird.svg", web=False),
          ("/assets/sprites/owl.svg", "the owl picture"))
    check("spoken name of a background", L.resolve_value("night sky", "#000", web=False)[1], "the night sky picture")


def t_watchfixes():
    """Fixes from the first live view run (tests/pc_watch.py, 2026-11-05)."""
    from fda import rules
    from fda.pc.services import speakable
    check("leak: '(should we open?) Actually need to open app.' is not said",
          speakable("Your game is ready. (should we open?) Actually need to open app. It's on the Flappy game tab."),
          "Your game is ready. It's on the Flappy game tab.")
    check("leak: plain facts in brackets stay", speakable("The bakery opens at nine (on weekdays)."),
          "The bakery opens at nine (on weekdays).")
    check("leak: 'Actually, I think ...' is talk, not thinking", rules.is_thinking("Actually, I think the blue one is nicer."),
          False)
    check("leak: new tool names count as thinking", [rules.is_thinking(t) for t in
          ("I'll call show_choices now.", "Use list_items for that.", "pick_choice two")], [True, True, True])

    # open_app while that very app is still being built: "it opens by itself", not a retry loop
    from fda.mcp_servers import tools_server as T
    import inspect
    src = inspect.getsource(T)
    check("open_app knows about a build in progress", "is still being built; it opens by itself" in src, True)

    # a request while a timer rings stops the ringing first (the person noticed it)
    from fda.pc.agent import Agent
    a = Agent.__new__(Agent)

    class Room:
        def __init__(self):
            self.state = {"ringing": "10 second timer"}
        def stop_ringing(self):
            self.state["ringing"] = None

    class Cfg:
        name = "Elkit"
    a.room, a.cfg = Room(), Cfg()

    async def no(*_a, **_k):
        return False
    for m in ("_continue", "_choice_direct", "_view_direct", "_media_direct", "_nav_direct", "_room_direct", "_quick",
              "_direct_tweak"):
        setattr(a, m, no)
    asyncio.run(a._intercept("What time is it?"))
    check("a request while a timer rings stops the ringing", a.room.state["ringing"], None)
    a.room.state["ringing"] = "alarm"
    asyncio.run(a._intercept("uh"))
    check("a single sound does not", a.room.state["ringing"], "alarm")

    # the far-away check is off while the alarm / music plays (the echo canceller turns the person down too)
    import base64 as b64
    import numpy as np
    from fda.pc.services import McpSTTService

    class Conn:
        async def call(self, *_a, **_k):
            return "What time is it?"
    stt = McpSTTService.__new__(McpSTTService)
    stt.conn, stt.far_db, stt.user_db, stt.far_dropped, stt.on_ignored, stt.own_name = Conn(), 10, -14.0, 0, None, "Elkit"
    stt.last_ms, stt._user_id = 0.0, "user"

    async def nothing():
        pass
    stt.start_processing_metrics = stt.stop_processing_metrics = nothing
    quiet = (np.sin(np.arange(32000) / 8) * 0.012 * 32767).astype("<i2").tobytes()      # about -41 dB
    for busy, want in ((False, 0), (True, 1)):
        stt.output_busy = (lambda b=busy: b)

        async def run():
            return [f async for f in stt.run_stt(quiet) if type(f).__name__ == "TranscriptionFrame"]
        got = asyncio.run(run())
        check(f"quiet speech {'during the alarm: heard' if busy else 'in a quiet room: far away, ignored'}",
              len(got), want)
    check("a level under the alarm does not change the learned level", stt.user_db, -14.0)


def t_mind():
    """User 2026-11-06: "the wait no, change it to ... concept should work on the agent totality. not just games"."""
    from fda.pc import mind as M
    for f, s, want, lead in [
            ("Set a timer for 5 minutes.", "Wait, no, 10 minutes.", "set a timer for 10 minutes",
             "Okay, not 5 minutes: 10 minutes instead."),
            ("Set a timer for 5 minutes.", "Actually, ten.", "set a timer for ten minutes", None),
            ("Turn the lights blue.", "Wait, no. I want them to be red instead.", "turn the lights red",
             "Okay, not blue: red instead."),
            ("Turn on the lights.", "Actually, turn them off.", "turn the lights off", "Okay, not on: off instead."),
            ("Add milk to my shopping list.", "No wait, eggs.", "add eggs to my shopping list",
             "Okay, not milk: eggs instead."),
            ("Add milk to my shopping list.", "I meant the to-do list.", "add milk to my to do list", None),
            ("Make me a snake game.", "Wait, no, a racing game instead.", "make me a racing game",
             "Okay, not snake game: racing game instead."),
            ("Open the room tab.", "I meant the office tab.", "open the office tab", None),
            ("Change the sky to the night sky.", "Wait, no, the beach instead.", "change the sky to the beach", None),
            ("Make me a website for my coffee shop called Bean There.", "Sorry, Bean Here.",
             "make me a website for my coffee shop called bean here", None),
            ("Set an alarm for 7 am.", "Wait, no, 8 am.", "set an alarm for 8 am", "Okay, not 7 am: 8 am instead."),
            ("Remind me to call mom at 5 pm.", "No wait, 6.", "remind me to call mom at 6 pm", None),
            ("Call mom.", "Wait no, dad.", "call dad", None),
            ("What is the weather in Manila?", "Actually, in Cebu.", "what is the weather in cebu", None),
            ("Show me a photo of a cat.", "Wait, no, a dog.", "show me a photo of a dog", None)]:
        m = M.merge(f, s)
        check(f"mind: {f!r} + {s!r}", (m and m["merged"], lead and m and M.lead(m)), (want, lead))

    async def go():
        a, ui = make_agent()
        a.worker = FakeWorker()
        said = lambda: a.worker.said[-1] if a.worker.said else ""      # noqa: E731
        await a._intercept("Set a timer for 5 minutes.")
        await a._intercept("Wait, no, 10 minutes.")
        check("mind: timer 5 -> 10: one timer, 10 minutes, 'not 5 minutes'",
              ([t["seconds"] for t in a.room.state["timers"]], said().startswith("Okay, not 5 minutes: 10 minutes instead.")),
              ([600], True))
        await a._intercept("Turn the lights blue.")
        await a._intercept("Actually, red.")
        check("mind: lights blue -> 'actually, red'", (a.room.state["lights"]["color"], said().startswith("Okay, not blue: red")),
              ("red", True))
        before = a.room.state["thermostat_c"]
        await a._intercept("Set the thermostat to 22 degrees. Wait, no, 20.")
        check("mind: in one breath: thermostat 20 (22 never set)", (before != 22, a.room.state["thermostat_c"],
                                                                  said().startswith("Okay, not 22 degrees")), (True, 20.0, True))
        await a._intercept("Add milk to my shopping list.")
        await a._intercept("No wait, eggs instead.")
        check("mind: list milk -> eggs", a.room.state["lists"].get("shopping"), ["eggs"])
        await a._intercept("Set a timer for 3 minutes.")
        n = len(a.room.state["timers"])
        await a._intercept("Actually, what time is it?")
        check("mind: 'actually, what time is it?' is a question: the timer stays", (len(a.room.state["timers"]) == n,
                                                                                    said().startswith("It's ")), (True, True))
        # not a fast request: the LLM does it, told what changed
        handled = await a._intercept("Remind me to call mom at 5 pm.")
        handled2 = await a._intercept("Wait, no, 6.")
        note = a.context.get_messages()[-1]
        check("mind: LLM request 'remind me at 5' -> 'wait, no, 6': the LLM is told the new one and the lead",
              (handled, handled2, note.get("role"), "remind me to call mom at 6 pm" in note.get("content", ""),
               "Okay, not 5 pm: 6 pm instead." in note.get("content", ""), "may be done already" in note.get("content", "")),
              (False, False, "system", True, True, True))
        await a._intercept("Set a timer for 5 minutes.")
        n = len(a.room.state["timers"])
        await a._intercept("We know make it 10 minutes.")
        check("mind: 'wait, no' misheard ('we know make it 10 minutes'): still the timer, 10 minutes",
              ([t["seconds"] for t in a.room.state["timers"]][-1], len(a.room.state["timers"]) == n,
               said().startswith("Okay, not 5 minutes")), (600, True, True))
        # fd-benchmark: a fresh session between trials (history, views, video, timers, lists)
        a._open_view("video", {"video_id": "v1", "title": "Bloom", "state": "playing"}, ["music video"], kindword="video")
        a.io.media_playing = True
        a.reset_session()
        from fda.pc.agent import FEW_SHOT
        check("reset_session: history, views, video, timers, lists all fresh",
              (len(a.context.get_messages()) == len(FEW_SHOT), a.views, a.io.media_playing, a.room.state["timers"],
               a.room.state["lists"], a._last_act), (True, [], False, [], {}, None))
        a.worker.said.clear()
        check("mind: 'sorry, I didn't hear you' is not a correction", await a._intercept("Sorry, I didn't hear you."),
              False)
    asyncio.run(go())

    # a build still running, then "wait, no, a racing game": the first build is dropped, never saved or announced
    import tempfile
    import threading
    from fda.mcp_servers import tools_server as T

    class Coder:
        engine, model = "fake", "fake"

        def __init__(self):
            self.asked, self.go = [], threading.Semaphore(0)

        def eta_s(self, kind):
            return 1

        def complete(self, msgs, max_tokens=0, temperature=0, stop_when=None):
            self.asked.append(msgs[-1]["content"])
            self.go.acquire(timeout=5)
            name = "Snake" if "snake" in msgs[-1]["content"] else "Racing"
            return f"<html><head><title>{name} game</title></head><body>ok</body></html>"

    d = Path(tempfile.mkdtemp(prefix="qa-mind-"))
    apps = T.Apps.__new__(T.Apps)
    apps.cfg, apps.dir, apps.coder, apps.max_tokens = None, d, Coder(), 1000
    apps.job, apps.lock, apps.current, apps.done = None, threading.Lock(), None, []
    apps.commit = lambda *a, **k: None
    saved = (T.live.run_check, T.js_syntax_error, T.live.runtime_problem)
    T.live.run_check, T.js_syntax_error, T.live.runtime_problem = (lambda h: {}), (lambda h: ""), (lambda r, g: "")
    try:
        r1 = apps.start("build", "a snake game", "")
        r2 = apps.start("build", "a racing game", "", replace=True)
        check("mind: build replaced while building", (r1.get("building"), r2.get("building")), (True, True))
        apps.coder.go.release()
        apps.coder.go.release()
        for _ in range(80):
            if apps.done:
                break
            time.sleep(0.05)
        time.sleep(0.2)
        check("mind: only the racing game is saved and announced",
              ([j["title"] for j in apps.done], sorted(p.name for p in d.glob("*.html"))),
              (["Racing game"], ["a-racing-game.html"]))
    finally:
        T.live.run_check, T.js_syntax_error, T.live.runtime_problem = saved


def t_appjobs():
    """A change asked for while an edit is still being made is queued and done right after (not lost)."""
    import tempfile
    import threading
    from pathlib import Path
    from fda.mcp_servers import tools_server as T

    class Coder:
        engine, model = "fake", "fake"
        def __init__(self):
            self.asked, self.go = [], threading.Semaphore(0)
        def eta_s(self, kind):
            return 1
        def complete(self, msgs, max_tokens=0, temperature=0, stop_when=None):
            self.asked.append(msgs[-1]["content"])
            self.go.acquire(timeout=5)             # "writing the code" until the test lets it finish
            return "<html><head><title>Bird game</title></head><body>ok</body></html>"

    d = Path(tempfile.mkdtemp(prefix="qa-apps-"))
    (d / "bird.html").write_text("<html><head><title>Bird game</title></head><body>v1</body></html>", encoding="utf-8")
    (d / "bird.json").write_text('{"title": "Bird game", "live": {}}', encoding="utf-8")
    apps = T.Apps.__new__(T.Apps)
    apps.cfg, apps.dir, apps.coder, apps.max_tokens = None, d, Coder(), 1000
    apps.job, apps.lock, apps.current, apps.done = None, threading.Lock(), "bird", []
    apps.commit = lambda *a, **k: None
    saved = (T.live.run_check, T.js_syntax_error, T.live.runtime_problem)
    T.live.run_check, T.js_syntax_error, T.live.runtime_problem = (lambda h: {}), (lambda h: ""), (lambda r, g: "")
    try:
        p = apps.find("")
        r1 = apps.start("edit", "make the bird green", "bird", base=p)
        r2 = apps.start("edit", "no wait, make it black instead", "bird", base=p)
        r3 = apps.start("edit", "and make the pipes bigger", "bird", base=p)
        r4 = apps.start("build", "a new snake game", "")
        check("edit starts", (r1["ok"], r1.get("building")), (True, True))
        check("change during an edit: queued, not refused", (r2["ok"], r2.get("queued"), r2.get("waiting")),
              (True, True, 1))
        check("second change also queued", r3.get("waiting"), 2)
        check("a new app during a build: still refused", r4["ok"], False)
        apps.coder.go.release()                    # the first edit finishes ...
        for _ in range(50):
            if len(apps.coder.asked) >= 2:
                break
            time.sleep(0.05)
        check("queued changes start by themselves after it", len(apps.coder.asked), 2)
        last = apps.coder.asked[-1]
        check("queued changes go in together, in order, later wins",
              ("(1) no wait, make it black instead" in last, "(2) and make the pipes bigger" in last,
               "later one wins" in last), (True, True, True))
        apps.coder.go.release()
        for _ in range(50):
            if len(apps.done) >= 2:
                break
            time.sleep(0.05)
        check("both edits finish and are announced", [j["state"] for j in apps.done], ["ready", "ready"])
        check("nothing left waiting", apps.job.get("queued"), None)
        r5 = apps.start("edit", "make it yellow", "bird", base=p)
        apps.coder.go.release()
        check("after that, a change starts at once again", (r5["ok"], r5.get("building")), (True, True))
        for _ in range(50):
            if len(apps.done) >= 3:
                break
            time.sleep(0.05)
    finally:
        T.live.run_check, T.js_syntax_error, T.live.runtime_problem = saved


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


# ---------------------------------------------------------------- the 3 tabs: Interface, Virtual Home, Virtual Office
def t_threetabs():
    from fda.pc import agent as A
    check("slides open in the Office tab", (A._app_kind("Q3 sales slides"), A._view_tab({"kindword": "slides"})),
          ("slides", "office"))
    check("a budget sheet / a report too", (A._app_kind("Family budget"), A._app_kind("Trip report")), ("sheet", "doc"))
    check("games and websites stay in Interface", (A._view_tab({"kindword": "game"}), A._view_tab({"kindword": "site"})),
          ("design", "design"))

    room, _ = make_room()
    a = A.Agent.__new__(A.Agent)

    class Cfg:
        name = "Elkit"

    class Ui:
        def __init__(self):
            self.sent = []

        def send(self, m):
            self.sent.append(m)

    class Ctx:
        def add_message(self, m):
            pass
    said = []

    async def announce(s):
        said.append(s)
    a.room, a.cfg, a.ui, a.context, a._announce = room, Cfg(), Ui(), Ctx(), announce
    a.views, a.active_view, a.cur_tab = [], None, "design"
    asyncio.run(a._room_direct("set a timer for 5 minutes"))
    check("set a timer from Interface: says it moves to the Virtual Home tab FIRST, then the timer",
          (said[-2], said[-1], a.cur_tab), ("Over to the Virtual Home tab.", "5 minute timer, starting now.", "room"))
    ti = [i for i, m in enumerate(a.ui.sent) if m == {"type": "tab", "tab": "room"}]
    check("the tab switch comes before the timer starts", bool(ti) and bool(room.state["timers"]), True)
    asyncio.run(a._room_direct("turn on the lights"))
    check("already on Virtual Home: no mention", said[-1].startswith("Over to"), False)
    a.on_tab({"tab": "office"})
    asyncio.run(a._room_direct("what time is it"))
    check("a question (no room change) does not move the screen", (a.cur_tab, said[-1].startswith("Over")),
          ("office", False))
    a.on_tab({"tab": "nonsense"})
    check("an unknown tab from the display is ignored", a.cur_tab, "office")

    for text, want in (("go to the office", "office"), ("show me the virtual home", "room"), ("home tab", "room"),
                       ("switch to the interface", "design"), ("go to the virtual studio", "design"), ("studio tab", "design")):
        a.cur_tab = "x"
        asyncio.run(a._view_direct(text))
        check(f"say {text!r}: the {want} tab", a.cur_tab, want)

    # the room tool path (the LLM): the result tells it to say the screen moved
    tools = A.room_tools(room, on_use=lambda: a._move_tab("room"))
    lights = next(t for t in tools if t.name == "lights")
    got = []

    class P:
        arguments = {"power": "off"}

        async def result_callback(self, r):
            got.append(r)
    a.cur_tab = "design"
    asyncio.run(lights.handler(P()))
    check("room tool from Interface: the result says to mention the move",
          "Over to the Virtual Home tab." in got[-1].get("screen", ""), True)
    asyncio.run(lights.handler(P()))
    check("room tool already on Home: nothing to mention", "screen" in got[-1], False)

    # any tool goes to its own tab first: slides from Interface -> Office, a game edit from Home -> Interface
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    order = []

    async def make(params):
        order.append(("ran", a.cur_tab))
        await params.result_callback({"ok": True, "result": "building"})
    fs = a._tab_first(FunctionSchema("create_app", "", {}, [], handler=make),
                      lambda x: "office" if A._app_kind(x.get("description", "")) in A.OFFICE_KINDS else "design")

    class P2:
        arguments = {"description": "a short slide presentation about Bean There"}

        async def result_callback(self, r):
            order.append(("result", r))
    a.cur_tab = "design"
    asyncio.run(fs.handler(P2()))
    check("slides asked on Interface: Office tab first, then the build",
          (order[0], "Over to the Virtual Office tab." in order[1][1].get("screen", "")), (("ran", "office"), True))
    a.views, a.active_view = [{"id": 1, "kind": "app", "kindword": "game", "label": "Flappy game", "key": "flappy"}], 1
    check("an edit to the game belongs to Interface", a._app_tab(), "design")
    check("an edit named for the slides belongs to Office", a._app_tab("Sales slides"), "office")

    # a ringing timer stops by itself after ring_max_s (10 s by default; 0.05 s here), the sound cut at once
    from fda.config import Config
    check("ring_max_s defaults to 10 s", Config().ring_max_s, 10.0)
    rroom, _ = make_room()
    rroom.cfg.ring_max_s = 0.05

    class Fx:
        loop = None

        def set_loop(self, x):
            self.loop = x
    rroom.mixer.fx = Fx()

    async def ring_then_wait():
        rroom.ring("10 second timer")
        on = (bool(rroom.state["ringing"]), rroom.mixer.fx.loop is not None)
        await asyncio.sleep(0.2)
        return on, (rroom.state["ringing"], rroom.mixer.fx.loop)
    on, off = asyncio.run(ring_then_wait())
    check("ringing, then silent by itself after ring_max_s", (on, off), ((True, True), (None, None)))

    page = (ROOT / "fda" / "web" / "index.html").read_text(encoding="utf-8")
    check("the page has the 3 tabs and the list pad",
          all(s in page for s in ('id="tDesign"', ">Virtual Home<", ">Virtual Office<", 'id="lists"')), True)


# ---------------------------------------------------------------- lists without the LLM; a made-up tool call is retried
def t_lists():
    from fda import rules
    room, _ = make_room()

    async def run(t):
        hit = rules.match(t, room.state)
        if hit is None:
            return None
        return [await room.execute(a) for a in hit[0]][-1].get("speak", "")
    check("add milk and eggs: straight to the list (no LLM)", asyncio.run(run("Add milk and eggs to my shopping list.")),
          "Added milk and eggs to your shopping list.")
    check("the items are on the list", room.state["lists"].get("shopping"), ["milk", "eggs"])
    check("what's on my shopping list", asyncio.run(run("What's on my shopping list?")),
          "Your shopping list has 2 items: milk and eggs.")
    check("take eggs off", asyncio.run(run("Take eggs off my shopping list")), "Removed eggs.")
    check("to do list", asyncio.run(run("Add call the plumber to my to do list")), "Added call the plumber to your to-do list.")
    for t in ("add a red button to the page", "make a list of birds", "add the owl to my game"):
        check(f"not a list command: {t!r}", rules.match(t, room.state), None)

    # gpt-oss called a tool that does not exist ('repo_browser.open_file') twice in a row: the retry tells it so
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.openai.llm import OpenAILLMService
    from fda.pc.fallback_llm import FallbackLLMService
    seen = []

    async def fake(self, context):
        seen.append([m["content"] for m in context.get_messages() if m.get("role") == "system"])
        if len(seen) < 3:
            raise RuntimeError("Error during completion: Tool call validation failed: tool call validation failed: "
                               "attempted to call tool 'repo_browser.open_file' which was not in request.tools")
        return "ok"
    old = OpenAILLMService._process_context
    OpenAILLMService._process_context = fake
    try:
        svc = FallbackLLMService.__new__(FallbackLLMService)
        ctx = LLMContext(messages=[{"role": "user", "content": "Add milk and eggs to my shopping list."}])
        got = asyncio.run(svc._process_context(ctx))
    finally:
        OpenAILLMService._process_context = old
    check("made-up tool twice: third try answers", (got, len(seen)), ("ok", 3))
    check("the retries tell it the tool does not exist", any("no tool called 'repo_browser.open_file'" in c
                                                              for c in seen[1]), True)
    check("the note is taken out again afterwards", [m["role"] for m in ctx.get_messages()], ["user"])


# ---------------------------------------------------------------- a bare refusal of a harmless request is retried
def t_refusal():
    import pipecat.processors.frame_processor as fp
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
    from pipecat.processors.frame_processor import FrameDirection as D
    from fda.pc.processors import RefusalGuard, _maybe_refusal
    fp.FrameProcessor.process_frame = lambda self, f, d: asyncio.sleep(0)

    def run(chunks):
        retried, out = [], []

        async def retry():
            retried.append(1)
        g = RefusalGuard(retry)

        async def push(f, d=None):
            out.append(f)
        g.push_frame = push

        async def go():
            await g.process_frame(LLMFullResponseStartFrame(), D.DOWNSTREAM)
            seen = []
            for c in chunks:
                await g.process_frame(LLMTextFrame(c), D.DOWNSTREAM)
                seen.append("".join(f.text for f in out if isinstance(f, LLMTextFrame)))
            await g.process_frame(LLMFullResponseEndFrame(), D.DOWNSTREAM)
            return seen
        seen = asyncio.run(go())
        return "".join(f.text for f in out if isinstance(f, LLMTextFrame)), retried, seen

    said, retried, _ = run(["I", "’m sorry", ", but I can", "’t help with that", "."])
    check("bare refusal: not spoken, asked again", (said, retried), ("", [1]))
    said, retried, seen = run(["The", " Saturn V", " roared to life", " on July 16, 1969."])
    check("a story: passes through at once (not held)", (said, retried, seen[1]),
          ("The Saturn V roared to life on July 16, 1969.", [], "The Saturn V"))
    said, retried, _ = run(["I'm sorry", " to hear that! Want me to play some music?"])
    check("'I'm sorry to hear that' is not a refusal", (said.startswith("I'm sorry to hear"), retried), (True, []))
    check("maybe-refusal prefix", (_maybe_refusal("I'm sor"), _maybe_refusal("It's ten")), (True, False))


# ---------------------------------------------------------------- faster replies: barge-in words, long sentences
def t_filler():
    """ThinkFiller: a cached filler when the answer is slow, only for what is clearly said to Elkit."""
    from fda.pc.processors import ThinkFiller, TurnClock, wants_filler
    from pipecat.frames.frames import (FunctionCallsStartedFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
                                       LLMTextFrame, TTSSpeakFrame)
    from pipecat.processors.frame_processor import FrameDirection
    yes = ["What's the weather in Davao?", "Turn the lights blue.", "Elkit, set a timer for 5 minutes",
           "Can you tell me a joke?", "Is morning the best time to exercise? [now: Thu 6 Nov, 15:20]"]
    no = ["Gianna, this coffee is legit amazing.", "I think the time is up", "Maybe for kids.",
          "Dr. Chen, I'll bring the lab results tomorrow.", "Mike, can you pass the salt?", "mm-hm",
          "What kind of movies do you think... I would like to share.",          # Candor clip: LLM stayed silent
          "What do you mean? I told you yesterday that we were going to the market together."]
    check("filler for questions / requests to Elkit", [wants_filler(t, "Elkit") for t in yes], [True] * len(yes))
    check("no filler for side talk, statements, backchannels", [wants_filler(t, "Elkit") for t in no], [False] * len(no))

    def run(frames, user="Turn the lights blue.", wait=0.15):
        async def go():
            f = ThinkFiller(after_s=0.05, last_user=lambda: user, own_name="Elkit", slow={"web_search"})
            out = []

            async def push(frame, direction=FrameDirection.DOWNSTREAM):
                out.append(frame)
            f.push_frame = push
            for fr in frames:
                if fr == "wait":
                    await asyncio.sleep(wait)
                else:
                    await _pf(f, fr)
            await asyncio.sleep(wait)
            return [x.text for x in out if isinstance(x, TTSSpeakFrame)]
        return asyncio.run(go())

    async def _pf(f, fr):            # ThinkFiller.process_frame without the pipecat setup (StartFrame etc.)
        from pipecat.processors.frame_processor import FrameProcessor
        orig = FrameProcessor.process_frame

        async def noop(self, frame, direction):
            return None
        FrameProcessor.process_frame = noop
        try:
            await ThinkFiller.process_frame(f, fr, FrameDirection.DOWNSTREAM)
        finally:
            FrameProcessor.process_frame = orig

    said = run([LLMFullResponseStartFrame(), "wait"])
    check("slow answer -> one filler from the 'do' set", (len(said), said[0] in ThinkFiller.DO if said else None),
          (1, True))
    said = run([LLMFullResponseStartFrame(), "wait"], user="What's the capital of Japan?")
    check("slow answer to a question -> a filler from the 'ask' set", said[0] in ThinkFiller.ASK if said else None,
          True)
    check("answer in time -> no filler", run([LLMFullResponseStartFrame(), LLMTextFrame("The lights are blue.")]), [])
    check("side talk -> no filler", run([LLMFullResponseStartFrame(), "wait"], user="Mike, can you pass the salt?"), [])
    check("tool-result round (last message not the user's) -> no filler",
          run([LLMFullResponseStartFrame(), "wait"], user=None), [])

    class _Call:
        def __init__(self, n):
            self.function_name, self.arguments = n, {}
    check("slow tool -> no filler here (ToolFiller speaks)",
          run([LLMFullResponseStartFrame(), FunctionCallsStartedFrame(function_calls=[_Call("web_search")]),
               LLMFullResponseEndFrame()]), [])
    said = run([LLMFullResponseStartFrame(), FunctionCallsStartedFrame(function_calls=[_Call("lights")]),
                LLMFullResponseEndFrame(), "wait"])
    check("room tool round trip slow -> filler still comes", len(said), 1)
    check("empty answer -> no filler", run([LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]), [])

    c = TurnClock()
    for n in ["speech_end", "heard", "sent", "filler", "first_audio", "first_audio", "first_word", "first_audio"]:
        c.mark(n)
    check("turn clock: filler kept apart, total = the answer's first audio",
          (set(c.last) >= {"filler", "total", "llm"}, c.last["filler"] <= c.last["total"]), (True, True))

    class _Aud:
        def __init__(self, ctx):
            self.context_id = ctx
    c = TurnClock()
    for n, fr in [("speech_end", None), ("heard", None), ("sent", None), ("filler", None), ("first_audio", _Aud("f")),
                  ("first_word", None), ("first_audio", _Aud("f")), ("first_audio", _Aud("f"))]:
        c.mark(n, fr)
        time.sleep(0.005)
    check("turn clock: later chunks of the filler are not the answer's first audio", c.last, {})
    c.mark("first_audio", _Aud("answer"))
    check("... the answer's own audio is", ("total" in c.last, c.last.get("filler", 99999) < c.last.get("total", 0)),
          (True, True))

    c = TurnClock()        # a tool turn: web_search, "Let me check that.", then the answer round
    for n, fr in [("speech_end", None), ("heard", None), ("sent", None), ("first_word", None), ("filler", None),
                  ("first_audio", _Aud("tf")), ("sent", None), ("first_word", None), ("first_audio", _Aud("tf")),
                  ("first_audio", _Aud("ans"))]:
        c.mark(n, fr)
        time.sleep(0.005)
    check("turn clock, tool turn: tool + llm2 steps, turn wait not inflated, one report",
          (sorted(k for k in c.last if k in ("tool", "llm2", "filler", "total")), c.last.get("turn wait", 0) < 20),
          (["filler", "llm2", "tool", "total"], True))

    from fda.pc.processors import ToolFiller
    tf = ToolFiller(slow={"web_search"}, skip=lambda: True)
    check("no second filler right after the first (ToolFiller skip)", tf.skip(), True)

    from fda.pc.services import split_long
    t2 = "The traffic on 85 South is moving slowly near the airport exit, but it is clearing."
    p2 = split_long(t2)
    check("split_long: two pieces with a short tail no longer crash (sentence was lost)",
          (" ".join(p2) == t2, len(p2)), (True, 1))

    from fda.pc.services import McpTTSService, speakable

    class _Conn:
        calls = 0

        async def call(self, name, args, timeout=0):
            _Conn.calls += 1
            return {"audio_b64": base64.b64encode(b"\0\0" * 3200).decode(), "sample_rate": 16000}

    async def tts_go():
        t = object.__new__(McpTTSService)
        t.conn, t.cache = _Conn(), {}
        await t.warm(["On it."])
        n0 = _Conn.calls
        hit = t.cache.get(speakable("On it."))
        return n0, hit is not None and hit[1] == 16000
    check("fillers are synthesized ahead (TTS cache)", asyncio.run(tts_go()), (1, True))


def t_realservice():
    """'Open YouTube' never builds a fake YouTube (2026-11-06); 'build a YouTube clone' still builds."""
    from fda.pc.agent import REAL_SERVICE_NOTE, SYSTEM, real_service
    opens = [("YouTube homepage", "Open YouTube."), ("YouTube", "Can you open YouTube for me?"),
             ("YouTube", "I'd like to open YouTube"), ("YouTube", "Go to YouTube"), ("Netflix", "open netflix")]
    builds = [("YouTube clone", "Build me a YouTube clone"), ("video site", "Make a site like YouTube"),
              ("Snake game", "Make a snake game"), ("Bakery site", "open the bakery site")]
    check("'open YouTube' (and Netflix...) -> create_app refused", [real_service(w, s) for w, s in opens],
          [True] * len(opens))
    check("asked to build (a YouTube clone, a site like it) -> builds", [real_service(w, s) for w, s in builds],
          [False] * len(builds))
    check("refusal tells the LLM to use play_video and ask what to watch",
          ("play_video" in REAL_SERVICE_NOTE, "ask what they want to watch" in REAL_SERVICE_NOTE), (True, True))
    check("prompt: 'Open YouTube' means the real YouTube", "\"Open YouTube\" means the real YouTube" in SYSTEM, True)


def t_quickbarge():
    """Quick (offline) overlap checks stop the bot only when sure; everything else waits for the cloud peeks."""
    import numpy as np
    from fda.config import Config
    from fda.mcp_servers.chain import ASRChain
    from fda.pc.processors import DuplexGuard
    from fda.pc.services import McpSTTService

    class Stt:
        level_db = staticmethod(McpSTTService.level_db)
        user_db = -18.0
    g = DuplexGuard.__new__(DuplexGuard)
    g.cfg, g.bot_text, g.echo_text, g.stt = Config(), (lambda: "The pasta needs plenty of boiling water."), \
        (lambda: ""), Stt()
    loud = (np.sin(np.arange(16000) / 3) * 0.2 * 32767).astype("<i2").tobytes()       # ~ -17 dB
    quiet = (np.sin(np.arange(16000) / 3) * 0.02 * 32767).astype("<i2").tobytes()     # ~ -37 dB: TV level
    sure = ["By the way, what's the weather?", "Can we talk about something else?", "Oh, before I forget, can we",
            "Actually, can we go", "Wait", "Hey, can I switch my phone?", "I'll do it myself"]
    unsure = ["Sarah you done with", "Ralph pizza's here", "Chris your laptop's", "uh-huh yeah", "Oh, how you?",
              "I see that", "I see the", "Oh, before I", "7. 7. 7.", "By the", "plenty of boiling water",
              "Welfth, welfth, welfth"]
    check("quick: sure interruptions stop", [g.sure_interruption(t, loud) for t in sure], [True] * len(sure))
    check("quick: names, backchannels, short, echo, loops wait", [g.sure_interruption(t, loud) for t in unsure],
          [False] * len(unsure))
    check("quick: quieter than the user (TV / background) waits",
          g.sure_interruption("By the way, what's the weather?", quiet), False)
    g.stt.user_db = None
    check("quick: no level of this person yet -> waits", g.sure_interruption("Can we talk about something", loud), False)
    check("quick: config switch on by default", Config().bargein_quick, True)
    check("chain: transcribe_local exists", callable(getattr(ASRChain, "transcribe_local", None)), True)


def t_nophotos():
    """Live changes never fetch web photos (Openverse is not an approved service): library art or emoji."""
    from fda.mcp_servers import live

    class NoWeb:
        def get(self, *a, **k):
            raise AssertionError("web called")
    got = {}
    for val in ["photo of ice", "ice", "photo of cactus", "a picture of bamboo", "zorblax machine"]:
        try:
            got[val] = live.resolve_value_plain(val, "x", NoWeb())[0]
        except AssertionError:
            got[val] = "WEB"
    check("web photos are off", live.PHOTOS, False)
    check("'photo of ice' -> the library's ice tile, no web", got["photo of ice"], "/assets/tiles/ice.svg")
    said = ["ice pipes", "pipes made of ice", "like ice", "icy", "frozen pipes"]
    check("'ice pipes' / 'made of ice' / 'icy' -> the ice tile",
          [live.resolve_value_plain(s, "pipe.svg", NoWeb())[0] for s in said], ["/assets/tiles/ice.svg"] * len(said))
    check("'cactus pipes' -> cactus emoji, 'blue pipes' -> blue tint",
          (live.resolve_value_plain("cactus pipes", "pipe.svg", NoWeb())[0],
           live.resolve_value_plain("blue pipes", "pipe.svg", NoWeb())[0]), ("🌵", "pipe.svg@blue"))
    check("'photo of cactus' / 'picture of bamboo' -> emoji, no web",
          (got["photo of cactus"], got["a picture of bamboo"]), ("\U0001F335", "\U0001F38B"))
    check("unknown thing -> no picture (the LLM offers something close), no web", got["zorblax machine"], None)


def t_roomhub():
    """Room tools over MCP: the tools server's room tools post to the agent's room hub, which runs the room."""
    import socket
    from fda.config import Config
    from fda.mcp_servers import tools_server
    from fda.pc import agent as A
    room, changes = make_room()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    async def go():
        a = object.__new__(A.Agent)
        cfg = Config()
        cfg.room_hub_port = port
        a.cfg, a.room = cfg, room
        hub = await a._start_room_hub()
        mcp, _ = tools_server.build(cfg)
        names = {t.name for t in await mcp.list_tools()}
        res = []
        for tool, args in [("lights", {"power": "on", "color": "red"}), ("timer", {"seconds": 90}),
                           ("list_items", {"action": "add", "items": "milk, eggs"}),
                           ("add_reminder", {"text": "call mother", "time": "18:00"})]:
            r = await mcp.call_tool(tool, args)
            res.append(r)
        await hub.cleanup()
        return names, res
    names, res = asyncio.run(go())
    check("tools MCP server has all 13 room tools", set(A.ROOM_TOOLS) <= names, True)
    check("room tools reach the room (lights red, timer 90 s)",
          (room.state["lights"]["power"], room.state["lights"]["color"],
           bool(room.state.get("timer") or room.state.get("timers"))), ("on", "red", True))
    lists = room.state.get("lists", {})
    check("list_items over MCP adds to the shopping list",
          any("milk" in str(v).lower() for v in lists.values()) if isinstance(lists, dict) else "milk" in str(lists),
          True)
    check("add_reminder over MCP", [r["text"] for r in room.state["reminders"]][-1:], ["call mother"])
    check("agent asks the MCP server for the room tools", set(A.ROOM_TOOLS) <= set(A.AGENT_MCP_TOOLS), True)


def t_speed():
    from fda import rules
    from fda.pc.services import split_long
    st = {"ringing": None, "music": {"state": "stopped"}}
    check("'Wait stop, what time is it?' -> instant time answer", (rules.match("Wait stop, what time is it?", st) or
                                                                 [None])[0], [])
    check("a lone 'stop' still means stop", rules.match("stop", st), ([], ""))
    check("'Hold on, set a timer for 2 minutes' -> timer", rules.match("Hold on, set a timer for 2 minutes", st)[0],
          [{"tool": "timer", "seconds": 120}])
    long = ("Back in 1969, a tiny rocket named Saturn V roared off the launch pad, carrying three brave astronauts "
            "who dreamed of walking on a world that only glittered in the night sky.")
    p = split_long(long)
    check("a long sentence is spoken in comma pieces (first piece shorter)", (len(p), " ".join(p) == long,
                                                                             len(p[0]) < len(long)), (2, True, True))
    check("a short sentence stays whole", split_long("It's 11:08 PM."), ["It's 11:08 PM."])
    import time as _t, concurrent.futures as _cf
    from fda.mcp_servers import chain as _ch

    class _Fake(_ch.CloudASR):
        def __init__(self, name, delay, text):
            _ch.Backend.__init__(self, name)
            self.delay, self.text, self.calls = delay, text, 0

        def __call__(self, x, rate, language):
            self.calls += 1
            _t.sleep(self.delay)
            return self.text

    def _chain(*bk, hedge=0.15):
        c = object.__new__(_ch.ASRChain)
        c.backends, c.hedge_s, c.hedged, c.last = list(bk), hedge, 0, ""
        c._pool = _cf.ThreadPoolExecutor(4)
        return c
    a, b = _Fake("groq", 0.05, "turbo"), _Fake("groq-v3", 0.05, "v3")
    c = _chain(a, b)
    check("ASR: turbo fast -> its text, no backup asked", (c.transcribe(None, 16000), b.calls, c.last),
          ("turbo", 0, "groq"))
    a, b = _Fake("groq", 0.8, "turbo"), _Fake("groq-v3", 0.05, "v3")
    c = _chain(a, b)
    t0 = _t.monotonic()
    out = c.transcribe(None, 16000)
    check("ASR: turbo slow -> the backup is asked and the first answer wins",
          (out, c.last, c.hedged, _t.monotonic() - t0 < 0.5), ("v3", "groq-v3", 1, True))
    a, b = _Fake("groq", 0.01, None), _Fake("groq-v3", 0.01, "v3")
    a.failed("x")
    check("ASR: turbo down -> the next engine as before", (_chain(a, b).transcribe(None, 16000), b.calls), ("v3", 1))
    from fda.pc.services import turn_wait_s as _tw
    check("turn end wait: a question at once, a statement 0.3 s, 'and...' 0.8 s (minus VAD 0.2 s)",
          [round(_tw(t) - 0.2, 2) for t in ("What's the population of Toronto?", "I practice every day.",
                                            "Yeah, the chords still feel tricky and", "So I went to the,",
                                            "Set a timer for 5 minutes.")],
          [0.05, 0.3, 0.8, 0.8, 0.3])
    import io as _io, numpy as _np, soundfile as _sf
    from fda.mcp_servers.chain import _upload, _wav
    x = (_np.sin(_np.arange(32000) / 9) * 0.3).astype("float32")
    name, data, mime = _upload(x, 16000)
    back, rate = _sf.read(_io.BytesIO(data), dtype="int16")
    check("the ASR upload is FLAC, smaller than WAV, same samples",
          (name, mime, len(data) < len(_wav(x, 16000)), rate, int(_np.abs(back.astype(int) - (x * 32767).astype(int)).max()) <= 1),
          ("turn.flac", "audio/flac", True, 16000, True))


# ---------------------------------------------------------------- no "it's ready" while the build is still running
def t_ready():
    from fda.pc.services import McpTTSService

    class Conn:
        def __init__(self):
            self.said = []

        async def call(self, name, args, timeout=0):
            self.said.append(args["text"])
            return {"audio_b64": "AAAA", "sample_rate": 24000}
    tts = McpTTSService.__new__(McpTTSService)
    tts.conn, tts._reply_id, tts._said_in_reply = Conn(), None, 0
    busy = [True]
    tts.building = lambda: busy[0]

    async def nothing(*_a, **_k):
        pass
    tts.start_tts_usage_metrics = tts.stop_ttfb_metrics = nothing

    async def frames(*_a, **_k):
        if False:
            yield None
    tts._stream_audio_frames_from_iterator = frames

    async def say(t, cid):
        return [f async for f in tts.run_tts(t, cid)]
    for t, cid in (("The game is building now.", "a"), ("It’s ready—enjoy the Flappy Bird game.", "a"),
                   ("Your 10 second timer is done.", "b"), ("I'll let you know when it's ready.", "d"),
                   ("The game is ready.", "e"),
                   # live 2026-11-06: said 10 s before the game was ready
                   ("The game is coming up.", "h"), ("Ready!", "h"), ("Try tapping to make the bird fly.", "h"),
                   ("Press space to jump.", "i"), ("All set.", "j")):
        asyncio.run(say(t, cid))
    check("while building: 'building now' / 'coming up' / 'I'll let you know when it's ready' said; "
          "'it's ready', 'Ready!', 'Try tapping…' not; timer is",
          tts.conn.said, ["The game is building now.", "Your 10 second timer is done.",
                          "I'll let you know when it's ready.", "The game is coming up."])
    # what is not said must not show in the caption / history either: pipecat calls run_tts, then
    # _push_spoken_text with the original text; only the sentences actually said may get a spoken-text frame
    from pipecat.services.tts_service import TTSService
    pushed, orig = [], TTSService._push_spoken_text

    async def rec(self, src, text, cid, app):
        pushed.append(text)
    TTSService._push_spoken_text = rec
    try:
        async def sentence(t, cid):
            await say(t, cid)
            await tts._push_spoken_text(None, t, cid, True)
        for t, cid in (("Building it now.", "f"), ("It's ready, enjoy!", "f"), ("(silent)", "g"), ("Okay.", "g")):
            asyncio.run(sentence(t, cid))
    finally:
        TTSService._push_spoken_text = orig
    check("an unsaid sentence ('It's ready' while building, '(silent)') gets no caption / history text",
          pushed, ["Building it now.", "Okay."])
    busy[0] = False
    asyncio.run(say("Flappy Bird is ready, it's on the Flappy game tab.", "c"))
    check("after the build: 'is ready' is said", tts.conn.said[-1], "Flappy Bird is ready, it's on the Flappy game tab.")
    from fda.mcp_servers import tools_server as T
    import inspect
    check("create_app's reply says it is NOT ready yet", "It is NOT ready yet" in inspect.getsource(T), True)
    from fda.pc import agent as A
    src = inspect.getsource(A)
    check("the agent gives the voice its build state (tts.building) and marks build starts",
          ("tts.building = " in src, "self._mark_build(fs)" in src, "            tts,\n" in src), (True, True, True))
    # a game that drew its bird as a colour ("#FFCC00" = a yellow square) gets the library bird
    check("bird as a colour -> the library bird; pipes / sky stay colours",
          T.sprite_fixes({"bird": "#FFCC00", "pipe": "#3a3", "sky": "#7cf", "speed": 3, "fish": "🐟"}),
          {"bird": "/assets/sprites/bird.svg"})
    check("the builder is told: creatures are pictures, never a plain colour", "never a plain colour" in T.art_rule(), True)
    # live 2026-11-06: "game over even when the bird is not touching the pipe": the collision used the bird's
    # centre, the picture was drawn with its corner there (22 px lower / to the right)
    g = ('if (bird.y - bird.r < top) gameOver();\nFDA.draw(ctx, "bird", bird.x, bird.y, bird.r * 2, bird.r * 2);\n'
         'FDA.draw(ctx, "coin", coin.x, coin.y, coin.r * 2, coin.r * 2);')
    fixed, n = T.centre_draw_fix(g)
    check("a centre-kept bird drawn from its corner is drawn around its centre; a corner-kept coin stays",
          (n, 'FDA.draw(ctx, "bird", bird.x - bird.r, bird.y - bird.r, bird.r * 2, bird.r * 2)' in fixed,
           'FDA.draw(ctx, "coin", coin.x, coin.y, coin.r * 2, coin.r * 2)' in fixed), (1, True, True))
    check("an already-correct draw is left alone", T.centre_draw_fix(fixed), (fixed, 0))
    check("the builder is told: FDA.draw is the top-left corner; collide with the box you draw",
          ("TOP-LEFT corner" in T.APP_SYSTEM, "you\n  SEE touches" in T.APP_SYSTEM or "you SEE touches" in T.APP_SYSTEM),
          (True, True))


def t_difficulty():
    """'Make it easier' / 'set it to hard' / 'pipes to cactus' while a game runs: instant, no rebuild."""
    from fda.pc import agent as A
    from fda.mcp_servers import live as L, tools_server as T
    got = {s: (lambda m: next((g for g in m.groups() if g), None) if m else None)(A._DIFFICULTY.match(s))
           for s in ("Make it easier.", "make the game a bit harder", "Set it to hard.", "Easy mode please.",
                     "It's too hard!", "change the difficulty to easy", "make the bird red", "make it hard to read",
                     "set a timer to 5 minutes")}
    check("difficulty phrases are heard; other requests are not",
          list(got.values()), ["easier", "harder", "hard", "Easy", "too hard", "easy", None, None, None])
    rv = lambda v, cur: L.resolve_value(v, cur, None)
    check("difficulty values: easier from normal = easy, harder = hard, 'hard' = 3, already easy = no change",
          (rv("easier", 2)[0], rv("harder", 2)[0], rv("hard", 1)[0], rv("easier", 1)), (1, 3, 3, (None, "it is already on easy")))
    check("speed still works the old way", rv("faster", 2)[0], 3.0)
    check("the builder gives every game a difficulty setting that changes speed, gaps and hitbox",
          ("difficulty: 2" in T.APP_SYSTEM, "C.difficulty" in T.APP_SYSTEM, "hitbox is" in T.APP_SYSTEM), (True, True, True))
    m = A._TWEAK.match("Change the pipes to cactus.")
    check("'change the pipes to cactus' -> the pipe setting becomes a cactus",
          (L.match_key(m.group(1), ["bird", "pipe", "sky"]), rv("cactus", "#228B22")[0]), ("pipe", "🌵"))

    class Conn:
        def __init__(self, ok=True, msg=""):
            self.calls, self.ok, self.msg = [], ok, msg

        async def call(self, name, args, timeout=0):
            self.calls.append(args["changes"])
            if not self.ok:
                return {"ok": False, "message": self.msg}
            ch = args["changes"]
            return {"ok": True, "patch": {k: (1 if v in ("easier", "easy") else 3) if k == "difficulty" else v
                                          for k, v in ch.items()}, "changed": str(ch)}

    class Ctx:
        def __init__(self):
            self.msgs = []

        def add_message(self, m):
            self.msgs.append(m)

    def run(keys, said, conn):
        a = A.Agent.__new__(A.Agent)
        a.app_keys, a.tools_conn, a.context, spoken = keys, conn, Ctx(), []

        async def nothing(*_a, **_k):
            return ""

        async def announce(t):
            spoken.append(t)
        a._go_first, a._app_tab, a._app_set, a._announce = nothing, (lambda *x: "design"), (lambda r: r), announce
        done = asyncio.run(a._difficulty_direct(said))
        return done, conn.calls, spoken
    check("new game: 'make it easier' -> difficulty, said 'on easy'",
          run(["bird", "pipe", "speed", "gap", "difficulty"], "Make it easier.", Conn()),
          (True, [{"difficulty": "easier"}], ["Done, it's on easy now."]))
    check("older game without difficulty: easier = slower + wider gaps",
          run(["bird", "pipe", "speed", "gap"], "It's too hard!", Conn()),
          (True, [{"speed": "slower", "gap": "bigger"}], ["Done, it's easier now: slower, with more room."]))
    check("already easiest: says so",
          run(["difficulty"], "make it easier", Conn(False, "I couldn't change that. difficulty: it is already on easy")),
          (True, [{"difficulty": "easier"}], ["It's already on easy."]))
    check("no game settings to change: left to the LLM", run(["title", "color"], "make it easier", Conn())[0], False)
    # found by going through every live edit (2026-11-06): "make the bird bigger" turned the bird into the word
    # "bigger"; "face right" flipped a bird that already faced right; "the pipes is purple"
    check("'bigger' / 'jump higher' on a picture or colour: refused (no size setting), never the word itself",
          (rv("bigger", "🐤|flip")[0], rv("jump higher", "🐤")[0], rv("a lot wider", "#228B22")[0], rv("bigger", "#fff")[0]),
          (None, None, None, None))
    st = {"bird": "🐤|flip", "pipe": "#228B22", "size": 44, "gravity": 0.5, "jump": 7.2, "pipeWidth": 64}
    check("'bird bigger' -> size, 'jump higher' -> jump, 'heavier' -> gravity, 'pipes wider' -> pipeWidth",
          (L.number_key_for("bird", "bigger", st), L.number_key_for("bird", "jump higher", st),
           L.number_key_for("bird", "heavier", st), L.number_key_for("pipe", "wider", st),
           L.number_key_for("bird", "bigger", {"bird": "x"})), ("size", "jump", "gravity", "pipeWidth", None))
    check("numbers: bigger 44 -> 61.6, higher 7.2 -> 9.36, more gravity 0.5 -> 0.65",
          (rv("bigger", 44)[0], rv("higher", 7.2)[0], rv("more", 0.5)[0]), (61.6, 9.36, 0.65))
    check("'face right' when it faces right: no change; 'face left': flipped back",
          (rv("face right", "🐤|flip"), rv("face left", "🐤|flip")[0], rv("face right", "🐤")[0]),
          ((None, "it already faces right"), "🐤", "🐤|flip"))

    def tweak(keys, said, reply):
        class C2(Conn):
            async def call(self, name, args, timeout=0):
                self.calls.append(args["changes"])
                return reply(args["changes"])
        a = A.Agent.__new__(A.Agent)
        a.app_keys, a.tools_conn, a.context, a.cfg, spoken = keys, C2(), Ctx(), type("c", (), {"name": "Elkit"})(), []

        async def nothing(*_a, **_k):
            return ""

        async def announce(t):
            spoken.append(t)
        a._go_first, a._app_tab, a._app_set, a._announce = nothing, (lambda *x: "design"), (lambda r: r), announce
        return asyncio.run(a._direct_tweak(said)), spoken
    ok = lambda ch: {"ok": True, "patch": ch, "changed": str(ch)}
    check("'the pipes are purple' (plural)", tweak(["bird", "pipe"], "Change the pipes to purple.", ok),
          (True, ["Done, the pipes are purple now."]))
    check("'make the bird face right' when it already does: says so, no flip",
          tweak(["bird", "pipe"], "Make the bird face right.",
                lambda ch: {"ok": False, "message": "I couldn't change that. bird: it already faces right"}),
          (True, ["The bird already faces right."]))
    check("'make the bird bigger' with no size setting: left to the LLM (edit_app), nothing said",
          tweak(["bird", "pipe"], "Make the bird bigger.",
                lambda ch: {"ok": False, "message": "I couldn't change that. bird: its size is not a live setting"}),
          (False, []))
    check("grammar: 'the bird jumps higher', 'the ship shoots faster', 'the cactus is rocks'",
          (tweak(["bird", "pipe", "jump"], "Make the bird jump higher.", ok)[1],
           tweak(["ship", "enemy", "fireRate"], "Make it shoot faster.", ok)[1],
           tweak(["dino", "cactus"], "Change the cactus to rocks.", ok)[1]),
          (["Done, the bird jumps higher now."], ["Done, the ship shoots faster now."], ["Done, the cactus is rocks now."]))
    # live 2026-11-06: "green... wait, no, black": black only, and Elkit says "not green" (user's choice)
    check("change of mind: only black is done, and Elkit says 'not green'",
          tweak(["bird", "pipe"], "Make the bird green. Wait, no. I want it to be black instead.", ok),
          (True, ["Okay, not green: black instead. The bird is black now."]))
    check("change of mind: 'the pipes red, actually blue'", tweak(["bird", "pipe"], "Make the pipes red, actually blue.", ok),
          (True, ["Okay, not red: blue instead. The pipes are blue now."]))
    def two(first, second, keys=("bird", "pipe")):
        a = A.Agent.__new__(A.Agent)
        calls, spoken = [], []

        class C3:
            async def call(self, name, args, timeout=0):
                calls.append(args["changes"])
                return ok(args["changes"])
        a.app_keys, a.tools_conn, a.context, a.cfg = list(keys), C3(), Ctx(), type("c", (), {"name": "Elkit"})()

        async def nothing(*_a, **_k):
            return ""

        async def announce(t):
            spoken.append(t)
        a._go_first, a._app_tab, a._app_set, a._announce = nothing, (lambda *x: "design"), (lambda r: r), announce

        async def go():
            return [await a._direct_tweak(first), await a._direct_tweak(second)]
        return asyncio.run(go()), calls, spoken[-1:]
    # the user's scene: green, a stop (Elkit takes it), then "wait, no... black instead" on its own
    check("'green' then, on its own, 'wait, no, black instead': black, and 'not green'",
          two("Make the bird green.", "Wait, no. I want it to be black instead."),
          ([True, True], [{"bird": "green"}, {"bird": "black"}], ["Okay, not green: black instead. The bird is black now."]))
    check("after a change, 'actually, what time is it?' is a question, not a correction",
          two("Make the bird green.", "Actually, what time is it?")[0], [True, False])
    check("after a change, 'no, I meant the pipes' goes to the LLM (which thing, not a value)",
          two("Make the bird green.", "No, I meant the pipes.")[0], [True, False])

    # live 2026-11-06: green showed at once, before "wait, no, black" came. Now a change is held a moment
    # ("Green, okay."); talking meanwhile keeps it held until the words come; "wait, no..." drops it
    def held(steps):
        a = A.Agent.__new__(A.Agent)
        calls, spoken = [], []

        class C4:
            async def call(self, name, args, timeout=0):
                calls.append(args["changes"])
                return ok(args["changes"])
        a.app_keys, a.tools_conn, a.context, a.cfg = ["bird", "pipe"], C4(), Ctx(), type("c", (), {"name": "Elkit"})()
        a.tweak_hold, a._held, a.io = 0.2, None, type("io", (), {"user_speaking": False})()

        async def nothing(*_a, **_k):
            return ""

        async def announce(t):
            spoken.append(t)
        a._go_first, a._app_tab, a._app_set, a._announce = nothing, (lambda *x: "design"), (lambda r: r), announce

        async def go():
            seen = []
            for kind, x in steps:
                if kind == "say":
                    await a._direct_tweak(x)
                elif kind == "voice":
                    a.io.user_speaking = x
                elif kind == "bot":
                    a.io.bot_speaking = x
                    if not x:
                        a.io.t_bot_stopped = time.monotonic()
                elif kind == "wait":
                    await asyncio.sleep(x)
                elif kind == "flush":
                    await a._flush_held()
                seen.append(list(calls))
            return seen
        seen = asyncio.run(go())
        return seen, spoken
    check("held: 'Green, okay.' at once, green only after the hold",
          held([("say", "Make the bird green."), ("wait", 0.6)]),
          ([[], [{"bird": "green"}]], ["Green, okay."]))
    check("held: while 'Green, okay.' is being said and a second after, green waits",
          held([("say", "Make the bird green."), ("bot", True), ("wait", 0.6), ("bot", False), ("wait", 0.1),
                ("wait", 0.4)]),
          ([[], [], [], [], [], [{"bird": "green"}]], ["Green, okay."]))
    check("held: talking during the hold, then 'wait, no, black instead': green is never made",
          held([("say", "Make the bird green."), ("voice", True), ("wait", 0.5), ("voice", False),
                ("say", "Wait, no. I want it to be black instead."), ("wait", 0.4)]),
          ([[], [], [], [], [{"bird": "black"}], [{"bird": "black"}]],
           ["Green, okay.", "Okay, not green: black instead. The bird is black now."]))
    check("held: another request in the hold makes the held change first",
          held([("say", "Make the bird green."), ("flush", None)]), ([[], [{"bird": "green"}]], ["Green, okay."]))
    check("held: 'make the bird face right' is not held (it may say 'already faces right')",
          held([("say", "Make the bird face right.")]), ([[{"bird": "face right"}]], ["Done, the bird faces right now."]))
    # live 2026-11-06: "open the Flappy Bird game" opened an old flappy-bird-game.html, not the one just made
    with tempfile.TemporaryDirectory() as d:
        D = Path(d)
        for stem, upd in (("flappy-bird-game", 100.0), ("flappy-bird", 200.0), ("neon-flappy", 50.0)):
            (D / f"{stem}.html").write_text("<html></html>", encoding="utf-8")
            (D / f"{stem}.json").write_text(json.dumps({"title": "Neon Flappy" if "neon" in stem else "Flappy Bird",
                                                        "updated": upd}), encoding="utf-8")
        ap = T.Apps.__new__(T.Apps)
        ap.dir, ap.current = D, None
        check("'Flappy Bird Game' opens the newest Flappy Bird; 'neon flappy' its own",
              (ap.find("Flappy Bird Game").stem, ap.find("flappy bird").stem, ap.find("neon flappy").stem),
              ("flappy-bird", "flappy-bird", "neon-flappy"))
    check("'make the sky blue, no clouds' is not a change of mind (left to the LLM)",
          tweak(["bird", "sky"], "Make the sky blue, no clouds.", ok), (False, []))
    check("the builder is told: C.lives is the start, the game counts down its own lives; tests start fresh",
          ("never change C.lives" in T.APP_SYSTEM, "reset() is called" in T.APP_SYSTEM), (True, True))
    # trial 2026-11-06: tests ran one after another in the same game (a blink, a turned snake, a game over)
    if shutil.which("node"):
        g = ('<!DOCTYPE html><html><head><title>t</title></head><body><canvas id="c"></canvas><script>'
             'const cv = document.getElementById("c"), ctx = cv.getContext("2d");'
             'let state = "ready", score = 0, used = false;'
             'function reset() { score = 0; used = false; state = "play"; }'
             'function update() { }'
             'function loop() { if (state === "play") update(); requestAnimationFrame(loop); } loop();'
             '</script><script type="application/json" id="fda-tests">['
             '{"name": "first", "setup": [], "before": "used", "do": ["run used = true"], "expect": "used === true"},'
             '{"name": "fresh", "setup": [], "before": "used", "do": [], "expect": "used === false"}'
             ']</script></body></html>')
        res = L.run_check(g) or {}
        check("behaviour tests: each starts from a fresh reset() game",
              [t.get("ok") for t in res.get("tests", [])], [True, True])
    # trial 2: a shooter that only failed a test was "repaired" into a syntax error and saved broken; a catcher's
    # tests lacked one "]" and the good game was reported as broken
    check("build problems ranked: none < 1 failing test < 2 failing tests < does not run",
          (T.problem_rank(None), T.problem_rank("Behaviour test failed: 'a': expected x"),
           T.problem_rank("Behaviour test failed: 'a': expected x; 'b': expected y"), T.problem_rank("Syntax error: y"),
           T.problem_rank("When it runs: TypeError: z"), T.problem_rank("The game never draws bullet: ...")),
          (0.0, 1.1, 1.2, 3.0, 3.0, 2.0))
    # trial 4: a counting loop ("spawn67: 0.8, spawn68: 0.8, ...") was not seen as the model repeating itself
    loop = "const C = {" + "".join(f"spawn{i}: 0.8, " for i in range(300))
    check("a counting loop is caught as the model repeating itself; a normal game is not",
          (T._looping(loop), any(T._looping(L.LIVE_JS[:i]) for i in range(1000, len(L.LIVE_JS), 250))), (True, False))
    import inspect
    src = inspect.getsource(T.Apps._build)
    check("the build keeps the best version when a repair made it worse",
          ("problem_rank(err) <= best[0]" in src, "html = best[1]" in src, "round_ == 2" in src), (True, True, True))
    bad = ('<script type="application/json" id="fda-tests">[\n  {"name": "a", "setup": ["tap", "run x = 1", '
           '"before": "score", "do": ["wait 3"], "expect": "score > before"},\n]</script>')
    fixed_t = L.tests_of(bad)
    check("tests with a missing ']' and a trailing comma are repaired",
          json.loads(fixed_t)[0]["setup"] if fixed_t and fixed_t.startswith("[") else fixed_t, ["tap", "run x = 1"])
    check("the builder: lists get their start value at the top", "draw() runs on the very first frame" in T.APP_SYSTEM, True)
    check("the builder adds size / gravity / jump number settings",
          ("size: 44, gravity: 0.5, jump: 7" in T.APP_SYSTEM, "its hitbox follow" in T.APP_SYSTEM), (True, True))


SECTIONS = {"voice": t_voice, "thinking": t_thinking, "stop": t_stop, "echo": t_echo, "rules": t_rules, "room": t_room,
            "replies": t_replies, "tabs": t_tabs, "video": t_video, "app": t_app, "tweak": t_tweak,
            "fastroom": t_fastroom, "dgx3": t_dgx3, "polish": t_polish, "assets": t_assets, "choices": t_choices, "watchfixes": t_watchfixes, "threetabs": t_threetabs, "lists": t_lists, "refusal": t_refusal, "speed": t_speed, "ready": t_ready, "difficulty": t_difficulty, "mind": t_mind, "appjobs": t_appjobs, "history": t_history, "tools": t_tools, "filler": t_filler, "roomhub": t_roomhub, "nophotos": t_nophotos, "realservice": t_realservice, "quickbarge": t_quickbarge}


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
