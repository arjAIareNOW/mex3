"""Checks "continue after an interruption" without an LLM or audio: a reply of three sentences starts playing, the
person cuts in during the second sentence, then says "okay, continue". FDA must resume from the start of the
second sentence (the one that was cut), not repeat the first and not skip the rest.

    .venv\\Scripts\\python scripts\\check_continue.py
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipecat.frames.frames import (InterruptionFrame, LLMFullResponseEndFrame,  # noqa: E402
                                   LLMFullResponseStartFrame, LLMTextFrame, TTSTextFrame)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

from fda.pc.processors import CONTINUE, NotForMe, SpokenTap  # noqa: E402

REPLY = ["Owls can turn their heads about two hundred seventy degrees. ",
         "They do this because their eyes cannot move in their sockets. ",
         "Their necks have fourteen bones, twice as many as ours."]


async def main():
    io = SimpleNamespace(last_heard="", bot_speaking=True)
    nfm = NotForMe(io=io)
    tap = SpokenTap(nfm)
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)
    nfm.push_frame = push
    tap.push_frame = push
    FrameProcessorPF = type(nfm).__mro__[1].process_frame

    async def feed(p, f):                        # skip the base class bookkeeping (no pipeline running)
        orig = type(p).__mro__[1].process_frame

        async def noop(self, frame, direction):
            return None
        type(p).__mro__[1].process_frame = noop
        try:
            await p.process_frame(f, FrameDirection.DOWNSTREAM)
        finally:
            type(p).__mro__[1].process_frame = orig

    await feed(nfm, LLMFullResponseStartFrame())
    for s in REPLY:                               # the LLM writes the whole reply (Groq is fast)
        for w in s.split(" "):
            await feed(nfm, LLMTextFrame(text=w + " "))
    await feed(nfm, LLMFullResponseEndFrame())
    await feed(tap, TTSTextFrame(text=REPLY[0].strip(), aggregated_by="sentence"))    # sentence 1 played
    await feed(tap, TTSTextFrame(text=REPLY[1].strip(), aggregated_by="sentence"))    # sentence 2 starts...
    await feed(nfm, InterruptionFrame())                                               # ...and the person cuts in
    for said in ["Okay, continue.", "go on please", "What were you saying?"]:
        assert CONTINUE.match(said), said
    rest, before, done = nfm.take_paused()
    print("heard before the cut:", repr(before))
    print("resumes with        :", repr(rest))
    print("reply fully written :", done)
    ok = rest.startswith("They do this") and rest.rstrip().endswith("as ours.") and "two hundred" not in rest
    print("PASS" if ok else "FAIL")
    print("asked twice         :", nfm.take_paused())   # used once only
    _ = FrameProcessorPF


asyncio.run(main())
