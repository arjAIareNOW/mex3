"""Pipecat processors that give the agent FDA's full-duplex manners.

DuplexGuard     the user starts talking while the bot talks -> duck the voice at once (no talking over each other),
                then: a backchannel ("uh huh", "okay") or a short noise -> back to full volume, keep talking;
                a real interruption (>= 3 words, or long sustained speech) -> stop (Pipecat interruption).
NotForMe        the LLM answers "(silent)" for speech that is not meant for it (people talking to each other, TV,
                a phone call): nothing is spoken. If that speech had interrupted the bot, it resumes what it was
                saying instead of going quiet.
UiTap           mirrors what was heard and said to the display.
"""
from __future__ import annotations

import asyncio
import base64
import re
import time

from loguru import logger

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame,
                                   FunctionCallsStartedFrame, InterruptionFrame, LLMMessagesAppendFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
                                   TranscriptionFrame, TTSSpeakFrame, TTSTextFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .. import rules

_STOP_WORDS = re.compile(r"\b(stop|stopp?|strop|stahp|wait|enough|quiet|shut up|hold on|hang on|never ?mind|"
                         r"cancel|be quiet)\b", re.I)
CONTINUE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|sure|sorry|alright|all right|right|please|so|and|um|uh|oh|fine|got it)[,.!\s]+)*"
    r"(?:please\s+)?(?:you can\s+)?"
    r"(?:continue|go on|keep going|carry on|go ahead|resume|proceed|you were saying|what were you saying|"
    r"where were we|as you were saying|finish (?:that|it|what you were saying|your (?:sentence|story|answer|thought))|"
    r"go back to what you were saying|back to what you were saying)"
    r"(?:\s+(?:please|then|now|with (?:that|it|the story|what you were saying)|what you were saying|the story))*"
    r"[,.!?\s]*$", re.I)
_TIME_Q = re.compile(r"\b(time|date|day|today|tonight|tomorrow|yesterday|clock|o'?clock|month|year)\b", re.I)
SILENT = re.compile(r"^\W*[(\[]\s*silen|^\W*silent\W*$", re.I)


def could_be_silent(text: str) -> bool:
    """True while the reply so far is still a possible start of "(silent)" (tokens arrive a few characters at a
    time: "(", "sil", "ent)"); False as soon as it is clearly a real reply, or clearly "(silent)" (then SILENT)."""
    t = re.sub(r"^[\s\W]+", "", text).lower()          # drop "(", "[", spaces
    if SILENT.match(text.strip()) or not t:
        return not SILENT.match(text.strip()) and len(text.strip()) < 3
    return "silent".startswith(t)


class DuplexGuard(FrameProcessor):
    """Talking over the bot. Ducks at once; while the overlap goes on, transcribes it early (at `peek_s`) so a
    real interruption ("no, I meant tomorrow") stops the bot in ~1 s instead of waiting for the person to finish.
    Backchannels, speech to someone else, TV-level speech and the bot's own echo never stop it."""

    def __init__(self, io, cfg, sustained_s: float = 3.0, unduck_s: float = 0.9, peek_s=(1.0, 1.6), **kw):
        super().__init__(**kw)
        self.io, self.cfg = io, cfg
        self.sustained_s, self.unduck_s, self.peek_s = sustained_s, unduck_s, tuple(peek_s)
        self.ducked = False
        self._watch: asyncio.Task | None = None
        self._unduck: asyncio.Task | None = None
        self.t_user = 0.0
        self.asr = None                  # McpConnection to the asr server (set by the agent; None = no early peek)
        self.stt = None                  # McpSTTService (its user level, for the far-speech check)
        self.bot_text = lambda: ""       # what the bot is saying now (echo check)
        self.on_bot = None               # callback(talking: bool) for the display
        self.pre = bytearray()           # last ~0.5 s of audio (VAD fires ~0.15 s after speech starts)
        self.buf = bytearray()           # overlap audio since the user started
        self.stats = {"ducks": 0, "resumed": 0, "sustained_stops": 0, "early_stops": 0}

    def _cancel(self, attr):
        t = getattr(self, attr)
        if t and not t.done():
            t.cancel()
        setattr(self, attr, None)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        io = self.io
        if isinstance(frame, InputAudioRawFrame):
            if self._watch is not None:
                self.buf += frame.audio
            else:
                self.pre += frame.audio
                del self.pre[:-16000]                     # keep 0.5 s at 16 kHz int16
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not io.bot_speaking and self.on_bot:
                self.on_bot(True)
            io.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if io.bot_speaking and self.on_bot:
                self.on_bot(False)
            io.bot_speaking = False
            self._set_duck(False)
            self._cancel("_watch")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            io.user_speaking = True
            self.t_user = time.monotonic()
            self._cancel("_unduck")
            if io.bot_speaking:
                self._set_duck(True)
                self.stats["ducks"] += 1
                self._cancel("_watch")
                self.buf = bytearray(self.pre)
                self._watch = asyncio.create_task(self._sustained())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            io.user_speaking = False
            peeked = self._watch is not None and self._watch.done()
            self._cancel("_watch")
            if self.ducked:
                # a short "stop" / "wait" ends before the first early check: check what was said right now
                if not peeked and self.asr is not None and len(self.buf) >= 6400:      # >= 0.2 s
                    self._unduck = asyncio.create_task(self._peek_then_unduck(bytes(self.buf)))
                else:
                    self._unduck = asyncio.create_task(self._unduck_later())
        elif isinstance(frame, InterruptionFrame):
            self.ducked = False
            self._cancel("_watch"); self._cancel("_unduck")
        await self.push_frame(frame, direction)

    def _set_duck(self, on: bool):
        if on != self.ducked:
            self.ducked = on
            self.io.duck(on)

    async def _sustained(self):
        t0 = time.monotonic()
        for at in self.peek_s if self.asr else ():
            await asyncio.sleep(max(0.0, at - (time.monotonic() - t0)))
            if not (self.io.user_speaking and self.io.bot_speaking):
                return
            if await self._peek():
                return
        await asyncio.sleep(max(0.0, self.sustained_s - (time.monotonic() - t0)))
        if self.io.user_speaking and self.io.bot_speaking:
            logger.info("user kept talking over the bot: stopping")
            self.stats["sustained_stops"] += 1
            await self.broadcast_interruption()

    async def _peek(self, audio: bytes | None = None, after_end: bool = False) -> bool:
        """Transcribe the overlap so far; stop the bot if it is a real interruption."""
        audio = bytes(self.buf) if audio is None else audio
        try:
            text = await self.asr.call("transcribe", {"audio_b64": base64.b64encode(audio).decode(),
                                                      "sample_rate": 16000}, timeout=5)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"peek failed: {e}")
            return False
        text = (text or "").strip() if isinstance(text, str) else ""
        if not text or not self.io.bot_speaking or not (after_end or self.io.user_speaking):
            return False
        why = self.not_an_interruption(text, audio)
        if why:
            logger.info(f"overlap {text!r}: {why}, keep talking")
            return why in ("to someone else", "echo")      # decided: no more peeks
        logger.info(f"interrupted by {text!r}: stopping")
        self.stats["early_stops"] += 1
        self.io.last_heard = text
        await self.broadcast_interruption()
        return True

    def not_an_interruption(self, text: str, audio: bytes) -> str:
        words = rules.norm(text).split()
        if _STOP_WORDS.search(text):
            return ""
        if rules.is_backchannel(text):
            return "backchannel"
        if rules.addressed_to_other(text, getattr(self.cfg, "name", "FDA")):
            return "to someone else"
        said = set(rules.norm(self.bot_text() or "").split())
        if words and said and sum(w in said for w in words) / len(words) > 0.7:
            return "echo"
        if len(words) < 3:
            return "too short"
        return ""

    async def _peek_then_unduck(self, audio: bytes):
        if await self._peek(audio=audio, after_end=True):
            return
        await self._unduck_later()

    async def _unduck_later(self):
        # the transcript decides within ~unduck_s (an interruption arrives first if it was a real turn)
        await asyncio.sleep(self.unduck_s)
        if self.ducked:
            self._set_duck(False)
            self.stats["resumed"] += 1


class NotForMe(FrameProcessor):
    """Between the LLM and TTS. Holds the first characters of every reply until it knows it is not "(silent)"."""

    def __init__(self, on_ignored=None, io=None, **kw):
        super().__init__(**kw)
        self.on_ignored, self.io = on_ignored, io
        self.held: list[Frame] = []
        self.text = ""
        self.state = "idle"          # idle | deciding | speaking | dropping
        self.interrupted_reply = ""  # what the bot was saying when it was last interrupted
        self.t_interrupt = 0.0
        self.reply_full = ""         # everything the LLM wrote for the reply now playing
        self.spoken = ""             # sentences of it the TTS has started
        self.last_sentence = ""      # the sentence playing now (cut off by an interruption: say it again on resume)
        self.reply_done = False      # the LLM finished writing the reply now playing
        self.paused = ""             # the rest of the last interrupted reply, from the cut sentence on
        self.paused_said = ""        # what was heard of it before the cut
        self.paused_done = False     # whether that reply was fully written (else the LLM must write the rest)
        self.paused_t = 0.0
        self.ignored = 0
        self.overruled = 0
        self.own_name = "FDA"
        self._retried_for = ""       # transcript we already overruled a "(silent)" for (only once per turn)
        self.turn_said: list[str] = []   # replies already spoken for the current user turn
        self._turn = -1
        self.repeats = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            words = self.reply_full.split()
            # resume from the start of the sentence that was cut off, so it sounds whole again
            done = max(0, len(self.spoken.split()) - len(self.last_sentence.split()))
            rest = " ".join(words[done:])
            if self.state in ("speaking", "idle") and rest.strip() and self.spoken:
                self.interrupted_reply, self.t_interrupt = rest.strip(), time.monotonic()
                self.paused, self.paused_said = rest.strip(), " ".join(words[:done])
                self.paused_done, self.paused_t = self.state == "idle" or self.reply_done, time.monotonic()
                logger.info(f"interrupted; kept for 'continue': {rest[:60]!r}")
            self.held, self.text, self.state = [], "", "idle"
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self.state, self.text, self.held = "deciding", "", [frame]
            return
        if self.state == "deciding":
            if isinstance(frame, LLMTextFrame):
                self.held.append(frame)
                self.text += frame.text
                if not could_be_silent(self.text):
                    if self._said_before() and len(self.text.split()) < 6:
                        return               # a second reply in this turn: wait for a few words, it may repeat
                    await self._decide()
                return
            if isinstance(frame, LLMFullResponseEndFrame):
                self.held.append(frame)
                await self._decide(end=True)
                return
            self.held.append(frame)
            return
        if self.state == "dropping":
            if isinstance(frame, LLMTextFrame):
                return
            if isinstance(frame, LLMFullResponseEndFrame):
                self.state = "idle"
                await self.push_frame(frame, direction)
                await self._maybe_resume()
                return
        if isinstance(frame, LLMTextFrame) and self.state == "speaking":
            self.reply_full += frame.text
        if isinstance(frame, LLMFullResponseEndFrame):
            if self.state == "speaking" and self.reply_full.strip():
                self.turn_said.append(self.reply_full)
            self.reply_done = True
            self.state = "idle"
        await self.push_frame(frame, direction)

    def _said_before(self) -> bool:
        turn = getattr(self.io, "turn", 0)
        if turn != self._turn:
            self._turn, self.turn_said = turn, []
        return bool(self.turn_said)

    def _is_repeat(self, text: str) -> bool:
        """After a tool call, a small model often says the same thing again ("I'm building it now")."""
        if not self._said_before():
            return False
        words = rules.norm(text).split()[:8]
        before = set(rules.norm(" ".join(self.turn_said)).split())
        return len(words) >= 3 and sum(w in before for w in words) / len(words) >= 0.7

    async def _decide(self, end: bool = False):
        held, self.held = self.held, []
        heard = getattr(self.io, "last_heard", "") or ""
        if SILENT.match(self.text.strip()) and heard and heard != self._retried_for and                 rules.probably_for_me(heard, self.own_name):
            # a 3B model sometimes goes "(silent)" on "Wait, what time is it?": ask once more, it was for us
            self._retried_for = heard
            self.overruled += 1
            logger.info(f"'(silent)' overruled for {heard[:60]!r}: asking again")
            for f in held:
                if not isinstance(f, LLMTextFrame):
                    await self.push_frame(f)
            self.state = "idle" if end else "dropping"
            await self.push_frame(LLMMessagesAppendFrame(messages=[{"role": "system", "content":
                "That was said to you, not to someone else. Answer it now, briefly. Do not reply (silent)."}],
                run_llm=True), FrameDirection.UPSTREAM)
            return
        if SILENT.match(self.text.strip()):
            self.ignored += 1
            logger.info("not for me: staying silent")
            if self.on_ignored:
                self.on_ignored()
            for f in held:                                   # keep the start/end markers, drop the text
                if not isinstance(f, LLMTextFrame):
                    await self.push_frame(f)
            self.state = "idle" if end else "dropping"
            if end:
                await self._maybe_resume()
            return
        if self._is_repeat(self.text):
            self.repeats += 1
            logger.info(f"repeat of what was just said, dropped: {self.text[:60]!r}")
            for f in held:
                if not isinstance(f, LLMTextFrame):
                    await self.push_frame(f)
            self.state = "idle" if end else "dropping"
            return
        self.state = "speaking" if not end else "idle"
        self.reply_full, self.spoken, self.last_sentence, self.reply_done = "", "", "", end
        for f in held:
            if isinstance(f, LLMTextFrame):
                self.reply_full += f.text
            await self.push_frame(f)

    async def _maybe_resume(self):
        """The bot was interrupted by speech that turned out not to be for it: finish the interrupted reply."""
        if _STOP_WORDS.search(getattr(self.io, "last_heard", "") or ""):
            self.interrupted_reply = ""          # "wait, stop": they wanted it to stop, even if the rest was unclear
            return
        if self.interrupted_reply and time.monotonic() - self.t_interrupt < 8.0:
            text, self.interrupted_reply = self.interrupted_reply, ""
            logger.info(f"resuming: {text[:60]!r}")
            self.reply_full, self.spoken, self.last_sentence, self.state = text, "", "", "idle"
            self.reply_done, self.paused = True, ""
            await self.push_frame(TTSSpeakFrame(text))

    def take_paused(self, max_age_s: float = 300.0):
        """(rest, said_before, fully_written) of the last interrupted reply, once; None if there is none."""
        if not self.paused or time.monotonic() - self.paused_t > max_age_s:
            return None
        out = (self.paused, self.paused_said, self.paused_done)
        self.paused = ""
        return out

class ToolFiller(FrameProcessor):
    """After the LLM: when it calls a slow tool (internet), say a short filler at once, so the person hears a
    reply within ~1 s instead of waiting for the tool and the second LLM call."""

    FILLERS = ["Let me check that.", "One moment, let me look.", "Checking that now.", "Let me see."]

    def __init__(self, slow: set[str], **kw):
        super().__init__(**kw)
        self.slow, self.n = slow, 0
        self.t_last = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, FunctionCallsStartedFrame) and direction == FrameDirection.DOWNSTREAM:
            names = {c.function_name for c in frame.function_calls}
            if any(c.function_name == "tweak_app" and re.search(r"photo|picture|image", str(c.arguments), re.I)
                   for c in frame.function_calls):
                names.add("tweak_app")           # finding a photo takes a few seconds
            if names & (self.slow | {"tweak_app"}) and time.monotonic() - self.t_last > 5:
                self.t_last = time.monotonic()
                self.n += 1
                await self.push_frame(TTSSpeakFrame(self.FILLERS[self.n % len(self.FILLERS)]))


class SpokenTap(FrameProcessor):
    """After the TTS: tells NotForMe which sentences have started playing, and shows them on the display."""

    def __init__(self, notforme: NotForMe, ui=None, **kw):
        super().__init__(**kw)
        self.nfm, self.ui = notforme, ui
        self.line = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and direction == FrameDirection.DOWNSTREAM and frame.text.strip()                 and not self.line.endswith(frame.text.strip()):      # the same text can come word- and sentence-wise
            self.nfm.spoken = (self.nfm.spoken + " " + frame.text).strip()
            self.nfm.last_sentence = frame.text.strip()
            self.line = (self.line + " " + frame.text).strip()
            if self.ui:
                self.ui.send({"type": "fda", "text": self.line, "final": False})
        elif isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame, BotStoppedSpeakingFrame)):
            # the reply is over when the bot stops talking (not when the next reply starts: by then the person's
            # next words are on the display, and the old reply would be shown a second time below them)
            if self.ui and self.line:
                self.ui.send({"type": "fda", "text": self.line, "final": True})
            self.line = ""
        await self.push_frame(frame, direction)


class HeardTap(FrameProcessor):
    """After the STT: shows transcripts on the display; drops pure backchannels while the bot is talking, so
    "uh huh" never becomes a turn of its own."""

    def __init__(self, io, ui=None, name: str = "FDA", on_heard=None, on_ignored=None, intercept=None, **kw):
        super().__init__(**kw)
        self.io, self.ui, self.own_name, self.on_heard, self.on_ignored = io, ui, name, on_heard, on_ignored
        self.intercept = intercept       # async (text) -> True when it handled the request itself
        self.backchannels = 0
        self.not_for_me = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self.io.last_heard = frame.text
            if self.io.bot_speaking and _STOP_WORDS.search(frame.text):
                logger.info(f"told to stop: {frame.text!r}")
                await self.broadcast_interruption()
                if len(rules.norm(frame.text).split()) <= 3:      # just "stop" / "okay, wait": go quiet, no reply
                    if self.ui:
                        self.ui.send({"type": "user", "text": frame.text})
                    return
            if self.io.bot_speaking and rules.is_backchannel(frame.text):
                self.backchannels += 1
                logger.info(f"backchannel while talking, ignored: {frame.text!r}")
                if self.ui:
                    self.ui.send({"type": "user", "text": frame.text, "ignored": True})
                return
            if rules.addressed_to_other(frame.text, self.own_name):
                self.not_for_me += 1
                logger.info(f"said to someone else, ignored: {frame.text!r}")
                if self.ui:
                    self.ui.send({"type": "user", "text": frame.text, "ignored": True})
                if self.on_ignored:
                    self.on_ignored()
                return
            if self.ui:
                self.ui.send({"type": "user", "text": frame.text})
            self.io.turn = getattr(self.io, "turn", 0) + 1
            if self.intercept is not None and await self.intercept(frame.text):
                return
            if self.on_heard:
                self.on_heard()
            if _TIME_Q.search(frame.text):
                # small models make up the time; the true one goes at the end of the prompt (keeps the cache)
                frame.text += time.strftime(" [now: %I:%M %p, %A %B %d %Y]")
        await self.push_frame(frame, direction)
