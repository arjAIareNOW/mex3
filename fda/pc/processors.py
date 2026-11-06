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

import numpy as np

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
# a bare "stop" ("stop", "okay, wait", "hold on", "that's enough", "Elkit, stop"): go quiet, never a reply.
# Only stop words and fillers: "stop the timer" / "stop the music" are commands and go on as usual.
_STOP_CORE = {"stop", "stopp", "strop", "stahp", "wait", "enough", "quiet", "shush", "shh", "hush", "silence"}
_STOP_FILL = {"okay", "ok", "please", "now", "just", "it", "that", "that's", "thats", "is", "alright", "all", "right",
              "fine", "hey", "oh", "um", "uh", "hold", "on", "hang", "never", "mind", "nevermind", "shut", "up", "be",
              "a", "s", "sec", "second", "moment", "minute", "already", "talking", "there", "you", "so", "no", "yes", "yeah"}


def is_bare_stop(text: str, name: str = "") -> bool:
    w = [x for x in rules.norm(strip_name(text, name)).split() if x]
    if not w or len(w) > 5:
        return False
    pairs = {" ".join(w[i:i + 2]) for i in range(len(w) - 1)}
    has_stop = bool(set(w) & _STOP_CORE) or "nevermind" in w or \
        bool(pairs & {"hold on", "hang on", "never mind", "shut up", "be quiet"})
    return has_stop and all(x in _STOP_CORE or x in _STOP_FILL for x in w)


_TIME_Q = re.compile(r"\b(time|date|day|today|tonight|tomorrow|yesterday|clock|o'?clock|month|year)\b", re.I)
SILENT = re.compile(r"^\W*[(\[]\s*silen|^\W*silent\W*$", re.I)


# while a video plays, the mic hears the video too: only these (or the agent's name) count as said to the agent
MEDIA_CMD = re.compile(r"\b(pause|resume|unpause|continue|play|stop|close|exit|quit|hide|turn|make|set|mute|unmute|louder|"
                       r"quieter|softer|volume|skip|rewind|restart|start|go|fast|next|show|previous)\b.{0,24}"
                       r"\b(video|youtube|it|this|that|sound|volume|up|down|ahead|forward|back|over|beginning|seconds|"
                       r"percent|photo|picture|image|pic|one|\d+)\b"
                       r"|^\W*(pause|resume|play|stop|close|mute|unmute|louder|quieter|softer|next|previous|another one|"
                       r"next one|next photo|next picture)\W*$", re.I)


def called_by_name(text: str, name: str) -> bool:
    """The name, also as ASR splits it ("Elkit", "El kit", "elk it")."""
    if not name:
        return False
    loose = r"\W*".join(re.escape(c) for c in name.lower() if c.isalnum())
    return re.search(rf"\b{loose}\b", text.lower()) is not None


def strip_name(text: str, name: str) -> str:
    """"Elkit, louder" -> "louder" (so commands match with or without the name)."""
    if not name:
        return text.strip()
    loose = r"\W*".join(re.escape(c) for c in name.lower() if c.isalnum())
    t = re.sub(rf"\b(hey\s+|okay\s+|ok\s+)?{loose}\b[\s,.!?]*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" ,.!?")


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

    # fd-benchmark 2026-11-06: it stopped 2.2 s after an interruption began (peeks at 1.0 / 1.6 s needed 3 words,
    # fallback at 3 s). Now: earlier peeks, 2 words are enough once the overlap is ~0.7 s long, fallback at 2 s.
    def __init__(self, io, cfg, sustained_s: float = 2.0, unduck_s: float = 0.9, peek_s=(0.5, 0.9, 1.3), **kw):
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
        self.echo_text = lambda: ""      # what it said in the last seconds (late echo check)
        self.on_bot = None               # callback(talking: bool) for the display
        self.on_user = None              # callback(event: dict) for the display: user voice on/off and its level
        self._t_mic = 0.0
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
            if io.user_speaking and self.on_user is not None:
                now = time.monotonic()
                if now - self._t_mic >= 0.06:             # ~16 a second: the display's "I hear you" glow
                    self._t_mic = now
                    x = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(x):
                        self.on_user({"type": "mic", "db": round(float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)), 1)})
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
            io.t_bot_stopped = time.monotonic()
            self._set_duck(False)
            self._cancel("_watch")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            io.user_speaking = True
            if self.on_user is not None:
                self.on_user({"type": "voice", "on": True})
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
            if self.on_user is not None:
                self.on_user({"type": "voice", "on": False})
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
            # decided it is not a turn: full voice again now (a short reply was played ducked to the end)
            if self._unduck is not asyncio.current_task():
                self._cancel("_unduck")
            if self.ducked:
                self._set_duck(False)
                self.stats["resumed"] += 1
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
        if rules.is_echo(text, self.echo_text()):        # its voice from a sentence or two ago (late through the room)
            return "echo"
        # 2 words stop it once the overlap has gone on ~0.7 s (the buffer also holds 0.5 s from before the VAD fired);
        # a shorter bit ("Not exactly." = its own echo, live 2026-11-05) still needs 3
        need = 2 if len(audio) >= int(1.2 * 32000) else 3
        if len(words) < need:
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
        self.interrupt_turn = 0      # the user turn count when it was interrupted
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
        self.recent: list[tuple[float, str]] = []   # (time, sentence) the speaker played lately (echo check)
        self.thinking = 0            # replies dropped because the model was thinking aloud ("The user wants ...")
        self._think_retry = False    # ask again for a direct answer when this reply ends
        self._think_retried_for = ""
        self._calls_in_reply = False

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
                self.interrupt_turn = getattr(self.io, "turn", 0)
                self.paused, self.paused_said = rest.strip(), " ".join(words[:done])
                self.paused_done, self.paused_t = self.state == "idle" or self.reply_done, time.monotonic()
                logger.info(f"interrupted; kept for 'continue': {rest[:60]!r}")
            self.held, self.text, self.state = [], "", "idle"
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self.state, self.text, self.held = "deciding", "", [frame]
            self._think_retry, self._calls_in_reply = False, False
            return
        if isinstance(frame, FunctionCallsStartedFrame):
            self._calls_in_reply = True
        if self.state == "deciding":
            if isinstance(frame, LLMTextFrame):
                self.held.append(frame)
                self.text += frame.text
                if not re.search(r"[A-Za-z0-9]", self.text) and len(self.text) < 200 and                         not could_be_silent(self.text):
                    return                   # only dots so far ("... …"): wait for words
                if not could_be_silent(self.text):
                    if self._said_before() and len(self.text.split()) < 6:
                        return               # a second reply in this turn: wait for a few words, it may repeat
                    if rules.could_be_thinking(self.text):
                        return               # "The user ..." / "We need ...": wait until it is clear
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
                if self._think_retry:
                    await self._ask_direct()
                else:
                    await self._maybe_resume()
                return
        if isinstance(frame, LLMTextFrame) and self.state == "speaking":
            self.reply_full += frame.text
            tail = " ".join(self.reply_full.split()[-12:])
            if rules.is_thinking(self.reply_full) or (len(tail.split()) >= 8 and rules.is_garbage(tail)):
                # it slid into thinking aloud, or into "... … …", mid-reply: stop there
                self.thinking += 1
                logger.info(f"thinking aloud mid-reply, rest dropped: {self.reply_full[-80:]!r}")
                self.state = "dropping"
                return
        if isinstance(frame, LLMFullResponseEndFrame):
            if self.state == "speaking" and self.reply_full.strip():
                self.turn_said.append(self.reply_full)
            self.reply_done = True
            self.state = "idle"
        await self.push_frame(frame, direction)

    def recent_text(self, window_s: float = 15.0) -> str:
        """What the bot said in the last `window_s` seconds (its voice can come back through the mic late)."""
        now = time.monotonic()
        self.recent = [(t, s) for t, s in self.recent if now - t < window_s]
        return " ".join(s for _, s in self.recent)

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
        if rules.is_thinking(self.text) or rules.is_garbage(self.text):
            self.thinking += 1
            logger.info(f"thinking aloud / garbled, not spoken: {self.text[:80]!r}")
            for f in held:
                if not isinstance(f, LLMTextFrame):
                    await self.push_frame(f)
            self._think_retry = True
            self.state = "dropping"
            if end:
                self.state = "idle"
                await self._ask_direct()
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

    async def _ask_direct(self):
        """The reply was the model thinking aloud: ask once for the real answer (not when a tool is running, its
        result brings a fresh reply anyway)."""
        self._think_retry = False
        heard = getattr(self.io, "last_heard", "") or ""
        if self._calls_in_reply or not heard or heard == self._think_retried_for:
            return
        self._think_retried_for = heard
        await self.push_frame(LLMMessagesAppendFrame(messages=[{"role": "system", "content":
            "Your last reply was your private reasoning, not an answer, so it was not spoken. Now say the answer "
            "itself to the person, in one or two short spoken sentences. Never describe what the user wants or "
            "what you will do, and never name tools."}], run_llm=True), FrameDirection.UPSTREAM)

    async def _maybe_resume(self, from_ignored: bool = False):
        """The bot was interrupted by speech that turned out not to be for it: finish the interrupted reply.
        from_ignored: called for speech that never reached the LLM (far away, side talk). Then only when no real
        request came since the interruption, else it would say the old sentence again on top of the new answer."""
        if _STOP_WORDS.search(getattr(self.io, "last_heard", "") or ""):
            self.interrupted_reply = ""          # "wait, stop": they wanted it to stop, even if the rest was unclear
            return
        turns = getattr(self.io, "turn", 0) - self.interrupt_turn
        # a real request counted just BEFORE the interruption caused it (it is counted first, the interruption
        # reaches here ~3 ms later): the old sentence must not come back when noise follows
        caused_by_request = getattr(self.io, "t_turn", -1e9) >= self.t_interrupt - 1.0
        if from_ignored and (turns > 0 or caused_by_request or self.state != "idle"):
            if self.interrupted_reply:
                logger.info(f"not resuming {self.interrupted_reply[:40]!r}: a new request came since")
            self.interrupted_reply = ""
            return
        if turns > 1:                            # two requests since: the old reply is long gone
            self.interrupted_reply = ""
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

_REFUSAL = re.compile(r"^\W*(?:i[’']?m|i am) sorry,? but i (?:can[’']?t|cannot|can not) (?:help|assist|do that)"
                      r"(?: with (?:that|this))?\W*$", re.I)


_REFUSAL_FORMS = [re.sub(r"[^a-z' ]", "", f) for f in (
    "i'm sorry, but i can't help with that", "i'm sorry, but i cannot help with that",
    "i am sorry, but i can't help with that", "i'm sorry, but i can't assist with that",
    "i'm sorry, but i cannot assist with that", "i'm sorry, but i can't do that", "i'm sorry, but i can't help with this")]


def _maybe_refusal(text: str) -> bool:
    """Could `text` still be (the start of) the bare refusal?"""
    n = re.sub(r"[^a-z' ]", "", re.sub(r"\s+", " ", text.lower().replace("’", "'"))).strip()
    return any(f.startswith(n) or n.startswith(f) for f in _REFUSAL_FORMS)


class RefusalGuard(FrameProcessor):
    """After the LLM: gpt-oss sometimes answers a harmless request ("tell me a long story about the first trip to
    the moon") with a bare "I'm sorry, but I can't help with that." -- the wrong feedback. The start of every reply
    is held back until it is clearly something else (~60 characters); a reply that is ONLY that refusal is not
    spoken, and `retry()` asks the LLM once more (at most once every 20 s, so a real refusal still comes through)."""

    def __init__(self, retry, **kw):
        super().__init__(**kw)
        self.retry = retry
        self.held: list[Frame] | None = None
        self.text = ""
        self.t_retry = 0.0
        self.stats = {"refusals_retried": 0}

    async def _flush(self):
        held, self.held = self.held or [], None
        for f in held:
            await self.push_frame(f)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self.held, self.text = [], ""
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and self.held is not None:
            self.held.append(frame)
            self.text += frame.text
            if not _maybe_refusal(self.text):
                await self._flush()                    # clearly not the refusal: let it through at once
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.held is not None and _REFUSAL.match(self.text.strip()) and \
                    time.monotonic() - self.t_retry > 20:
                logger.warning(f"LLM refused a request ({self.text.strip()!r}): not spoken, asking once more")
                self.held, self.t_retry = None, time.monotonic()
                self.stats["refusals_retried"] += 1
                await self.push_frame(frame, direction)
                await self.retry()
                return
            await self._flush()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self.held = None
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


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
            for c in frame.function_calls:                # in logs/agent.log: which tool, with what
                logger.info(f"tool call: {c.function_name} {str(c.arguments)[:200]}")
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
            self.nfm.recent.append((time.monotonic(), frame.text.strip()))
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

    def __init__(self, io, ui=None, name: str = "FDA", on_heard=None, on_ignored=None, intercept=None,
                 echo_text=None, **kw):
        super().__init__(**kw)
        self.echo_text = echo_text or (lambda: "")       # what the bot said in the last seconds
        self.echoes = 0
        self.io, self.ui, self.own_name, self.on_heard, self.on_ignored = io, ui, name, on_heard, on_ignored
        self.intercept = intercept       # async (text) -> True when it handled the request itself
        self.backchannels = 0
        self.not_for_me = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self.io.last_heard = frame.text
            if is_bare_stop(frame.text, self.own_name):
                # talking, thinking or quiet: stop everything and say nothing, so the person can just talk again
                logger.info(f"told to stop (no reply): {frame.text!r}")
                await self.broadcast_interruption()
                if self.ui:
                    self.ui.send({"type": "user", "text": frame.text})
                if self.intercept is not None:
                    await self.intercept(frame.text)     # the agent: stop a ringing alarm, pause a video (silently)
                return
            if self.io.bot_speaking and _STOP_WORDS.search(frame.text):
                logger.info(f"told to stop: {frame.text!r}")
                await self.broadcast_interruption()
                if len(rules.norm(frame.text).split()) <= 3:      # just "stop" / "okay, wait": go quiet, no reply
                    if self.ui:
                        self.ui.send({"type": "user", "text": frame.text})
                    return
            # a video only plays on a display: with none connected (closed page, benchmark) nothing plays, so she
            # must not go deaf (fd-benchmark 2026-11-06: a clip asked for a music video, then 12 turns were ignored)
            clients = getattr(self.ui, "clients", None)
            screen = self.ui is not None and (clients is None or bool(clients))
            if getattr(self.io, "media_playing", False) and screen and not called_by_name(frame.text, self.own_name) \
                    and not MEDIA_CMD.search(frame.text):
                self.not_for_me += 1
                logger.info(f"video playing, not called by name, ignored: {frame.text!r}")
                if self.ui:
                    self.ui.send({"type": "user", "text": frame.text, "ignored": True})
                if self.on_ignored:
                    self.on_ignored()
                return
            near_bot = self.io.bot_speaking or time.monotonic() - getattr(self.io, "t_bot_stopped", 0.0) < 4.0
            if near_bot and rules.is_echo(frame.text, self.echo_text()):
                self.echoes += 1                         # its own voice through the mic: not a turn, no reply
                logger.info(f"own voice heard back, ignored: {frame.text!r}")
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
            self.io.t_turn = time.monotonic()        # the interruption this request causes comes a moment later
            if self.intercept is not None and await self.intercept(frame.text):
                return
            if self.on_heard:
                self.on_heard()
            if _TIME_Q.search(frame.text):
                # small models make up the time; the true one goes at the end of the prompt (keeps the cache)
                frame.text += time.strftime(" [now: %I:%M %p, %A %B %d %Y]")
        await self.push_frame(frame, direction)


class TurnClock:
    """Where the time of one turn goes, written to the log once the first audio of the reply is out:
    speech end -> heard (ASR) -> sent to the LLM (turn detection) -> first word (LLM) -> first audio (TTS)."""

    STEPS = [("speech_end", "heard", "asr"), ("heard", "sent", "turn wait"), ("sent", "first_word", "llm"),
             ("first_word", "first_audio", "tts")]
    STALE_S = 3.0          # heard -> sent is ~0.05-0.8 s in a real turn

    def __init__(self):
        self.t: dict[str, float] = {}
        self.active = False
        self.last: dict[str, int] = {}

    def mark(self, name: str):
        now = time.monotonic()
        if name in ("speech_end", "heard"):           # the last one before the turn is sent counts
            if not self.active:
                self.t[name] = now
        elif name == "sent":
            # speech that was ignored (side talk, "not for me") leaves its stamps behind: a reply sent long after
            # (a tool result, a resumed answer) is not a spoken turn, so its timing must not start from them
            fresh = now - self.t.get("heard", now) <= self.STALE_S
            self.t = {k: v for k, v in self.t.items() if k in ("speech_end", "heard") and fresh}
            self.t["sent"], self.active = now, True
        elif self.active and name not in self.t:      # first_word, first_audio: once per turn
            self.t[name] = now
            if name == "first_audio":
                self.report()

    def report(self):
        self.active = False
        parts, ms = [], {}
        for a, b, label in self.STEPS:
            if a in self.t and b in self.t and self.t[b] >= self.t[a]:
                ms[label] = round((self.t[b] - self.t[a]) * 1000)
                parts.append(f"{label} {ms[label]}")
        start = self.t.get("speech_end", self.t.get("heard", self.t["sent"]))
        ms["total"] = round((self.t["first_audio"] - start) * 1000)
        self.last = ms
        logger.info(f"turn timing (ms): {', '.join(parts)} | total {ms['total']}")
        self.t = {}


class Mark(FrameProcessor):
    """Passes every frame on; marks the clock when the first frame of the given kinds goes down the pipeline."""

    def __init__(self, clock: TurnClock, kinds: dict, **kw):
        super().__init__(**kw)
        self.clock, self.kinds = clock, kinds        # {FrameClass: mark name}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            for kind, name in self.kinds.items():
                if isinstance(frame, kind):
                    self.clock.mark(name)
        await self.push_frame(frame, direction)
