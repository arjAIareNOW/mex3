"""Pipecat services backed by FDA's MCP servers: speech-to-text and text-to-speech are MCP tool calls.

    McpSTTService : one `transcribe` call per VAD-segmented utterance (asr MCP server)
    McpTTSService : one `synthesize` call per sentence (tts MCP server); Pipecat splits the reply into sentences,
                    so the first sentence plays while the LLM is still writing the rest
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from mcp.client.client import Client
from pipecat.frames.frames import ErrorFrame, Frame, STTMetadataFrame, TranscriptionFrame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from .. import rules


class McpConnection:
    """A persistent MCP session to one streamable-HTTP server, owned by a dedicated task (MCP sessions are
    task-bound). call() can be used from any task on the same loop and reconnects after a dropped session."""

    def __init__(self, url: str, name: str = ""):
        self.url, self.name = url, name or url
        self.client: Client | None = None
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None
        self._closing: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            if self.client is not None:
                return
            loop = asyncio.get_running_loop()
            self._ready, self._closing = loop.create_future(), asyncio.Event()
            self._task = asyncio.create_task(self._own(self._ready, self._closing))
            await self._ready

    async def _own(self, ready, closing):
        try:
            async with Client(self.url, read_timeout_seconds=600) as c:
                self.client = c
                ready.set_result(True)
                await closing.wait()
        except Exception as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e)
            else:
                logger.warning(f"MCP {self.name}: session ended: {e}")
        finally:
            self.client = None

    async def call(self, tool: str, args: dict, timeout: float = 120.0):
        """Returns the tool's structured result if it has one, else its text."""
        for attempt in range(2):
            if self.client is None:
                await self.start()
            try:
                r = await asyncio.wait_for(self.client.call_tool(tool, args), timeout)
                break
            except asyncio.TimeoutError:
                raise
            except Exception as e:  # noqa: BLE001  (dropped session: reconnect once)
                if attempt:
                    raise
                logger.warning(f"MCP {self.name}.{tool} failed ({e}); reconnecting")
                await self.close()
        if r.is_error:
            raise RuntimeError(" ".join(getattr(c, "text", "") for c in r.content)[:300])
        sc = r.structured_content
        if sc is not None:
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        text = "".join(getattr(c, "text", "") for c in r.content)
        try:
            return json.loads(text)
        except ValueError:
            return text

    async def close(self):
        if self._closing is not None:
            self._closing.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 5)
            except Exception:  # noqa: BLE001
                self._task.cancel()
        self.client, self._task = None, None


_UNFINISHED = re.compile(r"(?:,|\.\.\.|…|-|\b(?:and|but|or|so|because|um+|uh+|the|a|an|to|of|with|my|like))\s*$",
                         re.I)


def turn_wait_s(text: str) -> float:
    """Pipecat's turn-end wait after our transcript (ttfs_p99_latency; it waits that minus VAD stop_secs 0.2 s).
    A question is done: answer at once (0.05 s). A statement may go on after a pause ("I practice every day. Yeah,
    the chords...", fd-benchmark pause clips): 0.3 s. Cut off mid-thought ("and", "the", a comma): 0.8 s.
    If they start talking again, VAD cancels the wait anyway."""
    t = (text or "").strip()
    if _UNFINISHED.search(t):
        return 1.0
    if t.endswith("?"):
        return 0.25
    return 0.5


class McpSTTService(SegmentedSTTService):
    """Speech-to-text through the asr MCP server (Moonshine offline, or Groq Whisper)."""

    @property
    def wants_wav_segments(self) -> bool:
        return False                     # raw 16-bit PCM at 16 kHz

    def __init__(self, conn: McpConnection, far_db: float = 10.0, on_ignored=None, own_name: str = "FDA", **kwargs):
        super().__init__(sample_rate=16000, settings=STTSettings(model="mcp-asr", language=Language.EN), **kwargs)
        self.conn = conn
        self.last_ms = 0.0
        self.far_db = far_db             # quieter than the user's usual level by this much -> background (0 = off)
        self.user_db: float | None = None
        self.far_dropped = 0
        self.on_ignored = on_ignored
        self.own_name = own_name
        self.output_busy = None          # callable: an alarm or music is playing now (set by the agent)

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def level_db(audio: bytes) -> float:
        """Speech level of a segment: RMS of its loudest 30 % of 20 ms frames, in dBFS."""
        x = np.frombuffer(audio, np.int16).astype(np.float32) / 32768
        n = len(x) // 320
        if n == 0:
            return -100.0
        rms = np.sqrt(np.mean(x[: n * 320].reshape(n, 320) ** 2, axis=1) + 1e-12)
        top = np.sort(rms)[-max(1, int(n * 0.3)):]
        return float(20 * np.log10(np.mean(top) + 1e-9))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        t = time.perf_counter()
        try:
            text = await self.conn.call("transcribe", {"audio_b64": base64.b64encode(audio).decode(),
                                                       "sample_rate": 16000}, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self} transcribe failed: {e}")
            yield ErrorFrame(error=f"ASR MCP server: {e}")
            return
        finally:
            await self.stop_processing_metrics()
        self.last_ms = (time.perf_counter() - t) * 1000
        text = (text or "").strip() if isinstance(text, str) else str(text)
        if text:
            lvl = self.level_db(audio)
            logger.info(f"heard ({self.last_ms:.0f} ms, {len(audio) / 32000:.1f} s, {lvl:.0f} dB): {text}")
            # an alarm or music playing: the echo canceller also turns the person down (by ~10 dB), so a level
            # check would throw away "what time is it?" said right next to it. Judge only a quiet room.
            busy = bool(self.output_busy and self.output_busy())
            if busy:
                logger.info("speaker busy (alarm / music): no far-away check")
            if self.far_db and not busy and self.user_db is not None and lvl < self.user_db - self.far_db and \
                    self.own_name.lower() not in text.lower():          # "FDA, ..." from across the room still counts
                self.far_dropped += 1                      # TV / radio / someone across the room
                logger.info(f"far-away speech ({lvl:.0f} dB vs user {self.user_db:.0f} dB), ignored")
                if self.on_ignored:
                    self.on_ignored()
                return
            if not busy:                                   # a level under the alarm is not their usual level
                self.user_db = lvl if self.user_db is None else 0.7 * self.user_db + 0.3 * lvl
            # the whole utterance at once: this IS the final transcript. Not marked, pipecat's turn-end waits out its
            # STT safety timer every turn (fd-benchmark 2026-11-06: "turn wait" was a flat ~0.8 s on every turn)
            # how long the turn end waits after this text (see turn_wait_s): read by the stop strategy before the text
            yield STTMetadataFrame(service_name=getattr(self, "_name", "McpSTTService"), ttfs_p99_latency=turn_wait_s(text))
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, finalized=True)


# characters the voices read as noise or run words together on: web text is full of them
_UNI = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
                      "‌": "", "‍": "", "⁠": "", "﻿": "", "‘": "'", "’": "'",
                      "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"', "„": '"',
                      "«": '"', "»": '"', "‑": "-", "‐": "-", "–": ", ", "—": ", ", "―": ", ", "−": "-",
                      "…": "...", "•": ", ", "·": ", ", "→": " to ", "×": " times ",
                      "°": " degrees "})
# chatter tacked on the end of replies ("Let me know if you need anything else.")
_CHATTER = re.compile(r"(?i)\s*\b(let me know|feel free to (ask|let me know|reach out)|is there anything else|"
                      r"anything else i can|just let me know)\b[^.?!]*\b(else|next|other|more|changes|controls|help|"
                      r"further|do)\b[^.?!]*[.?!]*|\s*\bwhat would you like (to do )?next\b[^.?!]*[.?!]*"
                      r"|\s*\b(?:is there )?anything else\b[^.?!]*\?|\s*\banything else\s*[.?!]*$")
# stage directions the model writes instead of staying quiet: "(no further response)", "[pause]", "(end)"
_STAGE = re.compile(r"(?i)[(\[]\s*(?:no\s+(?:further\s+|more\s+|other\s+)?(?:response|reply|output|answer|comment|"
                    r"action|text)|silence|pause[sd]?|waiting|listening|end(?:s|ed)?|done|nothing(?:\s+more)?|"
                    r"none|no\s+reply|stops?|stopped|continues?|smiles?|laughs?|chuckles?|sighs?)[^)\]]{0,30}[)\]]")


# a bracketed aside that is the model talking to itself: a question, or a plan ("(should we open?)", "(maybe call
# open_app)", "(need to check)"); a plain fact in brackets ("(about 5 km)") stays
_ASIDE = re.compile(r"(?i)\s*[(\[](?=[^)\]]{0,120}[)\]])(?:[^)\]]*\?|\s*(?:should|shall|must|need|needs|maybe|"
                    r"perhaps|probably|let'?s|we|i\s+(?:need|should|will|must)|actually|note|thinking|todo|"
                    r"check|call|use|open)\b[^)\]]*)[)\]]")


def speakable(text: str) -> str:
    """Make a reply safe to say aloud: no markdown, links, "(silent)", citation marks, odd Unicode or chatter."""
    t = re.sub(r"₱\s?([0-9][0-9,]*(?:\.[0-9]+)?)", r"\1 pesos", text)      # ₱2,869 -> 2,869 pesos
    t = t.translate(_UNI)
    t = _ASIDE.sub("", t)                                        # "(should we open?)": a question to itself
    t = rules.drop_thinking(t)                                   # "The user wants ...": its thinking, not for us
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)             # [label](url) -> label
    t = re.sub(r"https?://\S+|www\.\S+", "", t)                  # bare urls
    t = re.sub(r"【[^】]*】|†\S*", "", t)       # gpt-oss citation marks: 【1†source】
    t = re.sub(r"(?i)[(\[]\s*silent\s*[)\]]", "", t)             # "(silent)" inside a real reply
    t = _STAGE.sub("", t)                                        # "(no further response)", "[pause]", "(waiting)"
    if re.fullmatch(r"\s*[(\[][^)\]]*[)\]]\s*[.!]?\s*", t):       # the whole sentence is a bracketed aside
        return ""
    t = re.sub(r"(^|\s)[-*#>]+\s+", r"\1", t)                    # list bullets, headings, quotes
    t = re.sub(r"[*_`~|]+", "", t)                               # bold / italics / code / tables
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", t)   # emoji
    t = _CHATTER.sub("", t)
    if rules.is_garbage(t):                                      # "... Weird……… … It's …": say nothing
        return ""
    t = re.sub(r"\.{3,}|(?:\.\s){2,}", " ", t)                   # runs of dots (the model stuck on "........")
    t = re.sub(r"\(\s*(the|a|an)?\s*\)|\(\s*(the|a|an)\s*$", "", t, flags=re.I)   # "(the )" left over from it
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t if re.search(r"[A-Za-z0-9]", t) else ""             # nothing but punctuation: say nothing


# a second confirmation tacked on after the real one ("Lights are on." + "All set."): said only as the first sentence
_EXTRA_OK = re.compile(r"(?i)^\W*(?:(?:all|you'?re all|you are all) set|done|there you go|that'?s done|okay|ok|"
                       r"got it|all done|sure)\W*$")


class McpTTSService(TTSService):
    """Text-to-speech through the tts MCP server (Piper offline, or Deepgram Aura)."""

    def __init__(self, conn: McpConnection, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model="mcp-tts", voice=None, language=None), **kwargs)
        self.conn = conn
        self._reply_id, self._said_in_reply = None, 0
        self.building = None            # callable -> True while an app is being built (set by the agent)
        self.cache: dict[str, tuple[bytes, int]] = {}    # fillers, synthesized ahead: no round trip when said

    async def warm(self, texts: list[str]):
        """Synthesize these now and keep them (the fillers). Called again later: the voice may have changed
        (DGX3 back after the VPN came up, or Piper while it is down)."""
        for t in texts:
            try:
                r = await self.conn.call("synthesize", {"text": speakable(t)}, timeout=30)
                self.cache[speakable(t)] = (base64.b64decode(r["audio_b64"]), int(r["sample_rate"]))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"filler {t!r} not cached: {e}")

    def can_generate_metrics(self) -> bool:
        return True

    async def _push_spoken_text(self, src_frame, text: str, context_id: str, append_to_context: bool):
        """A sentence run_tts did not say ("(silent)", a premature "It's ready", a second confirmation) gets no
        spoken-text frame either: it must not show as said, nor go into the LLM's history as said."""
        # pipecat runs run_tts (on the transformed text) and then this (on the original text) one after the
        # other for the same sentence, so a one-shot flag set by run_tts marks exactly this sentence.
        if text and getattr(self, "_unsaid", None) == context_id:
            self._unsaid = None
            return
        self._unsaid = None
        await super()._push_spoken_text(src_frame, text, context_id, append_to_context)

    def _not_said(self, raw: str, context_id: str):
        self._unsaid = context_id

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        raw = text
        text = speakable(text)
        if not text.strip():
            self._not_said(raw, context_id)
            return
        if context_id != self._reply_id:
            self._reply_id, self._said_in_reply = context_id, 0
        if self._said_in_reply and _EXTRA_OK.match(text):
            logger.info(f"second confirmation not said: {text!r}")
            self._not_said(raw, context_id)
            return
        if self.building and self.building() and _PREMATURE_READY.search(text) and \
                not re.search(r"timer|alarm|reminder|\blist\b|lights?|music|song|weather|temperature", text, re.I) and \
                not re.search(r"\b(?:when|once|until|till|as soon as|after)\s+(?:it[’']?s|it is|it[’']?ll be|is)\b|"
                              r"\blet you know\b|\bnot (?:ready|done)\b", text, re.I):
            # the LLM jumped ahead ("It's ready, enjoy!") while the build is still running: not said; the real
            # "... is ready" comes when it is
            logger.info(f"'ready' before the build finished, not said: {text!r}")
            self._not_said(raw, context_id)
            return
        self._said_in_reply += 1
        # a long sentence is synthesized in comma-sized pieces, all started at once: the first (short) piece is
        # ready sooner, so the voice starts sooner (synthesis time grows with length: ~0.7 s short, ~2 s long)
        pieces = split_long(text)
        cached = getattr(self, "cache", {}).get(text)

        async def from_cache():
            return {"audio_b64": base64.b64encode(cached[0]).decode(), "sample_rate": cached[1]}
        jobs = [asyncio.create_task(from_cache())] if cached else \
            [asyncio.create_task(self.conn.call("synthesize", {"text": p}, timeout=60)) for p in pieces]
        try:
            await self.start_tts_usage_metrics(text)
            for job in jobs:
                r = await job
                pcm = base64.b64decode(r["audio_b64"])
                rate = int(r["sample_rate"])

                async def chunks(pcm=pcm, rate=rate):
                    step = rate // 5 * 2                  # 200 ms per frame
                    for i in range(0, len(pcm), step):
                        yield pcm[i:i + step]

                async for frame in self._stream_audio_frames_from_iterator(chunks(), in_sample_rate=rate,
                                                                          context_id=context_id):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self} synthesize failed: {e}")
            yield ErrorFrame(error=f"TTS MCP server: {e}")
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()                       # interrupted: the rest is not needed
            await self.stop_ttfb_metrics()




_PREMATURE_READY = re.compile(r"\b(?:it[’']?s|it is|is|are)\s+(?:all\s+)?(?:ready|done|finished|built|live)\b|"
                              r"\benjoy\b|\bhave fun\b|\bgo ahead and (?:play|try)\b|"
                              # a bare "Ready!" / "Done." / "All set!", and how-to-play lines ("Try tapping to make
                              # the bird fly.", "Press space to jump.") are also "it's ready" said too early
                              r"^\W*(?:all\s+)?(?:ready|done|finished|all set|here it is|there you go)\b\W*$|"
                              r"^\W*(?:all\s+)?(?:ready|done|all set)\s*[!.]|"
                              r"\btry (?:tapping|clicking|pressing|it|playing|it out)\b|"
                              r"\b(?:tap|click|press|hit)\b.{0,25}\bto (?:flap|fly|jump|play|start|shoot|move)\b", re.I)


def split_long(text: str, longer_than: int = 70, min_piece: int = 25) -> list[str]:
    """'Back in 1969, a tiny rocket named Saturn V roared off the pad, carrying three astronauts ...' ->
    pieces at commas / semicolons / dashes, none shorter than `min_piece` characters. Short text: unchanged."""
    if len(text) <= longer_than:
        return [text]
    parts = re.split(r"(?<=[,;:—–])\s+", text)
    out: list[str] = []
    for p in parts:
        if out and (len(out[-1]) < min_piece or len(p) < min_piece // 2):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_piece:      # a short tail joins the piece before it
        out[-2] = f"{out[-2]} {out.pop()}"
    return out


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
