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


class DeepgramSegment:
    """One utterance streamed to Deepgram nova-3 while it is said (its own websocket, opened at VAD start).
    send() audio as it comes, finish() at VAD end (CloseStream): the whole text arrives in `done`.
    Measured on campus Wi-Fi (2026-11-07, 20 benchmark phrases): 0.37 s after the end of speech, WER 5 %,
    against Groq Whisper (one upload after the end) 3.1 s, WER 11 %."""

    def __init__(self, key: str, keyterms: list[str]):
        from urllib.parse import quote
        q = "&".join(f"keyterm={quote(k)}" for k in keyterms if k)
        self.url = ("wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en&encoding=linear16"
                    f"&sample_rate=16000&channels=1&interim_results=true&endpointing=300" + (f"&{q}" if q else ""))
        self.finished = False            # Finalize sent: the next speech_final / from_finalize result ends it
        self.key = key
        self.audio: asyncio.Queue = asyncio.Queue()
        self.done: asyncio.Future = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self._run())

    def send(self, pcm: bytes):
        if pcm and not self.done.done():
            self.audio.put_nowait(pcm)

    def finish(self):
        self.audio.put_nowait(None)

    def cancel(self):
        if not self.task.done():
            self.task.cancel()
        if not self.done.done():
            self.done.cancel()

    async def _run(self):
        import websockets
        parts: list[str] = []
        try:
            async with websockets.connect(self.url, additional_headers={"Authorization": f"Token {self.key}"},
                                          open_timeout=4, close_timeout=1, max_queue=None) as ws:
                async def sender():
                    while True:
                        b = await self.audio.get()
                        if b is None:
                            # CloseStream, not Finalize: audio that came in a burst (queued while the socket opened)
                            # is still being worked through, and Finalize covers only what is done so far (test:
                            # "What's a good routine" of "... for a beginner"). CloseStream: all of it, then closed.
                            await ws.send(json.dumps({"type": "CloseStream"}))
                            self.finished = True
                            return
                        await ws.send(b)
                st = asyncio.create_task(sender())
                try:
                    async for raw in ws:                       # until Deepgram closes: every final result
                        m = json.loads(raw)
                        if m.get("type") == "Results" and m.get("is_final"):
                            t = (m.get("channel", {}).get("alternatives") or [{}])[0].get("transcript", "")
                            if t:
                                parts.append(t)
                finally:
                    st.cancel()
            if not self.done.done():
                self.done.set_result(" ".join(parts).strip())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if not self.done.done():
                self.done.set_exception(e)


class McpSTTService(SegmentedSTTService):
    """Speech-to-text through the asr MCP server (Moonshine offline, or Groq Whisper); with stream="deepgram", each
    utterance is also streamed to Deepgram while it is said and that text is used when it comes in time."""

    @property
    def wants_wav_segments(self) -> bool:
        return False                     # raw 16-bit PCM at 16 kHz

    def __init__(self, conn: McpConnection, far_db: float = 10.0, on_ignored=None, own_name: str = "FDA",
                 stream: str = "", stream_wait_s: float = 1.5, keyterms: list[str] | None = None, **kwargs):
        super().__init__(sample_rate=16000, settings=STTSettings(model="mcp-asr", language=Language.EN), **kwargs)
        self.conn = conn
        self.last_ms = 0.0
        self.far_db = far_db             # quieter than the user's usual level by this much -> background (0 = off)
        self.user_db: float | None = None
        self.far_dropped = 0
        self.on_ignored = on_ignored
        self.own_name = own_name
        self.output_busy = None          # callable: an alarm or music is playing now (set by the agent)
        import os
        self.dg_key = os.environ.get("DEEPGRAM_API_KEY", "") if stream == "deepgram" else ""
        if stream == "deepgram" and not self.dg_key:
            logger.warning("stream_asr = deepgram but DEEPGRAM_API_KEY is not set: listening through the asr server")
        self.dg_wait_s = stream_wait_s
        self.dg_terms = [t for t in (keyterms or [own_name]) if t]
        self.dg_down_until = 0.0         # after a failure: the asr server only, for 30 s
        self._dg: DeepgramSegment | None = None
        self._dg_segs: dict[int, DeepgramSegment] = {}
        self.engine_used = "asr"

    # --- Deepgram streaming: one websocket per utterance, from VAD start (with the pre-roll) to VAD end ---
    async def _handle_user_started_speaking(self, frame):
        await super()._handle_user_started_speaking(frame)
        if self._dg is not None:
            self._dg.cancel()
            self._dg = None
        if self.dg_key and time.monotonic() >= self.dg_down_until:
            try:
                self._dg = DeepgramSegment(self.dg_key, self.dg_terms)
                self._dg.send(bytes(self._audio_buffer))          # the ~1 s before VAD noticed the speech
            except Exception as e:  # noqa: BLE001
                logger.warning(f"deepgram stream not started: {e}")
                self._dg = None

    async def process_audio_frame(self, frame, direction):
        await super().process_audio_frame(frame, direction)
        if self._dg is not None and self._user_speaking:
            self._dg.send(frame.audio)

    async def _handle_user_stopped_speaking(self, frame):
        if self._dg is not None:
            self._dg.finish()
            for n in [n for n in self._dg_segs if n < self._segment_count - 3]:   # never collected: drop
                self._dg_segs.pop(n).cancel()
            self._dg_segs[self._segment_count] = self._dg
            self._dg = None
        await super()._handle_user_stopped_speaking(frame)

    async def _stream_text(self) -> str | None:
        """Deepgram's text for the segment being transcribed; None = not usable (then the asr server)."""
        segs = getattr(self, "_dg_segs", None)
        seg = segs.pop(getattr(self, "_transcribing_segment", 0), None) if segs else None
        if seg is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(seg.done), self.dg_wait_s)
        except asyncio.TimeoutError:
            logger.warning(f"deepgram: no text {self.dg_wait_s:.1f} s after the end of speech: asking the asr server")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"deepgram failed ({e}): the asr server for 30 s")
            self.dg_down_until = time.monotonic() + 30
        seg.cancel()
        return None

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
        text = await self._stream_text()
        self.engine_used = "deepgram" if text is not None else "asr"
        if text is None:
            try:
                text = await self.conn.call("transcribe", {"audio_b64": base64.b64encode(audio).decode(),
                                                           "sample_rate": 16000}, timeout=30)
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self} transcribe failed: {e}")
                await self.stop_processing_metrics()
                yield ErrorFrame(error=f"ASR MCP server: {e}")
                return
        await self.stop_processing_metrics()
        self.last_ms = (time.perf_counter() - t) * 1000
        text = (text or "").strip() if isinstance(text, str) else str(text)
        if text:
            lvl = self.level_db(audio)
            via = " deepgram" if self.engine_used == "deepgram" else ""
            logger.info(f"heard ({self.last_ms:.0f} ms{via}, {len(audio) / 32000:.1f} s, {lvl:.0f} dB): {text}")
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

    def _fit_resampler(self, rate: int):
        """Pipecat's stream resampler is fixed to the first input rate it sees. The voice can change rate
        (DGX3 24 kHz, Piper 22.05 kHz when it takes over, a filler cached from the other one): a new
        resampler then, or every later sentence fails ('cannot be reused with different sample rates')."""
        rs = getattr(self, "_resampler", None)
        if rs is not None and getattr(rs, "_in_rate", None) not in (None, rate):
            from pipecat.audio.utils import create_stream_resampler
            logger.info(f"TTS audio now {rate} Hz (was {rs._in_rate:.0f}): new resampler")
            self._resampler = create_stream_resampler()

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
        # a long sentence is synthesized in comma-sized pieces: the first (short) piece is ready sooner, so the
        # voice starts sooner (synthesis time grows with length). The reply's first sentence splits sooner (> 40
        # characters). One piece at a time: on the Pi's 4 cores pieces started all at once slowed each other down
        # (first piece 0.65 s instead of 0.25 s); the next piece is made while the one before it plays.
        pieces = split_long(text, 40, 15) if self._said_in_reply == 1 else split_long(text)
        cached = getattr(self, "cache", {}).get(text)

        async def from_cache():
            return {"audio_b64": base64.b64encode(cached[0]).decode(), "sample_rate": cached[1]}

        def start(i):
            return asyncio.create_task(self.conn.call("synthesize", {"text": pieces[i]}, timeout=60))
        jobs = [asyncio.create_task(from_cache())] if cached else [start(0)]
        try:
            await self.start_tts_usage_metrics(text)
            i = 0
            while i < len(jobs):
                r = await jobs[i]
                if not cached and len(jobs) < len(pieces):
                    jobs.append(start(len(jobs)))       # the next piece, while this one is played
                i += 1
                pcm = base64.b64decode(r["audio_b64"])
                rate = int(r["sample_rate"])
                self._fit_resampler(rate)

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
        tail = out.pop()                              # (was out[-2] = ... out.pop(): with 2 pieces the pop ran
        out[-1] = f"{out[-1]} {tail}"                 # first, out[-2] no longer existed, the sentence was lost)
    return out


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
