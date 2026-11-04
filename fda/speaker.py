"""FDA's voice: streaming text -> phrases -> Qwen3-TTS (streamed, prefetched) -> mixer voice channel.

Knows exactly which words have reached the speaker (sample-accurate per phrase, proportional inside a phrase), so on
barge-in the conversation history records what the person actually heard, and the echo filter knows what FDA just said.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

import numpy as np
import soxr

log = logging.getLogger("fda.speaker")
_SENT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"')])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


class Segment:
    def __init__(self, text: str):
        self.text = text
        self.start = None        # sample index in the voice channel where it starts
        self.n = 0               # samples pushed so far
        self.done = False
        self.wall_start = 0.0    # estimated wall-clock time it starts / ends playing
        self.wall_end = 0.0


class Speaker:
    def __init__(self, cfg, servers, mixer, on_text=None):
        self.cfg, self.servers, self.mixer, self.ch = cfg, servers, mixer, mixer.voice
        self.on_text = on_text or (lambda *a: None)    # (text, final) for the display
        self.segments: list[Segment] = []
        self.pending = ""
        self.fed = 0
        self.final = False
        self.queue: asyncio.Queue | None = None
        self.task: asyncio.Task | None = None
        self.fetches: list[asyncio.Task] = []
        self.sem = asyncio.Semaphore(2)
        self.pushed = 0
        self.log: list[tuple[float, float, str]] = []   # (wall start, wall end, text) of everything spoken
        self.first_audio_at: float | None = None
        self.ducked = False

    # ---------- state ----------
    @property
    def active(self) -> bool:
        producing = self.task is not None and not self.task.done()
        return producing or self.ch.busy

    def played_text(self) -> str:
        played = self.ch.played
        words = []
        for s in self.segments:
            if s.start is None or played <= s.start:
                break
            if s.done and played >= s.start + s.n:
                words += s.text.split()
            else:
                w = s.text.split()
                frac = (played - s.start) / max(s.n if s.done else max(s.n, 1), 1)
                words += w[:int(len(w) * min(frac, 1.0))]
                break
        return " ".join(words)

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments) + (" " + self.pending if self.pending else "")

    def recent_spoken(self, since: float) -> str:
        now = time.time()
        spans = self.log + [(s.wall_start, s.wall_start + s.n / self.mixer.rate, s.text) for s in self.segments
                            if not s.done and s.start is not None and s.wall_start]
        return " ".join(t for (a, b, t) in spans if b >= since and a <= now + 0.5)

    def words_between(self, t0: float, t1: float, before: float = 0.3, after: float = 0.6) -> str:
        """The words FDA was playing between wall times t0 and t1 (with margins for latency and reverb). Word times are
        estimated from each phrase's start/end by character position; phrases still being generated count too."""
        spans = self.log + [(s.wall_start, s.wall_start + s.n / self.mixer.rate, s.text) for s in self.segments
                            if not s.done and s.start is not None and s.wall_start]
        out, lo, hi = [], t0 - before, t1 + after
        for a, b, text in spans:
            if b < lo or a > hi or not text:
                continue
            words, pos, n = text.split(), 0, max(len(text), 1)
            for w in words:
                t = a + (b - a) * (pos + len(w) / 2) / n
                pos += len(w) + 1
                if lo <= t <= hi:
                    out.append(w)
        return " ".join(out)

    # ---------- speaking ----------
    async def start(self):
        """Begin a new reply (stops any current one)."""
        if self.active:
            await self.interrupt(fade_ms=60)
        self.ch.clear(); self.ch.ramp(1.0, 1)
        self.segments, self.pending, self.final, self.fed = [], "", False, 0
        self.pushed, self.first_audio_at, self.ducked = 0, None, False
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self._play_loop())

    def feed(self, text_so_far: str):
        """Give the full reply text so far (it only grows); complete phrases go to TTS immediately."""
        if len(text_so_far) > self.fed:
            self.pending += text_so_far[self.fed:]
            self.fed = len(text_so_far)
        self._cut(final=False)

    def finish(self, text: str | None = None):
        if text is not None:
            self.feed(text)
        self._cut(final=True)
        self.final = True
        if self.queue:
            self.queue.put_nowait(None)

    async def say(self, text: str):
        await self.start()
        self.finish(text)

    def _cut(self, final: bool):
        while self.pending:
            parts = _SENT.split(self.pending, maxsplit=1)
            if len(parts) == 2:
                self._emit(parts[0]); self.pending = parts[1]; continue
            # long clause: cut at a comma (sooner for the first phrase, so audio starts fast)
            min_words = 3 if not self.segments else 7
            parts = _CLAUSE.split(self.pending, maxsplit=1)
            if len(parts) == 2 and len(parts[0].split()) >= min_words:
                self._emit(parts[0]); self.pending = parts[1]; continue
            if final:
                self._emit(self.pending); self.pending = ""
            break

    def _emit(self, phrase: str):
        phrase = phrase.strip()
        if not phrase or not re.search(r"\w", phrase):
            return
        seg = Segment(phrase)
        self.segments.append(seg)
        q: asyncio.Queue = asyncio.Queue()
        self.fetches.append(asyncio.create_task(self._fetch(phrase, q)))
        self.queue.put_nowait((seg, q))
        self.on_text(" ".join(s.text for s in self.segments), False)

    async def _fetch(self, phrase: str, q: asyncio.Queue):
        async with self.sem:
            try:
                async for pcm in self.servers.speak_stream(phrase):
                    q.put_nowait(pcm)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("TTS failed for %r: %s", phrase, e)
            finally:
                q.put_nowait(None)

    async def _play_loop(self):
        rate = self.mixer.rate
        while True:
            item = await self.queue.get()
            if item is None:
                break
            seg, q = item
            rs = soxr.ResampleStream(self.cfg.tts_rate, rate, 1, dtype="float32")
            first = True
            while True:
                pcm = await q.get()
                last = pcm is None
                y = rs.resample_chunk(np.zeros(0, np.float32) if last else pcm, last=last)
                if first and len(y) and self.pushed == 0:      # trim leading silence of the first phrase
                    nz = np.flatnonzero(np.abs(y) > 2e-3)
                    y = y[max(0, (nz[0] if len(nz) else len(y)) - int(0.02 * rate)):]
                if len(y):
                    if seg.start is None:
                        seg.start = self.pushed
                        ahead = max(0, self.ch.queued) / rate
                        seg.wall_start = time.time() + ahead
                    if first and self.first_audio_at is None:
                        self.first_audio_at = time.time()
                    first = False
                    self.ch.push(y); self.pushed += len(y); seg.n += len(y)
                if last:
                    break
            seg.done = True
            if seg.start is None:
                seg.start = self.pushed
            seg.wall_end = seg.wall_start + seg.n / rate
            self.log.append((seg.wall_start, seg.wall_end, seg.text))
            self.log = self.log[-40:]
        self.on_text(" ".join(s.text for s in self.segments), True)
        # stay "active" until the audio has actually played out
        while self.ch.busy and not self.ch.clear_when_silent:
            await asyncio.sleep(0.02)

    async def wait_done(self):
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    # ---------- barge-in ----------
    def duck(self):
        if not self.ducked:
            self.ducked = True
            self.ch.ramp(self.cfg.bargein_duck_gain, self.cfg.bargein_duck_ms)

    def unduck(self):
        if self.ducked:
            self.ducked = False
            self.ch.ramp(1.0, 150)

    async def interrupt(self, fade_ms: float | None = None) -> tuple[str, str]:
        """Fade out and stop. Returns (what was heard, what was not said yet)."""
        fade = self.cfg.bargein_fade_ms if fade_ms is None else fade_ms
        heard = self.played_text()
        full = self.full_text().strip()
        rest = full[len(heard):].strip() if full.startswith(heard) else ""
        self.ch.fade_out_and_clear(fade)
        for t in self.fetches:
            t.cancel()
        self.fetches = []
        if self.task and not self.task.done():
            self.task.cancel()
        # record partially played segments in the echo log
        now = time.time()
        if heard:
            self.log.append((now - 3, now, heard))
        await asyncio.sleep(fade / 1000 + 0.02)
        self.ducked = False
        return heard, rest
