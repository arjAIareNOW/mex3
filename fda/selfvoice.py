"""FDA knows its own voice: what the speaker plays is never taken as the user.

The echo canceller (AEC3) removes most of FDA's voice from the mic, but with the speaker next to the mic some always
leaks through (distortion, a moving echo path). Two checks use what FDA itself knows about its own output:

1. OwnVoiceDetector  (audio, every 16 ms, before any transcript)
   FDA has the exact reference of everything it plays (voice, music, alarms, at the current volume, ducked or not).
   It learns, per frequency band, how much of that reference still leaks through the AEC, and so predicts the leak
   at every moment: large on a loud syllable, nothing in a pause. Mic sound that this prediction explains is FDA
   itself: it does not duck FDA, does not count as the user talking, and does not start a turn. A person adds energy
   the prediction cannot explain.
   Simulated near-field room (tests): echo wrongly taken as a person 1.7-2.1 % of frames, versus 15-29 % for the old
   single "residual floor + 18 dB" rule.

2. strip_own_words  (text, per turn)
   The speaker logs when every phrase played. The words FDA was playing while the user spoke are removed from the
   transcript: "once upon a time stop what time is it" -> "stop what time is it"; pure echo -> "".

Tried and left out: a speaker-embedding voice print of FDA's voice. After the AEC the leaked echo no longer sounds
like FDA to the embedding (cosine 0.19-0.28, the same range as other people), so it could not tell them apart.
"""
from __future__ import annotations

import collections
import difflib
import re

import numpy as np

SR = 16000
HOP = 256                 # 16 ms frames
NFFT = 512
MAX_LAG = 30              # frames of delay + reverb covered (~480 ms)


def _bands(n_bands: int = 24, lo: float = 150.0, hi: float = 4000.0) -> np.ndarray:
    """Mel-spaced triangular filterbank (n_bands x NFFT/2+1)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(lo), mel(hi), n_bands + 2))
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    fb = np.zeros((n_bands, len(freqs)), np.float32)
    for i in range(n_bands):
        a, b, c = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - a) / (b - a), (c - freqs) / (c - b)), 0, None)
    return fb


class OwnVoiceDetector:
    """push(clean, ref, learn) with every block of echo-cancelled mic and the reference (same length, 16 kHz).
    is_own() -> True when the last 32 ms of mic sound is explained by what FDA is playing."""

    def __init__(self, margin_db: float = 2.0, k_spread: float = 2.0, ref_min_db: float = -55.0,
                 mic_min_db: float = -70.0, memory_frames: int = 600):
        self.fb = _bands()
        self.win = np.hanning(NFFT).astype(np.float32)
        self.margin, self.k, self.ref_min, self.mic_min = margin_db, k_spread, ref_min_db, mic_min_db
        self.buf_m = np.zeros(NFFT - HOP, np.float32)
        self.buf_r = np.zeros(NFFT - HOP, np.float32)
        self.ref_rows: collections.deque = collections.deque(maxlen=MAX_LAG + 1)   # band power (linear)
        self.ref_db: collections.deque = collections.deque(maxlen=MAX_LAG + 1)
        # leak per band (dB of echo-cancelled mic over the reference) over the last ~10 s of FDA playing. The median
        # and its spread are robust: the user talking now and then (well under half the time) cannot move them.
        self.mem: collections.deque = collections.deque(maxlen=memory_frames)
        self.bound: np.ndarray | None = None
        self._n_new = 0
        self.frames: collections.deque = collections.deque(maxlen=2)   # last two frames: (own?, excess dB)
        self.excess = 0.0

    def _rows(self, buf: np.ndarray, x: np.ndarray):
        buf = np.concatenate([buf, x])
        out = []
        while len(buf) >= NFFT:
            f = buf[:NFFT]
            buf = buf[HOP:]
            p = np.abs(np.fft.rfft(f * self.win)) ** 2
            out.append((self.fb @ p + 1e-12, 10 * np.log10(np.mean(f * f) + 1e-12)))
        return buf, out

    def push(self, clean: np.ndarray, ref: np.ndarray, learn: bool = False):
        self.buf_m, mrows = self._rows(self.buf_m, clean)
        self.buf_r, rrows = self._rows(self.buf_r, ref)
        for (mb, mdb), (rb, rdb) in zip(mrows, rrows):
            self.ref_rows.append(rb)
            self.ref_db.append(rdb)
            self._frame(mb, mdb, learn)

    def ref_active(self) -> bool:
        return bool(self.ref_db) and max(self.ref_db) >= self.ref_min

    def _frame(self, mb: np.ndarray, mdb: float, learn: bool):
        if not self.ref_active():
            self.frames.append((False, 99.0))       # FDA silent: whatever is heard is not FDA
            self.excess = 99.0
            return
        if mdb < self.mic_min:
            self.frames.append((True, 0.0))         # silence while FDA plays
            self.excess = 0.0
            return
        E = np.max(np.array(self.ref_rows), axis=0)            # loudest reference over delay + reverb
        obs = 10 * np.log10(mb) - 10 * np.log10(E)             # mic over reference, per band
        # learn only sound that looks like echo: a person talking over FDA before the VAD has noticed them would
        # otherwise be learned as "leak", raise the bound, and from then on be silenced as FDA's own voice
        if learn and self.bound is not None and float(np.clip(obs - self.bound, 0, None).mean()) > self.margin + 6.0:
            learn = False
        if learn:
            self.mem.append(obs)
            self._n_new += 1
            if self.bound is None and len(self.mem) >= 30 or self._n_new >= 16:
                self._n_new = 0
                M = np.array(self.mem)
                med = np.median(M, axis=0)
                spread = 1.4826 * np.median(np.abs(M - med), axis=0)       # robust standard deviation
                self.bound = med + self.k * np.maximum(spread, 1.0)
        if self.bound is None:                                 # nothing learned yet: no opinion
            self.frames.append((False, 99.0))
            self.excess = 99.0
            return
        ex = float(np.clip(obs - self.bound, 0, None).mean())
        self.excess = ex
        self.frames.append((ex < self.margin, ex))

    def is_own(self) -> bool:
        return len(self.frames) > 0 and all(o for o, _ in self.frames)

    @property
    def leak_db(self) -> float:
        """Learned leak bound through the AEC (dB, averaged over bands): how well echo cancellation works here."""
        return float(np.mean(self.bound)) if self.bound is not None else 0.0


# ------------------------------------------------------------------ own words out of the transcript
def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", w.lower())


def strip_own_words(heard: str, played: str) -> tuple[str, int]:
    """Remove from `heard` the runs of words FDA was playing at the time (`played`). -> (what is left, words removed).

    Runs of 2+ matching words in order are FDA's own (a person rarely repeats two of FDA's words in order at the same
    moment). A single matching word is removed only if it is all that was heard."""
    toks = heard.split()
    hw = [_norm(t) for t in toks]
    pw = [w for w in (_norm(t) for t in played.split()) if w]
    if not toks or not pw:
        return heard.strip(), 0
    drop = [False] * len(toks)
    sm = difflib.SequenceMatcher(None, hw, pw, autojunk=False)
    single = sum(1 for w in hw if w) == 1
    for b in sm.get_matching_blocks():
        if b.size >= 2 or (b.size == 1 and single):
            for i in range(b.a, b.a + b.size):
                drop[i] = True
    kept = [t for t, d, w in zip(toks, drop, hw) if not d and w]
    return " ".join(kept).strip(" ,.;:-"), sum(drop)
