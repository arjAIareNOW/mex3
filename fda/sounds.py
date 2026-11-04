"""Generated sounds (no files needed): alarm, timer chime, phone ringback, and a calm built-in music loop."""
from __future__ import annotations

import numpy as np


def _t(sec: float, rate: int) -> np.ndarray:
    return np.arange(int(sec * rate)) / rate


def _env(n: int, rate: int, a=0.01, r=0.08) -> np.ndarray:
    e = np.ones(n, np.float32)
    na, nr = int(a * rate), int(r * rate)
    e[:na] = np.linspace(0, 1, na); e[-nr:] = np.linspace(1, 0, nr)
    return e


def alarm(rate: int) -> np.ndarray:
    """Two-tone beep-beep, 1.2 s loop."""
    beep = lambda f: (np.sin(2 * np.pi * f * _t(0.15, rate)) * _env(int(0.15 * rate), rate, 0.005, 0.03))
    gap = np.zeros(int(0.08 * rate))
    x = np.concatenate([beep(880), gap, beep(1175), gap, beep(880), gap, beep(1175), np.zeros(int(0.46 * rate))])
    return (0.35 * x).astype(np.float32)


def chime(rate: int) -> np.ndarray:
    """Soft three-note chime for reminders and timers (one shot)."""
    out = []
    for f in (659.25, 783.99, 1046.5):
        t = _t(0.35, rate)
        out.append(np.sin(2 * np.pi * f * t) * np.exp(-t * 6))
    return (0.3 * np.concatenate(out + [np.zeros(int(0.6 * rate))])).astype(np.float32)


def ringback(rate: int) -> np.ndarray:
    """Phone ringing (440 + 480 Hz, 2 s on, 4 s off)."""
    t = _t(2.0, rate)
    on = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * 0.12 * _env(len(t), rate, 0.02, 0.05)
    return np.concatenate([on, np.zeros(int(4 * rate))]).astype(np.float32)


def music_loop(rate: int, seed: int = 0) -> np.ndarray:
    """A calm ~24 s lo-fi loop: Cmaj7-Am7-Fmaj7-G7 pads + arpeggio + soft bass."""
    rng = np.random.default_rng(seed)
    bpm = 80; beat = 60 / bpm; bar = 4 * beat
    chords = [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 65]]
    roots = [36, 33, 29, 31]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    total = int(2 * len(chords) * bar * rate)
    y = np.zeros(total)
    for rep in range(2):
        for ci, ch in enumerate(chords):
            start = int((rep * len(chords) + ci) * bar * rate)
            t = _t(bar, rate); n = len(t)
            pad = sum(np.sin(2 * np.pi * hz(m) * t + rng.uniform(0, 6)) for m in ch) / len(ch)
            y[start:start + n] += 0.16 * pad * _env(n, rate, 0.4, 0.5)
            b = np.sin(2 * np.pi * hz(roots[ci]) * t) * np.exp(-t * 1.2)
            y[start:start + n] += 0.22 * b
            for k in range(8):   # eighth-note arpeggio
                m = ch[[0, 1, 2, 3, 2, 1, 2, 3][k]] + 12
                s = start + int(k * beat / 2 * rate); tt = _t(beat / 2, rate)
                y[s:s + len(tt)] += 0.10 * np.sin(2 * np.pi * hz(m) * tt) * np.exp(-tt * 5)
    return (y / max(np.abs(y).max(), 1e-6) * 0.5).astype(np.float32)
