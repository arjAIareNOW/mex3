"""A simulated room for testing without a microphone or speaker (used on the build server and in tests).

The 'room' plays FDA's output back into the 'microphone' the way a real speaker + mic would: delayed, filtered,
with reverb and noise, and mixes in scripted user speech. Runs in real time, so server latencies are real.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .audio import AudioEngine


class SimAudioEngine(AudioEngine):
    def __init__(self, cfg, on_audio, echo_gain: float = 0.5, echo_delay_ms: float = 25.0, noise_db: float = -60.0,
                 seed: int = 0, drive: float = 0.0):
        """drive > 0: overdriven small speaker (tanh distortion), the echo the AEC's linear filter cannot remove."""
        self.drive = drive
        cfg._simulated = True
        super().__init__(cfg, on_audio)
        self.rng = np.random.default_rng(seed)
        rate = self.rate
        # room impulse response: direct path + a few decaying reflections
        n = int(rate * 0.12)
        ir = np.zeros(n, np.float32)
        d0 = int(rate * echo_delay_ms / 1000)
        ir[d0] = 1.0
        for k in range(1, 25):
            i = d0 + int(self.rng.uniform(0.002, 0.11) * rate)
            if i < n:
                ir[i] += self.rng.uniform(-0.35, 0.35) * np.exp(-k / 8)
        self.ir = ir * echo_gain
        self.hist = np.zeros(n, np.float32)
        self.noise = 10 ** (noise_db / 20)
        self.user: list[tuple[int, np.ndarray]] = []     # (start sample, audio at device rate)
        self.sample = 0
        self.played_out: list[np.ndarray] = []
        self.lock = threading.Lock()

    def say_at(self, t: float, audio: np.ndarray):
        """Schedule user speech (device-rate float32) at t seconds from now."""
        with self.lock:
            self.user.append((self.sample + int(t * self.rate), audio.astype(np.float32)))

    def now_s(self) -> float:
        return self.sample / self.rate

    def start(self):
        super().start()
        self._sim = threading.Thread(target=self._run, name="sim-device", daemon=True)
        self._sim.start()

    def _run(self):
        rate, n = self.rate, int(self.rate * self.cfg.block_ms / 1000)
        out_prev = np.zeros(n, np.float32)
        t_next = time.perf_counter()
        while not self._stop.is_set():
            # what the mic hears = echo of what was played + user speech + noise
            x = np.concatenate([self.hist, np.tanh(self.drive * out_prev) / np.tanh(self.drive) if self.drive else out_prev])
            echo = _fir(x, self.ir, n)
            self.hist = x[-len(self.ir):]
            mic = echo + self.rng.normal(0, self.noise, n).astype(np.float32)
            with self.lock:
                for st, a in self.user:
                    lo, hi = max(st, self.sample), min(st + len(a), self.sample + n)
                    if lo < hi:
                        mic[lo - self.sample:hi - self.sample] += a[lo - st:hi - st]
                self.user = [(st, a) for st, a in self.user if st + len(a) > self.sample]
            out_prev = self.callback(mic.astype(np.float32), n)
            self.sample += n
            t_next += n / rate
            d = t_next - time.perf_counter()
            if d > 0:
                time.sleep(d)


def _fir(x: np.ndarray, ir: np.ndarray, n: int) -> np.ndarray:
    """Last n samples of x convolved with ir (x already holds len(ir) samples of history)."""
    full = np.convolve(x, ir)[: len(x)]
    return full[-n:].astype(np.float32)
