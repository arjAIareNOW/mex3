"""Silero VAD v5 (ONNX, CPU): speech probability for every 32 ms of 16 kHz audio."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .config import PKG

WINDOW = 512      # 32 ms
CONTEXT = 64


class SileroVAD:
    def __init__(self, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.s = ort.InferenceSession(str(PKG / "models" / "silero_vad.onnx"), so, providers=["CPUExecutionProvider"])
        self.sr = np.array(16000, np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)
        self.pending = np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Feed any amount of audio; returns [(512-sample window, speech probability), ...]."""
        self.pending = np.concatenate([self.pending, x.astype(np.float32)])
        out = []
        while len(self.pending) >= WINDOW:
            w, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            inp = np.concatenate([self.context, w])[None]
            p, self.state = self.s.run(None, {"input": inp, "state": self.state, "sr": self.sr})
            self.context = w[-CONTEXT:]
            out.append((w, float(p[0, 0])))
        return out
