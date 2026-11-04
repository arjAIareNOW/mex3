"""Your MLEx-2-VCM intent Conformer (19 intents, 644 k parameters, ~20 ms on the Pi) as an instant second opinion.
Front end copied from MLEx-2-VCM- deployment/vcm_engine/features.py (MIT)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .config import PKG

INTENTS = ["ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS",
           "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN",
           "VOLUME_UP", "WEATHER"]
SR, N_FFT, WIN, HOP = 16000, 512, 400, 160
N_MELS, F_MIN, F_MAX, MAX_SAMPLES = 64, 20.0, 8000.0, 80000


def _mel_bank():
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    points = mel2hz(np.linspace(hz2mel(F_MIN), hz2mel(F_MAX), N_MELS + 2))
    diff = np.diff(points)
    slopes = points[None, :] - freqs[:, None]
    return np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))


_BANK = _mel_bank()
_WINDOW = np.zeros(N_FFT)
_WINDOW[(N_FFT - WIN) // 2:(N_FFT - WIN) // 2 + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)


def command_features(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)[:MAX_SAMPLES]
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _WINDOW, n=N_FFT)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _BANK, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    db = (db - db.mean()) / max(db.std(ddof=1), 1e-5)
    return db.T[None, None].astype(np.float32)


class VCMIntent:
    def __init__(self, threads: int = 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.s = ort.InferenceSession(str(PKG / "models" / "vcm_intent.onnx"), so, providers=["CPUExecutionProvider"])

    def predict(self, audio16k: np.ndarray) -> tuple[str, float]:
        f = command_features(audio16k)
        z = self.s.run(None, {"log_mel": f, "frame_lengths": np.array([f.shape[-1]], np.int64)})[0][0]
        p = np.exp(z - z.max()); p /= p.sum()
        k = int(p.argmax())
        return INTENTS[k], float(p[k])
