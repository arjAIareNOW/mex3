"""A Pipecat transport on FDA's full-duplex audio engine (mic + speaker open at the same time).

Pipecat's LocalAudioTransport has no echo canceller: with the speaker next to the mic the bot hears itself and
interrupts itself. This transport keeps FDA's audio path:

  speaker <- FDA mixer (voice | music | fx, smooth gain ramps) --+--> reference
  mic -----------------------------------------------------------+--> WebRTC AEC3 --> own-voice gate --> Pipecat

* AEC3 removes the bot's voice (and music/alarms) from the mic using exactly what was played as reference.
* The own-voice gate (fda/selfvoice.py) silences residual echo the AEC leaves, so Silero VAD never fires on the bot.
* Output is paced in real time into the mixer; an interruption fades the voice out in 80 ms (no click) and drops
  the rest; duck() lowers the voice while the bot decides whether the user is really taking the turn.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import soxr
from loguru import logger

from pipecat.frames.frames import (Frame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame, StartFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from ..audio import AudioEngine
from ..selfvoice import OwnVoiceDetector


class _Input(BaseInputTransport):
    def __init__(self, owner: "FdaAudioTransport", params: TransportParams):
        super().__init__(params)
        self.owner = owner
        self.loop: asyncio.AbstractEventLoop | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self.loop = asyncio.get_running_loop()
        self.owner._start_engine()
        await self.set_transport_ready(frame)

    async def stop(self, frame):
        await super().stop(frame)
        self.owner._stop_engine()

    async def cancel(self, frame):
        await super().cancel(frame)
        self.owner._stop_engine()

    def deliver(self, pcm: bytes):     # called from the audio worker thread
        if self.loop is not None and not self.loop.is_closed():
            f = InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(f), self.loop)


class _Output(BaseOutputTransport):
    def __init__(self, owner: "FdaAudioTransport", params: TransportParams):
        super().__init__(params)
        self.owner = owner
        self._rs: dict[int, soxr.ResampleStream] = {}
        self._t_level = 0.0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            self.owner.stop_voice()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        eng = self.owner.engine
        if eng is None:
            return False
        x = np.frombuffer(frame.audio, "<i2").astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            x = x.reshape(-1, frame.num_channels).mean(1)
        if frame.sample_rate != eng.rate:
            rs = self._rs.get(frame.sample_rate)
            if rs is None:
                rs = self._rs[frame.sample_rate] = soxr.ResampleStream(frame.sample_rate, eng.rate, 1,
                                                                       dtype="float32")
            x = rs.resample_chunk(x)
        voice = eng.mixer.voice
        voice.push(x)
        cb = self.owner.on_level
        if cb is not None and len(x):
            now = time.monotonic()
            if now - self._t_level >= 0.05:                  # ~20 updates a second for the speaking pulse
                self._t_level = now
                cb(float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)))
        # pace like a sound card: keep at most ~60 ms queued, so an interruption has little to throw away
        limit = int(eng.rate * 0.06)
        while voice.queued > limit and self.owner.engine is not None:
            await asyncio.sleep(0.005)
        return True


class FdaAudioTransport(BaseTransport):
    """Mic + speaker through FDA's AudioEngine (or the simulated room with sim=True)."""

    def __init__(self, cfg, params: TransportParams, sim: bool = False):
        super().__init__()
        self.cfg, self.params, self.sim = cfg, params, sim
        self.engine: AudioEngine | None = None
        self.own = OwnVoiceDetector(cfg.selfvoice_margin_db) if cfg.selfvoice else None
        self.bot_speaking = False
        self.user_speaking = False
        self.on_level = None             # callback(dB) with the voice level while the bot speaks (the display pulse)
        self.own_blocks = 0
        self.level_db = -120.0
        self._input = _Input(self, params)
        self._output = _Output(self, params)
        if sim:
            from ..sim import SimAudioEngine
            self.engine = SimAudioEngine(cfg, self._on_audio)
        else:
            self.engine = AudioEngine(cfg, self._on_audio)
        self.mixer = self.engine.mixer
        self._started = False

    def input(self):
        return self._input

    def output(self):
        return self._output

    # ---- engine ----
    def _start_engine(self):
        if not self._started:
            self._started = True
            self.engine.start()
            logger.info(f"FDA audio engine started (AEC {'on' if self.cfg.aec else 'off'}, own-voice gate "
                        f"{'on' if self.own else 'off'})")

    def _stop_engine(self):
        if self._started:
            self._started = False
            self.engine.stop()

    def _on_audio(self, clean: np.ndarray, ref: np.ndarray):
        """Worker thread: echo-cancelled 16 kHz mic + what was played. Residual echo is silenced here."""
        if self.own is not None:
            self.own.push(clean, ref, learn=self.bot_speaking and not self.user_speaking)
            if self.own.is_own() and self.own.ref_active():
                self.own_blocks += 1
                clean = clean * 0.0
        self.level_db = float(20 * np.log10(np.sqrt(np.mean(clean ** 2)) + 1e-9))
        self._input.deliver((np.clip(clean, -1, 1) * 32767).astype("<i2").tobytes())

    # ---- voice control (used by the duplex guard) ----
    def duck(self, on: bool):
        v = self.mixer.voice
        v.ramp(self.cfg.bargein_duck_gain if on else 1.0, self.cfg.bargein_duck_ms if on else 150.0)

    def stop_voice(self):
        self.mixer.voice.fade_out_and_clear(self.cfg.bargein_fade_ms)


class NullMixerTransport:
    """For benchmark mode (fdbench's own Pipecat transport): FDA's room still needs a mixer for music/alarms,
    which nobody hears there."""

    def __init__(self, cfg):
        from ..audio import Mixer
        self.mixer = Mixer(16000)
        self.bot_speaking = self.user_speaking = False
        self.own_blocks = 0

    def duck(self, on: bool):
        pass

    def stop_voice(self):
        pass
