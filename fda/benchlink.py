"""Benchmark mode: FDA talks to fd-benchmark (fdbench) over its direct-feed WebSocket instead of a mic and speaker.

    python -m fda bench                               # fdbench on this machine (ws://127.0.0.1:8791)
    python -m fda bench --url ws://LAPTOP-IP:8791     # FDA on the Pi, fdbench on the laptop

Protocol (fd-benchmark/fdbench/rig/direct.py): fdbench sends 16 kHz mono int16 PCM in real time (FDA's "mic");
FDA sends back 16 kHz mono int16 PCM (FDA's "speaker"), which fdbench plays out in real time.

The clock is fdbench's: every chunk it sends is one tick of FDA's sound card. For each chunk FDA renders exactly the
same number of samples from its mixer, so ducking, fades and barge-in behave exactly as on a real speaker (FDA's
voice is real-time paced; nothing is queued ahead that would need a "clear"). Silent blocks are not sent, so
fdbench's playout buffer never builds up delay.

No Pipecat, no API key: with the default offline backend everything runs on this machine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np

from .audio import AudioEngine

log = logging.getLogger("fda.bench")
BENCH_SR = 16000


class BenchAudioEngine(AudioEngine):
    """A sound card driven by fdbench: feed() takes one chunk of 'mic' audio and returns what FDA plays meanwhile."""

    def __init__(self, cfg, on_audio):
        cfg._simulated = True
        super().__init__(cfg, on_audio)
        self.block = int(self.rate * cfg.block_ms / 1000)
        self.pending = np.zeros(0, np.float32)
        self.gate = lambda: True          # False while FDA is still starting: input is dropped, nothing is played
        self.blocks = 0

    def feed(self, x: np.ndarray) -> np.ndarray | None:
        """x: float32 at 16 kHz. Returns FDA's output for the same span, or None if it is silent."""
        if not self.gate():
            return None
        buf = np.concatenate([self.pending, x])
        n, outs = self.block, []
        k = len(buf) // n
        for i in range(k):
            outs.append(self.callback(buf[i * n:(i + 1) * n], n))
        self.pending = buf[k * n:]
        self.blocks += k
        if not outs:
            return None
        y = np.concatenate(outs)
        return y if np.abs(y).max(initial=0.0) > 1e-4 else None


def bench_config(cfg):
    """Settings for benchmark mode: no devices, no browser, no echo canceller (there is no echo in direct feed)."""
    cfg.device_rate = BENCH_SR
    cfg.aec = False                 # fdbench's input never contains FDA's own voice
    cfg.noise_suppression = False
    cfg.aec_delay_ms = 0
    cfg.open_browser = False
    cfg.say_ready = False           # do not say "FDA is ready" into the first trial
    cfg.record_dir = ""
    return cfg


def _with_rate(url: str) -> str:
    u = urlparse(url)
    return urlunparse(u._replace(query=urlencode({"out_sr": BENCH_SR})))


def _i16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def wait_for_fdbench(url: str, quiet: bool = False):
    u = urlparse(url)
    host, port, said = u.hostname or "127.0.0.1", u.port or 8791, False
    while True:
        try:
            _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), 2)
            w.close()
            return
        except (OSError, asyncio.TimeoutError):
            pass
        if not said and not quiet:
            print(f"waiting for fdbench at {url} … (start it on that machine; Ctrl+C to quit)", flush=True)
            said = True
        await asyncio.sleep(1.0)


async def run_once(cfg, url: str, log_dir: Path | None = None) -> dict:
    """Start FDA, warm it up, connect to fdbench, serve until fdbench closes the socket. Returns a small summary."""
    import aiohttp
    from .agent import Agent

    bench_config(cfg)
    engine_box: list[BenchAudioEngine] = []

    def factory(c, cb):
        e = BenchAudioEngine(c, cb)
        engine_box.append(e)
        return e

    agent = Agent(cfg, ui=None, audio_factory=factory)
    eng = engine_box[0]
    eng.gate = lambda: agent.status != "starting"
    t0 = time.perf_counter()
    await agent.servers.warm_up()           # load models before connecting: fdbench starts testing 5 s after connect
    print(f"FDA models ready in {time.perf_counter() - t0:.1f} s ({agent.servers.kind} backend)", flush=True)

    stop = asyncio.Event()
    run_task = asyncio.create_task(agent.run(stop))
    while agent.status == "starting" and not run_task.done():
        await asyncio.sleep(0.05)
    if run_task.done():
        await run_task                      # raises the startup error
    sent = got = 0
    try:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(_with_rate(url), max_msg_size=0, autoping=True, heartbeat=None,
                                       compress=0) as ws:
                print(f"connected to fdbench at {url} · benchmark mode", flush=True)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        got += 1
                        x = np.frombuffer(msg.data, dtype="<i2").astype(np.float32) / 32768.0
                        y = eng.feed(x)
                        if y is not None:
                            await ws.send_bytes(_i16(y))
                            sent += 1
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
        print("fdbench closed the connection: run over, shutting FDA down", flush=True)
    finally:
        stop.set()
        try:
            await asyncio.wait_for(run_task, 15)
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
            if not isinstance(e, asyncio.TimeoutError):
                log.warning("agent stopped with %s", e)
            run_task.cancel()
        if log_dir is not None:
            _save_events(agent, Path(log_dir))
    return {"chunks_in": got, "chunks_out": sent, "dropped": eng.dropped}


def _save_events(agent, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"fda-events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in agent.events:
            e = {k: v for k, v in e.items() if k != "audio"}
            f.write(json.dumps(e, default=str, ensure_ascii=False) + "\n")
    print(f"FDA events: {path}", flush=True)


async def serve(cfg, url: str, once: bool = False, log_dir: Path | None = None):
    """Like fd-benchmark's launch_assistant.py: wait for fdbench, serve a run, start fresh for the next run."""
    from .config import Config
    runs = 0
    while True:
        await wait_for_fdbench(url)
        runs += 1
        print(f"--- run {runs}: starting FDA", flush=True)
        fresh = Config.load(getattr(cfg, "_path", None))
        fresh.backend = cfg.backend
        summary = await run_once(fresh, url, log_dir)
        print(f"--- run {runs} done: {summary}", flush=True)
        if once:
            return
        print("--- waiting for the next run (Ctrl+C to quit)", flush=True)
        await asyncio.sleep(2.0)
