"""python -m fda            run FDA (always listening, display on http://localhost:8765)
python -m fda devices    list audio devices (put a name in fda.toml: input_device / output_device)
python -m fda check      check both DGX3 servers, the API key and the voice (offline: speed of each local model)
python -m fda download   download the offline models (hearing, planner, voice) into offline_models/
python -m fda soundtest  beep + FDA's voice on the speaker, then the same through FDA's duplex audio path
python -m fda bench      benchmark mode: connect to fd-benchmark (fdbench) instead of the mic and speaker
python -m fda judge      a free local LLM judge for fd-benchmark's scores (no API key)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import Config


def main():
    ap = argparse.ArgumentParser(prog="fda", description="FDA: full-duplex voice agent")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "devices", "check", "soundtest", "download", "bench", "judge"])
    ap.add_argument("--url", default=None, help="bench: fdbench address (default $FDBENCH_URL or ws://127.0.0.1:8791)")
    ap.add_argument("--once", action="store_true", help="bench: exit after one run")
    ap.add_argument("--log-dir", default=None, help="bench: where FDA's event log goes")
    ap.add_argument("--model", default="qwen2.5-3b", help="judge: which downloaded planner model judges")
    ap.add_argument("--port", type=int, default=8092, help="judge: port")
    ap.add_argument("--config", help="path to fda.toml")
    ap.add_argument("--record", metavar="DIR", help="save mic / reference / echo-cancelled audio to DIR")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-aec", action="store_true", help="turn off FDA's echo canceller (if the OS already does it)")
    ap.add_argument("--sim", action="store_true", help="no hardware: simulated room (FDA hears its own echo)")
    ap.add_argument("--offline", action="store_true", help="use the models on this machine (no DGX3, no key)")
    ap.add_argument("--server", action="store_true", help="always use the DGX3 servers")
    args = ap.parse_args()

    if args.command == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        print("\nhost APIs:", [h["name"] for h in sd.query_hostapis()])
        return

    cfg = Config.load(args.config)
    if args.record:
        cfg.record_dir = args.record
    if args.no_browser:
        cfg.open_browser = False
    if args.no_aec:
        cfg.aec = False
    if args.offline:
        cfg.backend = "offline"
    elif args.server:
        cfg.backend = "server"
        if not cfg.api_key:
            raise SystemExit("QWEN_API_KEY is not set (put it in .env next to fda.toml)")
    if args.command == "download":
        from .local import download
        download(cfg)
        return
    if args.command == "judge":
        from .local import serve_judge
        serve_judge(args.model, args.port)
        return
    cfg._path = args.config
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "aiohttp.access", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "check":
        sys.exit(asyncio.run(_check(cfg)))
    if args.command == "soundtest":
        sys.exit(asyncio.run(_soundtest(cfg)))
    if args.command == "bench":
        import os
        from pathlib import Path
        from .benchlink import serve
        if not (args.offline or args.server):
            cfg.backend = "offline"     # benchmark default: everything on this machine, zero tokens
        url = args.url or os.environ.get("FDBENCH_URL") or "ws://127.0.0.1:8791"
        try:
            asyncio.run(serve(cfg, url, once=args.once, log_dir=Path(args.log_dir) if args.log_dir else None))
        except KeyboardInterrupt:
            print("stopped")
        return
    asyncio.run(_run(cfg, sim=args.sim))


async def _check(cfg) -> int:
    import time
    from .servers import make_servers
    s = make_servers(cfg)
    if s.kind == "offline":
        from .local import bench, missing
        gone = missing(cfg)
        if gone:
            print("offline models missing:", ", ".join(gone), "\nrun:  python -m fda download")
            return 1
        try:
            await bench(cfg, s)
        finally:
            await s.close()
        print("hearing:", "OK" if s.ok["asr"] else "FAIL", " planner:", "OK" if s.ok["llm"] else "FAIL",
              " voice:", "OK" if s.ok["tts"] else "FAIL")
        return 0 if all(s.ok.values()) else 1
    ok = await s.check()
    print(f"ASR  {cfg.asr_url}: {'OK' if ok['asr'] else 'FAIL'}")
    print(f"TTS  {cfg.tts_url}: {'OK' if ok['tts'] else 'FAIL'}")
    if all(ok.values()):
        t = time.perf_counter(); first = None
        async for _ in s.speak_stream("Hello, I'm ready."):
            first = first or time.perf_counter() - t
        print(f"TTS first audio: {first * 1000:.0f} ms (voice {cfg.voice})")
    await s.close()
    return 0 if all(ok.values()) else 1


async def _soundtest(cfg) -> int:
    import time
    import numpy as np
    import sounddevice as sd
    from .audio import AudioEngine, _find_device, resample
    from .servers import make_servers
    out = _find_device(cfg.output_device, "output"); inp = _find_device(cfg.input_device, "input")
    print("host API      :", sd.query_hostapis(sd.default.hostapi)["name"])
    print("output device :", sd.query_devices(out, "output")["name"])
    print("input device  :", sd.query_devices(inp, "input")["name"])
    print("\n1) beep (plain playback)..."); rate = 48000
    sd.play(0.3 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32), rate, device=out); sd.wait()
    s = make_servers(cfg)
    await s.warm_up()
    voice = np.concatenate([c async for c in s.speak_stream("This is FDA. Can you hear me?")])
    await s.close()
    print(f"2) FDA's voice (plain playback, {len(voice) / cfg.tts_rate:.1f} s, peak {np.abs(voice).max():.2f})...")
    sd.play(resample(voice, cfg.tts_rate, rate), rate, device=out); sd.wait()
    print("3) FDA's voice through the full-duplex path FDA really uses (mic + speaker + echo canceller)...")
    levels = []
    eng = AudioEngine(cfg, lambda x, ref: levels.append(float(np.abs(x).max())))
    eng.start()
    eng.mixer.voice.push(resample(voice, cfg.tts_rate, eng.rate))
    time.sleep(len(voice) / cfg.tts_rate + 0.8)
    eng.stop()
    print(f"   duplex stream: rate {eng.rate}, mic blocks {len(levels)}, dropped {eng.dropped}, "
          f"loudest mic after echo cancel {max(levels or [0]):.3f}")
    print("\nWhich of 1, 2, 3 did you hear?")
    return 0


async def _run(cfg, sim: bool = False):
    from .agent import Agent
    from .ui import UI
    ui = UI(cfg)
    factory = None
    if sim:
        from .sim import SimAudioEngine
        factory = lambda c, cb: SimAudioEngine(c, cb)
    agent = Agent(cfg, ui=ui, audio_factory=factory)
    ui.agent = agent
    await ui.start()
    try:
        await agent.run()
    finally:
        await ui.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
