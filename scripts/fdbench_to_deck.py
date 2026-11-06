"""fd-benchmark's own course results (stage-latency test, 10 phrases x 3 runs, on the Pi's clock) -> report/latency.json
for the deck (scripts/me_slides.py). The official numbers of the class harness, the same definitions for everyone.

    python scripts/fdbench_to_deck.py fd-benchmark/runs/fda-pc/2026-11-07_00-18-59 --cooling "case fan"

The report it replaces is kept as report/latency_<its run>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# results.csv key -> the deck's metric name (me_slides.metric() matches on the first word)
KEYS = {"course.e2e": "E2E latency (VAD end -> audio out)",
        "course.asr_final": "ASR final (VAD end -> transcript)",
        "course.llm_ttft": "LLM TTFT (transcript -> first token, incl. tool time)",
        "course.tts_ttfb": "TTS TTFB (text -> first audio chunk)",
        "course.barge_in": "Barge-in (speech -> playback stopped)",
        "course.endpointing": "VAD endpointing (true end of speech -> VAD end)",
        "course.tool_call": "MCP / tool call duration"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--cooling", default="")
    a = ap.parse_args()
    rows = list(csv.DictReader((a.run / "results.csv").open(encoding="utf-8")))
    by_key = {r["key"]: r for r in rows}
    by_metric = {r["metric"]: r for r in rows}

    def num(v):
        try:
            return round(float(v), 3)
        except (TypeError, ValueError):
            return None
    metrics = {}
    for key, name in KEYS.items():
        r = by_key.get(key) or next((x for m, x in by_metric.items()
                                     if m.lower().startswith(name.split(" (")[0].lower())), None)
        if r and num(r["mean"] or r["value"]) is not None:
            metrics[name] = {"n": int(r["n"] or 0), "mean": num(r["mean"] or r["value"]), "p50": num(r["median"]),
                             "p95": num(r["p95"])}
    wer_row = next((r for m, r in by_metric.items() if m.startswith("WER")), None)
    hello = json.loads((a.run / "pi_stats_hello.json").read_text(encoding="utf-8"))
    res = {r["metric"]: num(r["value"]) for r in rows if r["paper"] == "resources"}
    rep = {
        "run": a.run.name, "date": str(date.today()),
        "device": f"{hello.get('pi_model', 'Raspberry Pi')}, {round(hello.get('ram_mb', 0) / 1024)} GB RAM, "
                  f"{hello.get('os', '')}, Python {hello.get('python', '')}",
        "cooling": a.cooling, "network": ("Wi-Fi" if hello.get("network") == "wifi" else hello.get("network", "")) + " (same router as the laptop)",
        "llm_tools": "on",
        "source": "fd-benchmark course stage-latency test (10 phrases x 3 runs, the Pi's monotonic clock)",
        "method": "fd-benchmark 57eaddb: stage marks from Elkit's Pipecat pipeline; mean + p95 per stage",
        "metrics_s": metrics,
        "wer": {"wer": num(wer_row["value"]) if wer_row else None, "trials": int(wer_row["n"]) if wer_row else 0},
        "resources": {"peak_temp_c": res.get("Peak temperature (°C)"),
                      "throttled_rate": res.get("Time throttled or capped (rate)"),
                      "cpu_p95_pct": res.get("CPU p95, all cores (%)"),
                      "free_ram_min_mb": res.get("Lowest free RAM (MB)")},
    }
    out = ROOT / "report" / "latency.json"
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8")).get("run", "old")
        except ValueError:
            old = "old"
        if old != a.run.name:
            shutil.copy2(out, out.with_name(f"latency_{old}.json"))
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{out}  ({a.run.name})")
    for k, s in metrics.items():
        print(f"  {k:58} n={s['n']:>3}  mean {s['mean']:.3f}  p95 {s['p95']}")
    print(f"  WER {rep['wer']['wer']}  ·  peak {rep['resources']['peak_temp_c']} °C")


if __name__ == "__main__":
    main()
