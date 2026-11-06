"""Elkit latency + WER benchmark report (the standardized latency method of the machine exercise).

One fd-benchmark run (same 42 clips, seed 11273) + Elkit's own log of that run ->
  mean / p50 / p95 per stage, barge-in, WER  ->  <run>/latency_report.json  and  report/latency.json (for the deck)

    python scripts/latency_report.py fd-benchmark/runs/fda-pc/2026-11-06_13-28-19
    python scripts/latency_report.py <run dir> --log pi-assistant.log      # Elkit ran on the Pi: its log copied over

Clock: every stage is stamped on the machine Elkit runs on (Pipecat Mark processors -> 'turn timing (ms)' line):
  VAD end -> ASR final (asr) -> turn end (turn wait) -> LLM first token (llm) -> TTS first audio (tts);
  E2E = first audio - VAD end. Barge-in = fd-benchmark's stop latency on the interruption clips (speech start ->
  playback stopped). WER = what Elkit heard (its 'heard' lines, split per trial by the reset between trials) against
  each clip's own transcript.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TURN = re.compile(r"turn timing \(ms\): (.*?) \| total (\d+)")
HEARD = re.compile(r"heard \([^)]*\): (.*)$")
RESET = "fresh session"
DEVICE = re.compile(r"device: (.*)$")


def stats(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    p95 = xs[min(len(xs) - 1, max(0, round(0.95 * (len(xs) - 1))))] if len(xs) < 20 else \
        statistics.quantiles(xs, n=100, method="inclusive")[94]
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 3), "p50": round(statistics.median(xs), 3),
            "p95": round(p95, 3)}


FILLERS = {"um", "uh", "umm", "uhm", "mm", "hmm", "mhm", "er", "ah"}


def words(s: str) -> list[str]:
    """Standard WER normalisation: lower case, no punctuation, no [noise] tags or fillers (Whisper drops them)."""
    s = re.sub(r"\[[^\]]*\]", " ", s.lower()).replace("‑", "-").replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s.replace("-", " "))
    return [w for w in s.split() if w not in FILLERS]


def reference(meta: dict, variant: str) -> str:
    """What was said in the trial's audio (fd-benchmark manifest): the user's turn, plus what is said over Elkit
    (interruption, backchannel, side talk, background) unless it is the clean twin."""
    if "text" in meta:
        return meta["text"]
    parts = [meta.get("context_text", "")]
    if variant == "main":
        parts += [meta.get("overlap_text") or meta.get("current_turn_text", ""), meta.get("interrupt_text", "")]
    return " ".join(p for p in parts if p)


def edits(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def turn_stages(lines: list[str]) -> dict[str, list[float]]:
    out = {"asr": [], "turn wait": [], "llm": [], "tts": [], "e2e": [], "filler": [], "first sound": []}
    for ln in lines:
        m = TURN.search(ln)
        if not m:
            continue
        if "asr" not in m.group(1):                     # not a spoken turn (a tool result, a resumed answer)
            continue
        total = int(m.group(2)) / 1000
        out["e2e"].append(total)
        first = total
        for part in m.group(1).split(","):
            k, _, v = part.strip().rpartition(" ")
            if k in out and v.isdigit():
                out[k].append(int(v) / 1000)
                if k == "filler":
                    first = min(first, int(v) / 1000)
        out["first sound"].append(first)            # a filler ("On it.") or the answer, whichever came first
    return out


STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def run_window(run: Path, lines: list[str]) -> list[str]:
    """Only the log lines of this run: Elkit's own log (logs/agent.log on the Pi) also holds earlier sessions.
    Window = fd-benchmark's run_start - 10 s .. run_end + 15 s (both clocks on NTP, same time zone)."""
    from datetime import datetime, timedelta
    t0 = t1 = None
    for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("event") in ("run_start", "run_end"):
            t = datetime.fromisoformat(e["t"]).replace(tzinfo=None)
            t0, t1 = (t, t1) if e["event"] == "run_start" else (t0, t)
    if not t0 or not any(STAMP.match(ln) for ln in lines):
        return lines                      # fdbench's own assistant.log (no dates): already just this run
    lo, hi = t0 - timedelta(seconds=10), (t1 or datetime.max - timedelta(seconds=15)) + timedelta(seconds=15)
    out = []
    for ln in lines:
        m = STAMP.match(ln)
        if m and lo <= datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") <= hi:
            out.append(ln)
    return out


def trial_order(run: Path) -> list[str]:
    order = []
    for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("event") == "trial_start":
            order.append(e["trial"])
    return order


def wer(run: Path, lines: list[str]) -> dict:
    order = trial_order(run)
    segs, cur = [], []
    for ln in lines:
        if RESET in ln:
            segs.append(cur)
            cur = []
        else:
            m = HEARD.search(ln)
            if m:
                cur.append(m.group(1).strip())
    segs.append(cur)
    if len(segs) != len(order):
        return {"error": f"{len(segs)} log segments for {len(order)} trials: was the per-trial reset on?"}
    man = {}
    mp = Path(__file__).resolve().parents[1] / "fd-benchmark" / "datasets" / "manifest.json"
    for p in (run.parents[2] / "datasets" / "manifest.json", mp):
        if p.exists():
            for g in json.loads(p.read_text(encoding="utf-8"))["groups"].values():
                for c in (g.get("clips") if isinstance(g, dict) else g):
                    man[c["id"]] = c.get("meta") or {}
            break
    tot_e = tot_n = 0
    per = []
    for trial, heard in zip(order, segs):
        tj = run / "trials" / trial / "timing.json"
        if not tj.exists():
            continue
        t = json.loads(tj.read_text(encoding="utf-8"))
        meta = man.get(t.get("clip", "")) or t.get("clip_meta") or {}
        ref = words(reference(meta, t.get("variant", "main")))
        if not ref:
            continue
        hyp = words(" ".join(heard))
        e = edits(ref, hyp)
        tot_e, tot_n = tot_e + e, tot_n + len(ref)
        per.append({"trial": trial, "ref_words": len(ref), "errors": e, "heard": " ".join(heard),
                    "said": " ".join(ref)})
    return {"wer": round(tot_e / tot_n, 4) if tot_n else None, "errors": tot_e, "ref_words": tot_n,
            "trials": len(per), "per_trial": per}


def barge_in(run: Path) -> dict:
    p = run / "results.csv"
    if not p.exists():
        return {}
    for row in csv.DictReader(p.open(encoding="utf-8")):
        if row.get("key") == "v15_interruption.t_stop" and row.get("n") not in ("", "0"):
            f = lambda k: round(float(row[k]), 3) if row.get(k) else None   # noqa: E731
            return {"n": int(row["n"]), "mean": f("mean"), "p50": f("median"), "p95": f("p95")}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="fd-benchmark run dir (runs/<student>/<date_time>)")
    ap.add_argument("--log", type=Path, help="Elkit's log of that run (default: <run>/assistant.log)")
    ap.add_argument("--device", default="", help='e.g. "Raspberry Pi 5, 8 GB" (default: from the log)')
    ap.add_argument("--cooling", default="", help='e.g. "active cooler"')
    ap.add_argument("--network", default="", help='e.g. "Wi-Fi 5 GHz, ~40 Mbps"')
    a = ap.parse_args()
    every = (a.log or a.run / "assistant.log").read_text(encoding="utf-8", errors="replace").splitlines()
    lines = run_window(a.run, every)
    st = turn_stages(lines)
    dev = a.device or next((m.group(1) for ln in reversed(every) if (m := DEVICE.search(ln))), "")
    w = wer(a.run, lines)
    rep = {
        "run": a.run.name, "date": str(date.today()), "device": dev, "cooling": a.cooling, "network": a.network,
        "method": "fd-benchmark clips (same seed) on Elkit; per-stage stamps on the device's monotonic clock; "
                  "mean and p95 over all spoken turns of the run",
        "metrics_s": {
            "E2E latency (VAD end -> first audio)": stats(st["e2e"]),
            "ASR final (VAD end -> transcript)": stats(st["asr"]),
            "turn end wait (transcript -> turn end)": stats(st["turn wait"]),
            "LLM TTFT (turn end -> first token)": stats(st["llm"]),
            "TTS TTFB (first text -> first audio)": stats(st["tts"]),
            "Barge-in (speech -> playback stopped)": barge_in(a.run),
            **({"First sound (VAD end -> filler or answer)": stats(st["first sound"]),
                "Filler said (turns with one)": stats(st["filler"])} if st["filler"] else {}),
        },
        "wer": {k: v for k, v in w.items() if k != "per_trial"},
        "wer_per_trial": w.get("per_trial", []),
    }
    (a.run / "latency_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "latency.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"run {a.run.name}   device: {dev or '(not in the log: --device)'}")
    print(f"{'metric':44} {'n':>4} {'mean':>7} {'p50':>7} {'p95':>7}")
    for k, s in rep["metrics_s"].items():
        if s:
            fmt = lambda v: f"{v:7.2f}" if v is not None else "      -"   # noqa: E731
            print(f"{k:44} {s['n']:>4} {fmt(s['mean'])} {fmt(s['p50'])} {fmt(s['p95'])}")
    ww = rep["wer"]
    print(f"WER {ww.get('wer')}  ({ww.get('errors')} errors / {ww.get('ref_words')} words, {ww.get('trials')} trials)"
          if "wer" in ww else f"WER: {ww.get('error')}")
    print(f"-> {a.run / 'latency_report.json'}  and  report/latency.json")


if __name__ == "__main__":
    main()
