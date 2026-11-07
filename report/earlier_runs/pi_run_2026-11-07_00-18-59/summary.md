# Full-Duplex Voice Agent on a Raspberry Pi

Student fda-pc · run 2026-11-07_00-18-59 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 3.00 s | 5.72 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 0.79 s | 1.34 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 1.55 s | 5.25 s | 25 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.67 s | 1.25 s | 30 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.89 s | 1.50 s | 5 |
| WER | ASR accuracy on test set | < 15 % | 0.0 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 5 Model B Rev 1.1 · 8 GB
- Cooling: case fan · peak 63 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 5 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate) | 74 | 564 | 50 → 91 |
| Messages in context | 6 | 15 | 5 → 7 |
