# Full-Duplex Voice Agent on a Raspberry Pi

Student fda-pc · run 2026-11-07_01-12-53 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 2.94 s | 5.12 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 0.85 s | 1.81 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 1.68 s | 3.03 s | 24 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.48 s | 1.00 s | 30 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.90 s | 1.55 s | 5 |
| WER | ASR accuracy on test set | < 15 % | 0.0 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 5 Model B Rev 1.1 · 8 GB
- Cooling: case fan · peak 62 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 3 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0, ASR Groq whisper-large-v3-turbo + Moonshine, LLM Groq gpt-oss-120b, TTS Piper en_US-ryan-medium

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate) | 74 | 663 | 50 → 85 |
| Messages in context | 6 | 13 | 5 → 8 |
