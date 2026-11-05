# FDA engine combinations

Every part of FDA can use a different engine. Set them in `fda.toml`. Keys go in `.env`, never in `fda.toml`.
Several engines in a list are a **chain**: FDA uses the first one that works and skips a failed one for 30 s.
Speech chains always end with an offline engine, so FDA keeps listening and talking without internet.

## 1. Listening (speech to text): `asr_engine`

| Engine | Where it runs | Cost | Speed (measured on the laptop) | Notes |
|---|---|---|---|---|
| `dgx3` | School DGX3 (Qwen2-Audio-7B) | Free (needs VPN) | ~0.4 s | Good accuracy, needs `QWEN_API_KEY` |
| `groq` | Groq cloud (Whisper large-v3-turbo) | Free tier | ~0.5-1 s (3 s for the first call) | Best known name, multilingual, `GROQ_API_KEY` |
| `openai` | Any OpenAI-compatible server | Varies | Varies | `ASR_BASE_URL`, `ASR_API_KEY`, `ASR_MODEL` |
| `moonshine` | On the device (Moonshine v2 base) | Free, offline | 66 ms | Default offline listener, fine on the Pi |
| `moonshine-tiny` | On the device | Free, offline | Faster | Less accurate |
| `parakeet-small` | On the device (Parakeet TDT-CTC 110M) | Free, offline | 60-120 ms | Strong open model, fits the Pi |
| `parakeet` | On the device (Parakeet TDT 0.6B v2) | Free, offline | ~730 ms | Top open English accuracy, laptop only |

## 2. Voice (text to speech): `tts_engine`

| Engine | Where it runs | Cost | Speed (laptop) | Notes |
|---|---|---|---|---|
| `dgx3` | School DGX3 (Qwen3-TTS-1.7B) | Free (needs VPN) | ~0.6 s per sentence, first sound ~0.1 s | Most natural, voice set by `voice` |
| `deepgram` | Deepgram cloud (Aura-2) | Free sign-up credit | Fast | `DEEPGRAM_API_KEY` |
| `kokoro` | On the device (Kokoro-82M) | Free, offline | ~0.9 s per sentence | Natural, laptop (slow on a Pi) |
| `piper` | On the device (Piper ryan) | Free, offline | ~0.08 s per sentence | Clear, a bit robotic, fine on the Pi |

## 3. Thinking (the agent that talks and calls tools): `llm_engine`

| Engine | Model | Cost | Reply time | Notes |
|---|---|---|---|---|
| `groq` | gpt-oss-120b, one fallback gpt-oss-20b | Groq paid plan | 0.5-0.9 s | Default. Uses the fallback only when the main model fails |
| `local` | Qwen2.5-3B on the device | Free, offline | 3-6 s laptop, slower on a Pi | Works without internet |

## 4. App designer (writes apps and games): `coder_engine`

| Engine | Model | Cost | Full game | Notes |
|---|---|---|---|---|
| `groq` | qwen3.8-27b, one fallback gpt-oss-120b | Groq paid plan | ~5-10 s including tests and fixes | Default |
| `local` | Qwen2.5-Coder-3B | Free, offline | 1-3 min | Weak at games |

Default: `coder_engine = "groq"`. Rule: every part has one main engine and one fallback.

## Ready-made combinations

| Goal | `asr_engine` | `tts_engine` | `llm_engine` | `coder_engine` |
|---|---|---|---|---|
| **Groq main (current)** | `groq` (+ Moonshine) | `dgx3` (+ Piper) | `groq` | `groq` |
| Away from school (no VPN), Pi | `groq` (+ Moonshine) | `piper` | `groq` | `groq` |
| Fully offline, laptop | `parakeet` | `kokoro` | `local` | `local` |
| Fully offline, Pi | `parakeet-small` | `piper` | `local` | `local` (slow) |
| Cheapest cloud speech (no school) | `groq` | `deepgram` (+ Piper) | `groq` | `groq` |

(+ X) = the offline fallback that is added by itself (`offline_asr`, `offline_tts`).

Notes:
- Free tiers change; check each provider's limits. Groq's speech (Whisper) and chat limits are separate.
- On the Pi, FDA gives speech 2 of the 4 cores. Prefer `parakeet-small` or `moonshine`, and `piper`.
- Any offline engine needs its model downloaded once: `python -m fda download` (it reads `offline_asr` / `offline_tts`).
