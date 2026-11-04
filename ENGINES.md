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
| `groq` | gpt-oss-120b, then 20b, then Qwen | Free tier (8,000 tokens/min per model) | 0.5-0.9 s | Default. Switches model by itself at the limit, then SambaNova, then Gemini |
| `gemini` | Gemini Flash / Flash-Lite | Free tier | 1-3 s, sometimes "busy" | `GEMINI_API_KEY` |
| `local` | Qwen2.5-3B on the device | Free, offline | 3-6 s laptop, slower on a Pi | Works without internet |

## 4. App designer (writes apps and games): `coder_engine`

| Engine | Model | Cost | Full game | Notes |
|---|---|---|---|---|
| `groq` | gpt-oss-120b | Free tier | ~10 s including tests and fixes | Default first choice |
| `sambanova` | gemma-4-31B, gpt-oss-120b | Free | ~6 s when not busy | Often "high demand" |
| `gemini` | Gemini Flash | Free tier | 30-170 s | Slow but reliable backup |
| `local` | Qwen2.5-Coder-3B | Free, offline | 1-3 min | Weak at games |

Default chain: `coder_engine = "groq,sambanova,gemini"`.

## Ready-made combinations

| Goal | `asr_engine` | `tts_engine` | `llm_engine` | `coder_engine` |
|---|---|---|---|---|
| **Best quality at school (current)** | `dgx3` | `dgx3` | `groq` | `groq,sambanova,gemini` |
| **Whisper instead of Qwen** | `groq,dgx3,moonshine` | `dgx3,piper` | `groq` | `groq,sambanova,gemini` |
| Away from school (no VPN), laptop | `groq,parakeet` | `kokoro,piper` | `groq` | `groq,sambanova,gemini` |
| Away from school, Pi | `groq,parakeet-small` | `piper` | `groq` | `groq,sambanova,gemini` |
| Fully offline, laptop | `parakeet` | `kokoro` | `local` | `local` |
| Fully offline, Pi | `parakeet-small` | `piper` | `local` | `local` (slow) |
| Cheapest cloud speech (no school) | `groq` | `deepgram,piper` | `groq` | `groq,gemini` |

Notes:
- Free tiers change; check each provider's limits. Groq's speech (Whisper) and chat limits are separate.
- On the Pi, FDA gives speech 2 of the 4 cores. Prefer `parakeet-small` or `moonshine`, and `piper`.
- Any offline engine needs its model downloaded once: `python -m fda download` (it reads `offline_asr` / `offline_tts`).
