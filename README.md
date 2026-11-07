# FDA: Full-Duplex Agent

## Quick start (the final setup: Elkit on a Raspberry Pi 5)

Final configuration, as benchmarked (report/final_pi_run): **Groq** for listening (Whisper) and the LLM, **Piper** for
the voice on the Pi, everything else on the device. You need one key: a free Groq API key from console.groq.com.

```bash
git clone https://github.com/arjAIareNOW/mex3.git ~/fda && cd ~/fda
bash deploy/pi/install.sh             # venv, packages, Node + jsdom, offline models, boot service, kiosk,
                                      # fda.toml (Groq + Piper already chosen) and a .env template
nano .env                             # paste your key after GROQ_API_KEY= (the only key needed)
python -m fda.pc                      # run; the screen is at http://<pi-address>:8765
python tests/qa_offline.py            # offline test suite (no key, no tokens)
```

On a Windows laptop: Python 3.12, then `deploy\windows\setup.bat`, put `GROQ_API_KEY=...` in `.env`, and run
`python -m fda.pc`. The sections below describe the full system, including the earlier DGX3 (school server) setup.


A voice assistant you talk to like a person. It is **always listening** from boot, with no wake word. You can **interrupt
it mid-sentence**, and it **cancels its own echo**, so with an open speaker and mic it does not mistake its own voice
for you. It controls a virtual room shown on a 7" display (lights, thermostat, timers, alarms, reminders, music, volume,
calls, messages) and can check the weather and search the web.

Runs on a **Windows laptop** (for testing) and a **Raspberry Pi 5 (8 GB)**. The heavy models run on DGX3:

| Server | Model | Used for |
|---|---|---|
| `10.153.60.233:8000` (vLLM) | Qwen/Qwen2-Audio-7B-Instruct | hearing (transcript) + planner (decide, act, reply) |
| `10.153.60.233:8001` (vLLM-Omni) | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | voice (streamed 24 kHz PCM) |

On the laptop / Pi: echo cancellation (WebRTC AEC3), voice activity (Silero VAD), the fast path, your
**MLEx-2-VCM intent Conformer**, the agents and the display. Nothing goes to the cloud except web search and weather.

## FDA on Pipecat + MCP (the assignment architecture): `python -m fda.pc`

| Requirement | How FDA meets it |
|---|---|
| Agent harness with MCP support on the Pi | **Pipecat 1.15** pipeline (`fda/pc/agent.py`) with Pipecat's `MCPClient` |
| MCP servers for SOTA ASR and TTS, run on the Pi | `fda/mcp_servers/asr_server.py` (`transcribe`) and `tts_server.py` (`synthesize`): own processes, MCP streamable HTTP on `127.0.0.1:8101` / `8102` |
| VAD on the Pi | Silero VAD (Pipecat `VADProcessor`) + **Smart Turn v3** end-of-turn model (local ONNX) |
| Full-duplex agent doing real-world tasks | FDA's audio engine as a Pipecat transport (`fda/pc/transport.py`: AEC3 + own-voice gate, mic and speaker open together), duck on overlap, ≥3 words to interrupt, backchannels ignored, resume after a false interruption; tools MCP server `tools_server.py` on `:8103` |

```
mic ─► FDA audio engine (AEC3 + own-voice gate) ─► Silero VAD ─► DuplexGuard ─► STT ──MCP──► asr server (Groq Whisper, fallback Moonshine)
     ─► HeardTap ─► user turn (Smart Turn v3) ─► LLM (Groq gpt-oss-120b, fallback gpt-oss-20b) ◄─MCP─► tools server
     ─► NotForMe ("(silent)") ─► TTS ──MCP──► tts server (DGX3 Qwen3-TTS, fallback Piper) ─► FDA mixer ─► speaker
```

**Tools (all free, no key):** `create_app`, `edit_app`, `undo_app_change`, `list_apps`, `open_app` (build games, websites
and small tools as one HTML file, shown on the display), `get_weather` (Open-Meteo), `get_news` (Google News RSS),
`find_recipe` (TheMealDB), `wikipedia`, `web_search` (DuckDuckGo), `calculate`, `get_time`; plus FDA's room tools in
process (lights, thermostat, timers, alarms, reminders, music, volume, Sky Hop). Sky Hop is also in `apps/` (open it
with "open Sky Hop" or edit it like any built app).

**Full-duplex manners (`fda/pc/processors.py`):**
- someone talks over FDA -> its voice ducks at once; the overlap is transcribed early (at 1.0 s and 1.6 s), and a real
  interruption (>= 3 words, or "stop"/"wait") stops it right there; after 3 s of talking over it, it stops anyway
- backchannels ("uh huh", "okay"), speech to someone else ("Honey, ...", "Mateo, ..."), its own echo and far-away
  speech (> `far_speech_db` = 10 dB quieter than the user's usual level: TV, radio, across the room) never stop it;
  if they did stop it, it resumes the rest of what it was saying
- the LLM answers "(silent)" for speech not meant for it; when a small model does that to an obvious question or
  request ("Wait, what time is it?"), FDA overrules it once and asks again
- spoken replies are capped (~2 sentences) and stripped of markdown and links; slow internet tools get a short filler
  ("Let me check that.") so a reply starts within ~1 s; questions about the time carry the true time to the LLM

**Building apps by voice:** "make me a snake game", then "make the snake red", "undo that". The build runs in the background
(you can keep talking), the app opens in the free lower-right area of the display (⛶ = full screen), and FDA says when
it's ready. The coder is `qwen2.5-coder-3b` locally (snake game in 43 s on the laptop; several minutes on a Pi) or a
free cloud tier (`coder_engine = "groq"`: seconds). A JavaScript syntax check (`node --check`, if Node.js is installed)
sends errors back to the model for one repair round.

```powershell
cd C:\Users\asarn\sandbox\MachineExercise3\fda
.\.venv\Scripts\python.exe -m fda.pc                 # mic + speaker + display (starts the 3 MCP servers itself)
.\.venv\Scripts\python.exe -m fda.pc --sim           # no hardware: simulated room with echo
.\.venv\Scripts\python.exe -m fda.pc.launch          # only the MCP servers (e.g. as a service on the Pi)
.\.venv\Scripts\python.exe tests\pc_sim.py           # 8 end-to-end scenarios in the simulated room
.\.venv\Scripts\python.exe tests\pc_llm_eval.py      # tool choice of the LLM, 20 sentences
```

Settings (`fda.toml`): `llm_engine` (groq | local | openai), `agent_llm` (local model, default `qwen2.5-3b`),
`asr_engine` (groq | dgx3 | moonshine | openai), `tts_engine` (dgx3 | piper | kokoro | deepgram),
`coder_engine` (groq | local | same), `far_speech_db` (10; 0 = off). Groq is the main provider; each part has one
main model and one fallback (see `fda.toml`).
Keys go in `.env`: `GROQ_API_KEY` (main), `QWEN_API_KEY` (DGX3 voice), `DEEPGRAM_API_KEY` (optional).

Measured on the laptop (offline, zero tokens): ASR over MCP 154 ms, TTS over MCP 222 ms per sentence; LLM tool choice
16/20 (Qwen2.5-3B, median 1.3 s) vs 8/20 (1.5B); `tests/pc_sim.py` chat, weather, lights, news, not-for-me, backchannel,
interrupt and app all pass.

**Benchmark:** `fd-benchmark/assistant_entry.py` plugs this agent into fdbench's launcher (`launch_assistant.py --check`
→ READY). Free local judge for the scores: `python -m fda judge` (Qwen2.5-3B on :8092, no key).

## How a turn works (classic FDA, `python -m fda`)

```
mic ─► AEC3 (removes FDA's own voice, ~48 dB) ─► Silero VAD + level detector
         ▲ reference = exactly what the speaker plays (voice + music + alarms)
         │
 120 ms of silence ─► SPECULATIVE: transcript (Qwen2-Audio) ─► echo check ─► fast-path rules ─► planner (Qwen2-Audio, text)
 300-500 ms silence ─► COMMIT: act + speak                     (user keeps talking → speculative work is cancelled)

 planner JSON {for_me, say, do} streams: say → TTS phrase by phrase; do → device agent / search agent / scheduler
```

**Interruptions (barge-in)** while FDA is talking:
1. Your voice is detected (VAD or a level jump over the residual echo) → FDA's voice **ducks to 25 % in ~40 ms**.
2. The early transcript decides: real words ("stop", "what time is it") → **fade out in 80 ms** and answer;
   "uh huh", "yeah", "mm-hmm" → **volume comes back** and FDA keeps talking; FDA's own echo → ignored.
3. Talking over it for 1.5 s stops it even without a transcript.
4. History keeps only what you actually heard: `"Once upon a time, there was a [interrupted]"`.
5. If it stopped for something that turns out not to be for it, it **picks up where it left off**.

**Not for me**: "Honey, did you lock the door?", phone calls and TV news get `for_me=false`, so FDA stays quiet.

## Measured (simulated room on the build server, real DGX3 calls)

`python tests/sim_scenarios.py` plays FDA's output back into a simulated mic (delay, reverb, noise) and a second voice
(vivian) as the user. Results from the last two full runs (both 6/6):

| Scenario | Result |
|---|---|
| "Turn off the lights." (fast path) | first audio **443-455 ms** after you stop talking |
| "Make the lights blue and set a timer for two minutes" (planner) | **596-850 ms**, both actions done |
| Story, then "Stop. What time is it?" | ducks after **307-562 ms**, fades out at **~0.9-1.1 s**, answers the time |
| "Uh huh" while FDA talks | keeps talking |
| "Honey, did you remember to lock the front door…" | ignored (`for_me=false`) |
| "Who is the current president of France?" | web search → "Emmanuel Macron" |
| FDA's own echo while it talks | **0** false interruptions |

The simulated echo path is linear; a real speaker adds distortion, so tune on the real hardware (see below).
Planner accuracy on 23 typed test sentences (`tests/planner_order.py`): 21/23 with say-first (default).

## Offline (no DGX3, no key)

Everything can run on the laptop or the Pi itself. `backend = "auto"` (default) uses DGX3 when it answers and the
offline models otherwise; `--offline` / `--server` force one.

```
python -m fda download        # once: ~1.4 GB into offline_models/
python -m fda check --offline # speed of each part on this machine
python -m fda --offline       # run
```

| Part | Offline model | Laptop (measured) |
|---|---|---|
| hearing | Moonshine base (sherpa-onnx) | 0.1 s for 3.8 s of speech |
| planner | Qwen2.5-1.5B-Instruct Q4 (llama.cpp `llama-server`, started by FDA, 127.0.0.1) | first word 0.3-0.6 s, 19/23 on the planner test |
| voice | Piper "ryan" (default) or Kokoro (nicer, ~10x slower: laptop only) | 0.1 s to first audio, 25x real time |

End to end in the simulated room (offline): lights 0.3-0.5 s, planner commands ~1.2-2.3 s to first audio. Settings:
`offline_llm = "qwen2.5-3b"` (21/23, ~2x slower), `offline_tts`, `offline_voice`, `offline_threads`. The Pi has not
been measured yet: run `python -m fda check --offline` there. If the downloaded `llama-server` does not run on Pi OS,
build llama.cpp from source and put `llama-server` on the PATH; FDA uses it from there.

## FDA knows its own voice

With the mic next to the speaker, the echo canceller never removes all of FDA's voice. `fda/selfvoice.py` adds:
- **Predicted leak**: FDA has the exact audio it plays, learns how much of it leaks through the echo canceller per
  frequency band (a median over the last ~10 s, which the user's interruptions cannot move), and treats mic sound
  that this prediction explains as itself: it does not duck, stop, or start a turn on it.
- **Own words out of the transcript**: words FDA was playing while you spoke are removed ("once upon a time stop what
  time is it" -> "stop what time is it").

`tests/sim_scenarios.py --harsh` simulates a mic next to a distorting speaker; `--no-selfvoice` turns the checks off
to compare. A speaker that is loud *and* very close can drown out the user completely (20 dB louder than a person at
the mic): no software can hear through that, so keep the speaker volume moderate and point it away from the mic.

## Sky Hop (a game you change by voice)

"Let's play a game" opens Sky Hop, an original tap-to-fly game, in the lower right of the display (tap, click or
Space to fly). While it runs, say what to change and it changes live, without restarting: "make the plane red",
"night sky", "make the towers green", "faster", "make the gaps bigger", "lower gravity", "bigger plane", "pause the
game", "restart", "close the game". The settings are room state (`game` tool in `fda/brain.py`, `_do_game` in
`fda/room.py`); the display pushes each change into `fda/games/sky_hop.html`. While the game is open the display
keeps only the Lights and Music cards so the game fits the 800x480 screen. Offline planner: 12/12 on game commands.

## Windows laptop

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH").
2. Copy this folder to the laptop, then double-click `deploy\windows\setup.bat`. It creates `.env`: open it in Notepad
   and paste your key: `QWEN_API_KEY=...`. Setup ends with `python -m fda check` (both servers must say OK; the
   laptop needs the campus network or VPN to reach 10.153.60.233).
3. Double-click `deploy\windows\start_fda.bat`. The display opens in the browser. Talk.
4. Start on login: `deploy\windows\autostart_on.bat` (undo: `autostart_off.bat`).

**Windows tip:** many laptops already cancel echo on the built-in mic ("Audio enhancements"). Two cancellers fight. Try
FDA's own first (default). If voices sound choppy, either turn off *Settings → Sound → (mic) → Audio enhancements*, or
run `start_fda.bat --no-aec` to use only Windows'.

## Raspberry Pi 5

```bash
git clone <this repo> ~/fda && cd ~/fda       # or copy the folder
bash deploy/pi/install.sh                     # venv, packages, Node + jsdom, offline models, boot service, kiosk
nano .env                                     # GROQ_API_KEY, QWEN_API_KEY
python -m fda devices                         # find the USB mic/speaker names -> fda.toml input_device / output_device
sudo reboot                                   # FDA (python -m fda.pc) starts on boot, display full screen on the 7"
journalctl --user -u fda -f                   # live log
```

Best echo cancellation: mic and speaker on **one USB audio adapter** (shared clock). Separate devices drift apart.

**Moving the laptop folder to the Pi.** Copy the whole `fda` folder **except** these, which are built per machine
(install.sh makes them again on the Pi):
`.venv/` (Windows Python), `fda/mcp_servers/node_modules/` (jsdom), `offline_models/llama.cpp-*` (Windows binary),
`fd-benchmark/.venv/`. Keep `.env`, `fda.toml`, `apps/` (with its `.git` version history), `docs/`.
On the Pi prefer `offline_asr = "parakeet-small"` or `"moonshine-base"` and `offline_tts = "piper"` (Kokoro and
Parakeet 0.6B are laptop-speed). The DGX3 needs the school network or the VPN on the Pi as well; without it the
chains fall back to the offline engines. Engine choices: [ENGINES.md](ENGINES.md).

## Settings (`fda.toml`, see `fda.example.toml`; all defaults in `fda/config.py`)

| Setting | Default | What |
|---|---|---|
| `voice` | `ryan` | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |
| `input_device`, `output_device` | system default | part of the device name |
| `aec` | `true` | FDA's echo canceller |
| `endpoint_silence_s` / `early_endpoint_s` | 0.50 / 0.30 | silence that ends your turn (early when the command is clearly complete) |
| `bargein_margin_db` | 18 | how far above the residual echo your voice must be to duck FDA |
| `bargein_confirm_s` | 1.5 | talking over FDA this long stops it without a transcript |
| `planner_say_first` | `true` | ~190 ms faster first word; `false` = decide actions before speaking |
| `vcm_fast_path` | `true` | your VCM intent model acts instantly on very confident lights/pause/next/volume commands (undone if the transcript disagrees) |
| `record_dir` | off | saves `mic`, `ref`, `clean` WAVs for tuning echo cancellation |

## Commands

```
python -m fda                 run (always listening, display at http://localhost:8765)
python -m fda --sim           run without hardware (simulated room)
python -m fda --record DIR    also save mic / reference / echo-cancelled audio
python -m fda devices         list audio devices
python -m fda check           check servers, key and voice latency (--offline: local model speeds)
python -m fda download        download the offline models
python -m fda --offline       run with no DGX3 and no key
python tests/sim_scenarios.py end-to-end test in a simulated room
python tests/planner_order.py planner accuracy on typed sentences
```

## Files

```
fda/audio.py     duplex audio, mixer (voice | music | fx), AEC3 worker      fda/vad.py      Silero VAD
fda/agent.py     the talker: turn taking, barge-in, echo filter, agents     fda/speaker.py  streamed TTS, what was heard
fda/brain.py     transcript + planner prompts, JSON schemas, examples       fda/servers.py  DGX3 clients (streaming)
fda/rules.py     fast path + backchannels                                   fda/vcm.py      VCM intent Conformer
fda/room.py      virtual room, timers/alarms/reminders, music, phone        fda/search.py   web search + weather
fda/local.py     offline models (hearing, planner, voice)  fda/selfvoice.py FDA's own-voice checks
fda/ui.py        display server                                            fda/web/        7" display page
fda/sim.py       simulated room for testing                                 fda/models/     silero_vad.onnx, vcm_intent.onnx
fda/local.py     offline models: hearing, planner, voice, downloader        fda/selfvoice.py FDA's own-voice checks
```

## Known limits

- Qwen2-Audio is not a tool-calling model: the planner is held to a JSON schema with examples and backed by rules for
  the common commands, but unusual requests can still be misread (2/23 misses on the test sentences).
- English. Qwen2-Audio transcribed Filipino as Japanese in testing.
- Weather and web search need internet; everything else needs only the DGX3 servers.
- Calls and messages are simulated on the display.
- Licences: the VCM intent weights are research / education use only (trained on ai231); Silero VAD is MIT.
