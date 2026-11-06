"""All settings in one place. Override any of them in fda.toml (same names, see fda.example.toml)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent


def _load_env(path: Path):
    """Minimal .env reader (KEY=value per line); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Config:
    # --- where the models run ---
    backend: str = "auto"             # server = DGX3 | offline = this machine (no network, no key) |
                                      # auto = DGX3 when it answers, otherwise offline

    # --- servers (DGX3) ---
    asr_url: str = "http://10.153.60.233:8000"
    asr_model: str = "Qwen/Qwen2-Audio-7B-Instruct"
    tts_url: str = "http://10.153.60.233:8001"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    voice: str = "ryan"
    voice_style: str = ""            # e.g. "warm and friendly" -> Qwen3-TTS instructions
    language: str = "English"
    tts_rate: int = 24000             # Qwen3-TTS native rate (server supports 8000 / 24000)

    # --- offline models (python -m fda download) ---
    offline_asr: str = "moonshine-base"   # moonshine-base | moonshine-tiny (faster, less accurate)
    offline_llm: str = "qwen2.5-1.5b"     # qwen2.5-0.5b | qwen2.5-1.5b | qwen2.5-3b (smarter, ~2x slower)
    offline_tts: str = "piper"            # piper (fast, works on the Pi, voice: ryan) | kokoro (nicer, laptop only)
    offline_voice: str = "am_michael"     # kokoro: af, af_bella, af_nicole, af_sarah, af_sky, am_adam, am_michael,
                                          #         bf_emma, bf_isabella, bm_george, bm_lewis
    offline_speed: float = 1.0            # speaking speed
    offline_threads: int = 0              # CPU threads for the planner (0 = automatic)
    llm_port: int = 8091                  # local llama-server (127.0.0.1 only)

    # --- MCP servers (fda/mcp_servers, used by the Pipecat agent: python -m fda.pc) ---
    asr_engine: str = "moonshine"         # moonshine (offline) | groq (GROQ_API_KEY) | openai (ASR_BASE_URL...)
    tts_engine: str = "piper"             # piper | kokoro (offline) | deepgram (DEEPGRAM_API_KEY)
    llm_engine: str = "local"             # local (llama-server, offline_llm) | groq | openai (LLM_BASE_URL)
    llm_model: str = ""                   # cloud model name ("" = a sensible default per engine)
    agent_llm: str = "qwen2.5-3b"         # local model for the Pipecat agent (tool calling: 3b 16/20, 1.5b 8/20)
    coder_engine: str = "same"            # who writes apps for create_app: same (as llm_engine) | local | groq | ...
    mcp_host: str = "127.0.0.1"
    asr_mcp_port: int = 8101
    tts_mcp_port: int = 8102
    tools_mcp_port: int = 8103
    room_hub_port: int = 8104             # the agent's room, for the room tools of the tools MCP server (local only)
    far_speech_db: float = 10.0           # speech this much quieter than the user's level = TV/across the room (0 = off)
    docs_dir: str = "docs"                # documents FDA can answer questions about (PDF, DOCX, XLSX, CSV, MD, TXT)
    reports_dir: str = "reports"          # device check reports (audit-<time>.md)
    apps_dir: str = "apps"                # apps and games FDA builds (served on the display at /apps/)
    quick_answers: str = "quick_answers.txt"   # predefined answers, spoken without the LLM ("" = off)
    quick_match: float = 0.85             # how close what was heard must be to a question there (0-1)
    filler_after_s: float = 0.7           # no answer yet this long after the turn: a short filler ("On it.") from the
                                          # TTS cache, so the person hears Elkit at once (0 = off)

    # --- audio devices ---
    input_device: str | int | None = None    # name substring or index; None = system default
    output_device: str | int | None = None
    device_rate: int = 48000
    block_ms: int = 10
    aec: bool = True                  # WebRTC AEC3 in the app (turn off if Windows already cancels echo)
    noise_suppression: bool = True
    aec_delay_ms: int | None = None   # None = from the audio driver's reported latency

    # --- listening / turn taking ---
    vad_start: float = 0.55           # speech probability to start a user turn
    vad_end: float = 0.35             # below this counts as silence
    start_frames: int = 2             # 32 ms VAD windows above vad_start to start (64 ms)
    onset_margin_db: float = 20.0     # ...or this far above the room's noise floor for 3 windows (VAD can lag ~1 s)
    onset_min_db: float = -42.0
    speculative_silence_s: float = 0.12  # send the turn to Qwen early, after this much silence
    endpoint_silence_s: float = 0.50  # the turn is over after this much silence
    early_endpoint_s: float = 0.30    # ...or after this much, when the transcript is clearly a complete command
    incomplete_endpoint_s: float = 1.2  # ...but this long when it stops mid-sentence ("search the net for...")
    min_speech_s: float = 0.18        # shorter blips (cough, click) are ignored
    max_turn_s: float = 15.0
    preroll_s: float = 0.60           # audio kept from before speech was detected (VAD ramps up late)

    # --- barge-in (talking over FDA) ---
    bargein_vad: float = 0.50         # while FDA talks: 2 of the last 3 windows above this -> duck
    bargein_min_db: float = -48.0     # echo-cancelled mic level needed to count as the user
    bargein_margin_db: float = 18.0   # ...or this far above the residual-echo floor (energy detector)
    bargein_duck_gain: float = 0.25   # voice volume while deciding
    bargein_duck_ms: float = 40.0
    ring_max_s: float = 10.0                 # a ringing timer / alarm stops by itself after this (was 60 s)
    bargein_confirm_s: float = 1.50   # sustained speech that stops FDA on sound alone; shorter utterances
                                      # ("stop" vs "uh huh") are decided by their words while FDA is ducked
    bargein_fade_ms: float = 80.0
    resume_after_false_bargein: bool = True
    selfvoice: bool = True            # FDA predicts how much of its own playback leaks through the AEC; sound
                                      # that prediction explains never counts as the user (fda/selfvoice.py)
    selfvoice_margin_db: float = 2.0  # how far above the predicted leak counts as a person
    strip_own_words: bool = True      # remove the words FDA was playing from the transcript

    # --- agent ---
    name: str = "FDA"
    history_turns: int = 8
    planner_say_first: bool = True    # ~190 ms faster first words; False = decide actions first
    location: str = "Quezon City"
    asr_hint: str = ""                # names Whisper should spell right ("Mandaluyong, Davao"); the agent's name
                                      # and location are always in it
    music_dir: str = "music"
    state_file: str = "room_state.json"   # alarms, reminders and lists kept across restarts ("" = not kept)
    fast_path: bool = True
    vcm_fast_path: bool = True        # use the VCM intent Conformer as a second opinion
    vcm_min_conf: float = 0.95
    say_ready: bool = True

    # --- display ---
    ui_host: str = "0.0.0.0"
    ui_port: int = 8765
    open_browser: bool = True

    # --- debug ---
    record_dir: str = ""              # e.g. "recordings": saves mic / reference / cleaned WAVs
    log_level: str = "INFO"

    api_key: str = field(default="", repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        _load_env(ROOT / ".env")
        cfg = cls()
        p = Path(path) if path else ROOT / "fda.toml"
        if p.exists():
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k not in names:
                    raise ValueError(f"unknown setting in {p.name}: {k}")
                setattr(cfg, k, v)
        cfg.api_key = os.environ.get("QWEN_API_KEY", "")
        if cfg.backend not in ("server", "offline", "auto"):
            raise ValueError(f"backend must be server, offline or auto, not {cfg.backend!r}")
        if cfg.backend == "server" and not cfg.api_key:
            raise SystemExit("QWEN_API_KEY is not set (put it in .env next to fda.toml), or use backend = \"offline\"")
        return cfg
