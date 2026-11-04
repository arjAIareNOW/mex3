"""Draws the FDA pipeline diagram (Pillow only) -> MachineExercise3/FDA_pipeline.png

    .venv\\Scripts\\python scripts\\diagram.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[2] / "FDA_pipeline.png"
W, H = 2000, 1300
img = Image.new("RGB", (W, H), "#f7f8fa")
d = ImageDraw.Draw(img)
F = Path(r"C:\Windows\Fonts")


def font(bold, size):
    for name in (("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")):
        for base in (F, Path("/usr/share/fonts/truetype/dejavu")):
            if (base / name).exists():
                return ImageFont.truetype(str(base / name), size)
    return ImageFont.load_default()


f_title, f_sub, f_h, f_b, f_sec = font(1, 40), font(0, 22), font(1, 21), font(0, 17), font(1, 18)
INK, MUTED = "#1d2433", "#5b6475"
DEV = ("#e6f2ea", "#2f7d4f")      # runs on the Pi
CLOUD = ("#e8eefb", "#3559b7")    # outside service
MCP = ("#fdf0e1", "#b86a12")      # our MCP servers
PIPE = ("#ffffff", "#47506a")     # pipecat processors
G = "#9aa3b5"


def box(x, y, w, h, title, lines=(), style=PIPE, r=14):
    fill, edge = style
    d.rounded_rectangle([x, y, x + w, y + h], r, fill=fill, outline=edge, width=3)
    d.text((x + 14, y + 10), title, font=f_h, fill=edge if style != PIPE else INK)
    yy = y + 40
    for ln in lines:
        d.text((x + 14, yy), ln, font=f_b, fill=INK)
        yy += 23
    return (x, y, w, h)


def seg(p, q, color, width, dash):
    if not dash:
        d.line([p, q], fill=color, width=width)
        return
    n = max(2, int(math.dist(p, q) // 12))
    for i in range(0, n, 2):
        a = (p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
        b = (p[0] + (q[0] - p[0]) * (i + 1) / n, p[1] + (q[1] - p[1]) * (i + 1) / n)
        d.line([a, b], fill=color, width=width)


def head(a, b, color):
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    L = 16
    d.polygon([b, (b[0] - L * math.cos(ang - 0.45), b[1] - L * math.sin(ang - 0.45)),
               (b[0] - L * math.cos(ang + 0.45), b[1] - L * math.sin(ang + 0.45))], fill=color)


def path(pts, color="#47506a", width=4, dash=False, start_head=False):
    for a, b in zip(pts, pts[1:]):
        seg(a, b, color, width, dash)
    head(pts[-2], pts[-1], color)
    if start_head:
        head(pts[1], pts[0], color)


# ---------- title
d.text((40, 26), "FDA: full-duplex voice agent pipeline", font=f_title, fill=INK)
d.text((42, 80), "Pipecat 1.15 agent harness + three MCP servers we built (ASR, TTS, Tools), all running on the Raspberry Pi 5. "
       "Cloud models are optional; every chain ends offline.", font=f_sub, fill=MUTED)

# ---------- row 1: listening (left -> right)
y1, h1 = 150, 190
d.text((40, y1 - 4), "LISTENING", font=f_sec, fill=MUTED)
b_mic = box(40, y1 + 24, 190, h1, "USB mic", ["16 kHz audio", "always on, even", "while FDA talks"], DEV)
b_aec = box(270, y1 + 24, 260, h1, "Echo canceller", ["WebRTC AEC3 removes", "FDA's own voice", "+ own-voice gate"], DEV)
b_vad = box(570, y1 + 24, 270, h1, "Voice detection", ["Silero VAD (speech?)", "Smart Turn v3", "(has the person", "finished the turn?)"], DEV)
b_dg = box(880, y1 + 24, 300, h1, "DuplexGuard (barge-in)", ["ducks FDA's voice at once", "peeks at 1.0 / 1.6 s:",
                                                             "real words -> stop FDA", "echo / 'uh huh' -> carry on"])
b_stt = box(1220, y1 + 24, 250, h1, "Speech to text", ["sends the turn to", "the ASR MCP server", "(far-away speech", "is ignored)"])
b_heard = box(1510, y1 + 24, 450, h1, "HeardTap", ["'stop' / 'wait' -> silence at once",
                                                  "'uh huh' while talking -> ignored",
                                                  "said to someone else -> ignored",
                                                  "'continue' -> resumes the cut sentence",
                                                  "'bird -> butterfly' -> instant live tweak"])
for a, b in [(b_mic, b_aec), (b_aec, b_vad), (b_vad, b_dg), (b_dg, b_stt), (b_stt, b_heard)]:
    path([(a[0] + a[2], a[1] + a[3] / 2), (b[0] - 2, b[1] + b[3] / 2)])

# ---------- row 2: thinking + speaking (right -> left)
y2, h2 = 450, 190
d.text((40, y2 - 4), "THINKING AND SPEAKING", font=f_sec, fill=MUTED)
b_llm = box(1510, y2 + 24, 450, h2, "Agent LLM (thinking + tool calls)", ["Groq gpt-oss-120b (~0.5-0.9 s)",
                                                                          "rate limit -> gpt-oss-20b -> Qwen",
                                                                          "-> SambaNova -> Gemini (automatic)",
                                                                          "offline: Qwen2.5-3B on the device"], CLOUD)
b_nfm = box(1150, y2 + 24, 320, h2, "NotForMe + ToolFiller", ["'(silent)' when not for FDA", "drops repeated replies",
                                                              "'Let me check' while a", "slow tool runs"])
b_tts = box(860, y2 + 24, 250, h2, "Text to speech", ["each sentence to the", "TTS MCP server,", "streams as it comes"])
b_spoken = box(560, y2 + 24, 260, h2, "SpokenTap", ["tracks what was said", "(for 'continue')", "shows the chat", "on the display"])
b_spk = box(270, y2 + 24, 250, h2, "Speaker + display", ["USB speaker", "7-inch screen: chat,", "apps, progress bar"], DEV)
b_ctx = box(40, y2 + 24, 190, h2, "Context", ["conversation", "memory (recent", "turns + tools)"])
path([(b_heard[0] + b_heard[2] / 2, b_heard[1] + b_heard[3]), (b_llm[0] + b_llm[2] / 2, b_llm[1] - 2)])
for a, b in [(b_llm, b_nfm), (b_nfm, b_tts), (b_tts, b_spoken), (b_spoken, b_spk), (b_spk, b_ctx)]:
    path([(a[0], a[1] + a[3] / 2), (b[0] + b[2] + 2, b[1] + b[3] / 2)])
# the context goes back into the LLM on the next turn: out of its left side, along the gap above row 2
cy = b_ctx[1] + b_ctx[3] / 2
path([(b_ctx[0], cy), (20, cy), (20, 418), (1490, 418), (1490, b_llm[1] + 30), (b_llm[0] - 2, b_llm[1] + 30)],
     color=G, width=3)
d.text((560, 392), "next turn: the LLM sees what was really said (cut-off replies are trimmed)", font=f_b, fill=MUTED)

# ---------- row 3: our MCP servers
y3 = 760
d.text((40, y3 - 4), "OUR MCP SERVERS (on the Pi)", font=f_sec, fill=MUTED)
b_asr = box(40, y3 + 24, 560, 250, "ASR server :8101   tools: transcribe, asr_info", [
    "Chain, first one that works (failed engine skipped 30 s):",
    "  Whisper large-v3-turbo (Groq)   - optional",
    "  Qwen2-Audio-7B (school DGX3)     - via VPN",
    "  Moonshine v2 / Parakeet 110M or 0.6B  - offline",
    "Cleans the model's answer to just the words.",
    "Always ends offline: listening never stops."], MCP)
b_ttsm = box(640, y3 + 24, 520, 250, "TTS server :8102   tools: synthesize, tts_info", [
    "Chain:",
    "  Qwen3-TTS-1.7B (school DGX3), voice 'aiden'",
    "  Deepgram Aura   - optional",
    "  Kokoro-82M  - offline, natural (laptop)",
    "  Piper  - offline, fast (Pi)",
    "Returns 16-bit audio + its sample rate."], MCP)
b_tools = box(1200, y3 + 24, 760, 250, "Tools server :8103   (real-world tasks)", [
    "Apps & games: create_app, edit_app, tweak_app (live, while playing),",
    "   behaviour tests in a real DOM (jsdom) + auto-fix, git version history,",
    "   undo / restore_app_version.   Coder: Groq -> SambaNova -> Gemini",
    "Documents: ask_documents (PDF, Word, Excel, CSV, Markdown; BM25, offline)",
    "Device: device_check (ports, updates, SSH, disk, heat, power; read-only)",
    "Everyday: weather, news, recipes, web search, time, calculator"], MCP)
# MCP calls (dashed, both ways), routed through the gaps between boxes
xs = b_stt[0] + 125
path([(xs, b_stt[1] + b_stt[3] + 2), (xs, 400), (1130, 400), (1130, 720), (520, 720), (520, b_asr[1] - 2)],
     color=MCP[1], width=3, dash=True, start_head=True)
path([(b_tts[0] + 125, b_tts[1] + b_tts[3] + 2), (b_tts[0] + 125, b_ttsm[1] - 2)], color=MCP[1], width=3, dash=True,
     start_head=True)
path([(1800, b_llm[1] + b_llm[3] + 2), (1800, b_tools[1] - 2)], color=MCP[1], width=3, dash=True, start_head=True)
d.text((532, 730), "MCP: transcribe", font=f_b, fill=MCP[1])
d.text((b_tts[0] + 137, 730), "MCP: synthesize", font=f_b, fill=MCP[1])
d.text((1812, 700), "MCP: tool calls", font=f_b, fill=MCP[1])

# ---------- key
yl = 1085
d.text((40, yl), "KEY", font=f_sec, fill=MUTED)
for i, (style, label) in enumerate([(DEV, "runs on the Pi (hardware / on-device models)"), (PIPE, "Pipecat processor (our code)"),
                                    (MCP, "MCP server we built"), (CLOUD, "cloud model (optional, has an offline fallback)")]):
    x = 40 + i * 480
    d.rounded_rectangle([x, yl + 32, x + 40, yl + 62], 8, fill=style[0], outline=style[1], width=3)
    d.text((x + 52, yl + 35), label, font=f_b, fill=INK)
d.line([(40, yl + 100), (110, yl + 100)], fill="#47506a", width=4)
d.text((122, yl + 88), "audio / text frames", font=f_b, fill=INK)
seg((520, yl + 100), (590, yl + 100), MCP[1], 3, True)
d.text((602, yl + 88), "MCP calls (official MCP Python SDK, streamable HTTP)", font=f_b, fill=INK)
d.line([(1100, yl + 100), (1170, yl + 100)], fill=G, width=3)
d.text((1182, yl + 88), "conversation memory (next turn)", font=f_b, fill=INK)
d.text((40, yl + 140), "Measured on the laptop: reply ~0.5-0.9 s (Groq), voice first sound ~0.1 s (DGX3), listening 60-400 ms, "
       "a full game ~10 s, live tweak < 1 s, 'stop' interrupts at once.", font=f_b, fill=MUTED)
d.text((40, yl + 168), "Barge-in: VAD ducks the voice at once; stop words stop it at any length; 'continue' resumes the cut sentence "
       "without asking the LLM again.", font=f_b, fill=MUTED)

img.save(OUT)
print(OUT)
