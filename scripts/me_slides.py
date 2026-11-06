"""Machine exercise 2-page template, filled for Elkit (FDA): python scripts/me_slides.py -> report/machine_exercise_2page.pptx

Follows the class template (sections 1-6). Mean / p95 cells and the RPi state stay blank until the Raspberry Pi run.
Edit the text here and run it again; the .pptx is generated, not edited by hand."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

OUT = Path(__file__).resolve().parents[1] / "report" / "machine_exercise_2page.pptx"
NAVY, ORANGE, TEAL = RGBColor(0x17, 0x23, 0x3F), RGBColor(0xE8, 0x76, 0x2D), RGBColor(0x1A, 0x7F, 0x8E)
INK, GRAY, LIGHT = RGBColor(0x22, 0x2B, 0x3A), RGBColor(0x8A, 0x93, 0xA3), RGBColor(0xEE, 0xF1, 0xF5)
WHITE, LINE, CLOUD = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xC5, 0xCC, 0xD6), RGBColor(0xFD, 0xF1, 0xE8)
PIBG = RGBColor(0xE8, 0xF4, 0xF5)
SANS, MONO = "Arial", "Courier New"
REPORT = OUT.parent / "latency.json"        # written by scripts/latency_report.py


def pi_report() -> dict | None:
    """The latency report, only if it comes from the Raspberry Pi (laptop numbers stay off the deck)."""
    try:
        rep = json.loads(REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rep if "raspberry" in (rep.get("device") or "").lower() else None


def metric(rep: dict | None, prefix: str) -> tuple[str, str]:
    """(mean, p95) cells for one metric of the report, '' when there is no Pi report yet."""
    if not rep:
        return "", ""
    if prefix == "WER":
        w = (rep.get("wer") or {}).get("wer")
        return (f"{100 * w:.1f} %" if w is not None else "-"), "-"
    s = next((v for k, v in rep.get("metrics_s", {}).items() if k.startswith(prefix)), None) or {}
    f = lambda v: f"{v:.2f} s" if v is not None else "-"   # noqa: E731
    return f(s.get("mean")), f(s.get("p95"))


def text(sl, x, y, w, h, paras, size=10, color=INK, bold=False, font=SANS, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, margin=0.0, bullet_color=None, space=0):
    """paras: str | list of str | list of list of (text, {opts}) runs. A paragraph starting with '- ' gets a dash."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if space:
            para.space_after = Pt(space)
        runs = p if isinstance(p, list) else [(p, {})]
        if isinstance(p, str) and p.startswith("- "):
            runs = [("- ", {"color": bullet_color or ORANGE, "bold": True}), (p[2:], {})]
        for t, o in runs:
            r = para.add_run()
            r.text = t
            f = r.font
            f.name, f.size = o.get("font", font), Pt(o.get("size", size))
            f.bold, f.italic = o.get("bold", bold), o.get("italic", italic)
            f.color.rgb = o.get("color", color)
    return tb


def box(sl, x, y, w, h, fill=None, line=None, dash=False, shape=MSO_SHAPE.RECTANGLE, lw=0.75):
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def node(sl, x, y, w, h, title, sub="", fill=WHITE, line=TEAL, tcolor=INK):
    box(sl, x, y, w, h, fill, line, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    paras = [[(title, {"bold": True, "size": 8.5, "color": tcolor})]]
    if sub:
        paras.append([(sub, {"size": 7.5, "color": GRAY if tcolor == INK else tcolor})])
    text(sl, x, y, w, h, paras, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.03)


def arrow(sl, x1, y1, x2, y2, color=INK, both=False, dash=False):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.1)
    if dash:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = c.line._get_or_add_ln()
    for tag in (["a:headEnd"] if both else []) + ["a:tailEnd"]:
        e = etree.SubElement(ln, qn(tag))
        e.set("type", "triangle")
        e.set("w", "sm")
        e.set("len", "sm")
    return c


def header(sl, kicker, title, right):
    box(sl, 0, 0, 13.333, 1.05, NAVY)
    box(sl, 0, 1.05, 13.333, 0.06, ORANGE)
    text(sl, 0.6, 0.22, 9, 0.25, kicker, size=10, color=ORANGE, bold=True)
    text(sl, 0.6, 0.45, 9.5, 0.5, title, size=24, color=WHITE, bold=True)
    text(sl, 8.8, 0.5, 3.95, 0.35, right, size=11, color=RGBColor(0xC9, 0xD1, 0xDE), italic=True, align=PP_ALIGN.RIGHT)


def section(sl, x, y, n, title, note=""):
    box(sl, x, y + 0.02, 0.27, 0.27, ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(sl, x, y + 0.02, 0.27, 0.27, str(n), size=11, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    runs = [(title, {"bold": True, "size": 15, "color": NAVY})]
    if note:
        runs.append(("    " + note, {"italic": True, "size": 10, "color": GRAY}))
    text(sl, x + 0.38, y - 0.02, 7, 0.35, [runs], anchor=MSO_ANCHOR.MIDDLE)


def footer(sl, page):
    box(sl, 0.6, 7.08, 12.13, 0.01, LINE)
    text(sl, 0.6, 7.13, 4.5, 0.25, "Machine Exercise - Full-Duplex Voice Agent on Raspberry Pi", size=8.5, color=GRAY)
    text(sl, 4.6, 7.13, 4.2, 0.25, "Elkit (FDA)  ·  github.com/arjAIareNOW/mex3", size=8.5, color=GRAY,
         italic=True, align=PP_ALIGN.CENTER)
    text(sl, 9.5, 7.13, 3.23, 0.25, f"2-page template · {page}/2", size=8.5, color=GRAY, bold=True,
         align=PP_ALIGN.RIGHT)


def table(sl, x, y, widths, rows, row_h=0.26, size=8.5, head_fill=NAVY, blank_cols=()):
    t = sl.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)),
                            Inches(row_h * len(rows))).table
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = head_fill if i == 0 else (WHITE if i % 2 else LIGHT)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = val or " "          # an empty cell would take PowerPoint's 18 pt default height
            r.font.name, r.font.size = SANS, Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = WHITE if i == 0 else INK
    tblPr = t._tbl.tblPr
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "0")
    return t


# --------------------------------------------------------------------------------------------- page 1
def page1(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    header(sl, "MACHINE EXERCISE - ELKIT (FDA) v2.8", "Full-Duplex Voice Agent on a Raspberry Pi",
           "Raspberry Pi 5 · Pipecat 1.15 · MCP")

    section(sl, 0.6, 1.3, 1, "SYSTEM DIAGRAM", "the one diagram that explains the build")
    # boundaries
    box(sl, 0.6, 1.72, 4.75, 2.42, PIBG, TEAL, dash=True)
    text(sl, 0.7, 1.75, 4, 0.2, "RASPBERRY PI 5  (on device)", size=8, color=TEAL, bold=True)
    box(sl, 5.55, 1.72, 2.05, 2.42, CLOUD, ORANGE, dash=True)
    text(sl, 5.62, 1.75, 1.95, 0.2, "CLOUD  (APIs)", size=8, color=ORANGE, bold=True)
    # our MCP servers: one column, each in line with the cloud engine it uses
    box(sl, 3.74, 1.93, 1.56, 2.15, None, ORANGE, dash=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    node(sl, 3.82, 2.0, 1.4, 0.46, "asr-mcp :8101", "transcribe()", line=ORANGE)
    node(sl, 3.82, 3.1, 1.4, 0.42, "tts-mcp :8102", "synthesize()", line=ORANGE)
    node(sl, 3.82, 3.6, 1.4, 0.42, "tools-mcp :8103", "23 tools: apps, web", line=ORANGE)
    text(sl, 3.74, 2.52, 1.56, 0.2, "OUR MCP SERVERS", size=7, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    # audio in -> VAD -> guard -> asr-mcp
    node(sl, 0.72, 2.0, 0.62, 0.46, "Mic", "USB")
    node(sl, 1.5, 2.0, 0.9, 0.46, "Silero VAD", "16 kHz mono")
    node(sl, 2.56, 2.0, 1.0, 0.46, "DuplexGuard", "barge-in")
    arrow(sl, 1.34, 2.23, 1.5, 2.23)
    arrow(sl, 2.4, 2.23, 2.56, 2.23)
    arrow(sl, 3.56, 2.23, 3.82, 2.23)
    # agent: MCP client to the three servers, LLM by API
    node(sl, 0.72, 2.62, 2.84, 0.42, "Pipecat agent  (MCP client)", "context · room tools · display :8765",
         fill=NAVY, line=NAVY, tcolor=WHITE)
    arrow(sl, 3.06, 2.46, 3.06, 2.62)
    # voice out
    node(sl, 0.72, 3.1, 1.5, 0.42, "Speaker", "stops on barge-in")
    arrow(sl, 3.82, 3.31, 2.22, 3.31)
    text(sl, 0.72, 3.62, 2.9, 0.4, "agent calls asr / tts / tools over MCP (HTTP)\noffline on the Pi: Moonshine ASR · Piper TTS",
         size=7.5, color=TEAL, italic=True)
    # cloud
    node(sl, 5.65, 2.0, 1.85, 0.46, "Groq Whisper", "large-v3-turbo (+v3)", line=ORANGE)
    node(sl, 5.65, 2.62, 1.85, 0.42, "Groq LLM", "gpt-oss-120b", line=ORANGE)
    node(sl, 5.65, 3.1, 1.85, 0.42, "DGX3 Qwen3-TTS", "school GPU, VPN", line=ORANGE)
    node(sl, 5.65, 3.6, 1.85, 0.42, "Web APIs", "weather · search · news", line=ORANGE)
    arrow(sl, 5.22, 2.23, 5.65, 2.23, both=True)
    arrow(sl, 3.56, 2.83, 5.65, 2.83, both=True)
    arrow(sl, 5.22, 3.31, 5.65, 3.31, both=True)
    arrow(sl, 5.22, 3.81, 5.65, 3.81, both=True)

    box(sl, 0.6, 4.24, 7.0, 0.32, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(sl, 0.6, 4.24, 7.0, 0.32,
         "voice in -> Silero VAD -> asr-mcp (Whisper) -> gpt-oss-120b (+tools-mcp) -> tts-mcp -> voice out",
         size=8.5, color=WHITE, font=MONO, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 0.6, 4.58, 7.0, 0.25,
         "barge-in: VAD start -> duck -> ASR peek (2+ words) -> stop TTS + cancel LLM -> listen",
         size=8.5, color=ORANGE, bold=True, font=MONO, align=PP_ALIGN.CENTER)

    text(sl, 7.95, 1.75, 4.8, 3.1, [
        "- Mic -> Silero VAD (on the RPi) -> asr-mcp -> Pipecat agent -> tts-mcp -> speaker",
        "- Harness: Pipecat 1.15 with an MCP client (streamable HTTP)",
        "- LLM: Groq gpt-oss-120b (commercial API, streaming), fallback gpt-oss-20b; app coder qwen3.8-27b",
        "- Our 3 MCP servers, built by us, run on the RPi: asr :8101, tts :8102, tools :8103",
        "- Barge-in cancels TTS playback and the LLM stream; \"mm-hm\", side talk and TV are not barge-in",
        "- Each engine has a fallback: Whisper v3, Piper, Moonshine; Elkit keeps talking offline",
    ], size=10.5, space=5)

    section(sl, 0.6, 4.98, 2, "PROMPT", "kept in the repo, versioned (fda/pc/agent.py SYSTEM)")
    box(sl, 0.6, 5.36, 7.0, 1.62, LIGHT, LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(sl, 0.75, 5.42, 6.75, 1.5, [
        [("SYSTEM PROMPT (v2.8) - excerpt", {"bold": True, "color": TEAL})],
        "You are {name}, a voice assistant on a small screen. You speak out",
        "loud in real time: warm, relaxed and natural, usually one or two",
        "short sentences. Everything you write is spoken: no lists, no markdown.",
        "- If they interrupt you, drop what you were saying and answer what",
        "  they just said.",
        "- Do what the person asks, nothing more. After a task, a short",
        "  confirmation is enough. The links show on the screen by themselves.",
        "- A message ending in [now: ...] already has the true time and date:",
        "  use it, never call get_time then.",
    ], size=8, font=MONO, bullet_color=INK)

    section(sl, 7.95, 4.98, 3, "CONTEXT ENGINEERING")
    text(sl, 7.95, 5.36, 2.35, 0.22, "GOES IN:", size=10, color=TEAL, bold=True)
    text(sl, 7.95, 5.6, 2.35, 1.45, [
        "- system prompt + persona + few-shot",
        "- final ASR text per turn, stamped [now: date, time]",
        "- turn events: interrupted, ignored side talk",
        "- MCP tool results + room state",
        "- notes: \"they changed their mind\"",
    ], size=8.5, space=2)
    text(sl, 10.4, 5.36, 2.35, 0.22, "MANAGED:", size=10, color=TEAL, bold=True)
    text(sl, 10.4, 5.6, 2.35, 1.45, [
        "- few-shot + last 12 user turns kept",
        "- stable prefix first (Groq prompt cache)",
        "- idle -> listen -> think -> speak on the display",
        "- on barge-in: unsaid reply + stale audio dropped",
        "- per turn: stage timings + context size logged",
    ], size=8.5, space=2)
    footer(sl, 1)


# --------------------------------------------------------------------------------------------- page 2
def page2(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    header(sl, "MACHINE EXERCISE - ELKIT (FDA) v2.8", "MCP Services, Evaluation & Model Serving",
           "build + run servers on the RPi")

    section(sl, 0.6, 1.3, 4, "MCP SERVICES", "+ section 6: model serving (one table, one truth)")
    table(sl, 0.6, 1.68, [0.85, 2.05, 2.85, 3.2, 3.18], [
        ["Component", "MCP server / transport", "Tools", "Model", "Served where / notes"],
        ["ASR", "asr-mcp / HTTP :8101", "transcribe(audio) -> text, asr_info()",
         "Groq Whisper large-v3-turbo -> large-v3 -> Moonshine", "RPi server; final on VAD end; FLAC upload; backup after 0.7 s"],
        ["TTS", "tts-mcp / HTTP :8102", "synthesize(text, voice, speed) -> audio",
         "DGX3 Qwen3-TTS -> Piper (offline)", "RPi server; sentence by sentence, cancelled on barge-in"],
        ["LLM", "(Pipecat MCP client)", "function calling: MCP + room tools",
         "Groq gpt-oss-120b -> gpt-oss-20b", "commercial API; streaming tokens"],
        ["VAD", "in-process", "VADProcessor(frames)", "Silero VAD (0.6, start 0.15 s, stop 0.2 s)",
         "RPi on-device, 16 kHz mono"],
        ["Task", "tools-mcp / HTTP :8103", "create_app, edit_app, get_weather, web_search, get_news, play_video ... (23)",
         "app coder: Groq qwen3.8-27b; web APIs", "RPi server; room tools (lights, timers) in the agent"],
    ], row_h=0.3, size=8.5)

    section(sl, 0.6, 3.68, 5, "EVALUATION & BENCHMARK", "standardized latency method (classwork)")
    text(sl, 0.6, 4.08, 6.4, 1.6, [
        "- One clock: RPi monotonic; Pipecat Mark processors stamp every stage of each turn",
        "- VAD end -> ASR final -> LLM 1st token -> TTS 1st audio -> speaker (\"turn timing\" log)",
        "- E2E = first audio - VAD end; barge-in = playback stopped - user speech start",
        "- fd-benchmark: 42 clips (14 tasks x 3 rounds, seed 11273), same clips every run; "
        "mean + p95 over all spoken turns",
        "- WER: what Elkit heard vs each clip's transcript (normalised, word Levenshtein)",
        "- report: scripts/latency_report.py <run> -> latency_report.json",
    ], size=9.5, space=3)
    rep = pi_report()
    table(sl, 7.3, 4.05, [1.05, 2.4, 0.75, 0.6, 0.62], [
        ["Metric", "Definition", "Target", "Mean", "p95"],
        ["E2E latency", "end of speech -> audio out", "< 1.0 s", *metric(rep, "E2E")],
        ["ASR final", "end of speech -> transcript", "< 0.5 s", *metric(rep, "ASR")],
        ["LLM TTFT", "turn end -> first token", "< 0.5 s", *metric(rep, "LLM")],
        ["TTS TTFB", "text -> first audio chunk", "< 0.2 s", *metric(rep, "TTS")],
        ["Barge-in", "speech -> playback stopped", "< 1.0 s", *metric(rep, "Barge")],
        ["WER", "ASR accuracy on the clip set", "< 10 %", *metric(rep, "WER")],
    ], row_h=0.235, size=8.5)
    if rep:
        w = rep.get("wer") or {}
        results = [
            f"- {rep['device']}",
            f"- cooling: {rep.get('cooling') or '-'} · network: {rep.get('network') or '-'}",
            f"- run {rep['run']} · report {rep['date']} · WER {w.get('errors')}/{w.get('ref_words')} words",
            "- models: Whisper large-v3-turbo, gpt-oss-120b,",
            "  Qwen3-TTS / Piper, Silero VAD, Pipecat 1.15"]
    else:
        results = [
            [("Mean / p95: filled from the RPi run", {"italic": True, "color": GRAY})],
            [("RPi 5 · ___ GB RAM · cooling ___ · network ___ · date ___", {"italic": True, "color": GRAY})],
            "- models: Whisper large-v3-turbo, gpt-oss-120b,",
            "  Qwen3-TTS / Piper, Silero VAD, Pipecat 1.15",
            [("latency chart: per-stage bars, mean + p95", {"italic": True, "color": GRAY})]]

    for x, title, body in (
        (0.6, "DEMO (2-3 min)", [
            "- \"Turn the lights blue... wait, no, red\" (change of mind)",
            "- \"Weather in Quezon City?\" live tools-mcp call",
            "- talk over it: stops; \"mm-hm\" and side talk do not",
            "- \"Build me a snake game\": create_app on screen",
            "- recovery: VPN off -> Piper voice; slow Groq -> v3 backup"]),
        (4.72, "RESULTS (RPi run)" if rep else "RESULTS (after the RPi run)", results),
        (8.84, "CHALLENGES & REFERENCES", [
            "- flat 0.8 s turn wait -> final transcript + wait by text: turn-taking 4.30 -> 1.44 s*",
            "- late barge-in stop -> ASR peeks at 0.5/0.9/1.3 s: stop 2.2 -> 1.6-1.9 s*",
            "- deaf after a test video, slow lookups -> fixes: failed trials 12 -> 6 of 42*",
            [("* laptop, fd-benchmark. refs: modelcontextprotocol.io, docs.pipecat.ai, fd-benchmark, repo",
              {"size": 7.5, "color": GRAY, "italic": True})]]),
    ):
        box(sl, x, 5.82, 3.9, 1.18, LIGHT, LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        text(sl, x + 0.12, 5.86, 3.7, 0.22, title, size=10, color=TEAL, bold=True)
        text(sl, x + 0.12, 6.08, 3.7, 0.9, body, size=8, space=1)
    footer(sl, 2)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "Elkit - Full-Duplex Voice Agent on a Raspberry Pi"
    page1(prs)
    page2(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
