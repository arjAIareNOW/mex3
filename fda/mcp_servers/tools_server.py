"""MCP server: the agent's real-world tools. Every tool here is free and needs no API key.

    python -m fda.mcp_servers.tools_server                     # HTTP on 127.0.0.1:8103/mcp (default)

Tools
  create_app(description)        build a game / website / small tool as one HTML file, shown on the display
  edit_app(change, name="")      change the app that is open (or a named one): colours, rules, text, layout
  list_apps() / open_app(name)   the apps built so far
  app_status()                   is a build still running (and how far)
  get_weather(place)             Open-Meteo                     (free, no key)
  web_search(query)              DuckDuckGo                     (free, no key)
  get_news(topic="")             Google News RSS                (free, no key)
  find_recipe(dish)              TheMealDB                      (free, no key)
  calculate(expression)          safe arithmetic
  wikipedia(topic)               Wikipedia summary              (free, no key)
  read_page(link)                the text of a page from web_search (details the snippets lack)
  show_photo(what)               photos from the web for the display (Openverse, free licences, no key)
  play_video(what)               a YouTube video for the display (YouTube search page + oEmbed, free, no key)

Building an app takes an LLM writing ~150-300 lines of HTML. Who writes it is `coder_engine` (fda.toml):
  local  llama.cpp on this machine (offline_llm, or "qwen2.5-coder-3b" if downloaded): free, ~1-3 min on a laptop,
         several minutes on a Pi
  groq   the main provider: qwen3.8-27b, one fallback gpt-oss-120b, ~5-10 s (GROQ_API_KEY in .env)
  openai any OpenAI-compatible server (LLM_BASE_URL, LLM_API_KEY)
Builds run in the background: create_app returns at once and the agent says when the app is ready.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import operator
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field

from mcp.server.mcpserver import MCPServer

from ..config import ROOT, Config
from . import live
from . import photos
from . import llm as llmlib

log = logging.getLogger("fda.mcp.tools")

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
       81: "rain showers", 82: "heavy rain showers", 95: "thunderstorms", 96: "thunderstorms with hail",
       99: "thunderstorms with hail"}

def problem_rank(err: str | None) -> float:
    """How bad a build problem is: 0 none, 1.x some behaviour tests fail (x = how many), 2 it runs but something is
    not drawn with FDA.draw / nothing moves, 3 it does not run (syntax error, crash)."""
    if not err:
        return 0.0
    if err.startswith("Behaviour test failed"):
        return 1 + min(err.count("expected") + err.count("not valid JSON"), 9) / 10
    if err.startswith(("Syntax error", "When it runs")):
        return 3.0
    return 2.0


_CENTRE_DRAW = re.compile(r"FDA\.draw\(\s*ctx\s*,\s*(\"[^\"]+\"|'[^']+')\s*,\s*(\w+)\.x\s*,\s*\2\.y\s*,\s*"
                          r"\2\.(r|radius)\s*\*\s*2\s*,\s*\2\.\3\s*\*\s*2\s*\)")


def centre_draw_fix(html: str) -> tuple[str, int]:
    """A thing kept by its centre (bird.x, bird.y, bird.r; the collision uses bird.x - bird.r) but drawn with
    FDA.draw(ctx, "bird", bird.x, bird.y, bird.r * 2, bird.r * 2) shows its picture r lower and r to the right
    of where it really is: "game over" while the bird you see is clear of the pipe. Draw it around its centre."""
    def fix(m):
        o, r = m.group(2), m.group(3)
        if not re.search(rf"\b{o}\.[xy]\s*[-+]\s*{o}\.{r}\b", html):
            return m.group(0)                          # not a centre-based thing: leave it
        return f"FDA.draw(ctx, {m.group(1)}, {o}.x - {o}.{r}, {o}.y - {o}.{r}, {o}.{r} * 2, {o}.{r} * 2)"
    out = _CENTRE_DRAW.sub(fix, html)
    return out, len(_CENTRE_DRAW.findall(html)) - len(_CENTRE_DRAW.findall(out))


def sprite_fixes(settings: dict) -> dict:
    """Settings named after a library sprite whose value is only a colour -> {name: the sprite's path}.
    ("bird": "#FFCC00" draws a yellow square, not a bird; "pipe": "#3a3" stays: pipes are not a sprite.)"""
    groups = live.manifest().get("groups", {})
    # characters and things to collect, not obstacles / plants (pipes, rocks and walls are fine as coloured bars)
    keep = {m for g, v in groups.items() if g not in ("obstacle", "plant", "sky", "scene", "ground")
            for m in (v["members"] if isinstance(v, dict) else v)}
    out = {}
    for k, v in (settings or {}).items():
        if not isinstance(v, str) or v.startswith(("data:", "/assets/", "http")) or live._is_sprite(v):
            continue                                   # a picture, emoji or photo already (or not text)
        pic = live.asset_for(k)
        if pic and "/sprites/" in pic and pic.rsplit("/", 1)[-1][:-4] in keep:
            out[k] = pic
    return out


def art_rule() -> str:
    """The app writer's note about Elkit's own art library (fda/assets), with every name in it ("" if absent)."""
    man = live.manifest()
    items = man.get("items")
    if not items:
        return ""
    names = {k: ", ".join(i["name"] for i in items if i["kind"] == k) for k in ("sprite", "tile", "background", "icon")}
    pals = "; ".join(f"{n}: " + " ".join(p.values()) for n, p in man.get("palettes", {}).items())
    return f"""
Art library (on this device, works offline; use it instead of emoji when the thing is in it):
- Sprites "/assets/sprites/NAME.svg" (64x64, they already face RIGHT: no "|flip"): {names["sprite"]}.
- Tiles "/assets/tiles/NAME.svg" (repeat for ground, walls, platforms): {names["tile"]}.
- Backgrounds "/assets/backgrounds/NAME.svg" (800x480): {names["background"]}.
- Icons "/assets/icons/NAME.svg" (dark line icons; "/assets/icons/white/NAME.svg" on dark colours): {names["icon"]}.
Games use these paths as setting values (bird: "/assets/sprites/bird.svg", sky: "/assets/backgrounds/sky-day.svg"),
drawn with FDA.draw like any other value; websites put icons in <img src="/assets/icons/NAME.svg" width="24">.
A creature or object the player sees (the bird, a fish, a car, the player, an enemy, a coin) is ALWAYS a picture
value (a library path above, or an emoji) - never a plain colour like "#ffcc00", which draws only a square.
Only the names listed exist.
Colour schemes (background, cards, main colour, accent, text) - when one is named, use exactly these colours as
CSS variables: {pals}."""


APP_SYSTEM = """You write small web apps for a 800x480 touch screen. Output ONE complete HTML file and nothing else:
<!DOCTYPE html> ... </html>, with all CSS in one <style> and all JavaScript in one <script>.
Rules:
- No external scripts, CDNs, fonts or network requests. Games: draw with canvas or CSS, emoji are fine.
- It must work at once: fill the window (100vw x 100vh, no scrollbars), large readable text, big touch targets.
- Games: one <canvas> filling the window, a requestAnimationFrame loop that moves things every frame. Start on a
  tap/click or Space, show the score, show "Game over - tap to restart" on the canvas (never alert/confirm).
  Work with mouse, touch AND keyboard. Anything that changes (positions, speed, score) is a let, never a const.
  Give every list and object its start value where it is declared, at the top (let pipes = [], stars = [...],
  food = {x: 5, y: 5}): draw() runs on the very first frame, before any reset() or tap.
- Games get their look AND feel from live settings that people change by voice while it runs ("make the snake
  blue", "faster", "make it easier", "a bigger ship", "jump higher", "give me three lives"). Name each setting
  after a thing in the game; its value is a colour, an emoji or a number. Every game has: the player, each kind of
  obstacle / enemy / thing to collect, the background, text, and the numbers speed, size (the player's size in
  px), lives (1 unless asked) and difficulty (2). Add the numbers its kind of game has, with exactly these names:
    jumping, flapping or falling: gravity, jump (a positive number), gap (room to pass through)
    shooting: fireRate (shots per second; the first shot comes the moment fire is pressed), bulletSpeed
    ball and paddle: ballSpeed, paddleSize
    snake: grow (segments per food)                          catching, dodging, waves of enemies: spawnRate (per second)
  A flappy game:  const C = FDA.defaults({bird: "🐦|flip", pipe: "#2e8b57", sky: "#87ceeb", ground: "#deb887",
    text: "#ffffff", speed: 3, gap: 160, size: 44, gravity: 0.5, jump: 7, lives: 1, difficulty: 2});
  A space shooter: const C = FDA.defaults({ship: "🚀", enemy: "👾", bullet: "#ffee55", sky: "#050518",
    text: "#ffffff", speed: 3, size: 48, fireRate: 4, bulletSpeed: 9, spawnRate: 1, lives: 3, difficulty: 2});
  Draw every thing with FDA.draw(ctx, "pipe", x, y, w, h) (it draws a colour, emoji or photo), the background with
  FDA.draw(ctx, "sky", 0, 0, W, H) first in every frame, text with ctx.fillStyle = FDA.color("text").
  Read every setting from C every frame (never copy one into a constant), so a change shows at once: the player's
  drawn box AND its hitbox follow C.size.
  difficulty (1 easy, 2 normal, 3 hard) changes the whole game at once, with const d = C.difficulty: every speed
  (speed, ballSpeed, bulletSpeed of enemies) * (0.5 + 0.25 * d), gaps and room to move * (1.4 - 0.2 * d), spawnRate
  * (0.5 + 0.25 * d), and the hitbox is (0.5 + 0.15 * d) of the drawn size (easy = more forgiving).
  lives: C.lives is how many lives a game STARTS with (never change C.lives in the game); reset() sets
  let lives = C.lives, and the game counts that down. Show them (e.g. "❤️ 3"); losing one that is not the last
  puts the player back safely (a short blink, the danger right next to it cleared) and play goes on; game over
  only at 0 lives. Keep the best score in
  localStorage and show it on the game-over screen.
  Fair collisions: FDA.draw's x, y is the TOP-LEFT corner. A thing kept by its centre (bird.x, bird.y, bird.r) is
  drawn with FDA.draw(ctx, "bird", bird.x - bird.r, bird.y - bird.r, bird.r * 2, bird.r * 2). Collisions use
  exactly the box that is drawn, times the hitbox factor, so a near miss is not a crash: a life is lost only when
  the player you SEE touches a danger you SEE (an obstacle is hit with its full drawn size, including a pipe's
  cap). Scenery (clouds, stars, decorations) never collides.
  Emoji animals and vehicles face LEFT in most fonts. If a thing looks or moves to the right (a flappy-style bird,
  a runner, a car driving right), add "|flip" to its value so FDA.draw mirrors it: bird: "🐦|flip".
  FDA is already loaded by the page before your script: just use it. NEVER define FDA, FDA.defaults or FDA.draw
  yourself (no "const FDA = ..."), or the live settings stop working.
- Websites: look like a real, polished business site. Use the real name, wording and details you are given.
  Pages: put each page (Home, About, Services, Gallery, Contact...) in its own <section class="page" id="...">, show
  one at a time, and make EVERY menu link and button work (switch page with location.hash / hashchange, highlight
  the active link). Write real content: headings, short paragraphs, prices or services, a contact form that shows
  a thank-you message.
  Photos: AT MOST 3 photos per page or change (more come later when they ask for more photos). A photo is
  <img data-photo="2-4 plain English words of what it shows, e.g. white summer dress" alt="..."> with NO src:
  real photos are filled in for you. The same words on several images give different photos of that thing.
  Optional data-shape="portrait" (products, people) or "landscape" (banners). Every <img> gets object-fit:cover
  and a fixed height. Never use any other photo website. Other pictures (more products, icons, decorations):
  the art library or emoji. Do not write photo credits; they are added for you. Keep existing photos as they are.
  "Add photos" / "real photos as placeholders for our products": give up to 3 of the page's product (or main)
  pictures a data-photo with that product's own words (the product name, e.g. "linen summer dress"), no src,
  keeping the layout; the rest stay as they are.
  Scrolling inside the page is fine for websites.
- Tools: a clean, modern single page (cards, rounded corners, soft shadows), working buttons/forms (save data in
  localStorage if useful).
- After the app's <script>, add 2-4 behaviour tests that prove the main features really work, as JSON:
  <script type="application/json" id="fda-tests">[
  {"name": "tap makes the bird go up", "setup": ["tap", "wait 5"], "before": "bird.y", "do": ["tap", "wait 3"],
   "expect": "bird.y < before"},
  {"name": "15% tip of 100 is 15", "do": ["set #bill 100", "click #tip15"],
   "expect": "document.getElementById('tip').textContent.includes('15')"}]</script>
  Steps: "tap", "space", "key ArrowLeft", "click #id", "set #id value", "wait N" (N frames), "run <js>" (set up a
  situation directly, e.g. "run pipes = [{x: bird.x - 60, gapY: bird.y, scored: false}]"). "before" and "expect"
  are JavaScript run in the app's page and can read its variables (`before` = the value of "before" just before
  the "do" steps). Games start in "ready": put a "tap" in "setup" to start playing. Before each test the game's
  reset() is called, so every test starts from a fresh game. Tests read the game's own variables (lives, not
  C.lives). A situation set up with "run" must be one the game's update code really reacts to: put things exactly
  where it checks (the same names, the same hitbox), and wait enough frames for things to move there.
- Put the app's short title in <title>.
- Plain modern JavaScript, no frameworks. Make sure every function you call exists. No comments.
- Keep it short: games and tools about 80-150 lines, websites up to about 250. Stop right after </html>.
Game skeleton (fill it in):
<canvas id="c"></canvas><script>
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
let W, H; function fit() { W = cv.width = innerWidth; H = cv.height = innerHeight; } fit(); addEventListener("resize", fit);
const C = FDA.defaults({player: "🙂", enemy: "#e04040", sky: "#102030", text: "#ffffff", speed: 3, size: 44, lives: 1, difficulty: 2});
let state = "ready", score = 0;
function reset() { score = 0; state = "play"; }
function action() { if (state !== "play") { reset(); return; } }
addEventListener("keydown", e => { if (e.code === "Space" || e.code === "ArrowUp") { e.preventDefault(); action(); } });
cv.addEventListener("pointerdown", e => { e.preventDefault(); action(); });
function update() { }
function draw() { FDA.draw(ctx, "sky", 0, 0, W, H); ctx.fillStyle = FDA.color("text"); ctx.font = "bold 40px sans-serif";
  ctx.textAlign = "center"; ctx.fillText(score, W / 2, 60);
  if (state !== "play") ctx.fillText(state === "ready" ? "Tap to start" : "Game over - tap to restart", W / 2, H / 2); }
function loop() { if (state === "play") update(); draw(); requestAnimationFrame(loop); }
loop();
</script>""" + art_rule()

EDIT_SYSTEM = APP_SYSTEM + """
You get the current HTML file and a requested change. Apply the change (and only that), keep everything else
working, and output the whole new HTML file."""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return "-".join(s.split("-")[:5]) or "app"


def extract_html(raw: str) -> str | None:
    """The HTML document inside the model's answer (with or without ``` fences); repaired if cut short."""
    m = re.search(r"```(?:html)?\s*(.*?)```", raw, re.S | re.I)
    text = m.group(1) if m else raw
    start = re.search(r"<!DOCTYPE html|<html", text, re.I)
    if not start:
        return None
    html = text[start.start():]
    end = html.lower().rfind("</html>")
    if end != -1:
        return html[:end + 7]
    for tag in ("script", "body", "html"):        # cut off by the token limit: close what is open
        if html.lower().count(f"<{tag}") > html.lower().count(f"</{tag}>"):
            html += f"\n</{tag}>"
    return html


def _looping(text: str, tail: int = 160, times: int = 3) -> bool:
    """Small models sometimes repeat the same lines forever: the last `tail` characters occur `times`+ times.
    Digits are ignored, so counting loops ("spawn67: 0.8, spawn68: 0.8, spawn69: ...") are caught too."""
    if len(text) < tail * times:
        return False
    if text.count(text[-tail:]) >= times:
        return True
    flat = re.sub(r"\d+", "#", text[-tail * times * 4:])
    return len(flat) >= tail * times and flat.count(flat[-tail:]) >= times


def _html_done_or_looping(text: str) -> bool:
    return "</html>" in text[-400:].lower() or _looping(text)


def check_html(html: str) -> list[str]:
    """Cheap sanity checks (the agent tells the user if the app may be broken)."""
    problems = []
    low = html.lower()
    if "<script" in low:
        js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S | re.I))
        for a, b in ("()", "{}", "[]"):
            if abs(js.count(a) - js.count(b)) > 2:
                problems.append(f"unbalanced {a}{b} in the script")
    if re.search(r"<script[^>]+src=|<link[^>]+href=\"http|@import", low):
        problems.append("loads external files")
    err = js_syntax_error(html)
    if err:
        problems.append(f"JavaScript syntax error: {err}")
    return problems


def js_syntax_error(html: str) -> str | None:
    """`node --check` on the page's script when Node.js is installed (laptop; `sudo apt install nodejs` on a Pi)."""
    import shutil
    import subprocess
    import tempfile
    node = shutil.which("node")
    js = live.scripts(html)
    if not node or not js.strip():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(js)
    try:
        r = subprocess.run([node, "--check", f.name], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        Path(f.name).unlink(missing_ok=True)
    if r.returncode == 0:
        return None
    lines = [x for x in r.stderr.splitlines() if x.strip()]
    msg = next((x for x in lines if "Error" in x), lines[-1] if lines else "syntax error")
    where = next((x for x in lines if re.search(r":\d+$", x)), "")
    line_no = where.rsplit(":", 1)[-1] if where else ""
    return f"{msg.strip()}" + (f" (script line {line_no})" if line_no else "")


class Apps:
    """Builds and keeps the apps. One build at a time (it is the heavy part)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = Path(cfg.apps_dir) if Path(cfg.apps_dir).is_absolute() else ROOT / cfg.apps_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.coder = llmlib.make(cfg, cfg.coder_engine if cfg.coder_engine != "same" else cfg.llm_engine,
                                 role="coder")
        # cloud coders get more room (a whole game); a local 3B model loops if given too much
        self.max_tokens = 4000 if self.coder.engine == "local" else 12000
        log.info("app coder: %s / %s", self.coder.engine, self.coder.model)
        self.job: dict | None = None
        self.lock = threading.Lock()
        self.current: str | None = None       # the app shown last (edit_app's default)
        self.done: list[dict] = []            # finished builds the agent has not announced yet

    # ---------------- version history (git in the apps folder) ----------------
    def _git(self, *args: str) -> str | None:
        import shutil
        import subprocess
        if not shutil.which("git"):
            return None
        try:
            r = subprocess.run(["git", "-c", "user.name=Elkit", "-c", "user.email=elkit@localhost", "-C", str(self.dir),
                                *args], capture_output=True, text=True, encoding="utf-8", timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout if r.returncode == 0 else None

    def commit(self, p: Path, message: str):
        """Save this app's current version in the history (git), e.g. "build: snake game" or "tweak: bird -> 🦋"."""
        try:
            if not (self.dir / ".git").exists():
                if self._git("init", "-q") is None:
                    return
                (self.dir / ".gitignore").write_text("*.prev.html\n*.tmp\n", encoding="utf-8")
            files = [p.name] + ([p.with_suffix(".json").name] if p.with_suffix(".json").exists() else [])
            self._git("add", "--", *files)
            self._git("commit", "-q", "-m", message[:120], "--", *files)
        except Exception as e:  # noqa: BLE001
            log.warning("git commit failed: %s", e)

    def history(self, p: Path, n: int = 10) -> list[dict]:
        out = self._git("log", f"-{n}", "--format=%h|%cr|%s", "--", p.name) or ""
        return [dict(zip(("id", "when", "what"), line.split("|", 2))) for line in out.splitlines() if line.count("|") >= 2]

    def restore(self, p: Path, steps_back: int) -> dict:
        hist = self.history(p, 50)
        if steps_back < 1 or steps_back >= len(hist):
            return {"ok": False, "message": f"There are only {max(0, len(hist) - 1)} earlier versions."}
        target = hist[steps_back]
        for f in (p.name, p.with_suffix(".json").name):
            if self._git("checkout", target["id"], "--", f) is None and f == p.name:
                return {"ok": False, "message": "I couldn't restore that version."}
        self.commit(p, f"restore: back to '{target['what']}' ({target['when']})")
        self.current = p.stem
        return {"ok": True, "url": f"/apps/{p.name}", "message": f"Back to the version from {target['when']}: {target['what']}.",
                "settings": self.describe(self._meta(p).get("live") or {})}

    def path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.html"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = self._meta(p)
            out.append({"name": p.stem, "title": meta.get("title", p.stem), "url": f"/apps/{p.name}",
                        "about": meta.get("description", "")})
        return out

    def _meta(self, p: Path) -> dict:
        m = p.with_suffix(".json")
        try:
            return json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def describe(settings: dict) -> str:
        """'bird=..., pipe=#2e8b57, speed=3' (photos shown as 'photo') for the agent."""
        return ", ".join(f"{k}={'photo' if isinstance(v, str) and v.startswith(('data:', '(photo')) else v}"
                         for k, v in settings.items())

    def tweak(self, changes: dict, name: str, web) -> dict:
        p = self.find(name)
        if p is None:
            return {"ok": False, "message": "There is no app to change yet."}
        meta = self._meta(p)
        settings = dict(meta.get("live") or {})
        if not settings:
            return {"ok": False, "message": f"{meta.get('title', p.stem)} has no live settings; use edit_app."}
        patch, said, unknown = {}, [], []
        for k, v in (changes or {}).items():
            key = live.match_key(str(k), list(settings))
            if key is None:
                unknown.append(str(k))
                continue
            cur = settings.get(key)
            if not isinstance(cur, (int, float)) and live.SIZE_WORDS.match(str(v).strip()):
                key = live.number_key_for(key, str(v), settings) or key     # "bird bigger" -> birdSize / size
            val, how = live.resolve_value(v, settings.get(key), web)
            if val is None:
                said.append(f"{key}: {how}")
                continue
            patch[key] = val
            said.append(f"{key} -> {how}")
        if not patch:
            return {"ok": False, "message": ("I couldn't change that. " + "; ".join(said)).strip()
                    + f" This app's settings: {self.describe(settings)}. Bigger changes: edit_app."}
        settings.update(patch)
        meta["live"], meta["updated"] = settings, time.time()
        html = p.read_text(encoding="utf-8")
        stale = live.LIVE_JS not in html          # built with an older runtime: the open page must load the new one
        p.write_text(live.inject_live(html, settings), encoding="utf-8")
        p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.current = p.stem
        self.commit(p, "tweak: " + "; ".join(said))
        return {"ok": True, "url": f"/apps/{p.name}", "patch": patch, "changed": "; ".join(said), "reload": stale,
                "message": "Changed it live." + (f" Not found: {', '.join(unknown)}." if unknown else "")}

    def find(self, name: str) -> Path | None:
        if not name:
            return self.path(self.current) if self.current and self.path(self.current).exists() else None
        cands = [q for q in self.dir.glob("*.html") if not q.name.endswith(".prev.html")]

        def updated(q):
            return self._meta(q).get("updated") or q.stat().st_mtime
        p = self.path(name)
        if p.exists():
            # "open the Flappy Bird game" with several games called Flappy Bird: the one changed last (the one
            # just worked on), not an old one whose file happens to be named flappy-bird-game
            t = self._meta(p).get("title", "")
            same = [q for q in cands if t and self._meta(q).get("title", "") == t]
            return max(same + [p], key=updated)
        stop = {"a", "an", "the", "my", "app", "page", "site", "website", "game", "of", "for", "and", "to", "type"}
        words = {w for w in _slug(name).split("-") if len(w) > 1 and w not in stop}

        def score(q):
            title = _slug(self._meta(q).get("title", ""))
            return len(words & (set(q.stem.split("-")) | set(title.split("-"))))
        cur = self.path(self.current) if self.current and self.path(self.current).exists() else None
        best = max(cands, key=lambda q: (score(q), updated(q)), default=None)
        if best is None or score(best) == 0 or (cur is not None and score(cur) >= score(best)):
            return cur                    # no clear other app named: the one on screen
        return best

    def start(self, kind: str, description: str, name: str, base: Path | None = None, replace: bool = False) -> dict:
        with self.lock:
            if self.job and self.job["state"] == "building" and replace:
                # they changed their mind ("wait, no, a racing game"): the running build stops and is never saved
                log.info("build dropped for a change of mind: %r -> %r", self.job["title"], description[:80])
                self.job["cancelled"] = True
                self.job["state"] = "dropped"
            if self.job and self.job["state"] == "building":
                if kind == "edit":
                    # a change asked for while one is still being made: do it right after, not never
                    q = self.job.setdefault("queued", [])
                    q.append((description, name))
                    log.info("change queued while %s is building: %r", self.job["title"], description[:80])
                    return {"ok": True, "queued": True, "waiting": len(q),
                            "message": f"Still working on {self.job['title']}; this change comes right after it."}
                return {"ok": False, "message": f"Still building {self.job['title']}; one new app at a time."}
            title = name or description
            self.job = {"state": "building", "kind": kind, "title": title, "name": _slug(name or description),
                        "t0": time.time(), "description": description}
        threading.Thread(target=self._build, args=(kind, description, base), daemon=True).start()
        return {"ok": True, "building": True, "name": self.job["name"], "eta_s": self.coder.eta_s(kind),
                "message": f"Started the {kind} of {title}. It is NOT ready yet (about {self.coder.eta_s(kind)} s). "
                           "While it builds, tell them in two or three short sentences what you are building: "
                           "the main things in it (who or what they play or see, what to avoid or do, the score "
                           "or the parts), e.g. \"I'm building a game with a little bird that flaps between green "
                           "pipes. Each pipe you pass gives you a point.\" Do not say it is ready or done, do not "
                           "tell them how to play it yet, and do not tell them to enjoy or play it yet. "
                           "You will be told when it is ready."}

    def _build(self, kind: str, description: str, base: Path | None):
        job = self.job
        try:
            if kind == "edit":
                old = live.strip_live(base.read_text(encoding="utf-8"))
                msgs = [{"role": "system", "content": EDIT_SYSTEM},
                        {"role": "user", "content": f"Current file:\n```html\n{old}\n```\n\nChange: {description}"}]
                job["name"] = base.stem
            else:
                msgs = [{"role": "system", "content": APP_SYSTEM},
                        {"role": "user", "content": f"Make this: {description}"}]
            html = None
            job["expect"] = max(2500, len(old) if kind == "edit" else 3500)     # rough size, for the progress bar

            def progress(text: str) -> bool:                  # called while the code streams in
                job["chars"], job["code"] = len(text), text[-600:]
                return bool(job.get("cancelled")) or _html_done_or_looping(text)

            def dropped() -> bool:                            # a change of mind replaced this build
                if job.get("cancelled"):
                    log.info("dropped build stopped: %r", job["title"])
                return bool(job.get("cancelled"))

            for attempt in range(2):
                job["attempt"] = attempt + 1
                job["chars"], job["code"] = 0, ""
                raw = self.coder.complete(msgs, max_tokens=self.max_tokens, temperature=0.2 if attempt == 0 else 0.6,
                                          stop_when=progress)
                if dropped():
                    return
                if _looping(raw):
                    log.warning("the model started repeating itself; trying again")
                    continue
                html = extract_html(raw)
                if html:
                    break
            if not html:
                raise RuntimeError("the model did not write an HTML file")
            game = live.is_game(description if kind != "edit" else job["title"] + " " + description, html)
            best = None                       # (how bad, html): a repair can make it worse (a game that only failed
            for round_ in range(3):          # a test came back with a syntax error): keep the best version
                err = js_syntax_error(html)
                if err:
                    err = f"Syntax error: {err}"
                else:
                    err = live.runtime_problem(live.run_check(html), game)
                if dropped():
                    return
                if best is None or problem_rank(err) <= best[0]:
                    best = (problem_rank(err), html)
                if not err or round_ == 2:   # check, repair, check, repair, check
                    break
                log.warning("problem in the new app (%s): asking the model to fix it", err)
                job["fixing"] = err
                job["chars"], job["code"] = 0, ""
                fix = msgs + [{"role": "assistant", "content": html},
                              {"role": "user", "content": f"{err}. Fix it and output the whole corrected HTML file."}]
                fixed = extract_html(self.coder.complete(fix, max_tokens=self.max_tokens, temperature=0.2,
                                                         stop_when=progress) or "")
                if not fixed:
                    break
                html = fixed
            if best and best[1] is not html:
                log.warning("the repair made it worse: keeping the earlier version")
                html = best[1]
            html, n = centre_draw_fix(html)
            if n:
                log.info("%d thing(s) drawn from their centre point as if it were the corner: drawn around it", n)
            if not game and "data-photo" in html:
                # websites: real photos (Unsplash), 3 per request at most; games never (photos.py)
                html, job["photos"] = photos.fill(html)
            res = live.run_check(html) or {}
            settings = res.get("settings") or {}
            still = js_syntax_error(html) or live.runtime_problem(res, game)
            if dropped():
                return
            p = self.dir / f"{job['name']}.html"
            if kind == "edit" and p.exists():         # keep the previous version (undo)
                p.replace(p.with_suffix(".prev.html"))
            meta = self._meta(p) if kind == "edit" and p.exists() else {"description": description}
            saved = {k: v for k, v in (meta.get("live") or {}).items() if k in settings}   # keep earlier tweaks
            # a thing drawn as a bare colour ("bird": "#FFCC00" = a yellow square) gets its library picture
            for k, pic in sprite_fixes(settings).items():
                if k not in saved:
                    saved[k] = pic
                    log.info("setting %r was only a colour (%r): the library picture %s instead", k, settings[k], pic)
            p.write_text(live.inject_live(html, saved), encoding="utf-8")
            t = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
            meta.update({"title": (t.group(1).strip() if t else job["title"])[:60], "updated": time.time(),
                         "live": {**settings, **saved}})
            p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            problems = check_html(html) + ([still] if still else [])
            tests = res.get("tests") or []
            job["tests"] = (f"{sum(bool(t.get('ok')) for t in tests)}/{len(tests)} behaviour tests passed"
                            if tests else "no behaviour tests")
            job.update(state="ready", url=f"/apps/{p.name}", title=meta["title"], problems=problems,
                       seconds=round(time.time() - job["t0"], 1), settings=self.describe(meta["live"]))
            self.current = p.stem
            self.commit(p, f"{kind}: {description[:90]} ({job['tests']})")
            log.info("%s %s ready in %.1f s (%d chars), %s%s", kind, p.name, job["seconds"], len(html), job["tests"],
                     f" problems: {problems}" if problems else "")
        except Exception as e:  # noqa: BLE001
            if job.get("cancelled"):
                return                         # dropped anyway: nothing to say
            log.exception("build failed")
            job.update(state="failed", error=str(e)[:200])
        with self.lock:
            queued = job.pop("queued", [])
            self.done.append(dict(job))
        if queued:
            self._start_queued(queued)

    def _start_queued(self, queued: list[tuple[str, str]]):
        """The changes asked for during the last build, as ONE edit: in order, a later one wins
        ("make it green" ... "no, black instead" -> black)."""
        if len(queued) == 1:
            change = queued[0][0]
        else:
            change = ("Make these changes, in this order; when two of them disagree, the later one wins: " +
                      " ".join(f"({i + 1}) {d}" for i, (d, _) in enumerate(queued)))
        base = self.find(queued[-1][1])
        if base is None:
            log.warning("queued change dropped: the app is gone")
            return
        r = self.start("edit", change, base.stem, base=base)
        log.info("queued change started: %s", r.get("message"))


def _calc(expr: str) -> float:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
           ast.UAdd: operator.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / % ** ( )")

    expr = expr.replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"\1/100*", expr)
    return ev(ast.parse(expr, mode="eval").body)


def build(cfg: Config) -> tuple[MCPServer, Apps]:
    mcp = MCPServer("fda-tools", instructions="Real-world tools for the FDA voice agent. All free, no API keys.")
    web = httpx.Client(timeout=8.0, headers={"User-Agent": "FDAVoiceAgent/1.0 (student project) httpx/0.28.1"}, follow_redirects=True)
    browser = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/124.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}

    apps = Apps(cfg)
    geo: dict[str, tuple] = {}

    @mcp.tool()
    def create_app(description: str, name: str = "", instead: bool = False) -> dict:
        """Build a new app, game or simple website from a description (e.g. "a snake game with a neon look",
        "a tip calculator", "a page about our coffee shop with a menu"). Runs in the background; the app appears on
        the display when ready. name: a short name for it (optional). instead: true when they changed their mind
        about the app being built right now ("wait, no, a racing game"): that build is dropped and this one starts."""
        return apps.start("build", description, name, replace=instead)

    @mcp.tool()
    def edit_app(change: str, name: str = "", instead: bool = False) -> dict:
        """Change an app that was built before: "make the snake red", "add a high score", "bigger buttons".
        name: which app (default: the one built or opened last). instead: true when they changed their mind about
        the change being made right now ("wait, no, blue buttons"): that change is dropped and this one is made."""
        p = apps.find(name)
        if p is None:
            return {"ok": False, "message": "There is no app to change yet. Ask me to build one first."}
        return apps.start("edit", change, p.stem, base=p, replace=instead)

    @mcp.tool()
    def undo_app_change(name: str = "") -> dict:
        """Go back to the version of an app before its last change."""
        p = apps.find(name)
        if p is None or not p.with_suffix(".prev.html").exists():
            return {"ok": False, "message": "Nothing to undo."}
        prev = p.with_suffix(".prev.html")
        tmp = p.with_suffix(".tmp")
        p.replace(tmp); prev.replace(p); tmp.replace(prev)
        return {"ok": True, "url": f"/apps/{p.name}", "message": "Back to the previous version."}

    @mcp.tool()
    def app_history(name: str = "") -> dict:
        """The saved versions of an app, newest first (every build, change and tweak is saved)."""
        p = apps.find(name)
        if p is None:
            return {"ok": False, "message": "There is no app yet."}
        return {"ok": True, "title": apps._meta(p).get("title", p.stem), "versions": apps.history(p)}

    @mcp.tool()
    def restore_app_version(steps_back: int = 1, name: str = "") -> dict:
        """Bring back an earlier version of an app: steps_back=1 is the version before the last change,
        2 the one before that ("go back two versions"). Use app_history to see them."""
        p = apps.find(name)
        if p is None:
            return {"ok": False, "message": "There is no app yet."}
        return apps.restore(p, int(steps_back))

    @mcp.tool()
    def tweak_app(changes: dict, name: str = "") -> dict:
        """Change how a built app or game looks while it runs - instant, no rebuild. changes: setting -> new value,
        e.g. {"bird": "butterfly"}, {"bird": "red"}, {"pipe": "ice"}, {"sky": "#000033"},
        {"speed": "faster"}, {"pipe": "cactus"}, {"difficulty": "easier"} / "harder" / "easy" / "hard".
        Values: a colour, a thing (becomes an emoji or a picture from the art library), or a number.
        "Make the pipes look like ice" -> {"pipe": "ice"} (just the thing, no "photo of").
        name: which app (default: the one shown last). For new features or rules use edit_app instead."""
        return apps.tweak(changes, name, web)

    @mcp.tool()
    def app_settings(name: str = "") -> dict:
        """The live settings of a built app or game (what tweak_app can change)."""
        p = apps.find(name)
        if p is None:
            return {"ok": False, "message": "There is no app yet."}
        meta = apps._meta(p)
        return {"ok": True, "title": meta.get("title", p.stem), "settings": apps.describe(meta.get("live") or {})}

    @mcp.tool()
    def list_apps() -> list[dict]:
        """The apps, games and websites built so far (newest first)."""
        return apps.list()[:15]

    @mcp.tool()
    def open_app(name: str) -> dict:
        """Show an app that was built before on the display."""
        job = apps.job
        if job and job.get("state") == "building" and job.get("kind") == "build":
            # asking for the app that is still being made (the LLM tried again and again): it opens by itself
            if not name or _slug(name) == job["name"] or name.lower() in job["title"].lower() or \
                    apps.find(name) is None:
                return {"ok": False, "building": True,
                        "message": f"{job['title']} is still being built; it opens by itself when it is ready. "
                                   "Do not call open_app for it."}
        p = apps.find(name)
        if p is None:
            return {"ok": False, "message": f"I don't have an app called {name}."}
        apps.current = p.stem
        meta = apps._meta(p)
        return {"ok": True, "url": f"/apps/{p.name}", "title": meta.get("title", p.stem),
                "settings": apps.describe(meta.get("live") or {})}

    @mcp.tool()
    def app_status() -> dict:
        """Whether an app is being built right now, and the builds that finished since the last call."""
        with apps.lock:
            done, apps.done = apps.done, []
        job = apps.job if apps.job and apps.job["state"] == "building" else None
        st = {"building": bool(job), "elapsed_s": round(time.time() - job["t0"]) if job else 0,
              "title": job["title"] if job else "", "finished": done}
        if job:
            st.update(kind=job.get("kind", "build"), change=job.get("description", "") if job.get("kind") == "edit" else "",
                      chars=job.get("chars", 0), expect=job.get("expect", 3500), code=job.get("code", ""),
                      step="fixing a bug" if job.get("fixing") else
                           ("second try" if job.get("attempt", 1) > 1 else "writing the code"),
                      eta_s=apps.coder.eta_s(job["kind"]))
        return st

    @mcp.tool()
    def get_weather(place: str = "") -> str:
        """Current weather and today's forecast for a place (default: the user's home)."""
        place = (place or cfg.location).strip()
        try:
            if place not in geo:
                g = (web.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": place, "count": 1}).json().get("results") or [None])[0]
                if not g:
                    return f"I couldn't find a place called {place}."
                geo[place] = (g["latitude"], g["longitude"], g["name"])
            lat, lon, name = geo[place]
            w = web.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "current": "temperature_2m,weather_code,apparent_temperature",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 1}).json()
            c, d = w["current"], w["daily"]
            rain = d["precipitation_probability_max"][0]
            return (f"In {name} it's {round(c['temperature_2m'])} degrees and {WMO.get(int(c['weather_code']), '')}, "
                    f"feels like {round(c['apparent_temperature'])}. Today's high is "
                    f"{round(d['temperature_2m_max'][0])}" + (f", {rain} percent chance of rain." if rain is not None
                                                              else "."))
        except Exception as e:  # noqa: BLE001
            log.warning("weather: %s", e)
            return "The weather service is not reachable right now."

    from .device import Device
    from .docs import Docs
    here = lambda d: Path(d) if Path(d).is_absolute() else ROOT / d      # noqa: E731
    docs = Docs(here(cfg.docs_dir))
    device = Device(here(cfg.reports_dir))

    @mcp.tool()
    def ask_documents(question: str) -> dict:
        """Search the user's own documents (PDF, Word, Excel, CSV, notes in the docs folder) for the answer to a
        question, offline. Returns the best passages with file and section: answer in one or two sentences from
        them and say which file (and page/section) it came from. If nothing fits, say it is not in the documents."""
        try:
            hits = docs.search(question)
        except Exception as e:  # noqa: BLE001
            log.warning("docs: %s", e)
            return {"ok": False, "message": "I couldn't read the documents."}
        if not docs.files:
            return {"ok": False, "message": f"There are no documents yet. Put files in the {cfg.docs_dir} folder."}
        return {"ok": bool(hits), "passages": hits} if hits else {"ok": False, "message": "Nothing about that in the documents."}

    @mcp.tool()
    def list_documents() -> dict:
        """The documents FDA can answer questions about."""
        return {"folder": cfg.docs_dir, "documents": docs.list()}

    @mcp.tool()
    def device_check() -> dict:
        """A read-only health and security check of this device (the Pi): services open to the network, waiting
        updates, SSH settings, files anyone can change, disk, memory, temperature and power. Nothing is changed.
        Say the summary and the most important finding with its fix, briefly."""
        try:
            return device.check()
        except Exception as e:  # noqa: BLE001
            log.exception("device check")
            return {"summary": f"The check failed: {e}"}

    @mcp.tool()
    def web_search(query: str) -> list[dict] | str:
        """Search the web (current facts, prices, flights, schedules, people). Returns the top results with their
        links; the links are shown on the screen."""
        try:
            from ddgs import DDGS
            res = DDGS().text(query, max_results=8)
        except Exception as e:  # noqa: BLE001
            log.warning("search: %s", e)
            return "The search is not reachable right now."
        ads = ("/aclick", "duckduckgo.com/y.js", "googleadservices", "doubleclick.net")
        res = [r for r in res if r.get("href") and not any(a in r["href"] for a in ads)][:5]
        return [{"title": r.get("title", "")[:90], "snippet": r.get("body", "")[:300], "link": r["href"]}
                for r in res] or "No results."

    @mcp.tool()
    def read_page(link: str) -> str:
        """The text of a web page from web_search (for details the snippets lack: the airline, the exact price,
        the date). Returns the first part of the page as plain text."""
        try:
            r = httpx.get(link, headers=browser, timeout=10, follow_redirects=True)
        except Exception as e:  # noqa: BLE001
            log.warning("read page: %s", e)
            return "That page is not reachable right now."
        html = re.sub(r"(?is)<(script|style|noscript|svg|head|template)\b[^>]*>.*?</\1\s*>", " ", r.text)
        text = re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"(?s)<[^>]+>", " ", html))
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        import html as _html
        return _html.unescape(text)[:4000] or "The page has no readable text."

    @mcp.tool()
    def get_news(topic: str = "") -> list[dict] | str:
        """Latest news headlines, optionally about a topic (Philippines by default)."""
        url = ("https://news.google.com/rss/search" if topic else "https://news.google.com/rss")
        params = {"hl": "en-PH", "gl": "PH", "ceid": "PH:en", **({"q": topic} if topic else {})}
        try:
            root = ET.fromstring(web.get(url, params=params).content)
        except Exception as e:  # noqa: BLE001
            log.warning("news: %s", e)
            return "The news feed is not reachable right now."
        items = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").rsplit(" - ", 1)
            items.append({"headline": title[0][:110], "source": title[1] if len(title) > 1 else "",
                          "link": it.findtext("link") or ""})
            if len(items) == 4:
                break
        return items or "No news found."

    @mcp.tool()
    def find_recipe(dish: str) -> dict | str:
        """A recipe for a dish: ingredients and steps."""
        try:
            meals = web.get("https://www.themealdb.com/api/json/v1/1/search.php", params={"s": dish}).json()["meals"]
        except Exception as e:  # noqa: BLE001
            log.warning("recipe: %s", e)
            return "The recipe service is not reachable right now."
        if not meals:
            return f"No recipe found for {dish}. Try web_search."
        m = meals[0]
        ingredients = [f"{(m.get(f'strMeasure{i}') or '').strip()} {m[f'strIngredient{i}'].strip()}".strip()
                       for i in range(1, 21) if (m.get(f"strIngredient{i}") or "").strip()]
        steps = [s.strip() for s in re.split(r"\r?\n+|(?<=\.)\s+(?=[A-Z])", m.get("strInstructions") or "") if
                 len(s.strip()) > 3]
        return {"name": m["strMeal"], "ingredients": ingredients[:12], "steps": [x[:160] for x in steps[:8]]}

    @mcp.tool()
    def wikipedia(topic: str) -> str:
        """A short encyclopedia summary of a topic, person or place."""
        try:
            hits = web.get("https://en.wikipedia.org/w/api.php", params={      # exact title first (case matters)
                "action": "opensearch", "search": topic, "limit": 1, "format": "json"}).json()[1]
            if not hits:
                return f"Wikipedia has no page on {topic}."
            r = web.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + hits[0].replace(" ", "_"))
            text = r.json().get("extract") or ""
            return " ".join(re.split(r"(?<=[.!?])\s+", text)[:3])[:500] or f"Wikipedia has no summary of {topic}."
        except Exception as e:  # noqa: BLE001   (Wikimedia sometimes refuses scripted clients: use the search)
            log.warning("wikipedia: %s; using web search", e)
            try:
                from ddgs import DDGS
                res = DDGS().text(f"{topic} wikipedia", max_results=2)
                return " ".join(r.get("body", "") for r in res)[:800] or f"Nothing found on {topic}."
            except Exception:  # noqa: BLE001
                return "Wikipedia is not reachable right now."

    @mcp.tool()
    def calculate(expression: str) -> str:
        """Exact arithmetic: "17.5 * 3", "2 ** 10", "15 % of 240", "(3 + 4) / 7"."""
        try:
            v = _calc(expression)
        except Exception as e:  # noqa: BLE001
            return f"I can't calculate that: {e}"
        return f"{v:,.6g}" if isinstance(v, float) else f"{v:,}"

    @mcp.tool()
    def play_video(what: str) -> dict | str:
        """Find a YouTube video and play it on the screen. `what` = exactly what the person asked for
        ("the Bohemian Rhapsody music video", "how to tie a tie", "cat videos"). Only call this when they asked
        for a video and said what it is about."""
        if not what.strip():
            return "Ask which video they want."
        try:
            page = httpx.get("https://www.youtube.com/results", params={"search_query": what}, headers=browser,
                             timeout=8, follow_redirects=True).text
        except Exception as e:  # noqa: BLE001
            log.warning("video search: %s", e)
            return "YouTube is not reachable right now."
        ids = list(dict.fromkeys(re.findall(r'"videoRenderer":\{"videoId":"([\w-]{11})"', page)))
        for vid in ids[:5]:
            # oEmbed answers 401/403 for videos whose owner does not allow playing them on other sites
            try:
                o = web.get("https://www.youtube.com/oembed",
                            params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"})
            except Exception:  # noqa: BLE001
                continue
            if o.status_code == 200:
                d = o.json()
                return {"ok": True, "video_id": vid, "title": d.get("title", "")[:100],
                        "channel": d.get("author_name", "")[:60]}
        return f"I couldn't find a video for {what} that can play here."

    @mcp.tool()
    def show_photo(what: str) -> dict | str:
        """Show photos from the web on the screen (free-licence photos, Openverse). `what` = what they want to see
        ("Mount Apo", "a red panda", "the Eiffel Tower at night")."""
        try:
            r = web.get("https://api.openverse.org/v1/images/", params={"q": what, "page_size": 8}, timeout=12)
            res = r.json().get("results", [])
        except Exception as e:  # noqa: BLE001
            log.warning("photos: %s", e)
            return "The photo search is not reachable right now."
        items = []
        for it in res:
            if not it.get("url"):
                continue
            url = it["url"]
            m = re.match(r"https://upload\.wikimedia\.org/wikipedia/commons/(\w)/(\w\w)/([^/?#]+)$", url)
            if m and (it.get("width") or 0) > 960:    # big originals are slow on the Pi: a 960 px copy (Wikimedia resize)
                url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{m[1]}/{m[2]}/{m[3]}/960px-{m[3]}"
            items.append({"title": (it.get("title") or what)[:80], "creator": (it.get("creator") or "")[:40],
                          "license": it.get("license") or "", "url": url, "thumb": it["url"]})
            if len(items) == 5:
                break
        return {"ok": True, "photos": items} if items else f"No photos found for {what}."

    @mcp.tool()
    def get_time() -> str:
        """The current local time, day and date (for "what time is it", "what day is today")."""
        return time.strftime("%I:%M %p, %A %B %d %Y").lstrip("0")

    room_tools(mcp, cfg)
    return mcp, apps


When = Annotated[str, Field(description="a time like 15:00 (leave out for none)")]
Repeat = Annotated[str, Field(description="for a repeating reminder: daily, weekdays, weekends, or a day like monday; "
                                          "leave out for once")]


def room_tools(mcp: MCPServer, cfg: Config):
    """The room (lights, thermostat, timers, alarms, reminders, lists, music, volume) as MCP tools. The room itself
    lives in the agent (it plays the alarm and the music, and draws the Virtual Home): each tool hands the call to
    the agent's room hub (http://127.0.0.1:<room_hub_port>/room/<tool>, fda/pc/agent.py) and returns its answer."""
    hub = f"http://127.0.0.1:{getattr(cfg, 'room_hub_port', 8104)}/room/"
    client = httpx.Client(timeout=10.0)

    def room(tool: str, **args) -> dict:
        args = {k: v for k, v in args.items() if v not in (None, "")}      # left out = not given
        try:
            r = client.post(hub + tool, json=args)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            return {"ok": False, "result": f"The room is not reachable right now ({type(e).__name__})."}

    @mcp.tool()
    def lights(power: Literal["on", "off"] | None = None,
               brightness: Annotated[int | None, Field(ge=0, le=100)] = None, color: str = "") -> dict:
        """Turn the room lights on/off, set brightness (0-100) or colour."""
        return room("lights", power=power, brightness=brightness, color=color)

    @mcp.tool()
    def thermostat(celsius: float) -> dict:
        """Set the room temperature in degrees Celsius."""
        return room("thermostat", celsius=celsius)

    @mcp.tool()
    def timer(seconds: Annotated[int, Field(ge=1)], label: str = "") -> dict:
        """Start a countdown timer (seconds)."""
        return room("timer", seconds=seconds, label=label)

    @mcp.tool()
    def cancel_timer() -> dict:
        """Cancel the running timer."""
        return room("cancel_timer")

    @mcp.tool()
    def alarm(time: str) -> dict:
        """Set an alarm at a time like 06:30."""
        return room("alarm", time=time)

    @mcp.tool()
    def cancel_alarm(time: Annotated[str, Field(description="the alarm's time like 06:30, or 'all'; leave out when "
                                                            "there is only one")] = "") -> dict:
        """Cancel an alarm (its time, or 'all')."""
        return room("cancel_alarm", time=time)

    @mcp.tool()
    def add_reminder(text: str, time: When = "", repeat: Repeat = "") -> dict:
        """Remember something for the user, optionally at a time like 15:00."""
        return room("add_reminder", text=text, time=time, repeat=repeat)

    @mcp.tool()
    def list_reminders() -> dict:
        """List the user's reminders."""
        return room("list_reminders")

    @mcp.tool()
    def delete_reminder(which: Annotated[str, Field(description="words from the reminder, its number (1 = first), "
                                                                "'last', or 'all'")]) -> dict:
        """Delete reminders: by words from it, its number, 'last', or 'all'."""
        return room("delete_reminder", which=which)

    @mcp.tool()
    def edit_reminder(which: Annotated[str, Field(description="words from the reminder, its number, or 'last'")],
                      text: Annotated[str, Field(description="new text (leave out to keep it)")] = "",
                      time: Annotated[str, Field(description="new time like 18:00 (leave out to keep it)")] = "",
                      repeat: Repeat = "") -> dict:
        """Change a reminder's text or time ("scratch that, call father instead": edit the last one)."""
        return room("edit_reminder", which=which, text=text, time=time, repeat=repeat)

    @mcp.tool()
    def list_items(action: Literal["add", "remove", "show", "clear"],
                   items: Annotated[str, Field(description="the items, comma separated (add / remove)")] = "",
                   list: Annotated[str, Field(description="which list: shopping (default), to-do, or a name")] = ""
                   ) -> dict:  # noqa: A002 (the argument the agent's room expects is called "list")
        """The person's lists (shopping, to-do, ...): add or remove items, read a list out, or clear it."""
        return room("list_items", action=action, items=items, list=list)

    @mcp.tool()
    def music(action: Literal["play", "pause", "resume", "stop", "next"]) -> dict:
        """Play, pause, resume, stop or skip the music."""
        return room("music", action=action)

    @mcp.tool()
    def volume(change: Literal["up", "down"] | None = None,
               level: Annotated[int | None, Field(ge=0, le=100)] = None) -> dict:
        """Change the speaker volume (up/down or a level 0-100)."""
        return room("volume", change=change, level=level)

    @mcp.tool()
    def call(contact: Annotated[str, Field(description="who, as they said it: wife, mom, dad, John")]) -> dict:
        """Call someone on the phone in the Virtual Home (a pretend phone: it rings, nobody answers)."""
        return room("call", contact=contact)

    @mcp.tool()
    def end_call() -> dict:
        """Hang up the phone (end the call), or put the phone away."""
        return room("end_call")

    @mcp.tool()
    def message(contact: Annotated[str, Field(description="who: wife, mom, dad, a name")],
                text: Annotated[str, Field(description="the message; empty if they did not say it yet")] = "") -> dict:
        """Send a text from the phone in the Virtual Home (pretend). If they did not say what it should say, leave
        text empty and ask them (the phone shows quick replies)."""
        return room("message", contact=contact, text=text)


def main():
    ap = argparse.ArgumentParser(description="FDA tools MCP server")
    ap.add_argument("--transport", default="streamable-http", choices=["streamable-http", "stdio"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8103)
    args = ap.parse_args()
    cfg = Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpcore", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    mcp, _ = build(cfg)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
