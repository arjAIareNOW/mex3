"""Live design of built apps: change how a running game looks by voice, without rebuilding it.

Every app FDA builds gets `live.js` (window.FDA): the app takes its look from named settings
(`const C = FDA.defaults({bird: "🐦", pipe: "#3a3", sky: "#7cf"})`) and draws things with `FDA.draw(...)`.
`tweak` turns "make the bird a butterfly" / "the pipes a photo of bamboo" into a settings patch: the display
sends it into the running app at once, and it is saved in the file so it stays after a reload.

`run_check` runs a new app for ~1.5 s in Node with a stand-in page (runcheck.js) to catch errors that only show
when it runs, and reads its live settings.
"""
from __future__ import annotations

import base64
import difflib
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
LIVE_JS = (HERE / "live.js").read_text(encoding="utf-8")
log = logging.getLogger("fda.mcp.live")

_LIVE_BLOCK = re.compile(r"<script id=\"fda-live\">.*?</script>\s*", re.S)


def scripts(html: str) -> str:
    """The app's own JavaScript (not the live runtime, not its JSON tests)."""
    return "\n".join(re.findall(r"<script(?![^>]*id=[\"']fda-(?:live|tests))(?![^>]*application/json)[^>]*>(.*?)</script>",
                                 html, re.S | re.I))


_TESTS = re.compile(r"<script[^>]*id=[\"']fda-tests[\"'][^>]*>(.*?)</script>", re.S | re.I)


def tests_of(html: str) -> str | None:
    """The app's own behaviour tests (JSON text), if it has them."""
    m = _TESTS.search(html)
    return m.group(1).strip() if m else None


def strip_live(html: str) -> str:
    return _LIVE_BLOCK.sub("", html)


def inject_live(html: str, saved: dict | None = None) -> str:
    """Put the live runtime (and the saved settings) at the start of <head>."""
    html = strip_live(html)
    block = (f'<script id="fda-live">window.FDA_SAVED = {json.dumps(saved or {}, ensure_ascii=False)};\n'
             f"{LIVE_JS}</script>\n")
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        return html[:m.end()] + "\n" + block + html[m.end():]
    m = re.search(r"<html[^>]*>", html, re.I)
    if m:
        return html[:m.end()] + "\n<head>\n" + block + "</head>\n" + html[m.end():]
    return block + html


def run_check(html: str) -> dict | None:
    """Run the app's script for a moment in Node. None if Node.js is not installed."""
    node = shutil.which("node")
    js = scripts(html)
    if not node or not js.strip():
        return None
    with tempfile.TemporaryDirectory() as d:
        live, app = Path(d) / "live.js", Path(d) / "app.js"
        live.write_text("window.FDA_SAVED = {};\n" + LIVE_JS, encoding="utf-8")
        app.write_text(js, encoding="utf-8")
        if (HERE / "node_modules" / "jsdom").exists():         # a real DOM (npm install in fda/mcp_servers)
            page = Path(d) / "page.html"
            page.write_text(re.sub(r"<script\b[^>]*>.*?</script>", "", strip_live(html), flags=re.S | re.I),
                            encoding="utf-8")
            args = [node, str(HERE / "runcheck_dom.js"), str(live), str(app), str(page)]
        else:
            args = [node, str(HERE / "runcheck.js"), str(live), str(app)]
        tests = tests_of(html)
        if tests:
            (Path(d) / "tests.json").write_text(tests, encoding="utf-8")
            args.append(str(Path(d) / "tests.json"))
        try:
            r = subprocess.run(args, capture_output=True,
                               text=True, timeout=20, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            return {"error": "the app hangs (endless loop?)", "line": None, "draws": 0, "dom": 0, "settings": {}}
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    res["own_fda"] = bool(re.search(r"(?:const|let|var)\s+FDA\s*=|window\.FDA\s*=", js))
    if res.get("error") and res.get("line"):
        lines = js.splitlines()
        n = res["line"] - 1
        if 0 <= n < len(lines):
            res["code_line"] = lines[n].strip()[:160]
    return res


def runtime_problem(res: dict | None, is_game: bool) -> str | None:
    """What to tell the coder, or None if it runs fine."""
    if not res:
        return None
    if res.get("error"):
        where = f" at script line {res['line']}: `{res.get('code_line', '')}`" if res.get("line") else ""
        hint = " (a value that changes must be declared with let, not const)" if "constant" in res["error"] else ""
        return f"When it runs: {res['error']}{where}{hint}"
    if is_game and not res.get("draws") and res.get("dom", 0) < 6:
        return "It runs but draws nothing on the canvas: draw every frame in the requestAnimationFrame loop"
    failed = [t for t in res.get("tests") or [] if not t.get("ok")]
    if failed:
        return ("Behaviour test failed: " + "; ".join(f"'{t['name']}': {t.get('error', 'failed')}" for t in failed[:3])
                + ". Fix the app so it really does this (only change a test if the test itself is wrong)")
    if is_game and not res.get("settings") and res.get("own_fda"):
        return ('The game defines its own FDA object ("const FDA = ..." / "let FDA" / "var FDA"): delete that whole '
                'definition. FDA is already loaded by the page; keep using FDA.defaults, FDA.draw and FDA.color')
    if is_game and not res.get("settings"):
        return ('The game does not use live settings: put its look in const C = FDA.defaults({...}) (one setting '
                'per thing: colours, emoji, numbers) and draw every thing with FDA.draw(ctx, "name", x, y, w, h)')
    if is_game and res.get("settings"):
        drawn = res.get("drawn") or {}
        things = [k for k, v in res["settings"].items()
                  if isinstance(v, str) and k.lower() not in ("sky", "background", "bg", "text", "score", "font")]
        missing = [k for k in things if k not in drawn]
        if missing:
            return (f"The game never draws {', '.join(missing)}: write the real game logic in update() and draw "
                    f"every thing with FDA.draw(ctx, \"{missing[0]}\", x, y, w, h) each frame")
        if things and all(drawn.get(k, 0) <= 1 for k in things):
            return ("Nothing moves after the player taps: write the game logic in update() so things move "
                    "every frame (velocity, gravity, obstacles moving), and the tap/Space action changes the game")
    return None


def is_game(description: str, html: str) -> bool:
    return bool(re.search(r"\bgame|play|snake|bird|flap|jump|shoot|race|pong|tetris|maze\b", description, re.I)
                or ("<canvas" in html.lower() and "requestanimationframe" in html.lower()))


# ------------------------------------------------------------------ settings changes
CSS_COLORS = set("""aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown
burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod
darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen
darkslateblue darkslategray darkturquoise darkviolet deeppink deepskyblue dimgray dodgerblue firebrick floralwhite
forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo
ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray
lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightsteelblue lightyellow lime
limegreen linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue
mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace
olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru
pink plum powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna
silver skyblue slateblue slategray snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white
whitesmoke yellow yellowgreen""".split())

EMOJI = {
    "butterfly": "🦋", "bird": "🐦", "eagle": "🦅", "owl": "🦉", "duck": "🦆", "chicken": "🐔", "penguin": "🐧",
    "parrot": "🦜", "bat": "🦇", "bee": "🐝", "ladybug": "🐞", "fish": "🐟", "shark": "🦈", "whale": "🐳",
    "dolphin": "🐬", "octopus": "🐙", "cat": "🐱", "dog": "🐶", "rabbit": "🐰", "bunny": "🐰", "frog": "🐸",
    "monkey": "🐵", "pig": "🐷", "cow": "🐮", "unicorn": "🦄", "dragon": "🐉", "dinosaur": "🦖", "snake": "🐍",
    "turtle": "🐢", "mouse": "🐭", "fox": "🦊", "bear": "🐻", "panda": "🐼", "lion": "🦁", "tiger": "🐯",
    "ghost": "👻", "alien": "👽", "robot": "🤖", "rocket": "🚀", "ufo": "🛸", "plane": "✈️", "airplane": "✈️",
    "helicopter": "🚁", "car": "🚗", "bus": "🚌", "ball": "⚽", "football": "🏈", "basketball": "🏀",
    "star": "⭐", "heart": "❤️", "fire": "🔥", "apple": "🍎", "banana": "🍌", "cherry": "🍒", "pizza": "🍕",
    "burger": "🍔", "donut": "🍩", "cookie": "🍪", "cake": "🍰", "tree": "🌳", "cactus": "🌵", "flower": "🌸",
    "mushroom": "🍄", "rock": "🪨", "cloud": "☁️", "sun": "☀️", "moon": "🌙", "balloon": "🎈", "bomb": "💣",
    "coin": "🪙", "gem": "💎", "diamond": "💎", "crown": "👑", "smile": "🙂", "face": "🙂", "skull": "💀",
    "zombie": "🧟", "ninja": "🥷", "wizard": "🧙", "princess": "👸", "astronaut": "🧑‍🚀", "bamboo": "🎋",
}

SYNONYMS = [
    {"player", "bird", "character", "hero", "ship", "plane", "snake", "car", "ball"},
    {"pipe", "pipes", "obstacle", "obstacles", "wall", "walls", "column", "tower", "enemy", "enemies"},
    {"sky", "background", "bg", "backdrop", "scenery"},
    {"ground", "floor", "land", "grass"},
    {"food", "fruit", "apple", "coin", "item", "pickup", "target"},
    {"text", "score", "font", "label"},
    {"speed", "pace", "velocity"},
]


def match_key(key: str, keys: list[str]) -> str | None:
    """'pipes' -> 'pipe', 'background' -> 'sky', 'the bird' -> 'bird'."""
    k = re.sub(r"^(the|a|an)\s+", "", key.strip().lower()).replace(" ", "")
    low = {x.lower(): x for x in keys}
    if k in low:
        return low[k]
    for x in keys:
        if k.rstrip("s") == x.lower().rstrip("s"):
            return x
    for group in SYNONYMS:                       # before substrings: "background" is not "ground"
        if k in group or k.rstrip("s") in group:
            for x in keys:
                if x.lower() in group or x.lower().rstrip("s") in group:
                    return x
    for x in keys:
        if len(x) > 2 and (k.startswith(x.lower()) or x.lower().startswith(k)):
            return x
    close = difflib.get_close_matches(k, list(low), n=1, cutoff=0.75)
    return low[close[0]] if close else None


def photo(query: str, web) -> str | None:
    """A free photo for `query` (Openverse, no key), shrunk to a small JPEG data URL."""
    try:
        r = web.get("https://api.openverse.org/v1/images/", params={"q": query, "page_size": 5}, timeout=12)
        for item in r.json().get("results", []):
            url = item.get("thumbnail") or item.get("url")
            if not url:
                continue
            im = web.get(url, timeout=10)
            if im.status_code != 200 or not im.headers.get("content-type", "").startswith("image"):
                continue
            data = im.content
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((256, 256))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
                data = buf.getvalue()
            except Exception:  # noqa: BLE001  (no Pillow: keep the thumbnail as it is)
                pass
            log.info("photo for %r: %s (%s, %d bytes)", query, item.get("title"), item.get("license"), len(data))
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception as e:  # noqa: BLE001
        log.warning("photo search failed: %s", e)
    return None


def _is_sprite(v) -> bool:
    """An emoji/text sprite (not a colour, photo or number)."""
    if not isinstance(v, str) or v.startswith("data:") or re.match(r"^(#|rgb|hsl)", v):
        return False
    return v.split("@")[0].lower().replace(" ", "") not in CSS_COLORS


def resolve_value(value, current, web) -> tuple[object, str]:
    """A spoken value -> what the app understands, and how to describe it.
    Numbers stay numbers; colours and emoji pass; "photo of bamboo" fetches a photo; "butterfly" -> 🦋."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, str(value)
    v = str(value).strip()
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            return float(m.group()), m.group()
        factor = {"faster": 1.5, "slower": 0.67, "bigger": 1.4, "smaller": 0.7, "double": 2.0, "half": 0.5}
        for w, f in factor.items():
            if w in v.lower():
                return round(current * f, 2), f"{w} ({round(current * f, 2)})"
    m = re.match(r"^(?:an?\s+)?(?:photo|picture|image|pic)\s+(?:of\s+)?(?:an?\s+)?(.+)$", v, re.I)
    if m:
        url = photo(m.group(1), web)
        return (url, f"a photo of {m.group(1)}") if url else (None, f"no photo of {m.group(1)} found")
    low = v.lower().replace(" ", "")
    if re.match(r"^#[0-9a-f]{3,8}$", low) or low in CSS_COLORS or re.match(r"^(rgb|hsl)a?\(", low):
        if _is_sprite(current):                    # "make the bird red": keep the bird, tint it
            return f"{current.split('@')[0]}@{low}", f"{v} {current.split('@')[0]}"
        return low, v
    if not re.search(r"[a-z]", v, re.I):           # emoji / symbols
        return v, v
    word = re.sub(r"^(an?|the)\s+", "", v.lower()).rstrip("s")
    if word in EMOJI or v.lower() in EMOJI:
        e = EMOJI.get(word) or EMOJI[v.lower()]
        return e, f"{v} {e}"
    url = photo(v, web)                            # some other thing: find a picture of it
    return (url, f"a picture of {v}") if url else (v, v)
