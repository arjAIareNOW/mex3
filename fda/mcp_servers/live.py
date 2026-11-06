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
    if not m:
        return None
    text = m.group(1).strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        pass
    # small slips the model makes: a "setup" list not closed before "before" / "do" / "expect", a trailing comma,
    # a // comment. Repaired here, so a good app is not reported as broken for its tests' punctuation.
    fixed = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
    fixed = re.sub(r'(\[(?:\s*"(?:[^"\\]|\\.)*"\s*,?)*?)\s*,\s*("(?:before|do|expect)"\s*:)', r"\1], \2", fixed)
    try:
        json.loads(fixed)
        return fixed
    except ValueError:
        return text


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


# Web photos (Openverse) for live changes: OFF. Not an approved service (2026-11-06: "make the pipes look like
# ice" fetched a photo of a person and tiled it into the pipes). Pictures come from the art library or emoji.
PHOTOS = False


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
    """An emoji/text sprite or a picture from Elkit's art library (not a colour, photo or number)."""
    if not isinstance(v, str) or v.startswith("data:") or re.match(r"^(#|rgb|hsl)", v):
        return False
    return v.split("@")[0].lower().replace(" ", "") not in CSS_COLORS


# ---------- Elkit's own art library (fda/assets, made by scripts/make_assets.py; offline) ----------
ASSETS = Path(__file__).resolve().parents[1] / "assets"
_asset_index: dict[str, str] | None = None
_KIND_ORDER = {"sprite": 0, "tile": 1, "background": 2, "icon": 3}   # "star": the sprite before the icon


def _asset_key(s: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split()
    words = [w for w in words if w not in ("a", "an", "the", "of", "picture", "sprite", "icon", "image")]
    return " ".join(sorted(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words))


_asset_say: dict[str, str] = {}
_manifest: dict = {}


def manifest(reload: bool = False) -> dict:
    """fda/assets/manifest.json: items, groups (choices) and palettes (colour schemes); {} if not built."""
    global _manifest
    if not _manifest or reload:
        try:
            _manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _manifest = {}
    return _manifest


def asset_index(reload: bool = False) -> dict[str, str]:
    """{"fish": "/assets/sprites/fish.svg", "night sky": "/assets/backgrounds/sky-night.svg", ...}"""
    global _asset_index
    if _asset_index is None or reload:
        idx: dict[str, str] = {}
        items = manifest(reload).get("items", [])
        for it in sorted(items, key=lambda i: _KIND_ORDER.get(i.get("kind"), 9)):
            idx.setdefault(_asset_key(it["name"]), it["path"])
            if it.get("say"):
                idx.setdefault(_asset_key(it["say"]), it["path"])
                _asset_say[it["path"]] = it["say"]
        _asset_index = idx
    return _asset_index


def asset_for(word) -> str | None:
    """A spoken thing -> the library picture of it ("spaceships" -> /assets/sprites/spaceship.svg), or None."""
    return asset_index().get(_asset_key(word)) if word else None


def is_asset(v) -> bool:
    return isinstance(v, str) and v.startswith("/assets/")


def _label(v: str) -> str:
    """How to say a value: a library path by its name ("/assets/sprites/fish.svg" -> "fish")."""
    v = v.split("@")[0].replace("|flip", "")
    if not is_asset(v):
        return v
    asset_index()
    return _asset_say.get(v) or Path(v).stem.replace("-", " ")


_FLIP = re.compile(r"\b(flip(ped)?|mirror(ed)?|reverse(d)?|turn(ed)?( it| him| her)? around|the other way|"
                   r"other direction|opposite (way|direction)|wrong way|(face|facing|look|looking|point|pointing)"
                   r"( to the| to)? (right|left|forward|backward|the other way))\b", re.I)


def is_flipped(v) -> bool:
    return isinstance(v, str) and "|flip" in v


# "bigger", "a lot wider", "jump higher": a change of size or strength, which only a number setting can do
SIZE_WORDS = re.compile(r"^(?:(?:a )?(?:lot|bit|little)\s+)?(?:much\s+)?"
                        r"(?:(?:jump|fly|flap|bounce|shoot|fire|move|run|fall|spawn|come|grow)s?\s+)?"
                        r"(bigger|larger|smaller|tinier|wider|narrower|thicker|thinner|taller|shorter|longer|"
                        r"higher|lower|heavier|lighter|stronger|weaker|faster|quicker|slower|more|fewer|less|"
                        r"more often|less often)\W*$", re.I)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "a dozen": 12, "twenty": 20}
_FACE = re.compile(r"\b(?:face|facing|look|looking|point|pointing|go|going|fly|flying)(?: to the| to| towards the)? "
                   r"(right|left|forwards?|backwards?)\b", re.I)


def number_key_for(key: str, value: str, settings: dict) -> str | None:
    """'bird' + 'bigger' -> the number setting that does it ('birdSize', 'size'); 'jump higher' -> 'jump'."""
    nums = {k.lower().replace("_", ""): k for k, v in settings.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    v, k = value.lower(), key.lower()
    if re.search(r"\b(shoots?|fires?)\b", v):                       # "shoot faster" / "fire more"
        names = ["firerate", "shootrate", "bulletspeed"]
    elif re.search(r"\b(spawns?|comes?)\b|more often|less often|\b(more|fewer|less)\b", v):
        names = [f"{k}rate", f"{k}spawn", "spawnrate", f"{k}s", f"{k}count", "enemies", "count"]   # "more enemies"
    elif re.search(r"\bgrows?\b", v):
        names = ["grow", "growth"]
    elif re.search(r"\b(jump|fly|flap|bounce)s?\b|higher|lower", v):
        names = ["jump", "flap", "lift", "jumpheight", "flapstrength", "bounce"]
    elif re.search(r"heavier|lighter", v):
        names = ["gravity", "weight"]
    elif re.search(r"faster|quicker|slower", v):                    # "the ball faster" -> ballSpeed
        names = [f"{k}speed", "bulletspeed" if k in ("bullet", "bullets", "shot", "laser") else "", "speed"]
    else:
        names = [f"{k}size", f"{k}width" if re.search(r"wider|narrower|thicker|thinner", v) else "", "size", "scale"]
    return next((nums[n] for n in names if n and n in nums), None)


def resolve_value(value, current, web) -> tuple[object, str]:
    """resolve_value_plain, plus mirroring: "flip it" / "face right" toggles "|flip" on the current sprite or photo,
    and a new sprite for a flipped thing stays flipped (a butterfly replacing a flipped bird faces the same way)."""
    v = str(value).strip()
    face = _FACE.search(v)
    if isinstance(current, str) and current and face and not re.search(r"\d", v):
        right = face.group(1).lower() in ("right", "forward", "forwards")   # forward = the way the game moves
        if _faces_right(current) == right:
            return None, f"it already faces {face.group(1).lower()}"
        return (current.replace("|flip", "") if is_flipped(current) else current + "|flip"), f"faces {face.group(1).lower()}"
    if isinstance(current, str) and _FLIP.search(v) and not re.search(r"\d", v):
        if is_flipped(current):
            return current.replace("|flip", ""), "turned back"
        return current + "|flip", "flipped"
    new, said = resolve_value_plain(value, current, web)
    if not (isinstance(new, str) and isinstance(current, str) and current) or is_flipped(new):
        return new, said
    if _is_sprite(new) and _is_sprite(current):
        # keep the way it faces: emoji face left, library art faces right (a mirrored emoji bird faces right,
        # so the library spaceship replacing it is NOT mirrored; a library bird replaced by an emoji is)
        if _faces_right(current) != _faces_right(new):
            new += "|flip"
    elif is_flipped(current) and new.startswith(("data:", "http")):
        new += "|flip"
    return new, said


def _faces_right(v: str) -> bool:
    return is_asset(v) != is_flipped(v)


LEVELS = {"easy": 1, "normal": 2, "medium": 2, "hard": 3}
LEVEL_NAMES = {1: "easy", 2: "normal", 3: "hard"}


def resolve_value_plain(value, current, web) -> tuple[object, str]:
    """A spoken value -> what the app understands, and how to describe it.
    Numbers stay numbers; colours and emoji pass; "photo of bamboo" fetches a photo; "butterfly" -> 🦋."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, str(value)
    v = str(value).strip()
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            return float(m.group()), m.group()
        w = next((n for word, n in _WORD_NUM.items() if re.search(rf"\b{word}\b", v.lower())), None)
        if w is not None:                              # "three lives"
            return w, str(w)
        lv = v.lower()
        lvl = re.fullmatch(r"(?:on |to )?(easy|normal|medium|hard)(?: mode| level)?", lv)
        step = -1 if re.search(r"easier|less (?:difficult|hard)|too (?:hard|difficult)", lv) else \
            1 if re.search(r"harder|more (?:difficult|challenging)|too easy", lv) else 0
        if lvl or step:                                # difficulty: 1 easy, 2 normal, 3 hard
            new = LEVELS[lvl.group(1)] if lvl else min(3, max(1, round(current) + step))
            if new == round(current) and not lvl:
                return None, f"it is already on {LEVEL_NAMES[new]}"
            return new, f"{LEVEL_NAMES[new]} ({new})"
        factor = {"faster": 1.5, "slower": 0.67, "bigger": 1.4, "smaller": 0.7, "double": 2.0, "half": 0.5,
                  "larger": 1.4, "wider": 1.4, "taller": 1.4, "thicker": 1.4, "longer": 1.4, "tinier": 0.7,
                  "narrower": 0.7, "thinner": 0.7, "shorter": 0.7, "higher": 1.3, "lower": 0.77, "heavier": 1.3,
                  "lighter": 0.77, "stronger": 1.3, "weaker": 0.77, "more": 1.3, "less": 0.77}
        for w, f in factor.items():
            if w in v.lower():
                return round(current * f, 2), f"{w} ({round(current * f, 2)})"
    if SIZE_WORDS.match(v):                         # "make the bird bigger": not a new look (the bird would turn
        return None, "its size is not a live setting"   # into the word "bigger"): a number setting or edit_app
    base = v.split("@")[0].replace("|flip", "")
    if is_asset(base) and (ASSETS.parent / base.lstrip("/")).is_file():
        return v, f"the {_label(v)} picture"            # a library picture by its path (the pick pop-up)
    m = re.match(r"^(?:an?\s+)?(?:photo|picture|image|pic)\s+(?:of\s+)?(?:an?\s+)?(.+)$", v, re.I)
    if m and PHOTOS:
        url = photo(m.group(1), web)
        return (url, f"a photo of {m.group(1)}") if url else (None, f"no photo of {m.group(1)} found")
    if m:                                          # "photo of ice" -> the library's ice, or an ice emoji
        v = m.group(1).strip()
    # "ice pipes" / "pipes made of ice" / "like ice" / "icy": the material, not the thing being changed
    # (live 2026-11-06: "change the pipes to ice pipes" found nothing)
    v2 = re.sub(r"^(?:(?:make|turn|change)\s+(?:it|them)\s+(?:into|to)\s+)?(?:(?:something\s+)?like\s+|"
                r"(?:pipes?|walls?|obstacles?|columns?|towers?|blocks?|platforms?)\s+(?:made\s+)?(?:of|from)\s+)",
                "", v, flags=re.I)
    v2 = re.sub(r"\s+(?:pipes?|walls?|obstacles?|columns?|towers?|blocks?|platforms?|ones?|style|looking)$", "",
                v2, flags=re.I).strip()
    v2 = {"icy": "ice", "frozen": "ice", "icicle": "ice", "icicles": "ice", "wooden": "wood", "stony": "stone",
          "fiery": "fire", "watery": "water", "bricks": "brick"}.get(v2.lower(), v2)
    if v2 and v2.lower() != v.lower() and re.search(r"[a-z]", v2, re.I):
        v = v2
    low = v.lower().replace(" ", "")
    if re.match(r"^#[0-9a-f]{3,8}$", low) or low in CSS_COLORS or re.match(r"^(rgb|hsl)a?\(", low):
        if _is_sprite(current):                    # "make the bird red": keep the bird, tint it
            return f"{current.split('@')[0]}@{low}", f"{v} {_label(current)}"
        return low, v
    if not re.search(r"[a-z]", v, re.I):           # emoji / symbols
        return v, v
    word = re.sub(r"^(an?|the)\s+", "", v.lower()).rstrip("s")
    art = asset_for(v)
    if art and is_asset(current):                  # the app draws library art: stay in that style
        return art, f"the {_label(art)} picture"
    if word in EMOJI or v.lower() in EMOJI:
        e = EMOJI.get(word) or EMOJI[v.lower()]
        return e, f"{v} {e}"
    if art:                                        # no emoji for it, but the library has it (works offline)
        return art, f"the {_label(art)} picture"
    if PHOTOS:
        url = photo(v, web)                        # some other thing: find a picture of it
        return (url, f"a picture of {v}") if url else (v, v)
    return None, (f"there is no picture or emoji for '{v}': offer something close (show_choices shows the "
                  "library), or a colour")
