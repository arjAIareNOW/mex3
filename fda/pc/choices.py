"""The pick pop-up: "show me other birds" / "show me colour schemes" puts 4 numbered choices on the display, and
"number two" / "the owl" / "the blue one" picks one. Pure logic here (no pipeline), so it is easy to test; the agent
shows the pop-up, applies the pick (a live tweak, or an edit / a note for the next build) and closes it.

Choices come from Elkit's art library (fda/assets/manifest.json): its `groups` (bird: bird, parrot, owl ...) and
`palettes` (ocean, sunset, forest ...)."""
from __future__ import annotations

import re

from ..mcp_servers import live

SHOW = 4
_NUM = {"one": 1, "first": 1, "1": 1, "two": 2, "second": 2, "2": 2, "to": 2, "too": 2, "three": 3, "third": 3,
        "3": 3, "four": 4, "fourth": 4, "for": 4, "4": 4, "last": -1}
PALETTE_WORDS = re.compile(r"colou?r(?:s| scheme| schemes| palette| palettes| combination| theme)?|palettes?|"
                           r"schemes?|themes?|looks?|styles?", re.I)
_FILLER = r"(?:um+|uh+|okay|ok|so|yeah|yes|hmm|well|i (?:think|guess|want|like|choose|pick|will take|'ll take|'d like)" \
          r"|let'?s (?:go with|use|take|do)|go with|give me|use|take|pick|choose|i'?ll go with|maybe|please|the|that|" \
          r"option|number|choice|elkit|i|like|want|it|is|good|looks|nice|great|go|with)"


def palette_choice(offset: int = 0) -> list[dict]:
    pals = list(live.manifest().get("palettes", {}).items())
    if not pals:
        return []
    out = []
    for i in range(min(SHOW, len(pals))):
        name, colors = pals[(offset + i) % len(pals)]
        out.append({"name": name, "say": name, "colors": colors})
    return out


def group_of(word: str, current: str | None = None) -> tuple[str, list[str]] | None:
    """The choice group for a spoken thing ("birds", "the car", "background") or for the picture shown now."""
    man = live.manifest()
    groups: dict[str, list[str]] = man.get("groups", {})
    if not groups:
        return None
    names = {i["path"]: i["name"] for i in man.get("items", [])}
    cur = names.get(str(current or "").split("@")[0].replace("|flip", ""))
    w = live._asset_key(word or "")
    syn = {"background": "sky", "backdrop": "sky", "sky": "sky", "place": "scene", "world": "scene", "level": "scene",
           "player": "hero", "character": "hero", "person": "hero", "people": "hero", "animal": "pet", "monster": "enemy",
           "baddie": "enemy", "vehicle": "car", "plane": "flyer", "ship": "flyer", "spaceship": "flyer",
           "item": "treasure", "coin": "treasure", "floor": "ground", "platform": "ground", "tile": "ground",
           "tree": "plant", "fruit": "food", "obstacle": "obstacle", "wall": "material",
           # long things a game tiles: offered as materials (ice, metal, candy ...), not as other sprites
           "pipe": "material", "column": "material", "pillar": "material", "tower": "material", "material": "material",
           "texture": "material", "block": "material", "brick": "material", "beam": "material", "pole": "material"}
    for cand in (w, w.split()[-1] if w else "", syn.get(w), syn.get(w.split()[-1]) if w else None):
        if cand and cand in groups:
            return cand, groups[cand]
    for cand in (w, w.split()[-1] if w else ""):          # a member's name: "parrot" -> the bird group
        for g, members in groups.items():
            if cand and cand in members:
                return g, members
    if cur:
        for g, members in groups.items():
            if cur in members:
                return g, members
    return None


def art_choice(members: list[str], current: str | None = None, offset: int = 0) -> list[dict]:
    """Up to 4 pictures from a group, without the one showing now; `offset` = "show me more"."""
    items = {i["name"]: i for i in live.manifest().get("items", [])}
    cur = str(current or "").split("@")[0].replace("|flip", "")
    pool = [items[m] for m in members if m in items and items[m]["path"] != cur]
    if not pool:
        return []
    start = (offset * SHOW) % len(pool)
    pool = (pool[start:] + pool[:start])[:SHOW]
    return [{"name": i["name"], "say": i.get("say", i["name"]), "img": i["path"]} for i in pool]


def parse_pick(text: str, options: list[dict]) -> int | None:
    """'number two', 'the second one', 'two', 'the owl', 'I'll take the parrot', 'the blue one' -> index; else None.
    Only short answers count (a long sentence is a new request, not a pick)."""
    t = re.sub(r"[^a-z0-9' ]", " ", (text or "").lower()).strip()
    words = t.split()
    if not words or len(words) > 8 or not options:
        return None
    named = []                                            # a name: "the owl", "race car", "ocean" (longest wins:
    for i, o in enumerate(options):                       # "race car" is not "car")
        for nm in {o.get("say", o["name"]).lower(), o["name"].lower().replace("-", " ")}:
            if re.search(rf"\b{re.escape(nm)}s?\b", t):
                named.append((len(nm), i))
    if named:
        return max(named)[1]
    rest = re.sub(rf"\b{_FILLER}\b", " ", t).split()
    if len(rest) == 2 and "one" in rest:
        rest.remove("one")                                # "the second one", "the dark one"
    if len(rest) == 1 and rest[0] in _NUM:                # only a bare number counts ("the dark one" is not 1)
        n = _NUM[rest[0]]
        n = len(options) if n == -1 else n
        if 1 <= n <= len(options):
            return n - 1
    for i, o in enumerate(options):                       # "the blue one": a palette's main colour by name
        c = o.get("colors")
        if c and re.search(r"\b(blue|red|green|orange|purple|pink|brown|black|grey|gray|yellow|dark|white|light)\b", t):
            if _colour_word(c.get("primary", "")) in t or (("dark" in t or "black" in t) and _is_dark(c.get("background", ""))):
                return i
    return None


def is_more(text: str) -> bool:
    return bool(re.search(r"\b(more|other|others|different|next|else|new ones?)\b", text or "", re.I)) and \
        len((text or "").split()) <= 9 and not re.search(r"\b(make|build|create|change)\b", text or "", re.I)


def is_close(text: str) -> bool:
    return bool(re.fullmatch(r"\W*(?:(?:no|nah|none|nope)(?: of (?:them|these|those))?(?:,? thanks?| thank you)?|"
                             r"never ?mind|close(?: it| that| this)?|cancel|forget it|keep (?:it|the old one|"
                             r"what (?:i|we) had)|you (?:pick|choose|decide)|surprise me)\W*", (text or "").lower()))


def you_pick(text: str) -> bool:
    return bool(re.search(r"\byou (?:pick|choose|decide)|surprise me\b", (text or "").lower()))


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        return 128, 128, 128


def _is_dark(h: str) -> bool:
    r, g, b = _rgb(h)
    return 0.299 * r + 0.587 * g + 0.114 * b < 80


def _colour_word(h: str) -> str:
    """'#1e6fd9' -> 'blue' (rough; for 'the blue one')."""
    import colorsys
    r, g, b = (x / 255 for x in _rgb(h))
    hh, l, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.18:
        return "black" if l < 0.25 else "grey"
    d = hh * 360
    if l < 0.3 and 15 < d < 50:
        return "brown"
    for lim, name in ((15, "red"), (45, "orange"), (65, "yellow"), (170, "green"), (255, "blue"), (290, "purple"),
                      (340, "pink"), (361, "red")):
        if d < lim:
            return name
    return "grey"


def palette_note(o: dict) -> str:
    """How the coder is told a chosen palette."""
    c = o["colors"]
    return (f"the {o['name']} colour scheme: page background {c['background']}, cards {c['surface']}, main colour "
            f"{c['primary']}, accent {c['accent']}, text {c['text']}")


def popup(title: str, options: list[dict]) -> dict:
    """The display message."""
    return {"type": "choices", "title": title,
            "options": [{"n": i + 1, "label": o["say"].capitalize(), **({"img": o["img"]} if o.get("img") else {}),
                         **({"colors": o["colors"]} if o.get("colors") else {})} for i, o in enumerate(options)]}


def spoken_list(options: list[dict]) -> str:
    xs = [f"{i + 1}, {o['say']}" for i, o in enumerate(options)]
    return "; ".join(xs)
