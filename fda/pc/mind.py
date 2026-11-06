"""Change of mind for any request (user 2026-11-06: "the wait no, change it to ... concept should work on the agent
totality"). merge(first, correction) gives the request they want now:

    "Set a timer for 5 minutes."  + "Wait, no, 10 minutes."            -> "set a timer for 10 minutes"
    "Turn the lights blue."       + "Actually, red."                   -> "turn the lights red"
    "Add milk to my shopping list" + "No wait, eggs."                  -> "add eggs to my shopping list"
    "Make me a snake game"        + "Wait, no, a racing game instead." -> "make me a racing game"
    "Open the room tab"           + "I meant the office tab."          -> "open the office tab"

with what was dropped (v1) and what replaces it (v2), for "Okay, not 5 minutes: 10 minutes instead."
Pure text work, no side effects (the agent undoes and redoes)."""
from __future__ import annotations

import re

from .. import rules

COLORS = sorted(set(rules.COLORS) | {"black", "brown", "grey", "gray", "gold", "silver", "navy", "lime", "beige",
                                       "maroon", "turquoise", "light blue", "dark blue", "dark green", "light green"},
                key=len, reverse=True)
_COLOR = r"(?:" + "|".join(re.escape(c) for c in COLORS) + r")"
_NUMWORD = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
            r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half")
_UNIT = (r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|percent|degrees?(?: celsius)?|am|pm|o ?clock|lives?|"
         r"days?|weeks?|times?|points?)")
# a number with its unit: "5 minutes", "twenty five percent", "7:30 pm", "an hour"
_NUM = rf"(?:\d+(?::\d\d)?|(?:(?:{_NUMWORD})(?:[ -](?:and )?(?:{_NUMWORD}))*)|(?:an?(?= {_UNIT})))"
_NUM_PHRASE = re.compile(rf"\b{_NUM}(?: {_UNIT})?\b")
_ONLY_NUM = re.compile(rf"^{_NUM}(?: {_UNIT})?$")

# the correction's own words around the new value: "I want it to be", "make it", "instead"
_LEAD = re.compile(r"^(?:no |nope |wait |sorry |actually |i meant |i mean |oh |um |uh |hmm |okay |ok )+")
_SAY = re.compile(r"^(?:i want (?:it|that|them|one|the \w+) (?:to be |to |at |for )?|i want |i d like (?:it |them )?(?:to be )?|"
                  r"i would like (?:it )?|make (?:it|that|them|those) (?:be )?|change (?:it|that|them) to |"
                  r"set (?:it|that) (?:to |for |at )?|put (?:it|that) (?:on |at |to )?|let s (?:do |go with |have |make it )?|"
                  r"go with |it should be |they should be |(?:it s|that s|its) |do |use |try |how about |what about |"
                  r"rather |better |just |please )+")
_INSTEAD = re.compile(r"(?: (?:instead|rather)(?: of (?:that|it|this|\w+))?| not (?:that|\w+))$")
VERBS = re.compile(r"^(?:turn|switch|set|make|change|add|put|remove|take|delete|open|close|show|play|call|phone|ring|go|"
                   r"create|build|search|look|find|tell|remind|cancel|start|stop|dim|pause|resume|read|send|text|"
                   r"what|who|where|when|how|is|are|can|could|do|does|give|let|move|switch|paint|colou?r|draw|write)\b")
_PREP = re.compile(r"\b(?:called|named|titled|about|into|to|for|with|as|of|at|on|in|from)\b")


def norm(text: str) -> str:
    return rules.TAIL.sub("", re.sub(r"\s+", " ", re.sub(r"[^a-z0-9: ]+", " ", text.lower().replace("%", " percent")
                                                         .replace("°", " degrees").replace("'", ""))).strip()).strip()


def clean(correction: str) -> str:
    """'Wait, no. I want it to be black instead.' -> 'black'"""
    s = norm(correction)
    for _ in range(3):
        s2 = _SAY.sub("", _LEAD.sub("", s)).strip()
        s2 = _INSTEAD.sub("", s2).strip()
        if s2 == s:
            break
        s = s2
    return s


def _words(s: str) -> list[str]:
    return s.split()


def _bare(s: str) -> str:
    return re.sub(r"^(?:the|a|an|my|our|some) ", "", s)


def _diff(a: str, b: str) -> tuple[str, str]:
    """The part that differs: 'turn the lights blue' / 'turn the lights red' -> ('blue', 'red')."""
    x, y = _words(a), _words(b)
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    j = 0
    while j < min(len(x), len(y)) - i and x[-1 - j] == y[-1 - j]:
        j += 1
    a, b = x[i:len(x) - j], y[i:len(y) - j]
    only_a, only_b = [w for w in a if w not in b], [w for w in b if w not in a]
    if only_a and only_b and (len(only_a) < len(a) or len(only_b) < len(b)):    # "on the lights" / "the lights off"
        a, b = only_a, only_b
    return " ".join(a), " ".join(b)


def _replace_last(text: str, rx: re.Pattern, new: str) -> tuple[str, str] | None:
    hits = list(rx.finditer(text))
    hits = [h for h in hits if h.group(0).strip() and h.group(0) not in ("a", "an")]
    if not hits:
        return None
    h = hits[-1]
    return text[:h.start()] + new + text[h.end():], h.group(0)


def merge(first: str, correction: str) -> dict | None:
    """-> {"merged", "v1", "v2", "how"} or None when it is not clear what the correction changes."""
    f, s = norm(first), clean(correction)
    f = rules.LEAD.sub("", f).strip()
    if not f or not s or s == f:
        return None
    # 1. a whole new request: "turn the lights red", "open the office tab"; "turn them off": them = the lights
    if (VERBS.match(s) and len(_words(s)) >= 3) or rules.match(s, {}) is not None:
        thing = re.search(rf"\b(?:the|my|all the) (?!(?:same|{_COLOR})\b)[a-z]+", f)
        if thing and re.search(r"\b(?:them|it|that|those)\b", s):
            s = re.sub(r"\b(?:them|it|that|those)\b", thing.group(0), s, count=1)
        v1, v2 = _diff(f, s)
        return {"merged": s, "v1": v1 or f, "v2": v2 or s, "how": "whole"}
    # 2. a colour for a colour
    if re.fullmatch(rf"(?:a |the )?{_COLOR}(?: one| colou?r)?", s):
        c2 = re.search(_COLOR, s).group(0)
        r = _replace_last(f, re.compile(rf"\b{_COLOR}\b"), c2)
        if r:
            return {"merged": r[0], "v1": r[1], "v2": c2, "how": "colour"}
    # 3. a number for a number: "10 minutes", "seven", "8 am", "25 percent"
    if _ONLY_NUM.match(s):
        r = _replace_last(f, _NUM_PHRASE, s)
        if r:
            old = r[1]
            unit = re.search(rf" ({_UNIT})$", old)
            new = s if re.search(rf" {_UNIT}$", s) or not unit else f"{s} {unit.group(1)}"
            merged = f[:f.rfind(old)] + new + f[f.rfind(old) + len(old):]
            return {"merged": merged, "v1": old, "v2": new, "how": "number"}
    # 4. a room command: the thing in it ("add milk to my list" + "eggs"; "call mom" + "dad")
    hit = rules.match(f, {})
    if hit and hit[0]:
        a = hit[0][0]
        want_list = s.endswith(" list")
        for field in (("list",) if want_list else ("items", "contact", "text", "label")):
            old = str(a.get(field) or "")
            if old and re.search(rf"\b{re.escape(old)}\b", f):
                new = _bare(re.sub(r" list$", "", s)) if want_list else s
                return {"merged": re.sub(rf"\b{re.escape(old)}\b", new, f, count=1), "v1": old, "v2": new,
                        "how": "room"}
    # 5. same last word: "a snake game" + "a racing game" -> the words before it
    fw, sw = _words(f), _words(s)
    if len(sw) >= 2 and sw[-1] in fw:
        end = len(fw) - 1 - fw[::-1].index(sw[-1])
        start = end
        while start > 0 and fw[start - 1] not in ("me", "make", "build", "create", "open", "play", "show", "is", "be") \
                and not _PREP.fullmatch(fw[start - 1]) and end - start < len(sw) + 1:
            start -= 1
        if start < end:
            old = " ".join(fw[start:end + 1])
            merged = " ".join(fw[:start] + sw + fw[end + 1:])
            return {"merged": merged, "v1": old, "v2": s, "how": "same end"}
    # 6. after the last "to / for / called / about": "change the sky to the night sky" + "the beach"
    preps = list(_PREP.finditer(f))
    if preps:
        p = next((m for m in reversed(preps) if m.group(0) in ("called", "named", "titled")), preps[-1])
        old = f[p.end():].strip()
        if s.startswith(p.group(0) + " "):           # "the weather in Manila" + "in Cebu": one "in"
            s = s[len(p.group(0)) + 1:]
        if old and len(_words(old)) <= 6:
            return {"merged": f"{f[:p.end()]} {s}", "v1": old, "v2": s, "how": "after " + p.group(0)}
    # 7. a short request ending in the thing: "play jazz" + "rock"
    if len(fw) >= 2 and len(sw) <= 3 and VERBS.match(f):
        old = fw[-1]
        return {"merged": " ".join(fw[:-1] + sw), "v1": old, "v2": s, "how": "last word"}
    return None


def lead(m: dict) -> str:
    """'Okay, not 5 minutes: 10 minutes instead.' (a long first part: just 'Okay, instead: ...')"""
    v1, v2 = _bare(m["v1"]), _bare(m["v2"])
    if len(_words(v1)) > 5 or not v1:
        return "Okay, changing that."
    return f"Okay, not {v1}: {v2} instead."
