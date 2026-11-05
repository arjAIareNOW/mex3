"""Fast path: precise rules for the common room commands (the VCM intents). A match acts immediately, with no LLM call;
anything the rules are not sure about goes to the planner. Rules only fire on the WHOLE utterance being a command,
so "and then she said turn off the lights in the hallway" does not match.
"""
from __future__ import annotations

import re
import time

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "warm white", "cool white",
          "cyan", "magenta", "violet", "teal"]
_UNITS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600}
_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_NUM.update({w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())})
_NUM.update({"a": 1, "an": 1, "hundred": 100})

LEAD = re.compile(r"^(?:(?:hey|hi|ok|okay|yo)\s+)?(?:fda[,\s]+)?(?:(?:can|could|would|will) you\s+)?(?:please\s+)?")
TAIL = re.compile(r"(?:\s+(?:please|for me|now|fda|thanks|thank you))+$")


def norm(text: str) -> str:
    t = text.lower().replace("%", " percent").replace("°", " degrees")
    t = re.sub(r"[^a-z0-9: ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = LEAD.sub("", t)
    return TAIL.sub("", t).strip()


def number(words: str) -> int | None:
    words = words.strip()
    if re.fullmatch(r"\d+", words):
        return int(words)
    total, ok = 0, False
    for w in words.replace("-", " ").split():
        if w in _NUM:
            total = total * 100 if _NUM[w] == 100 and total else total + _NUM[w]; ok = True
        elif w != "and":
            return None
    return total if ok else None


_NUMW = r"(\d+|(?:(?:a|an|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|" \
        r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and)" \
        r"(?:[ -]|$))+)"


def match(text: str, room: dict) -> tuple[list[dict], str] | None:
    """-> (actions, what to say) or None. `room` is the current room state (to say "stop" right)."""
    t = norm(text)
    if not t:
        return None
    L = r"(?:the |all the |all )?lights?"
    m = re.fullmatch(rf"(?:turn|switch|put) (on|off) {L}|(?:turn|switch|put) {L} (on|off)|{L} (on|off)", t)
    if m:
        p = next(g for g in m.groups() if g)
        return [{"tool": "lights", "power": p}], f"Lights {p}."
    m = re.fullmatch(rf"(?:dim|set|turn|put|change|make) {L} (?:to |at |down to |up to )?{_NUMW}\s*percent(?: brightness)?"
                     rf"|(?:set )?brightness (?:to )?{_NUMW}\s*percent", t)
    if m:
        n = number(next(g for g in m.groups() if g))
        if n is not None and 0 <= n <= 100:
            return [{"tool": "lights", "power": "on", "brightness": n}], f"Lights at {n} percent."
    cm = "|".join(sorted(COLORS, key=len, reverse=True))
    m = re.fullmatch(rf"(?:make|turn|set|change|switch) {L} (?:to )?({cm})|({cm}) {L}", t)
    if m:
        c = next(g for g in m.groups() if g)
        return [{"tool": "lights", "power": "on", "color": c}], f"Lights are {c} now."
    m = re.fullmatch(rf"set (?:a |an )?timer (?:for )?{_NUMW}\s*(second|sec|minute|min|hour)s?|{_NUMW}\s*(second|sec|minute|min|hour)s? timer", t)
    if m:
        g = [x for x in m.groups() if x]
        n = number(g[0])
        if n:
            secs = n * _UNITS[g[1]]
            return [{"tool": "timer", "seconds": secs}], f"{_say_duration(secs)} timer, starting now."
    m = re.fullmatch(rf"set (?:the )?(?:thermostat|temperature|aircon|ac) (?:to )?{_NUMW}\s*(?:degrees)?(?: celsius)?", t)
    if m:
        n = number(m.group(1))
        if n and 10 <= n <= 35:
            return [{"tool": "thermostat", "celsius": n}], f"Thermostat set to {n} degrees."
    if re.fullmatch(r"(?:stop|stop it|stop that|be quiet|quiet|silence|enough|shut up|cancel|never mind|nevermind)", t):
        if room.get("ringing"):
            return [{"tool": "stop_ringing"}], "Okay."
        if room.get("music", {}).get("state") == "playing":
            return [{"tool": "music", "action": "stop"}], ""
        return [], ""            # just stop talking (barge-in already did it)
    if re.fullmatch(r"(?:snooze|turn off (?:the )?(?:alarm|timer)|stop (?:the )?(?:alarm|timer|ringing))", t):
        return [{"tool": "stop_ringing"}], "Okay."
    if re.fullmatch(r"(?:pause|pause (?:the )?music|pause (?:the )?song)", t):
        return [{"tool": "music", "action": "pause"}], ""
    if re.fullmatch(r"(?:resume|resume (?:the )?music|unpause|continue (?:the )?music)", t):
        return [{"tool": "music", "action": "resume"}], ""
    if re.fullmatch(r"(?:play (?:some |the )?music|play (?:a |some )?songs?|music)", t):
        return [{"tool": "music", "action": "play"}], "Here's some music."
    if re.fullmatch(r"(?:stop (?:the )?music|stop (?:the )?song|music off)", t):
        return [{"tool": "music", "action": "stop"}], ""
    if re.fullmatch(r"(?:next|next song|next track|skip|skip (?:this )?(?:song|track))", t):
        return [{"tool": "music", "action": "next"}], ""
    if re.fullmatch(r"(?:louder|volume up|turn (?:it|the volume|the music) up|turn up (?:the )?(?:volume|music)|increase (?:the )?volume)", t):
        return [{"tool": "volume", "change": "up"}], ""
    if re.fullmatch(r"(?:quieter|softer|volume down|turn (?:it|the volume|the music) down|turn down (?:the )?(?:volume|music)|lower (?:the )?volume|decrease (?:the )?volume)", t):
        return [{"tool": "volume", "change": "down"}], ""
    m = re.fullmatch(rf"(?:set )?(?:the )?volume (?:to )?{_NUMW}\s*(?:percent)?", t)
    if m:
        n = number(m.group(1))
        if n is not None and 0 <= n <= 100:
            return [{"tool": "volume", "level": n}], f"Volume {n}."
    if re.fullmatch(r"(?:what time is it|what s the time|whats the time|what is the time|time|tell me the time)", t):
        return [], "It's " + time.strftime("%I:%M %p").lstrip("0") + "."
    if re.fullmatch(r"(?:what s|whats|what is) (?:the )?(?:date|day) (?:today)?|what day is it(?: today)?", t):
        return [], "It's " + time.strftime("%A, %B ") + str(int(time.strftime("%d"))) + "."
    if re.fullmatch(r"(?:what are my reminders|what s on my reminders|whats on my reminders|list (?:my )?reminders|"
                    r"(?:do i have )?any reminders|show (?:me )?(?:my )?reminders|read (?:me )?my reminders)", t):
        return [{"tool": "list_reminders"}], ""
    m = re.fullmatch(r"(?:call|phone|ring|dial) ([a-z]+(?: [a-z]+)?)", t)
    if m and m.group(1) not in ("it", "me", "that", "this", "the police"):
        return [{"tool": "call", "contact": m.group(1)}], f"Calling {m.group(1).title()}."
    return None


def _say_duration(s: int) -> str:
    if s % 3600 == 0:
        return f"{s // 3600} hour"
    if s % 60 == 0:
        return f"{s // 60} minute"
    return f"{s} second"


BACKCHANNEL = {"yeah", "yes", "yep", "uh huh", "uh-huh", "mm hmm", "mhm", "mm", "hmm", "okay", "ok", "right", "sure",
               "i see", "oh", "ah", "wow", "nice", "cool", "got it", "alright", "all right", "true",
               "aha", "ah ha", "oh ho", "uh hu", "uhuh", "uh uh", "mm mm", "mmm", "hm", "huh", "oh okay", "oh ok",
               "oh wow", "oh nice", "yeah yeah", "okay okay", "i see i see", "go on", "really", "oh really", "interesting"}


_FILLER = re.compile(r"^(?:u+h*|u+m+|m+|m+h+m*|h+m+|a+h+|o+h+|h+a+|h+o+|h+u+h+|u+h+u+h+|m+h+m+|ye+a*h*|yep|ok(?:ay)?|right|sure|wow)$")


_MUSIC_WORDS = re.compile(r"\b(?:music|song|songs|play|playing|playlist|lofi|lo-fi|track|tune|tunes|radio|beat|beats|"
                          r"album|artist|listen|jam|spotify|put on)\b", re.I)


_GAME_WORDS = re.compile(r"\b(?:game|games|play|sky ?hop|flappy|fly|flying|plane)\b", re.I)


_COLOR = r"red|green|blue|yellow|orange|purple|pink|white|cyan|violet|teal|magenta|colou?r"
# Evidence a person asked for this tool: at least one of these (word starts) must be in what was actually heard.
_EVIDENCE = {
    "lights": r"light|lamp|bright|dark|dim|bulb|glow|" + _COLOR,
    "thermostat": r"temp|degree|warm|cold|hot|cool|chill|heat|freez|air ?con|aircon|thermostat|celsius|stuffy|sweat",
    "timer": r"timer|minute|second|hour|countdown",
    "cancel_timer": r"timer|cancel",
    "alarm": r"alarm|wake",
    "cancel_alarm": r"alarm|wake|cancel|delet|remov|turn off|clear",
    "stop_ringing": r"stop|enough|quiet|shut|off|okay|ok\b|dismiss",
    "add_reminder": r"remind|remember|forget",
    "list_reminders": r"remind|list|schedule|agenda|to ?do",
    "delete_reminder": r"remind|delet|remov|cancel|scratch|clear|forget|eras|drop|get rid|done|all|number|first|last",
    "edit_reminder": r"remind|chang|edit|move|updat|instead|scratch|renam|make it|switch|replac|actually",
    "music": r"music|song|track|tune|play|pause|stop|next|skip|resume|continue|listen|lofi|lo-fi|radio",
    "volume": r"volume|loud|quiet|soft|mute|turn (?:it|that) (?:up|down)|can't hear|cannot hear",
    "call": r"call|phone|ring|dial",
    "message": r"text|messag|send|tell|sms",
    "weather": r"weather|rain|sun|forecast|temperature|hot|cold|umbrella|wind|storm|cloud|typhoon|humid",
}
_EVIDENCE_RE = {k: re.compile(r"\b(?:" + v + ")", re.I) for k, v in _EVIDENCE.items()}


def plausible(action: dict, heard: str) -> bool:
    """Small offline planners add actions nobody asked for ("how are you" -> play music, "can you make an app" ->
    lights off) and then copy them from the conversation into later turns. An action needs evidence in what was
    actually said. Web search is never blocked (questions can be about anything)."""
    tool = action.get("tool")
    if tool == "music" and action.get("action") in ("play", "resume"):
        return bool(_MUSIC_WORDS.search(heard))
    if tool == "game":
        return action.get("action") != "open" or bool(_GAME_WORDS.search(heard))
    rx = _EVIDENCE_RE.get(tool)
    return rx is None or bool(rx.search(heard))


_UNFINISHED_END = {"for", "on", "about", "the", "a", "an", "and", "or", "but", "to", "of", "with", "in", "at", "from",
                   "my", "your", "like", "is", "are", "search", "find", "play", "set", "make", "turn", "remind", "me",
                   "um", "uh", "so", "because", "if", "when", "that", "into", "some", "what", "how"}


def sounds_unfinished(text: str) -> bool:
    """'can you search the net for?' / 'remind me to' / 'set the lights to': the person paused mid-sentence."""
    words = norm(text).split()
    return len(words) >= 2 and words[-1] in _UNFINISHED_END


_VOCATIVE = re.compile(
    r"^\W*(?:hey\s+|oh\s+)?(honey|hon|babe|baby|sweetie|sweetheart|darling|dear|mom|mum|mommy|mama|ma|dad|daddy|papa|"
    r"pa|grandma|grandpa|lola|lolo|nanay|tatay|ate|kuya|guys|everyone|son|kids|kiddo|bro|dude|sis)\b\s*[,!?.]", re.I)
_NAME_COMMA = re.compile(r"^\W*(?:(?:[Hh]ey|[Hh]i|[Oo]h|[Oo]kay|OK|[Oo]k)\s+)?([A-Z][a-z]{2,})\s*,")
_VERBS = {   # "Translate, ..." / "Describe, ..." are commands, not names
    "translate", "describe", "explain", "tell", "play", "make", "show", "give", "set", "turn", "open", "close", "call",
    "find", "search", "remind", "help", "read", "write", "start", "pause", "check", "list", "add", "remove", "take",
    "bring", "build", "create", "define", "spell", "count", "compare", "summarize", "summarise", "name", "guess",
    "imagine", "pretend", "say", "repeat", "continue", "go", "send", "text", "book", "order", "convert", "calculate",
    "recommend", "suggest", "teach", "sing", "draw", "plan", "schedule", "cancel", "skip", "resume", "lower", "raise",
    "dim", "brighten", "increase", "decrease", "honestly", "seriously", "basically", "anyway", "besides", "however",
    "otherwise", "finally", "first", "second", "third", "lastly", "meanwhile", "personally", "luckily", "sadly"}
_NOT_NAMES = {"hey", "okay", "ok", "please", "well", "so", "yes", "yeah", "no", "oh", "hi", "hello", "sorry", "thanks",
              "wait", "now", "alright", "also", "actually", "um", "uh", "and", "but", "then", "listen", "look", "right",
              "sure", "great", "cool", "fine", "good", "nice", "wow", "hmm", "anyway", "first", "next", "today",
              "tomorrow", "tonight", "maybe", "honestly", "basically", "excuse", "pardon", "quick", "question",
              "what", "who", "why", "how", "where", "when", "which", "really", "seriously", "come", "hold", "stop",
              "okay", "god", "jeez", "man", "boy", "darn", "damn", "geez", "huh", "ah", "ahh", "ooh", "aha", "yep", "nope"}
_PHONE = re.compile(r"\b(talk to you later|i'?ll call you back|bye[- ]bye|love you,? bye|can you hear me on the line)\b",
                    re.I)


def addressed_to_other(text: str, own_name: str = "FDA") -> bool:
    """'Honey, did you buy milk?' / 'John, pass the salt' / '...talk to you later, bye': said to another person."""
    t = text.strip()
    if _VOCATIVE.match(t) or _PHONE.search(t):
        return True
    m = _NAME_COMMA.match(t)
    if not m:
        return False
    w = m.group(1).lower()
    return w not in _NOT_NAMES and w not in _VERBS and w != own_name.lower() and not w.endswith(("ly", "ing"))


_FOR_ME = re.compile(r"\?|\b(you|your|can|could|would|will|tell|what|who|how|why|when|where|which|please|"
                     r"make|build|create|turn|set|play|open|show|give|help|stop|wait|hey|add|change|remove|put|switch|replace|find|remind)\b", re.I)


def probably_for_me(text: str, own_name: str = "FDA") -> bool:
    """A question or request a person in the room would say to the assistant ("Wait, what time is it?",
    "Tell me a story"). Used to overrule a "(silent)" from a small LLM; side talk and broadcasts rarely match."""
    t = text.strip()
    if len(norm(t).split()) < 2 or addressed_to_other(t, own_name):
        return False
    return bool(_FOR_ME.search(t)) or own_name.lower() in t.lower()


def is_backchannel(text: str) -> bool:
    """'uh huh', 'mhm-hmm', 'yeah yeah', 'oh wow', 'I see'... (listener noises, not a turn)."""
    t = norm(text)
    if not t:
        return False
    if t in BACKCHANNEL:
        return True
    words = t.split()
    return len(words) <= 3 and all(_FILLER.match(w) for w in words)


# VCM intent -> action, for the instant path before the transcript (only argument-free intents)
VCM_ACTIONS = {"LIGHT_ON": [{"tool": "lights", "power": "on"}], "LIGHT_OFF": [{"tool": "lights", "power": "off"}],
               "PAUSE": [{"tool": "music", "action": "pause"}], "NEXT": [{"tool": "music", "action": "next"}],
               "VOLUME_UP": [{"tool": "volume", "change": "up"}], "VOLUME_DOWN": [{"tool": "volume", "change": "down"}]}


# gpt-oss sometimes writes its private thinking into the reply ("We We Okay The user wants Articuno bird. We have
# editapp ..."). Those sentences must never be spoken, nor kept in the conversation (the model copies them).
_THINK_LEAD = r"(?:(?:we|so|okay|ok|now|thus|then|alright|hmm|well|but)[\s,.:]+)*"
THINKING = re.compile(
    r"(?i)^\s*" + _THINK_LEAD + r"(?:the\s+)?(?:user|person|human)\s+(?:just\s+|now\s+|probably\s+)?"
    r"(?:wants?|asks?|asked|is\s+asking|says?|said|requests?|requested|needs?|mentioned|wanted|likely|might)\b"
    r"|^\s*" + _THINK_LEAD + r"we\s+(?:need|have|should|must|can|could|will|might|may|are\s+to|'ll|'d)\s+(?:to\s+)?"
    r"(?:call|use|respond|answer|reply|ask|check|output|produce|provide|say|give|comply|follow|invoke|run|"
    r"edit|create|open|tweak|search|fetch|look|figure|decide|clarify|interpret|keep|be|not|just|now|also|first|"
    r"the|a|an|this|that|it|tool|function)\b"
    r"|^\s*(?:we\s+){2,}|^\s*(?:okay|ok|so|alright)[\s,.:]+we\b"
    r"|^\s*let'?s\s+(?:call|use|respond|answer|output|produce|craft|reply|think|see\s+what|check\s+(?:the|if|what))\b"
    r"|^\s*(?:probably|maybe|likely|perhaps)\s+(?:answer|respond|reply|say|call|use|just|we|they|the\s+user)\b"
    r"|^\s*(?:need\s+to|must|should)\s+(?:call|use|respond|answer|reply|output|produce|ask|say)\b"
    r"|^\s*(?:answer|response|reply|final|output|analysis)\s*:"
    r"|\b(?:edit_?app|create_?app|tweak_?app|open_?app|undo_?app|app_?settings|list_?apps|web_?search|"
    r"read_?page|play_?video|get_?news|get_?weather|get_?time|video_?control|app_?control|interface_?tab|"
    r"show_?screen|function\s+call|tool\s+call|the\s+tools?\s+(?:to|we)|system\s+prompt|developer\s+message|"
    r"the\s+instructions?\s+say)\b")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def is_thinking(text: str) -> bool:
    """True if any sentence of `text` is the model thinking aloud rather than talking to the person."""
    return any(THINKING.search(s) for s in _SENT.split(text or "") if s.strip())


def drop_thinking(text: str) -> str:
    """`text` without the sentences that are the model thinking aloud."""
    return " ".join(s for s in _SENT.split(text or "") if s.strip() and not THINKING.search(s)).strip()


def could_be_thinking(text: str) -> bool:
    """While a reply is still only a few words, could it turn out to be thinking aloud? (then wait for more)"""
    w = re.findall(r"[a-z']+", (text or "").lower())
    return bool(w) and len(w) < 7 and w[0] in {"the", "user", "we", "so", "okay", "ok", "now", "thus", "let's",
                                                "lets", "probably", "maybe", "likely", "perhaps", "need",
                                                "alright", "hmm", "well", "answer", "final", "analysis", "person",
                                                "must", "should", "then", "but"}


def is_garbage(text: str) -> bool:
    """The model stuck on dots: "... … --- … The … … We" (mostly tokens without a letter or digit)."""
    t = (text or "").strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return len(t) >= 3
    toks = t.split()
    # junk = runs of dots / ellipses / "---" (emoji and a list "-" are not junk)
    junk = sum(1 for w in toks if re.fullmatch(r"[.…​‌‍]+[,!?]*|-{2,}|[.…]*-{2,}[.…]*", w))
    return len(toks) >= 4 and junk / len(toks) >= 0.5
