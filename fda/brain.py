"""The two Qwen2-Audio calls of a turn (same server, port 8000):

1. hear    : user audio -> {"heard": "..."}   (JSON-constrained so the transcript comes back clean, ~150-300 ms)
2. planner : transcript + conversation -> {"for_me", "do", "say"}   (text only, schema-constrained, few-shot)

Measured on DGX3: the split is far more reliable than asking for everything from the audio in one call (that version
claimed actions without doing them, answered news from memory and replied to the TV). Order inside the planner JSON:
for_me (stop early if not for FDA) -> say (streams into TTS at once) -> do. Measured on 23 test sentences: say-first
21/23 correct with the first word after ~135 ms; do-first 20/23 after ~330 ms (cfg.planner_say_first).
"""
from __future__ import annotations

import json
import time

TOOLS = {
    "lights": {"power": {"enum": ["on", "off"]}, "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
               "color": {"type": "string"}},
    "thermostat": {"celsius": {"type": "number"}},
    "timer": {"seconds": {"type": "integer", "minimum": 1}, "label": {"type": "string"}},
    "cancel_timer": {},
    "alarm": {"time": {"type": "string"}},
    "cancel_alarm": {"time": {"type": "string", "description": "the alarm's time like 06:30, or 'all'; leave out "
                              "when there is only one"}},
    "stop_ringing": {},
    "add_reminder": {"text": {"type": "string"}, "time": {"type": "string"},
                     "repeat": {"type": "string", "description": "for a repeating reminder: daily, weekdays, weekends, or a day like monday; leave out for once"}},
    "list_reminders": {},
    "delete_reminder": {"which": {"type": "string", "description": "words from the reminder, its number "
                                  "(1 = first), 'last', or 'all'"}},
    "edit_reminder": {"which": {"type": "string", "description": "words from the reminder, its number, or 'last'"},
                      "text": {"type": "string", "description": "new text (leave out to keep it)"},
                      "time": {"type": "string", "description": "new time like 18:00 (leave out to keep it)"},
                      "repeat": {"type": "string", "description": "for a repeating reminder: daily, weekdays, weekends, or a day like monday; leave out for once"}},
    "list_items": {"action": {"enum": ["add", "remove", "show", "clear"]},
                   "items": {"type": "string", "description": "the items, comma separated (add / remove)"},
                   "list": {"type": "string", "description": "which list: shopping (default), to-do, or a name"}},
    "music": {"action": {"enum": ["play", "pause", "resume", "stop", "next"]}},
    "volume": {"change": {"enum": ["up", "down"]}, "level": {"type": "integer", "minimum": 0, "maximum": 100}},
    "call": {"contact": {"type": "string"}},
    "message": {"contact": {"type": "string"}, "text": {"type": "string"}},
    "weather": {"place": {"type": "string"}},
    "web_search": {"query": {"type": "string"}},
    "game": {"action": {"enum": ["open", "close", "set", "restart", "pause", "resume"]},
             "plane_color": {"type": "string"}, "tower_color": {"type": "string"},
             "sky": {"enum": ["sunset", "day", "night", "space"]},
             "speed": {"enum": ["slower", "faster", "slow", "normal", "fast"]},
             "gap": {"enum": ["narrower", "wider", "narrow", "normal", "wide"]},
             "gravity": {"enum": ["lower", "higher", "low", "normal", "high"]},
             "size": {"enum": ["smaller", "bigger", "small", "normal", "big"]},
             "title": {"type": "string"}},
}
REQUIRED = {"thermostat": ["celsius"], "timer": ["seconds"], "alarm": ["time"], "add_reminder": ["text"], "delete_reminder": ["which"],
            "edit_reminder": ["which"], "list_items": ["action"],
            "music": ["action"], "call": ["contact"], "message": ["contact", "text"], "web_search": ["query"],
            "game": ["action"]}

HEAR_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"heard": {"type": "string"}},
               "required": ["heard"]}
HEAR_PROMPT = "Transcribe exactly what is said in the audio, word for word. Do not answer it."


def _order(say_first: bool) -> list[str]:
    return ["for_me", "say", "do"] if say_first else ["for_me", "do", "say"]


def plan_schema(say_first: bool = True) -> dict:
    items = [{"type": "object", "additionalProperties": False,
              "properties": {"tool": {"const": name}, **props}, "required": ["tool", *REQUIRED.get(name, [])]}
             for name, props in TOOLS.items()]
    props = {"for_me": {"type": "boolean"}, "do": {"type": "array", "maxItems": 4, "items": {"anyOf": items}},
             "say": {"type": "string"}}
    order = _order(say_first)
    return {"type": "object", "additionalProperties": False, "properties": {k: props[k] for k in order},
            "required": order}


PLANNER = """You are {name}, a friendly voice assistant in a room, talking with people in real time.
You get the text the microphone just heard. Decide: for_me (is the person talking TO you?), do (actions), say (your
short spoken reply, one or two sentences, natural, no lists or markdown).
- You CAN control lights, thermostat, timers, alarms, reminders, music, volume, calls and messages: always use the tool.
- The game tool opens Sky Hop (a tap-to-fly game) on the display and changes it live while it is open: plane and
  tower colours, sky, speed, gap, gravity, size, title. "The plane" or "the game" means the game, not the lights.
- Weather -> weather tool. Current facts that change (news, sports results, who holds an office, prices, schedules)
  -> web_search, and say only "Let me check." Never answer those from memory.
- Timeless general knowledge, stories, jokes, advice, chit-chat: just answer in say, briefly (2-3 sentences).
- Never say something is done unless its tool is in do. Say durations and times like people do (two minutes, 7 PM).
- for_me is false if they address another person by name or relation (honey, mom, John), sound like a phone call, or
  sound like TV/radio/news. Then do=[] and say="".
- If your last reply ends with [interrupted], they only heard the part before it."""
CONTEXT = "Now: {now}. Location: {location}. Room: {room}"

EXAMPLES = [   # order matters: "not for me" examples are spread out, and the last ones are for_me=true
    ("turn off the lights", True, [{"tool": "lights", "power": "off"}], "Lights off."),
    ("John, pass me the salt", False, [], ""),
    ("make it cooler, eighteen degrees", True, [{"tool": "thermostat", "celsius": 18}], "Setting it to 18 degrees."),
    ("set a timer for five minutes", True, [{"tool": "timer", "seconds": 300}], "Five minute timer started."),
    ("is it going to rain later", True, [{"tool": "weather"}], "Let me check."),
    ("stocks fell sharply in afternoon trading, analysts said", False, [], ""),
    ("play some music", True, [{"tool": "music", "action": "play"}], "Here's some music."),
    ("who won the world cup", True, [{"tool": "web_search", "query": "latest FIFA World Cup winner"}], "Let me check."),
    ("remind me to stretch at 3 pm", True, [{"tool": "add_reminder", "text": "stretch", "time": "15:00"}],
     "I'll remind you to stretch at 3 PM."),
    ("mom, are you coming home for dinner", False, [], ""),
    ("cancel the timer", True, [{"tool": "cancel_timer"}], "Timer cancelled."),
    ("wake me up at six thirty", True, [{"tool": "alarm", "time": "06:30"}], "Alarm set for 6:30."),
    ("text Anna that I'm running late", True, [{"tool": "message", "contact": "Anna", "text": "I'm running late"}],
     "Sent to Anna."),
    ("yes I'll send the report tomorrow, talk to you later", False, [], ""),
    ("it's too bright in here", True, [{"tool": "lights", "brightness": 40}], "Dimmed the lights to 40 percent."),
    ("what are my reminders", True, [{"tool": "list_reminders"}], ""),
    ("who is the president of France", True, [{"tool": "web_search", "query": "current president of France"}],
     "Let me check."),
    ("call mom", True, [{"tool": "call", "contact": "mom"}], "Calling Mom."),
    ("can you search the internet for a chicken adobo recipe", True,
     [{"tool": "web_search", "query": "chicken adobo recipe"}], "Let me check."),
    ("search wikipedia on plants", True, [{"tool": "web_search", "query": "plants wikipedia"}], "Let me check."),
    ("what else can you do", True, [],
     "I can control the lights, temperature, timers, alarms, reminders and music, check the weather, search the web, "
     "and open a little game."),
    ("let's play a game", True, [{"tool": "game", "action": "open"}], "Here's Sky Hop. Tap to fly through the gaps."),
    ("can you make me a flappy bird game", True, [{"tool": "game", "action": "open"}],
     "I can't make that exact game, but here's Sky Hop, a tap-to-fly game like it."),
    ("can you make an app for me", True, [], "I can't build apps yet, but I can open Sky Hop, a little game. Want it?"),
    ("make the plane red and the sky night", True, [{"tool": "game", "action": "set", "plane_color": "red", "sky": "night"}],
     "Red plane, night sky."),
    ("make it faster with bigger gaps", True, [{"tool": "game", "action": "set", "speed": "faster", "gap": "wider"}],
     "Faster, with wider gaps."),
    ("close the game", True, [{"tool": "game", "action": "close"}], "Game closed."),
    ("make the lights green and set a timer for ten minutes", True,
     [{"tool": "lights", "color": "green"}, {"tool": "timer", "seconds": 600}],
     "Lights are green, and your ten minute timer is running."),
    ("tell me something interesting about octopuses", True, [],
     "Octopuses have three hearts and blue blood, and each arm can taste what it touches."),
    ("I'm a bit chilly", True, [{"tool": "thermostat", "celsius": 26}], "Warming it up to 26 degrees."),
    ("how are you", True, [], "I'm doing great, thanks! How about you?"),
    ("what's two plus two", True, [], "That's four."),
]


def _answer(order: list[str], for_me: bool, do: list, say: str) -> str:
    v = {"for_me": for_me, "do": do, "say": say}
    return json.dumps({k: v[k] for k in order})


def _example_turns(say_first: bool) -> list[dict]:
    out = []
    for heard, for_me, do, say in EXAMPLES:
        out += [{"role": "user", "content": heard},
                {"role": "assistant", "content": _answer(_order(say_first), for_me, do, say)}]
    return out


_EX = {True: _example_turns(True), False: _example_turns(False)}


def hear_messages(audio_part: dict) -> list[dict]:
    return [{"role": "user", "content": [audio_part, {"type": "text", "text": HEAR_PROMPT}]}]


def plan_messages(cfg, room_state: dict, history: list[dict], heard: str, static_prefix: bool = False) -> list[dict]:
    """static_prefix=True (offline): the rules and examples come first and never change, so the local model keeps
    them cached and only reads the new turn; the clock and room state go in a short system note at the end."""
    rules_ = PLANNER.format(name=cfg.name)
    ctx = CONTEXT.format(now=time.strftime("%A %d %B %Y, %I:%M %p"), location=cfg.location,
                         room=json.dumps(room_state, separators=(",", ":")))
    hist = []
    for m in history[-2 * cfg.history_turns:]:   # past turns in the same JSON style the model answers in
        if m["role"] == "user":
            hist.append(m)
        else:
            hist.append({"role": "assistant",
                         "content": _answer(_order(cfg.planner_say_first), True, m.get("do", []), m["content"])})
    if static_prefix:
        return ([{"role": "system", "content": rules_}] + _EX[cfg.planner_say_first] + hist
                + [{"role": "system", "content": ctx}, {"role": "user", "content": heard}])
    return ([{"role": "system", "content": rules_ + "\n" + ctx}] + _EX[cfg.planner_say_first] + hist
            + [{"role": "user", "content": heard}])


SEARCH_PROMPT = """You are {name}, a voice assistant. Answer the question in one or two short spoken sentences using
only these web results. No lists, no URLs, no markdown. If the results don't answer it, say you couldn't find it.
Today is {today}.
Question: {q}
Results:
{results}"""
