"""FDA on Pipecat: the agent harness with MCP.

    transport.input()                      FDA audio engine: mic + speaker, AEC3, own-voice gate  (or fdbench)
    -> VADProcessor(Silero)                voice activity, on the Pi
    -> DuplexGuard                         duck on overlap, stop on real interruptions, resume after backchannels
    -> McpSTTService ---- MCP ---->        asr server (Moonshine offline / Groq Whisper)
    -> HeardTap                            display; "uh huh" while talking is not a turn
    -> user aggregator                     Smart Turn v3 (end of turn), >= 3 words to interrupt the bot
    -> LLM (OpenAI-compatible)             Groq gpt-oss-120b (fallback gpt-oss-20b) or local llama.cpp, tool calling
         tools ---- MCP ---->              tools server (create_app, edit_app, weather, news, recipes, search...)
         tools (in process)                FDA's room: lights, thermostat, timers, alarms, reminders, music, game
    -> NotForMe                            "(silent)" replies are not spoken; resume an interrupted reply
    -> McpTTSService ---- MCP ---->        tts server (Piper offline / Deepgram Aura)
    -> SpokenTap -> transport.output() -> assistant aggregator
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from pathlib import Path

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (FunctionCallsStartedFrame, LLMContextFrame, LLMMessagesAppendFrame, LLMTextFrame,
                                   TranscriptionFrame, TTSAudioRawFrame, TTSSpeakFrame, VADUserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (LLMContextAggregatorPair,
                                                                   LLMUserAggregatorParams)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from mcp.client.session_group import StreamableHttpParameters

from .. import brain, rules
from . import fallback_llm, mind
from ..mcp_servers import llm as llmlib
from ..room import Room
from ..config import ROOT
from .processors import (CONTINUE, MEDIA_CMD, DuplexGuard, HeardTap, Mark, NotForMe, RefusalGuard, SpokenTap, ThinkFiller, ToolFiller, TurnClock,
                         is_bare_stop,
                         strip_name)
from .quick import QuickAnswers
from .services import McpConnection, McpSTTService, McpTTSService

SYSTEM = """You are {name}, a voice assistant on a small screen. You speak out loud in real time: warm, relaxed and
natural, usually one or two short sentences. Everything you write is spoken: no lists, no markdown, no emoji,
numbers as words people say.
How you work:
- Questions get spoken answers. "What options are there for the bird", "show me other birds", "what could I use":
  show_choices puts four numbered pictures on the screen (birds, cars, fish, heroes, pets, enemies, backgrounds,
  ground); for anything else say two or three choices out loud in one sentence. Then wait for them to pick (by number
  or name: pick_choice). Do not build, change or open anything until they choose. create_app only when they ask you
  to make something new; a change to what is on the screen is tweak_app or edit_app, never a new app.
- Websites: before create_app for a website, if they did not say the colours, show_choices "colour scheme" with
  for_new and ask which one they like; then create_app with the picked colours in the description (exact hex
  values), plus anything else they picked. "You pick" or "just build it": choose yourself and start.
- Do what the person asks, nothing more. Do not steer the talk to any one thing, do not ask what to do next or what
  to build, and do not list what you can do unless they ask. After a task, a short confirmation is enough.
- Things you can do when asked: build websites, apps and small games live on the screen; the weather and the time
  (get_weather, get_time); the room on the screen: lights, thermostat, timers, alarms, reminders, music and volume.
  A message ending in [now: ...] already has the true time and date: use it, never call get_time then. It is only
  there in case they ask: "the time is up", "a long time" are not questions about the time.
- Building: when they ask for an app, a website or a game and the idea is clear enough, start at once with create_app
  and a rich, specific description (purpose, sections or gameplay, layout, colours, fonts, mood, controls). If it is
  very vague, ask one quick question, or suggest a direction and start. While it builds, describe in two or three
  short sentences what you are building (what is in it, what to do or avoid); never say it is ready before you are
  told it is ready, then say one short thing about it.
- Remember what was built and what they liked or disliked, and build on it: "it", "that", "the button", "make it
  pop" mean the app on screen. Look changes to a game (colours, swap the bird for a butterfly, a photo for the
  pipes, a cactus for the pipes, faster, easier / harder = its difficulty setting): tweak_app.
  Any other change, a new section or feature: edit_app with a clear instruction.
- Changing their mind, about ANYTHING ("green... wait, no, black", "a timer for 5 minutes. Actually, 10", "make me a
  snake game. Wait, no, a racing game instead", "remind me at 5. No wait, 6"): do only the new one; if the first
  one was already done, change or undo it so only the new one is left (edit the reminder, cancel the timer,
  create_app with instead=true while that build still runs). Say it first: "Okay, not green: black instead."
  "Undo that": undo_app_change. "The earlier version": restore_app_version (app_history lists them).
  "Show me the snake game": open_app.
- What you already know well, answer at once with no tool: the population or capital of a country or big city,
  a word in another language (say it as a sentence: "In Italian, thank you is grazie."), how to do something,
  science, history, definitions. A lookup makes them wait several seconds in silence.
- Looking things up: for prices, flights, schedules, news, people or anything current or that you are not sure of,
  use web_search (get_news for headlines, wikipedia for background). If the snippets lack a detail they asked for
  (the airline, the exact price, the date), read_page the best link. Then say the answer in one or two sentences:
  the number, the name, and which site it is from. The links show on the screen by themselves: say "the link is on
  the screen", never read a web address out loud. If the results do not say it, say so; never make up a price.
- Videos: only when they ask for a video and say what it is, play_video with exactly what they asked; never pick
  one they did not ask for. "Play a video" with no subject: ask which. "Open YouTube" means the real YouTube:
  ask what they want to watch, then play_video. Never create_app a copy of YouTube or any real site or app
  unless they say to build one. While a video is on the screen, pause, play,
  mute, louder, quieter, skip, restart or close mean the video: video_control (not the room volume or music).
  While a video plays you only hear requests that say your name.
- The person has no mouse or keyboard: everything is by voice. To use the app on the screen (go to a tab, page or
  section, click a button, scroll, type into a field, press a key in a game, read what it says), use app_control.
  If it says not found, choose from the options it lists, or tell them in a few words what is there.
- The screen has three tabs: Virtual Studio (games, websites, videos, photos, searches), Virtual Home (the room: lights,
  thermostat, timers, alarms, reminders, lists, music) and Virtual Office (slides, documents, spreadsheets: build
  them with create_app like any app; they open in the Office tab). "Show me the room" is show_screen room, "the
  office" is show_screen office, "back to the studio" is show_screen interface. When a tool moves the
  screen to another tab, say so first in a few words. Virtual Studio has a sub-tab for every app, game, video, photo set and search,
  each with its own name ("Snake game", "Davao links"); the tool results say the name. Switch or close one with
  interface_tab (action go / close / list); when you open something new, you may say which tab it is on.
- A story, a joke, a poem or an explanation they ask for: tell it (a short version, a few sentences, and keep
  going if they want more). Never answer a harmless request with "I'm sorry, but I can't help with that."
- Talk like a person: short acknowledgements, a little enthusiasm, an opinion when it helps. Do not repeat what they
  said, do not list options unless they asked for options, do not explain your tools.
- Stop when the answer is said: never add offers like "let me know if you need anything else" or "what would
  you like next".
- Everything you write is spoken aloud to them. Think silently; write only the words you say TO them. Never
  write "the user wants", "we need to", or plans about which tool to call. No stage directions or asides in
  brackets like "(no further response)" or "(pause)": when you have nothing more to say, just stop.
- Never say something is done unless the tool said so. If something fails, say so and offer to try again.
- If they interrupt you, drop what you were saying and answer what they just said.
- Before something that cannot be undone or costs a lot (clearing a whole list, deleting all reminders or alarms,
  going back to an old version of an app, building a big new app they only wondered about), ask once in a few words
  ("Clear the whole shopping list?") and do it only after a yes. One clear request for one thing (cancel the timer,
  remove milk, turn off the lights) needs no question: just do it.
- Only when speech is clearly not meant for you (a TV, or someone talking to another person by name), reply with
  exactly (silent) and nothing else."""

FEW_SHOT = [   # how "(silent)" works; short so they cost little prompt
    {"role": "user", "content": "Stocks fell sharply in afternoon trading, analysts said, as investors weighed the "
                                "latest inflation figures."},
    {"role": "assistant", "content": "(silent)"},
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hey! What can I do for you?"},
]

_TWEAK = re.compile(r"^\W*(?:(?:can|could) you\s+|please\s+|now\s+|and\s+|okay,?\s+)*"
                    r"(?:change|make|turn|replace|swap|set|switch|paint|colou?r)\s+(?:the\s+|my\s+|its\s+)?"
                    r"([a-z]+(?:\s[a-z]+)?)\s+(?:to|into|with|for|as|be)\s+(.+)$", re.I)

_TWEAK_SHORT = re.compile(r"^\W*(?:(?:can|could) you\s+|please\s+|now\s+|okay,?\s+)*(?:make|turn|paint|colou?r)\s+"
                          r"(?:the\s+|my\s+|(?=it\s|him\s|her\s))([a-z]+)\s+((?:a (?:lot|bit|little) )?[a-z]+(?:\s[a-z]+)?)\W*$", re.I)

# "make the bird green. wait, no. black instead" / "the pipes red, actually blue" / "make it green, sorry, black":
# a correction needs a cue (wait / actually / sorry / instead), so "make the sky blue, no clouds" is not one
_CORRECTION = re.compile(r"^(?P<first>.+?)[.,!]?\s+(?:(?:wait|hold on)[.,!]?\s+(?:no|nope)|actually|sorry|no wait|"
                         r"(?:no|nope)(?=[.,!]?\s+.+\binstead\b))[.,!]?\s+(?:no[.,!]?\s+)?(?P<second>.+)$", re.I)

_CORRECTION_ALONE = re.compile(r"^(?:(?:wait|hold on)[.,!]?\s+)?(?:no[.,!]?\s+wait|no|nope|actually|sorry|i meant)[.,!]?\s+"
                               r"(?:no[.,!]?\s+)?(?P<second>.+)$", re.I)

_LIVES = re.compile(r"^\W*(?:(?:can|could) you\s+|please\s+|okay,?\s+)*(?:give me|i want|let me have|set (?:it to|the lives to)"
                    r"|make it|change (?:it|the lives) to)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                    r"\s+(?:lives|life|tries|chances)(?:\s+please)?\W*$", re.I)

# "make it easier", "make the game a bit harder", "set it to hard", "change the difficulty to easy", "easy mode",
# "it's too hard", "difficulty hard" -> one of the groups holds what they want
_DIFFICULTY = re.compile(
    r"^\W*(?:(?:can|could) you\s+|please\s+|now\s+|okay,?\s+|let'?s\s+)*(?:"
    r"(?:make|set|put|change|turn|switch)\s+(?:it|this|the game|the difficulty|the level|difficulty)?\s*"
    r"(?:(?:a )?(?:bit|little|lot)\s+)?(?:to\s+|on\s+|into\s+)?(easier|harder|more difficult|less difficult|"
    r"more challenging|easy|normal|medium|hard)(?:\s+mode|\s+level|\s+difficulty)?"
    r"|(easy|normal|medium|hard)\s+(?:mode|level|difficulty)"
    r"|difficulty\s+(?:to\s+|on\s+)?(easy|normal|medium|hard)"
    r"|(?:it'?s|it is|this is|the game is|that'?s)\s+(too hard|too difficult|too easy)"
    r")(?:\s+please)?\W*$", re.I)

# MCP tools the agent gets (the server has more; `wikipedia` is left out: general knowledge is answered directly,
# which is ~5-8 s faster with a local model; current facts still go through web_search)
# the tools the agent is given (the tools server has more: news, web search, documents, device check...)
AGENT_MCP_TOOLS = ["create_app", "edit_app", "tweak_app", "undo_app_change", "app_history", "restore_app_version",
                   "list_apps", "open_app", "app_settings", "get_weather", "get_time",
                   "web_search", "read_page", "get_news", "wikipedia", "play_video",
                   # the room, served by the tools MCP server through the agent's room hub (_start_room_hub)
                   "lights", "thermostat", "timer", "cancel_timer", "alarm", "cancel_alarm", "add_reminder",
                   "list_reminders", "delete_reminder", "edit_reminder", "list_items", "music", "volume"]
# a short filler is spoken at once for these (internet, 1-3 s; photos for tweak_app are handled too)
TURN_TTFS_S = 0.5          # the start value; each transcript then sets its own (services.turn_wait_s: 0.05 s after
                           # a question, 0.3 s after a statement, 0.8 s after "and..."); was a flat 0.8 s
SLOW_TOOLS: set[str] = {"web_search", "read_page", "get_news", "wikipedia", "play_video"}

# ---- Interface sub-tabs: every app, video, photo set or search opens its own tab, named so that no two tabs share
# their key word ("Snake game", "Bakery site", "Davao links", "Tie video"): the person switches by voice ----
_GAME_WORDS = {"game", "snake", "flappy", "tetris", "pong", "puzzle", "quiz", "maze", "racing", "racer", "shooter",
               "platformer", "breakout", "memory", "tic", "chess", "runner", "jump", "hop", "arcade", "dodge", "invaders"}
_SITE_WORDS = {"website", "site", "landing", "page", "portfolio", "shop", "store", "blog", "homepage", "restaurant",
               "bakery", "business", "company", "studio", "agency", "services", "salon", "cafe"}
_VIEW_STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with", "from", "my", "me", "your",
              "our", "this", "that", "it", "is", "are", "be", "can", "could", "would", "will", "you", "please", "show",
              "play", "find", "search", "look", "up", "get", "give", "open", "make", "build", "create", "want", "need",
              "some", "any", "what", "whats", "who", "how", "when", "where", "which", "why", "is", "does", "do", "about",
              "video", "videos", "clip", "youtube", "photo", "photos", "picture", "pictures", "pic", "pics", "image",
              "images", "link", "links", "tab", "app", "application", "website", "site", "page", "web", "landing",
              "game", "simple", "new", "small", "little", "nice", "cool", "one", "type", "kind", "like", "official",
              "elkit", "hey", "okay", "now", "again", "just", "really", "very", "best", "top", "latest", "music",
              "song", "watch", "see", "tell", "let", "lets", "us", "there", "here", "cheapest", "much", "many"}
_POOL = ["Amber", "Coral", "Maple", "Olive", "Ruby", "Jade", "Cedar", "Pearl", "Indigo", "Saffron", "Willow", "Topaz"]
_KIND_SAY = {"video": {"video", "youtube", "clip"}, "photos": {"photo", "photos", "picture", "pictures", "pics", "images"},
             "links": {"links", "link", "search", "results"}, "game": {"game"}, "site": {"site", "website", "page"},
             "app": {"app", "application"}}
MAX_VIEWS = 8


# real sites and apps: "open YouTube" means the real one, never a copy built with create_app (YouTube itself cannot
# be shown in the display's frame; its videos can: play_video). Only "build / make / clone a YouTube-like app" builds.
_REAL_SERVICES = r"you\s?tube|netflix|spotify|facebook|instagram|tik\s?tok|twitter|google|gmail|messenger|whatsapp|" \
                 r"reddit|amazon|shopee|lazada|wikipedia|twitch|disney\+?|hbo|prime video"
_REAL = re.compile(rf"\b(?:{_REAL_SERVICES})\b", re.I)
_BUILD_WORDS = re.compile(r"\b(?:build|make|create|design|code|clone|copy|replica|similar|mock\s?up|version of)\b",
                          re.I)        # not "like": "I'd like to open YouTube" is an open
REAL_SERVICE_NOTE = ("NOT built: they want to open the real service, not a copy. For YouTube use play_video (if they "
                     "did not say what to watch, ask what they want to watch). For other sites say you cannot open "
                     "that app here and offer a web search or a video instead. Never build a copy unless they "
                     "ask to build one.")


def real_service(what: str, said: str) -> bool:
    """create_app for a real site (YouTube, Netflix...) when the person did not ask to build anything."""
    return bool(_REAL.search(what or "") and _REAL.search(said or "") and not _BUILD_WORDS.search(said or ""))


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9']+", (text or "").lower().replace("'s", ""))
            if len(w) > 2 and w not in _VIEW_STOP]


# the three screen tabs (the display's ids -> what people call them)
TAB_NAMES = {"design": "Virtual Studio", "room": "Virtual Home", "office": "Virtual Office"}   # "design" was "Interface"
TAB_SETTLE_S = 0.7          # after moving to the Virtual Home: let the tab fade in before the room changes
# office work opens in the Virtual Office tab (same words as OFFICE_WORDS in web/index.html)
_SLIDE_WORDS = {"slides", "slide", "presentation", "deck", "ppt", "powerpoint"}
_SHEET_WORDS = {"spreadsheet", "sheet", "budget", "invoice", "table"}
_DOC_WORDS = {"document", "doc", "report", "letter", "resume", "cv", "memo", "essay"}
OFFICE_KINDS = {"slides", "sheet", "doc"}


def _app_kind(title: str) -> str:
    w = set(re.findall(r"[a-z]+", (title or "").lower()))
    if w & _GAME_WORDS:
        return "game"
    if w & _SLIDE_WORDS:
        return "slides"
    if w & _SHEET_WORDS:
        return "sheet"
    if w & _DOC_WORDS:
        return "doc"
    return "site" if w & _SITE_WORDS else "app"


def _view_tab(v: dict | None) -> str:
    return "office" if v and v.get("kindword") in OFFICE_KINDS else "design"


# what each video_control action did (the tool result for the LLM, and the context line for direct commands)
VIDEO_ACTIONS = {"pause": "Paused the video.", "resume": "The video is playing.", "mute": "Muted the video.",
                 "unmute": "Unmuted the video.", "louder": "Turned the video up.", "quieter": "Turned the video down.",
                 "volume": "Set the video volume.", "forward": "Skipped ahead.", "back": "Went back.",
                 "restart": "Started the video over.", "close": "Closed it.",
                 "next": "Showing the next photo.", "previous": "Showing the previous photo."}

ROOM_TOOLS = {
    "lights": "Turn the room lights on/off, set brightness (0-100) or colour.",
    "thermostat": "Set the room temperature in degrees Celsius.",
    "timer": "Start a countdown timer (seconds).",
    "cancel_timer": "Cancel the running timer.",
    "alarm": "Set an alarm at a time like 06:30.",
    "cancel_alarm": "Cancel an alarm (its time, or 'all').",
    "add_reminder": "Remember something for the user, optionally at a time like 15:00.",
    "list_reminders": "List the user's reminders.",
    "delete_reminder": "Delete reminders: by words from it, its number, 'last', or 'all'.",
    "edit_reminder": "Change a reminder's text or time (\"scratch that, call father instead\": edit the last one).",
    "list_items": "The person's lists (shopping, to-do, ...): add or remove items, read a list out, or clear it.",
    "music": "Play, pause, resume, stop or skip the music.",
    "volume": "Change the speaker volume (up/down or a level 0-100).",
}


def room_tools(room: Room, on_use=None) -> list[FunctionSchema]:
    out = []
    for name, desc in ROOM_TOOLS.items():
        props = brain.TOOLS[name]

        async def handler(params, _name=name):
            moved = (on_use() or "") if on_use else ""
            if moved:
                await asyncio.sleep(TAB_SETTLE_S)          # the Virtual Home shows first, then the change happens
            r = await room.execute({"tool": _name, **(params.arguments or {})})
            await params.result_callback({"ok": r.get("ok", True), "result": r.get("note", ""),
                                          **({"say": r["speak"]} if r.get("speak") else {}),
                                          **({"screen": "the screen just moved to the Virtual Home tab: "
                                              f"start your reply with \"{moved}\""} if moved else {})})

        out.append(FunctionSchema(name=name, description=desc, properties=props,
                                  required=brain.REQUIRED.get(name, []), handler=handler))
    return out


class Agent:
    """Builds and runs the pipeline. `io` is the FdaAudioTransport (duck/stop control, mixer) or NullMixerTransport."""

    def __init__(self, cfg, transport, io, mcp_urls: dict, ui=None, log_dir: Path | None = None,
                 out_rate: int = 24000):
        self.cfg, self.transport, self.io, self.ui = cfg, transport, io, ui
        self.urls, self.log_dir, self.out_rate = mcp_urls, log_dir, out_rate
        self.worker: PipelineWorker | None = None
        self.app_url, self.app_keys = "", []          # the app on screen and its live settings
        self.views: list[dict] = []                   # Interface sub-tabs (apps, videos, photos, links)
        self.cur_tab = "design"                        # the tab on screen: design (Interface) / room / office
        self._building, self._build_t0 = False, 0.0    # an app build running (for the premature "ready" filter)
        # a voice change to a game waits this long before it is made, so "green... wait, no, black" never shows green
        self.tweak_hold = 1.0          # (after "Green, okay." is said)
        self._held: dict | None = None
        self.active_view: int | None = None
        self._view_n = 0
        self.context: LLMContext | None = None
        self.quick: QuickAnswers | None = None      # predefined answers (quick_answers.txt), set in build()
        self.tasks: list[asyncio.Task] = []
        self.events: list[dict] = []
        self.choice: dict | None = None               # the pick pop-up on the display (choices.py)
        self.picked: dict[str, str] = {}              # what was picked for something not built yet

    def _event(self, kind: str, **kw):
        self.events.append({"t": time.time(), "kind": kind, **kw})

    async def build(self) -> PipelineWorker:
        cfg = self.cfg
        self.asr = McpConnection(self.urls["asr"], "asr")
        self.tts_conn = McpConnection(self.urls["tts"], "tts")
        self.tools_conn = McpConnection(self.urls["tools"], "tools")      # app_status polling
        await asyncio.gather(self.asr.start(), self.tts_conn.start(), self.tools_conn.start())

        ep = llmlib.make(cfg, cfg.llm_engine, role="agent")
        if isinstance(ep, llmlib.LocalEndpoint):
            await asyncio.to_thread(ep._ensure)
            self.local_llm = ep
        logger.info(f"LLM: {ep.engine} / {ep.model} at {ep.base_url}")
        self.system = SYSTEM.format(name=cfg.name, location=cfg.location)
        entries = fallback_llm.chain(ep.model) if ep.engine != "local" else []
        if entries:
            logger.info("LLM fallback chain: " + " -> ".join(f"{e['engine']}/{e['model']}" for e in entries))
        llm = (fallback_llm.FallbackLLMService if entries else OpenAILLMService)(
                               **({"entries": entries} if entries else {"api_key": ep.key or "none",
                                                                        "base_url": ep.base_url}),
                               settings=OpenAILLMService.Settings(
                                   model=ep.model, temperature=0.3, system_instruction=self.system,
                                   # thinking models count their thinking in max_tokens: give them room
                                   max_tokens=400 if llmlib.reasoning_extra(ep.model) else 90,
                                   extra=llmlib.reasoning_extra(ep.model)))

        self.room = Room(cfg, self.io.mixer, on_change=self._room_changed, on_ring=self._announce)
        qa = getattr(cfg, "quick_answers", "")
        self.quick = QuickAnswers(ROOT / qa, name=cfg.name, threshold=getattr(cfg, "quick_match", 0.85)) if qa \
            else None
        mcp_tools = MCPClient(server_params=StreamableHttpParameters(url=self.urls["tools"], timeout=60,
                                                                    sse_read_timeout=600),
                              tools_filter=AGENT_MCP_TOOLS,
                              tools_output_filters={**{n: self._show_app for n in ("open_app", "undo_app_change", "restore_app_version")},
                                                    "tweak_app": self._app_set, "play_video": self._show_video,
                                                    "show_photo": self._show_photo,
                                                    "web_search": self._show_links, "get_news": self._show_links})
        self.mcp_tools = mcp_tools
        remote = await mcp_tools.tools()
        show = FunctionSchema(
            name="show_screen", description="Switch the screen tab: 'room' is the Virtual Home (lights, "
            "thermostat, timers, alarms, reminders, lists, music); 'interface' shows games, websites, videos, links "
            "and photos; 'office' is the Virtual Office (slides, documents, spreadsheets).",
            properties={"screen": {"type": "string", "enum": ["interface", "room", "office"]}}, required=["screen"],
            handler=self._show_screen)
        video = FunctionSchema(
            name="video_control", description="Control the video or photos on the screen. Video: pause, resume, "
            "mute, unmute, louder, quieter, volume (set `level` 0-100), forward / back (`seconds`, default 10), restart. "
            "Photos: next, previous. Both: close.",
            properties={"action": {"type": "string", "enum": list(VIDEO_ACTIONS)},
                        "level": {"type": "integer", "description": "volume 0-100, for action volume"},
                        "seconds": {"type": "integer", "description": "for forward / back"}},
            required=["action"], handler=self._video_control)
        nav = FunctionSchema(
            name="app_control", description="Use the app on the screen hands-free, like a mouse and keyboard. "
            "go_to: a tab, page, section or link by its name. click: a button or link. scroll: target up, down, "
            "top or bottom. type: `text` into the field named `target` (or the first empty one). press: a key "
            "(space, enter, up, down, left, right, escape, or a letter) for games. back: the previous page. "
            "look: what is on the screen (headings, buttons, some text). If the target is not found, the result "
            "lists what can be clicked.",
            properties={"action": {"type": "string", "enum": ["go_to", "click", "scroll", "type", "press", "back",
                                                              "look"]},
                        "target": {"type": "string", "description": "the name as it shows on screen, or the key"},
                        "text": {"type": "string", "description": "for type: what to write"}},
            required=["action"], handler=self._app_control)
        subtab = FunctionSchema(
            name="interface_tab", description="The sub-tabs of Virtual Studio and Virtual Office (one per app, game, "
            "slides, video, photos, search). "
            "go: show the tab called `name`; close: close it (no name: the one showing); list: the tabs.",
            properties={"action": {"type": "string", "enum": ["go", "close", "list"]},
                        "name": {"type": "string", "description": "the tab's name or a word of it"}},
            required=["action"], handler=self._interface_tab)
        offer = FunctionSchema(
            name="show_choices", description="Show 4 numbered choices on the screen to pick from: pictures from the "
            "library (what: birds, cars, fish, heroes, pets, enemies, backgrounds, ground ...) or colour schemes "
            "(what: 'colour scheme'). For the app on screen, setting = which of its settings it is for. for_new = "
            "true when choosing for something not built yet. more = true for the next four.",
            properties={"what": {"type": "string"}, "setting": {"type": "string"},
                        "for_new": {"type": "boolean"}, "more": {"type": "boolean"}},
            required=["what"], handler=self._show_choices_tool)
        pick = FunctionSchema(
            name="pick_choice", description="The person picked one of the choices on the screen: its number or "
            "name as they said it ('two', 'the owl', 'the blue one'), or 'none' to close them.",
            properties={"choice": {"type": "string"}}, required=["choice"], handler=self._pick_choice_tool)
        # every tool goes to its tab first: a new build by what it is (slides / docs -> Office), a change to an
        # app -> that app's tab, videos and searches -> Interface (room tools: room_tools(on_use=...) below)
        new_tab = (lambda a: "office" if _app_kind(f"{a.get('name', '')} {a.get('description', '')}")
                   in OFFICE_KINDS else "design")
        app_tab = (lambda a: self._app_tab(a.get("name", "")))
        picks = {"create_app": new_tab, "edit_app": app_tab, "tweak_app": app_tab, "undo_app_change": app_tab,
                 "restore_app_version": app_tab, "open_app": app_tab,
                 "play_video": lambda a: "design", "web_search": lambda a: "design", "get_news": lambda a: "design"}
        # room tools come from the tools MCP server (it calls back into our room hub); the Virtual Home shows first
        picks.update({n: (lambda a: "room") for n in ROOM_TOOLS})
        for fs in remote.standard_tools:
            if fs.name in picks:
                self._tab_first(fs, picks[fs.name])
            if fs.name in ("create_app", "edit_app"):
                self._mark_build(fs)
        self._tab_first(nav, lambda a: self._app_tab())
        self._tab_first(video, lambda a: None if a.get("action") == "close" else "design")
        self.room_hub = await self._start_room_hub()
        own_room = []
        if not (self.room_hub and set(ROOM_TOOLS) <= {fs.name for fs in remote.standard_tools}):
            # an older tools server (no room tools) or the hub port taken: the agent's own room tools, as before
            logger.warning("room tools served by the agent itself, not over MCP")
            remote.standard_tools[:] = [t for t in remote.standard_tools if t.name not in ROOM_TOOLS]
            own_room = room_tools(self.room, on_use=lambda: self._move_tab("room"))
        tools = ToolsSchema(standard_tools=[*remote.standard_tools, *own_room, show, video, nav, subtab, offer, pick])
        logger.info(f"tools: {[t.name for t in tools.standard_tools]}")

        self.context = LLMContext(messages=[*FEW_SHOT], tools=tools)
        agg = LLMContextAggregatorPair(self.context, user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]),
            user_turn_stop_timeout=3.0))

        self.guard = DuplexGuard(self.io, cfg)
        self.guard.on_bot = lambda talking: self._status("speaking" if talking else "listening")
        self.guard.on_user = (lambda ev: self.ui.send(ev)) if self.ui else None
        self.nfm = NotForMe(on_ignored=lambda: self._event("not_for_me"), io=self.io)
        self.nfm.own_name = cfg.name
        # ttfs_p99_latency: our ASR gives its text only after the speech ends, so pipecat has already started the turn
        # (min 3 words) and forgotten the VAD stop; it then ends the turn on a fallback timer of ttfs - stop_secs
        # (1.0 - 0.2 = a flat 0.8 s on every turn, fd-benchmark 2026-11-06). 0.25 -> 0.05 s: when the text comes,
        # the person has been quiet for 0.2 s (VAD) + the ASR time (~0.4-0.8 s) already; if they start again,
        # VAD start cancels this timer (pipecat _discard_pending_end_of_turn), so a pause mid-sentence still waits.
        stt = McpSTTService(self.asr, far_db=cfg.far_speech_db, on_ignored=self._resume_soon,
                             own_name=cfg.name, ttfs_p99_latency=TURN_TTFS_S)
        self.guard.asr, self.guard.stt = self.asr, stt
        stt.output_busy = lambda: bool(self.room.state.get("ringing")) or \
            self.room.state.get("music", {}).get("state") == "playing"
        self.guard.bot_text = lambda: self.nfm.reply_full
        self.guard.echo_text = lambda: self.nfm.recent_text()      # its voice can come back late (Pi + PipeWire)
        self.clock = TurnClock()          # 'turn timing (ms)' line in the log, per turn
        tts = McpTTSService(self.tts_conn)
        # while a build runs, a spoken "it's ready / enjoy" is dropped (the LLM jumps ahead); app_status is polled
        # every 1.5 s, so the first seconds after create_app / edit_app count as building too
        tts.building = lambda: self._building or time.monotonic() - self._build_t0 < 6.0
        self._heard_tap = HeardTap(self.io, self.ui, name=cfg.name, on_heard=self._heard, on_ignored=self._resume_soon,
                                   intercept=self._intercept, echo_text=lambda: self.nfm.recent_text())
        # a filler when the answer is slow (cached audio), and none twice in a row (ToolFiller checks t_fill)
        think = ThinkFiller(after_s=getattr(cfg, "filler_after_s", 0.7), last_user=self._last_user_text,
                            own_name=cfg.name, slow=SLOW_TOOLS | {"tweak_app"}, on_fill=lambda: self.clock.mark("filler"))
        pipeline = Pipeline([
            self.transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.6, start_secs=0.15,
                                                                         stop_secs=0.2))),
            self.guard,
            Mark(self.clock, {VADUserStoppedSpeakingFrame: "speech_end"}),
            stt,
            self._heard_tap,
            Mark(self.clock, {TranscriptionFrame: "heard"}),
            agg.user(),
            Mark(self.clock, {LLMContextFrame: "sent"}),
            llm,
            RefusalGuard(self._retry_refusal),
            Mark(self.clock, {LLMTextFrame: "first_word", FunctionCallsStartedFrame: "first_word"}),
            think,
            ToolFiller(slow=SLOW_TOOLS, on_fill=lambda: self.clock.mark("filler"),
                       skip=lambda: time.monotonic() - think.t_fill < 4.0),
            self.nfm,
            tts,
            Mark(self.clock, {TTSAudioRawFrame: "first_audio"}),
            SpokenTap(self.nfm, self.ui),
            self.transport.output(),
            agg.assistant(),
        ])
        if isinstance(ep, llmlib.LocalEndpoint):
            await self._warm_up(ep, llm, tools)
        self.worker = PipelineWorker(pipeline, params=PipelineParams(
            audio_in_sample_rate=16000, audio_out_sample_rate=self.out_rate, enable_metrics=True),
            idle_timeout_secs=None)
        self.tasks.append(asyncio.create_task(self._watch_apps()))
        self.tasks.append(asyncio.create_task(self.room.tick_forever()))      # timers and alarms
        if getattr(cfg, "filler_after_s", 0.7) > 0:
            self.tasks.append(asyncio.create_task(self._warm_fillers(tts)))
        return self.worker

    async def _start_room_hub(self):
        """The room for the tools MCP server's room tools: POST /room/<tool> {args} -> what the room did. Local only
        (127.0.0.1). None when the port is taken: the agent then keeps its own room tools."""
        from aiohttp import web

        async def call(req):
            name = req.match_info["tool"]
            if name not in ROOM_TOOLS:
                return web.json_response({"ok": False, "result": f"no room tool {name}"}, status=404)
            try:
                args = await req.json()
            except ValueError:
                args = {}
            r = await self.room.execute({"tool": name, **(args if isinstance(args, dict) else {})})
            return web.json_response({"ok": r.get("ok", True), "result": r.get("note", ""),
                                      **({"say": r["speak"]} if r.get("speak") else {})})
        app = web.Application()
        app.router.add_post("/room/{tool}", call)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", getattr(self.cfg, "room_hub_port", 8104)).start()
            return runner
        except OSError as e:
            logger.warning(f"room hub not started ({e})")
            await runner.cleanup()
            return None

    def _last_user_text(self) -> str | None:
        """What the person just said, if the LLM is answering it (None on a tool-result round)."""
        msgs = self.context.messages if self.context else []
        if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") != "user":
            return None
        c = msgs[-1].get("content")
        return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))

    async def _warm_fillers(self, tts: McpTTSService):
        """The fillers (ThinkFiller) synthesized ahead, again every 5 min: the voice follows DGX3 / Piper."""
        while True:
            await tts.warm(ThinkFiller.phrases())
            await asyncio.sleep(300)

    async def _retry_refusal(self):
        """RefusalGuard held back a bare "I'm sorry, but I can't help with that.": ask once more."""
        if self.worker is not None:
            await self.worker.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "system", "content":
                "Your last reply refused a harmless request. Answer it now: a short version is fine."}],
                run_llm=True)])

    def _tab(self, tab: str):
        self.cur_tab = tab
        if self.ui:
            self.ui.send({"type": "tab", "tab": tab})

    def on_tab(self, msg: dict):
        """The display says which tab is on screen (a tap, or a switch it made itself)."""
        if msg.get("tab") in TAB_NAMES:
            self.cur_tab = msg["tab"]

    def _move_tab(self, tab: str) -> str:
        """Switch to `tab`; if the screen was on another tab, the words that say so ("Over to the Virtual Home
        tab."), so the person knows why the screen changed."""
        was = getattr(self, "cur_tab", "design")
        self._tab(tab)
        return "" if was == tab else f"Over to the {TAB_NAMES[tab]} tab."

    def _app_tab(self, name: str = "") -> str:
        """The tab an app lives in: the one called `name` (or what its name says), else the app on screen."""
        if name:
            v = self._find_view(name)
            if v is not None:
                return _view_tab(v)
            return "office" if _app_kind(name) in OFFICE_KINDS else "design"
        return _view_tab(next((x for x in self.views if x["id"] == self.active_view), None))

    async def _go_first(self, tab: str, speak: bool = True) -> str:
        """Before doing something that shows on `tab`: go there first (and say so), let the tab settle, so the
        change is seen happening. Returns the words said ("" when it was already there)."""
        moved = self._move_tab(tab)
        if moved:
            if speak:
                await self._announce(moved)
            await asyncio.sleep(TAB_SETTLE_S)
        return moved

    def _mark_build(self, fs: FunctionSchema) -> FunctionSchema:
        """create_app / edit_app: note when a build starts (before app_status shows it)."""
        inner = fs.handler

        async def handler(params):
            if fs.name == "create_app":
                said = self._last_heard()
                what = f"{(params.arguments or {}).get('name', '')} {(params.arguments or {}).get('description', '')}"
                if real_service(what, said):
                    # "Open YouTube" is never a build (2026-11-06: it built a fake YouTube in 45 s)
                    logger.info(f"create_app refused, they want the real thing: {said!r}")
                    return await params.result_callback({"ok": False, "result": REAL_SERVICE_NOTE})
            self._build_t0 = time.monotonic()
            return await inner(params)
        if inner is not None:
            fs._handler = handler
        return fs

    def _last_heard(self) -> str:
        """The person's last words in the context (also while a tool call is under way)."""
        for m in reversed(self.context.messages if self.context else []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
        return ""

    def _tab_first(self, fs: FunctionSchema, pick) -> FunctionSchema:
        """Wrap a tool: go to the tab `pick(arguments)` names before it runs; the result tells the LLM to say so."""
        inner = fs.handler

        async def handler(params):
            tab = pick(params.arguments or {})
            moved = await self._go_first(tab, speak=False) if tab else ""
            if moved:
                cb = params.result_callback
                hint = f'the screen just moved to the {TAB_NAMES[tab]} tab: start your reply with "{moved}"'

                async def with_hint(result, *a, **k):
                    if isinstance(result, dict):
                        result = {**result, "screen": hint}
                    else:
                        result = f"{result}\n({hint})"
                    return await cb(result, *a, **k)
                params.result_callback = with_hint
            return await inner(params)
        if inner is not None:
            # (FunctionSchema.handler is read-only; keep the inner handler's @tool_options call options)
            for k, v in vars(inner).items() if hasattr(inner, "__dict__") else ():
                setattr(handler, k, v)
            fs._handler = handler
        return fs

    # ---------------- the pick pop-up: 4 numbered choices on the display (choices.py) ----------------
    async def _app_settings(self) -> dict:
        """The live settings of the app on screen, as {key: value}."""
        try:
            r = await self.tools_conn.call("app_settings", {}, timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"app_settings: {e}")
            return {}
        out = {}
        if isinstance(r, dict) and r.get("ok"):
            for part in str(r.get("settings", "")).split(", "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip()
        return out

    async def _offer(self, what: str, setting: str = "", for_new: bool = False, more: bool = False) -> dict:
        """Put up to 4 choices on the display. Returns the tool result for the LLM."""
        from . import choices as C
        from ..mcp_servers import live
        prev = self.choice if more else None
        if prev:                                       # "show me more": the same kind of thing, the next four
            what, setting, for_new = prev["what"], prev.get("key") or "", prev["for_new"]
        palette = bool(re.search(r"palette|scheme|theme|colou?rs?\b", what, re.I)) and C.group_of(what) is None
        if palette:
            off = (prev["offset"] + C.SHOW) if prev else 0
            options, key, title = C.palette_choice(off), "", "Colour schemes"
        else:
            settings = {} if for_new else await self._app_settings()
            key = live.match_key(setting or what, list(settings)) if settings else None
            current = settings.get(key) if key else None
            grp = C.group_of(what, current)
            if grp is None:
                return {"ok": False, "result": f"The picture library has no choices for {what!r}. Say two or three "
                                               "ideas out loud instead and let them pick."}
            off = (prev["offset"] + 1) if prev else 0
            options = C.art_choice(grp[1], current, off)
            title = f"Pick a {(setting or what).strip()}" if key or for_new else "Pick one"
        if not options:
            return {"ok": False, "result": "Nothing to choose from."}
        self.choice = {"what": what, "key": key, "palette": palette,
                       # a picture with no setting to put it in, or anything asked for a new build: kept for the build
                       "for_new": for_new or (not palette and not key),
                       "offset": off, "options": options, "t": time.monotonic()}
        # the pop-up shows over the app's tab: go there first (a new build: Interface, or Office for slides / docs)
        moved = await self._go_first("design" if for_new else self._app_tab(), speak=False)
        if self.ui:
            self.ui.send(C.popup(title, options))
        logger.info(f"choices shown ({what}): {[o['name'] for o in options]}")
        return {"ok": True, "result": f"Showing on the screen, numbered: {C.spoken_list(options)}. Ask in a few words "
                                      "which one they like (a number or a name); then pick_choice.",
                **({"screen": f'the screen just moved to that tab: start your reply with "{moved}"'} if moved else {})}

    def _close_choices(self):
        self.choice = None
        if self.ui:
            self.ui.send({"type": "choices", "close": True})

    async def _apply_pick(self, i: int) -> tuple[str, str]:
        """Use choice i. Returns (what to say, a note for the LLM's history)."""
        from . import choices as C
        ch = self.choice
        o = ch["options"][i]
        self._close_choices()
        if ch["palette"]:
            note = C.palette_note(o)
            if ch["for_new"] or not self.app_url:
                self.picked["colours"] = note
                return f"{o['say'].capitalize()} it is.", f"[picked on the screen: {note}; use it when building]"
            try:
                r = await self.tools_conn.call("edit_app", {"change": f"Use {note} for the whole page (as CSS "
                                                                      "variables); keep everything else."}, timeout=30)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"edit_app (palette): {e}")
                r = {}
            if isinstance(r, dict) and r.get("ok"):
                return f"Okay, switching it to {o['say']}. It takes a moment.", f"[picked: {note}; edit started]"
            return "Sorry, I couldn't change the colours.", f"[picked {o['say']}, but the edit failed]"
        if ch["for_new"]:
            self.picked[ch["what"]] = o["img"]
            return f"The {o['say']} it is.", f"[picked on the screen: {ch['what']} = {o['img']}; use it when building]"
        try:
            r = await self.tools_conn.call("tweak_app", {"changes": {ch["key"]: o["img"]}}, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tweak_app (pick): {e}")
            r = {}
        if isinstance(r, dict) and r.get("ok"):
            self._app_set(r)
            return f"Done, the {ch['key']} is the {o['say']} now.", f"[picked: {ch['key']} = {o['say']}]"
        return "Sorry, I couldn't change it.", f"[picked {o['say']}, but the change failed]"

    async def _show_choices_tool(self, params):
        a = params.arguments or {}
        r = await self._offer(str(a.get("what", "")), str(a.get("setting", "") or ""), bool(a.get("for_new")),
                              more=bool(a.get("more")) and self.choice is not None)
        await params.result_callback(r)

    async def _pick_choice_tool(self, params):
        from . import choices as C
        c = str((params.arguments or {}).get("choice", ""))
        if self.choice is None:
            await params.result_callback({"ok": False, "result": "There are no choices on the screen."})
            return
        if C.is_close(c):
            self._close_choices()
            await params.result_callback({"ok": True, "result": "closed the choices"})
            return
        i = C.parse_pick(c, self.choice["options"])
        if i is None:
            await params.result_callback({"ok": False, "result": "Which one? On the screen: " +
                                          C.spoken_list(self.choice["options"])})
            return
        say, note = await self._apply_pick(i)
        await params.result_callback({"ok": True, "result": f"{say} {note}"})

    async def _choice_direct(self, text: str) -> bool:
        """While choices are on the display: "number two", "the owl", "show me more", "none of them" at once,
        without the LLM. Anything else goes on as usual (the choices stay up)."""
        from . import choices as C
        if self.choice is None:
            return False
        if time.monotonic() - self.choice["t"] > 300:     # nobody picked in 5 minutes: put them away
            self._close_choices()
            return False
        said = strip_name(text, self.cfg.name)
        if C.is_close(said):
            self._close_choices()
            if C.you_pick(said):
                return False                               # "you pick": the LLM chooses and carries on
            msg, note = "Okay.", "[closed the choices]"
        else:
            i = C.parse_pick(said, self.choice["options"])
            if i is not None and self.choice["for_new"]:
                # picked for something not built yet ("the dark one" before the website): the LLM carries on
                # with the build, the pick in hand (not a dead end with "Midnight it is." and nothing built)
                _, note = await self._apply_pick(i)
                await self._carry_on(text, note)
                return True
            if i is not None:
                msg, note = await self._apply_pick(i)
            elif C.is_more(said):
                r = await self._offer("", more=True)
                msg, note = ("Here are some more." if r.get("ok") else "That's all I have."), "[showed more choices]"
            else:
                return False
        logger.info(f"choice (direct): {text!r} -> {msg!r}")
        self.context.add_message({"role": "user", "content": f"{text} {note}"})
        self.context.add_message({"role": "assistant", "content": msg})
        await self._announce(msg)
        return True

    async def _carry_on(self, text: str, note: str):
        """A pick for something not built yet: tell the LLM and let it continue (build, with the pick)."""
        logger.info(f"choice for a new build: {note}")
        if self.worker is None:
            return
        await self.worker.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "user", "content":
            f"{text} {note} Carry on with what they asked for, using this pick."}], run_llm=True)])

    def on_choice(self, msg: dict):
        """A tap on a choice card on the display."""
        if self.choice is None:
            return
        try:
            i = int(msg.get("n", 0)) - 1
        except (TypeError, ValueError):
            return
        if not 0 <= i < len(self.choice["options"]):
            return
        if self.choice["for_new"]:
            async def build():
                _, note = await self._apply_pick(i)
                await self._carry_on(f"(tapped choice {i + 1})", note)
            self.tasks.append(asyncio.create_task(build()))
            return

        async def go():
            say, note = await self._apply_pick(i)
            self.context.add_message({"role": "user", "content": f"(tapped choice {i + 1}) {note}"})
            self.context.add_message({"role": "assistant", "content": say})
            await self._announce(say)
        self.tasks.append(asyncio.create_task(go()))

    async def _show_screen(self, params):
        tab = (params.arguments or {}).get("screen", "interface")
        t = tab if tab in ("room", "office") else "design"
        self._tab(t)
        await params.result_callback({"ok": True, "result": f"the {TAB_NAMES[t]} tab is showing"})

    # ---- Interface sub-tabs ----
    def _active(self) -> dict | None:
        return next((v for v in self.views if v["id"] == self.active_view), None)

    @property
    def video(self) -> dict | None:
        """The video the controls mean: the one showing, else the latest one."""
        a = self._active()
        if a and a["kind"] == "video":
            return a
        return next((v for v in reversed(self.views) if v["kind"] == "video"), None)

    @property
    def photo(self) -> dict | None:
        a = self._active()
        return a if a and a["kind"] == "photos" else None

    def _name_view(self, kindword: str, sources: list[str], reverse: bool) -> tuple[str, str]:
        used = {v["key"] for v in self.views}
        cands = []
        for src in sources:
            ws = _content_words(src)
            cands += list(reversed(ws)) if reverse else ws
        key = next((w for w in cands if w not in used), None) or \
            next(p.lower() for p in _POOL if p.lower() not in used)
        return f"{key.capitalize()} {kindword}", key

    def _open_view(self, kind: str, data: dict, sources: list[str], match: tuple[str, str] | None = None,
                   reverse: bool = True, kindword: str | None = None) -> dict:
        """Show `data` in its own sub-tab (or the one already showing the same app / video) and make it the active one."""
        v = next((x for x in self.views if match and x.get(match[0]) == match[1]), None)
        if v is not None:
            v.update(data)
            v["v"] = v.get("v", 0) + 1                 # the display reloads an app whose version changed
        else:
            kw = kindword or kind
            label, key = self._name_view(kw, sources, reverse)
            self._view_n += 1
            v = {"id": self._view_n, "kind": kind, "kindword": kw, "label": label, "key": key, "v": 0, **data}
            self.views.append(v)
            while len(self.views) > MAX_VIEWS:         # the oldest goes (never the new one)
                self.views.pop(0)
        self._activate(v)
        return v

    def _activate(self, v: dict | None, sync: bool = True):
        self.active_view = v["id"] if v else None
        if v and v["kind"] == "app":
            self.app_url = v["url"]
            stem = v["url"].rsplit("/", 1)[-1].split("?")[0].rsplit(".", 1)[0]
            try:
                asyncio.get_running_loop().create_task(self._server_open(stem))
            except RuntimeError:                       # no loop (tests): nothing to sync
                pass
        self.io.media_playing = bool(v and v["kind"] == "video" and v.get("state", "playing") == "playing")
        if sync:
            self._sync_views()

    async def _server_open(self, stem: str):
        try:                                           # edit_app / tweak_app change the app that is showing
            await self.tools_conn.call("open_app", {"name": stem}, timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"open_app sync: {e}")

    def _close_view(self, v: dict):
        self.views = [x for x in self.views if x["id"] != v["id"]]
        if self.active_view == v["id"]:
            self._activate(self.views[-1] if self.views else None, sync=False)
        if v["kind"] == "app" and v.get("url") == self.app_url and not any(x["kind"] == "app" for x in self.views):
            self.app_url = ""
        self._sync_views()

    def _sync_views(self):
        if not self.ui:
            return
        keep = ("id", "kind", "kindword", "label", "v", "url", "title", "video_id", "photos", "i", "items")
        self.ui.send({"type": "views", "active": self.active_view,
                      "views": [{k: x[k] for k in keep if k in x} for x in self.views]})
        if self.active_view is not None:
            self._tab(_view_tab(next((x for x in self.views if x["id"] == self.active_view), None)))

    def _find_view(self, target: str) -> dict | None:
        """The tab someone means: its key word ("the snake one"), else its kind if that is clear ("the video")."""
        words = set(re.findall(r"[a-z]+", (target or "").lower()))
        hit = [v for v in self.views if v["key"] in words or v["label"].lower() == (target or "").lower().strip()]
        if hit:
            return hit[-1]
        for kind, say in _KIND_SAY.items():
            if words & say:
                same = [v for v in self.views if v["kindword"] == kind or v["kind"] == kind]
                if same:
                    return same[-1]
        return None

    def on_view(self, msg: dict):
        """A tap on a sub-tab (when someone does use the screen)."""
        v = next((x for x in self.views if x["id"] == msg.get("id")), None)
        if v:
            self._activate(v)

    async def _interface_tab(self, params):
        a = params.arguments or {}
        action, name = a.get("action", "list"), a.get("name", "")
        tabs = [v["label"] for v in self.views]
        if action == "list":
            act = self._active()
            await params.result_callback({"ok": True, "tabs": tabs, "showing": act["label"] if act else ""})
            return
        v = self._find_view(name) if name else self._active()
        if v is None:
            await params.result_callback({"ok": False, "result": f"no tab called {name}", "tabs": tabs})
            return
        if action == "close":
            self._close_view(v)
            await params.result_callback({"ok": True, "result": f"closed the {v['label']} tab"})
        else:
            self._activate(v)
            await params.result_callback({"ok": True, "result": f"the {v['label']} tab is showing"})

    async def _view_direct(self, text: str) -> bool:
        """"Go to the snake tab", "switch to the video", "close the Davao links", "show me the room": no LLM."""
        t = strip_name(text, self.cfg.name).lower().strip(" .!?")
        t = re.sub(r"^(please |can you |could you |now |okay |ok )+", "", t)
        t = re.sub(r" please$", "", t)
        go = r"(?:go|switch|change|jump|move|head|take me|flip)(?: back)?(?: over)? to (?:the |my )?"
        show = r"(?:show me|open|show) (?:the |my )?"
        home = r"(?:room|home|house|virtual (?:home|room|house))"
        if re.fullmatch(rf"{go}{home}(?: tab)?|{show}{home}(?: tab)?|(?:the )?{home} tab", t):
            self._tab("room")
            return self._noted(text, "showing the Virtual Home tab")
        office = r"(?:virtual )?office"
        if re.fullmatch(rf"{go}{office}(?: tab)?|{show}{office}(?: tab)?|(?:the )?{office} tab", t):
            mine = [v for v in self.views if _view_tab(v) == "office"]
            a = next((x for x in self.views if x["id"] == self.active_view), None)
            if mine and _view_tab(a) != "office":
                self._activate(mine[-1])               # its newest slides / document
            self._tab("office")
            return self._noted(text, "showing the Virtual Office tab")
        studio = r"(?:virtual studio|studio|interface)"            # "interface": its old name still works
        if re.fullmatch(rf"{go}{studio}(?: tab)?|{show}{studio}(?: tab)?|(?:the )?{studio} tab", t):
            mine = [v for v in self.views if _view_tab(v) == "design"]
            a = next((x for x in self.views if x["id"] == self.active_view), None)
            if mine and _view_tab(a) != "design":
                self._activate(mine[-1])
            self._tab("design")
            return self._noted(text, "showing the Virtual Studio tab")
        m = re.fullmatch(r"(next|previous|last|other) tab|(?:go|switch) to the (next|previous|other) tab", t)
        if m and len(self.views) > 1:
            i = next((n for n, v in enumerate(self.views) if v["id"] == self.active_view), 0)
            step = -1 if (m.group(1) or m.group(2)) in ("previous", "last") else 1
            v = self.views[(i + step) % len(self.views)]
            self._activate(v)
            return self._noted(text, f"showing the {v['label']} tab")
        if t in ("close this tab", "close the tab", "close tab", "close this one", "close the current tab", "close that tab"):
            v = self._active()
            if v is None:
                return False
            self._close_view(v)
            return self._noted(text, f"closed the {v['label']} tab")
        m = re.fullmatch(r"close (?:the |this |that )?(.+?)(?: tab)?", t)
        if m and ("tab" in t or self._find_view(m.group(1))):
            target = m.group(1)
            v = self._active() if target in ("this", "that", "it", "this one", "current") else self._find_view(target)
            if v is None or (v["kind"] in ("video", "photos") and "tab" not in t and target in ("it", "this")):
                return False
            self._close_view(v)
            return self._noted(text, f"closed the {v['label']} tab")
        m = re.fullmatch(r"(?:go|switch|change|jump|move|head|take me|flip)(?: back)?(?: over)? to (?:the )?(.+?)"
                         r"(?: tab| one)?|(?:open|show me|show|bring up) (?:the )?(.+?) tab|(?:the )?(.+?) tab", t)
        if not m:
            m2 = re.fullmatch(r"(?:open|show me|show|bring up|back to) (?:the )?(.+)", t)
            target = m2.group(1) if m2 else None
            v = None
            if target:                                 # "open the snake game": only a clear key-word match
                words = set(re.findall(r"[a-z]+", target))
                v = next((x for x in reversed(self.views) if x["key"] in words), None)
        else:
            v = self._find_view(m.group(1) or m.group(2) or m.group(3))
        if v is None:
            return False                               # maybe a section inside the app ("go to contact")
        self._activate(v)
        return self._noted(text, f"showing the {v['label']} tab")

    def _noted(self, text: str, did: str) -> bool:
        logger.info(f"tabs (direct): {text!r} -> {did}")
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": f"({did})"})
        return True

    def _heard_text(self) -> str:
        return getattr(self.io, "last_heard", "") or ""

    def _show_photo(self, output):
        try:
            d = json.loads(output) if isinstance(output, str) else output
        except (ValueError, TypeError):
            return output
        if isinstance(d, dict) and d.get("photos"):
            v = self._open_view("photos", {"photos": d["photos"], "i": 0, "title": d["photos"][0]["title"]},
                                [self._heard_text(), d["photos"][0]["title"]])
            logger.info(f"photos -> tab {v['label']!r}: {[p['title'] for p in d['photos']]}")
            return json.dumps({"ok": True, "tab": v["label"], "showing": d["photos"][0]["title"],
                               "photos": len(d["photos"]), "titles": [p["title"] for p in d["photos"]]},
                              ensure_ascii=False)
        return output

    def _show_video(self, output):
        try:
            d = json.loads(output) if isinstance(output, str) else output
        except (ValueError, TypeError):
            return output
        if isinstance(d, dict) and d.get("video_id"):
            v = self._open_view("video", {"video_id": d["video_id"], "title": d.get("title", ""), "state": "playing"},
                                [self._heard_text(), d.get("title", "")], match=("video_id", d["video_id"]))
            self.io.media_playing = True          # until the display says otherwise
            logger.info(f"video -> tab {v['label']!r}: {d.get('title')!r} ({d['video_id']})")
            self._event("video", title=d.get("title", ""))
            return json.dumps({"ok": True, "tab": v["label"], "playing": d.get("title", ""),
                               "channel": d.get("channel", "")}, ensure_ascii=False)
        return output

    def _show_links(self, output):
        """web_search / get_news: the links get their own tab (the LLM only says the answer)."""
        items, dec, text, i = [], json.JSONDecoder(), output if isinstance(output, str) else "", 0
        while i < len(text):                       # results arrive as JSON objects one after another
            while i < len(text) and (text[i].isspace() or text[i] in ",[]"):
                i += 1
            if i >= len(text):
                break
            try:
                obj, i = dec.raw_decode(text, i)
            except ValueError:
                break
            for o in obj if isinstance(obj, list) else [obj]:
                if isinstance(o, dict) and o.get("link"):
                    items.append({"title": o.get("title") or o.get("headline") or o["link"], "link": o["link"],
                                  "source": o.get("source", ""), "snippet": (o.get("snippet") or "")[:160]})
        if items:
            v = self._open_view("links", {"items": items[:6], "title": self._heard_text()}, [self._heard_text()])
            logger.info(f"links -> tab {v['label']!r}")
        return output

    async def _video_control(self, params):
        a = params.arguments or {}
        action = a.get("action", "pause")
        ok = self._video_cmd(action, level=a.get("level"), seconds=a.get("seconds"))
        await params.result_callback({"ok": ok, "result": VIDEO_ACTIONS.get(action, "done") if ok
                                      else "no video is on the screen"})

    def _video_cmd(self, action: str, level=None, seconds=None) -> bool:
        if not self.ui or action not in VIDEO_ACTIONS:
            return False
        ph, vd = self.photo, self.video
        if ph and action in ("next", "previous", "close"):
            if action == "close":
                self._close_view(ph)
            else:
                n = len(ph["photos"])
                ph["i"] = (ph.get("i", 0) + (1 if action == "next" else -1)) % n
                self.ui.send({"type": "video_cmd", "view": ph["id"], "action": action, "i": ph["i"]})
            return True
        if not vd or action in ("next", "previous"):
            return False
        if action == "close":
            self._close_view(vd)
            return True
        if action in ("resume", "restart") and self.active_view != vd["id"]:
            self._activate(vd)                         # "play the video" from another tab: show it
        msg = {"type": "video_cmd", "view": vd["id"], "action": action}
        if level is not None:
            msg["level"] = max(0, min(100, int(level)))
        if seconds:
            msg["seconds"] = max(1, min(600, int(seconds)))
        self.ui.send(msg)
        if action == "pause":
            vd["state"] = "paused"
            self.io.media_playing = False          # the display confirms; resume is confirmed by it too
        return True

    # ---- hands-free use of the app on the screen (the display finds and clicks things inside it) ----
    async def _app_cmd(self, cmd: dict, timeout: float = 3.0) -> dict:
        if not self.ui or not self.ui.clients:
            return {"ok": False, "did": "the screen is not connected"}
        self._nav_id = getattr(self, "_nav_id", 0) + 1
        fut = asyncio.get_running_loop().create_future()
        self._nav_wait = getattr(self, "_nav_wait", {})
        self._nav_wait[self._nav_id] = fut
        self.ui.send({"type": "app_nav", "id": self._nav_id, **cmd})
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "did": "the screen did not answer"}
        finally:
            self._nav_wait.pop(self._nav_id, None)

    def on_app_result(self, msg: dict):
        fut = getattr(self, "_nav_wait", {}).get(msg.get("id"))
        if fut is not None and not fut.done():
            fut.set_result({k: v for k, v in msg.items() if k not in ("type", "id")})

    async def _app_control(self, params):
        a = params.arguments or {}
        r = await self._app_cmd({"action": a.get("action", "look"), "target": a.get("target", ""),
                                 "text": a.get("text", "")})
        logger.info(f"app_control {a} -> {r.get('did')}")
        await params.result_callback(r)

    _KEYS = {"space", "spacebar", "enter", "return", "up", "down", "left", "right", "escape", "esc", "tab"}
    _NOT_APP = {"room", "the room", "room tab", "lights", "lights panel", "design", "the app", "app", "gallery",
                "interface", "the interface", "interface tab"}

    async def _nav_direct(self, text: str) -> bool:
        """"Go to the contact tab", "scroll down", "click send", "press space": straight to the screen, no LLM."""
        a = self._active()
        if not a or a["kind"] != "app":
            return False
        t = strip_name(text, self.cfg.name).lower().strip(" .!?")
        t = re.sub(r"^(please |can you |could you |now )+", "", t)
        t = re.sub(r" please$", "", t)
        cmd = None
        m = re.fullmatch(r"(?:scroll|go|move) (up|down)(?: a bit| a little| more| again)?|page (up|down)", t)
        if m:
            cmd = {"action": "scroll", "target": m.group(1) or m.group(2)}
        m = cmd is None and re.fullmatch(r"(?:scroll|go|jump|take me) (?:back )?(?:to )?(?:the )?(top|bottom)"
                                         r"(?: of the page)?", t)
        if m:
            cmd = {"action": "scroll", "target": m.group(1)}
        if cmd is None and t in ("go back", "back", "previous page", "go to the previous page"):
            cmd = {"action": "back"}
        m = cmd is None and re.fullmatch(r"(?:press|hit|tap) (?:the )?(\w+)(?: key| button)?", t)
        if m and m.group(1) in self._KEYS | set("abcdefghijklmnopqrstuvwxyz"):
            cmd = {"action": "press", "target": m.group(1)}
        m = cmd is None and re.fullmatch(r"(?:go|take me|jump|navigate|switch|move|head)(?: back)? to (?:the )?(.+?)"
                                         r"(?: tab| page| section| part| menu| screen)?", t)
        if m and m.group(1) not in self._NOT_APP:
            cmd = {"action": "go_to", "target": m.group(1)}
        m = cmd is None and re.fullmatch(r"(?:open|show me|show) (?:the )?(.+?) (?:tab|page|section|menu)", t)
        if m and m.group(1) not in self._NOT_APP:
            cmd = {"action": "go_to", "target": m.group(1)}
        m = cmd is None and re.fullmatch(r"(?:click|press|tap|hit|select|choose)(?: on)? (?:the )?(.+?)"
                                         r"(?: button| link| tab)?", t)
        if m:
            cmd = {"action": "click", "target": m.group(1)}
        if cmd is None:
            return False
        await self._go_first(_view_tab(a))            # the app's own tab first (Interface or Office)
        r = await self._app_cmd(cmd)
        if not r.get("ok"):
            return False                           # not found: the LLM gets it (it can look and choose)
        logger.info(f"app nav (direct): {cmd} -> {r.get('did')}")
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": f"({r.get('did', 'done')})"})
        return True                                # the screen moving is the answer

    def on_media(self, msg: dict):
        """The display reports a video: playing / paused / ended / closed."""
        v = next((x for x in self.views if x["id"] == msg.get("view")), None)
        if v is None:
            return
        v["state"] = msg.get("state")
        self.io.media_playing = self.active_view == v["id"] and v["state"] == "playing"

    async def _media_direct(self, text: str) -> bool:
        """"Elkit, pause the video" without the LLM (quick, and the video is loud)."""
        text = strip_name(text, self.cfg.name)
        if not (self.video or self.photo) or not MEDIA_CMD.search(text) or len(text.split()) > 8:
            return False
        if self.photo:                             # photos showing: next / previous / close
            w = set(re.findall(r"[a-z]+", text.lower()))
            action = ("close" if w & {"close", "exit", "hide", "quit"} else
                      "previous" if w & {"previous", "back", "last", "before"} else
                      "next" if w & {"next", "another", "more", "other"} else None)
            if action is not None:
                await self._go_first("design")         # photos live in Interface
            if action is None or not self._video_cmd(action):
                return False
            self.context.add_message({"role": "user", "content": text})
            self.context.add_message({"role": "assistant", "content": VIDEO_ACTIONS[action]})
            return True
        t = text.lower()
        w = set(re.findall(r"[a-z]+", t))
        if w & {"light", "lights", "lamp", "thermostat", "temperature", "music", "song", "timer", "alarm", "heat"}:
            return False                           # "turn the lights up" is for the room, not the video
        nums = [int(n) for n in re.findall(r"[0-9]+", t)]
        up, down = bool(w & {"up", "louder", "higher"}), bool(w & {"down", "quieter", "softer", "lower"})
        action = ("close" if w & {"close", "exit", "quit", "hide"} else
                  "unmute" if "unmute" in w else
                  "mute" if "mute" in w else
                  "restart" if w & {"restart", "beginning", "over"} else
                  "forward" if w & {"skip", "ahead", "forward"} else
                  "back" if w & {"rewind", "back"} and not up else
                  "volume" if nums and w & {"volume", "percent"} else
                  "louder" if up and not w & {"pause"} else
                  "quieter" if down else
                  "resume" if w & {"resume", "unpause", "continue", "play"} else
                  "pause" if w & {"pause", "stop"} else None)
        level = nums[0] if action == "volume" else None
        seconds = nums[0] if action in ("forward", "back") and nums else None
        if action is not None and action != "close":
            await self._go_first("design")             # the video lives in Interface
        if action is None or not self._video_cmd(action, level=level, seconds=seconds):
            return False
        logger.info(f"video {action} (direct)")
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": VIDEO_ACTIONS[action]})
        return True                                # the video changing is the answer: nothing spoken over it

    def _status(self, status: str):
        """The speaking pulse on the display: listening | thinking | speaking."""
        if self.ui:
            self.ui.send({"type": "status", "status": status})

    def _heard(self):
        self._trim_context()
        self._status("thinking")

    async def fresh_session(self):
        """reset_session, and first cut off any reply still being written or spoken (fd-benchmark: the last trial's
        answer was still playing when the next clip started: 3 failures that were not hers to fail)."""
        tap = getattr(self, "_heard_tap", None)
        if tap is not None:
            try:
                await tap.broadcast_interruption()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"fresh session: interruption failed: {e}")
        if getattr(self, "nfm", None) is not None:
            self.nfm.paused = ""                       # nothing left to "continue" from the last trial
        self.reset_session()

    def reset_session(self):
        """A fresh session (fd-benchmark between trials: each clip is its own conversation, as in the papers):
        the history back to the few-shot, no views or video, the room empty and quiet. Nothing is saved."""
        if self.context is not None:
            self.context.set_messages([*FEW_SHOT])
        self.views, self.active_view, self.app_url, self.app_keys = [], None, "", []
        self.io.media_playing = False
        self._last_act, self._lead = None, None
        self._drop_held()
        room = getattr(self, "room", None)
        if room is not None:
            if room.state.get("ringing"):
                room.stop_ringing()
            if room.state["music"]["state"] != "stopped":
                room.state["music"] = {"state": "stopped", "track": "", "index": 0}
                room._apply_music()
            for k in ("timers", "alarms", "reminders", "messages"):
                room.state[k] = []
            room.state["lists"] = {}
            room._changed()
        logger.info("fresh session (reset between benchmark trials)")

    def _trim_context(self, keep_turns: int = 12):
        """Keep the few-shot plus the last `keep_turns` user turns (a whole design session; a cloud LLM reads this
        in a blink, a local one would want fewer).
        Cuts only at a user message, so tool calls and their results stay together."""
        msgs = self.context.get_messages()
        n_fixed = len(FEW_SHOT)
        users = [i for i, m in enumerate(msgs) if i >= n_fixed and isinstance(m, dict) and m.get("role") == "user"]
        if len(users) > keep_turns:
            cut = users[-keep_turns]
            self.context.set_messages(msgs[:n_fixed] + msgs[cut:])

    def _resume_soon(self):
        """Speech that interrupted the bot turned out not to be for it (side talk, TV): finish the reply."""
        if self.io.bot_speaking:
            return                                    # it was only ducked, it is still talking
        self.tasks.append(asyncio.create_task(self.nfm._maybe_resume(from_ignored=True)))

    async def _warm_up(self, ep, llm, tools):
        """Put the system prompt + tool list into llama.cpp's prompt cache, so the first turn is not slow."""
        import httpx
        t = time.perf_counter()
        body = {"model": ep.model, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "system", "content": self.system}, *FEW_SHOT,
                             {"role": "user", "content": "hello"}],
                "tools": llm.get_llm_adapter().to_provider_tools_format(tools)}
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                r = await c.post(f"{ep.base_url}/chat/completions", json=body)
            logger.info(f"LLM warm-up {time.perf_counter() - t:.1f} s (HTTP {r.status_code}, "
                        f"{(r.json().get('usage') or {}).get('prompt_tokens', '?')} prompt tokens)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM warm-up failed: {e}")

    # ---------------- display + announcements ----------------
    def _room_changed(self, state):
        if self.ui:
            from ..agent import _public
            self.ui.send({"type": "room", "state": _public(state)})

    async def _announce(self, text: str):
        lead = getattr(self, "_lead", None)
        if lead:                                     # a change of mind: "Okay, not 5 minutes: 10 minutes instead."
            self._lead = None
            text = f"{lead} {text}"
        if self.worker is not None:
            await self.worker.queue_frames([TTSSpeakFrame(text)])

    def _app_set(self, output):
        """tweak_app changed live settings: push them into the running app on the display."""
        try:
            d = json.loads(output) if isinstance(output, str) else output
            if isinstance(d, dict) and d.get("ok") and d.get("patch") and self.ui:
                self.app_url = d.get("url") or self.app_url
                if d.get("reload") or any("|flip" in str(x) for x in d["patch"].values()):
                    # an app built before flipping existed has the old runtime in the open page: load it again
                    base = (d.get("url") or self.app_url).split("?")[0]
                    v = next((x for x in self.views if x["kind"] == "app" and x.get("url") == base), None)
                    if v is not None:
                        v["v"] = v.get("v", 0) + 1
                        self._sync_views()
                self.ui.send({"type": "app_set", "url": d.get("url"), "patch": d["patch"]})
                logger.info(f"live change: {d.get('changed')}")
                d = {k: v for k, v in d.items() if k != "patch"}      # photos are big: keep them out of the LLM
                return json.dumps(d, ensure_ascii=False)
        except (ValueError, TypeError):
            pass
        return output

    def _remember_app(self, d: dict):
        if isinstance(d, dict) and d.get("url"):
            self.app_url = d["url"]
            if d.get("settings") is not None:
                self.app_keys = [x.split("=", 1)[0].strip() for x in str(d["settings"]).split(", ") if "=" in x]

    async def _intercept(self, text: str) -> bool:
        """Requests handled without the LLM: "continue" after an interruption, predefined answers, live app tweaks."""
        if is_bare_stop(text, self.cfg.name):          # "stop": no reply, ever (HeardTap already cut the speech)
            if self.room.state.get("ringing"):
                self.room.stop_ringing()               # an alarm or timer ringing: "stop" means that
            elif self.video and self.io.media_playing:
                self._video_cmd("pause")               # a video playing: pause it
            self.context.add_message({"role": "user", "content": text})
            self.context.add_message({"role": "assistant", "content": "(stopped)"})
            return True
        if self.room.state.get("ringing") and len(re.findall(r"[a-z']+", text.lower())) >= 2:
            # someone talks to it while a timer / alarm rings: they noticed it, so it stops; then the request
            logger.info(f"ringing stopped by a request: {text!r}")
            self.room.stop_ringing()
        held = getattr(self, "_held", None)
        if held and not held["task"].done():
            said = strip_name(text, self.cfg.name).strip()
            if not (_CORRECTION_ALONE.match(said) or _CORRECTION.match(said)):
                await self._flush_held()           # a new request, not "wait, no...": the held change is made first
        if await self._mind_direct(text):
            return True
        if await self._chain(text):
            return True
        self._last_act = {"text": text, "t": time.monotonic(), "undo": None, "kind": "llm"}
        return False

    async def _chain(self, text: str) -> bool:
        """The requests done without the LLM, in order. What was done is kept (_last_act) for "wait, no, ..."."""
        self._act_undo = None
        for fn in (self._continue, self._choice_direct, self._view_direct, self._media_direct, self._nav_direct,
                   self._room_direct, self._quick, self._direct_tweak):
            if await fn(text):
                if fn != self._continue:
                    self._last_act = {"text": text, "t": time.monotonic(), "undo": self._act_undo,
                                      "kind": fn.__name__}
                return True
        return False

    async def _mind_direct(self, text: str) -> bool:
        """'Wait, no, ...' for ANY request, not only games: "Set a timer for 5 minutes. Wait, no, 10." / "Turn the
        lights blue." ... "Actually, red." / "Make me a snake game." ... "Wait, no, a racing game instead."
        The first one is undone (or never done), the new one is done, and Elkit says "Okay, not 5 minutes:
        10 minutes instead." What is not a fast request goes to the LLM, told exactly what changed."""
        said = strip_name(text, self.cfg.name).strip()
        combined = _CORRECTION.match(said)
        alone = None if combined else _CORRECTION_ALONE.match(said)
        last = getattr(self, "_last_act", None)
        fresh = last is not None and time.monotonic() - last["t"] < 60
        loose = None
        if not (combined or alone) and fresh and time.monotonic() - last["t"] < 20 and last["kind"] != "_direct_tweak":
            # live 2026-11-06: "Wait, no, make it 10 minutes" was heard as "We know make it 10 minutes": "make it
            # <a number / colour>" just after a request still changes that one (only a number or a colour)
            lm = re.search(r"\b(?:make it|make that|change it to|change that to|i meant)\s+(.+?)\W*$", said, re.I)
            if lm and len(lm.group(1).split()) <= 3:
                m0 = mind.merge(last["text"], lm.group(1))
                if m0 and m0["how"] in ("number", "colour"):
                    loose = lm.group(1)
        if not (combined or alone or loose):
            return False
        if self.app_keys and (combined or (fresh and last["kind"] == "_direct_tweak")) \
                and await self._correction_direct(said, text):
            return True                              # a game: its own wording ("The bird is black now.")
        if combined:
            first, second, undo = combined["first"], combined["second"], None
            if not (mind.VERBS.match(mind.norm(first)) or rules.match(first, {})):
                return False                         # "Hello. Actually, how are you": not a request changed
        elif fresh:
            first, second, undo = last["text"], (alone["second"] if alone else loose), last
        else:
            return False
        new = mind.clean(second)
        if re.match(r"(?:what|who|where|when|why|how|is|are|can|could|do|does|will|tell)\b", new) and \
                not re.match(r"(?:what|how) about\b", mind.norm(second)):
            return False                             # "Actually, what time is it?": a new question
        strong = loose or re.search(r"\binstead\b|^\W*(?:wait|hold on)\W+no|^\W*no\W+wait|\bi meant\b", said, re.I)
        if not (combined or strong) and (len(new.split()) > 3 or re.match(r"(?:i|you|we)\b", new)):
            return False                             # "Sorry, I didn't hear you": not a correction
        m = mind.merge(first, second)
        if m is None:
            return False
        lead = mind.lead(m)
        logger.info(f"change of mind ({m['how']}): {first!r} -> {m['merged']!r} | {lead!r}")
        self._drop_held()
        if undo and undo.get("undo"):
            self._undo_room(undo["undo"])
        self._lead = lead
        if await self._chain(m["merged"]):
            if self._lead:                           # the fast path said nothing: the lead on its own
                self._lead = None
                await self._announce(lead)
            return True
        self._lead = None
        done = undo is not None and undo.get("kind") == "llm"
        self.context.add_message({"role": "system", "content":
            f'They changed their mind. Before: "{first}". Now they want: "{m["merged"]}". '
            + ("The first one may be done already: change or cancel it (edit or delete the reminder or alarm, cancel "
               "the timer, edit_app, or create_app with instead=true while that build is still running) so only "
               "the new one is left. " if done else "Do only the new one. ")
            + f'Start your answer with "{lead}"'})
        self._last_act = {"text": m["merged"], "t": time.monotonic(), "undo": None, "kind": "llm"}
        return False

    _UNDO_KEYS = {"lights": "lights", "thermostat": "thermostat_c", "timer": "timers", "cancel_timer": "timers",
                  "alarm": "alarms", "cancel_alarm": "alarms", "list_items": "lists", "volume": "volume",
                  "add_reminder": "reminders", "delete_reminder": "reminders", "edit_reminder": "reminders"}

    def _undo_room(self, u: dict):
        """Put back what the last room command changed (the 5-minute timer goes, the lights are blue again)."""
        snap = self.room.snapshot()
        for k in u["keys"]:
            snap[k] = copy.deepcopy(u["before"][k])
        self.room.restore(snap)
        logger.info(f"undone for the change of mind: {u['keys']}")

    async def _room_direct(self, text: str) -> bool:
        """"Turn on the lights", "set a timer for 5 minutes", "volume 30", "what time is it": the room at once,
        without the LLM (instant, no tokens). Only whole commands match (rules.match); anything else goes on."""
        room = getattr(self, "room", None)
        if room is None:
            return False
        hit = rules.match(strip_name(text, self.cfg.name), room.state)
        if hit is None:
            return False
        actions, say = hit
        if not actions and not say:
            return False                               # a bare "stop": handled before this
        # it shows in the Virtual Home: go there first and say so, let the tab settle, THEN change the room,
        # so the change is seen happening (the lamp fades to blue) instead of arriving already done
        moved = await self._go_first("room") if actions else ""
        before = room.snapshot()
        keys = sorted({self._UNDO_KEYS[a["tool"]] for a in actions if a.get("tool") in self._UNDO_KEYS and
                       not (a["tool"] == "list_items" and a.get("action") == "show")})
        if keys:
            self._act_undo = {"before": before, "keys": keys}     # for "wait, no, 10 minutes"
        t = time.perf_counter()
        for a in actions:
            r = await room.execute(a)
            if not r.get("ok", True):
                say = r.get("speak") or "Sorry, I couldn't do that."
                break
            say = r.get("speak") or say
        logger.info(f"room (direct,{1000 * (time.perf_counter() - t):.0f} ms): {text!r} -> {actions} "
                    f"{(moved + ' ' if moved else '') + (say or '')!r}")
        self.context.add_message({"role": "user", "content": text})          # the LLM still knows it happened
        self.context.add_message({"role": "assistant", "content": f"{moved} {say}".strip() or "(done)"})
        if say:
            await self._announce(say)
        return True

    async def _quick(self, text: str) -> bool:
        """A question from quick_answers.txt: say its answer at once (no LLM round trip, no tokens)."""
        if self.quick is None or self.worker is None:
            return False
        t = time.perf_counter()
        hit = self.quick.match(text)
        if hit is None:
            return False
        answer, score = hit
        logger.info(f"quick answer ({score:.2f}, {1000 * (time.perf_counter() - t):.0f} ms): {answer[:70]!r}")
        self._event("quick_answer", heard=text, answer=answer, score=round(score, 2))
        self.context.add_message({"role": "user", "content": text})        # the LLM still knows it was said
        self.context.add_message({"role": "assistant", "content": answer})
        await self._announce(answer)
        return True

    async def _continue(self, text: str) -> bool:
        """"Okay, continue" / "go on" after the person cut in: say the rest of the interrupted reply at once,
        starting from the sentence that was cut. If that reply was not fully written yet, the LLM finishes it."""
        if not CONTINUE.match(text.strip()) or self.worker is None:
            return False
        got = self.nfm.take_paused()
        if got is None:
            return False                               # nothing to continue: the LLM answers "go on" normally
        rest, said, done = got
        if done:
            logger.info(f"continue: {rest[:70]!r}")
            self.nfm.reply_full, self.nfm.spoken, self.nfm.last_sentence, self.nfm.reply_done = rest, "", "", True
            await self.worker.queue_frames([TTSSpeakFrame(rest)])
        else:
            logger.info("continue: the reply was not finished, the LLM writes the rest")
            await self.worker.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "system", "content":
                f"You were interrupted. You had said: \"{said[-300:]}\" and were about to say: \"{rest[:300]}\". "
                "They asked you to continue: carry on from exactly there, without repeating what you said."}],
                run_llm=True)])
        return True

    async def _difficulty_direct(self, said: str) -> bool:
        """'Make it easier', 'set it to hard', 'easy mode', 'it's too hard' while a game is on screen: the game's
        difficulty setting (1 easy, 2 normal, 3 hard) at once. A game built before it had that setting gets
        slower + wider gaps (easier) or faster + narrower gaps (harder)."""
        lives = _LIVES.match(said.strip())
        key = next((k for k in self.app_keys if k.lower() == "lives"), None)
        if lives and key:                               # "give me three lives" / "five lives please"
            r = await self.tools_conn.call("tweak_app", {"changes": {key: lives.group(1)}}, timeout=30)
            if not (isinstance(r, dict) and r.get("ok")):
                return False
            self._app_set(r)
            n = (r.get("patch") or {}).get(key)
            msg = f"Done, you have {n if n is None else int(float(n))} {'life' if n == 1 else 'lives'} now."
            self.context.add_message({"role": "user", "content": said})
            self.context.add_message({"role": "assistant", "content": msg})
            await self._announce(msg)
            return True
        m = _DIFFICULTY.match(said.strip())
        if not m:
            return False
        want = next(g for g in m.groups() if g).lower()
        if want in ("more difficult", "more challenging", "too easy"):
            want = "harder"
        elif want in ("less difficult", "less hard", "too hard", "too difficult"):
            want = "easier"
        elif want == "medium":
            want = "normal"
        key = next((k for k in self.app_keys if k.lower() == "difficulty"), None)
        if key:
            changes = {key: want}
        else:                                           # an older game: speed and gaps instead
            easier = want in ("easier", "easy")
            changes = {}
            if want == "normal":
                return False                            # no level to go back to: let the LLM say so
            if "speed" in self.app_keys:
                changes["speed"] = "slower" if easier else "faster"
            gap = next((k for k in self.app_keys if k.lower() in ("gap", "gapsize", "gap_size", "space")), None)
            if gap:
                changes[gap] = "bigger" if easier else "smaller"
            if not changes:
                return False
        await self._go_first(self._app_tab())
        logger.info(f"difficulty by voice: {changes}")
        try:
            r = await self.tools_conn.call("tweak_app", {"changes": changes}, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tweak_app: {e}")
            return False
        if not isinstance(r, dict):
            return False
        if not r.get("ok"):
            already = re.search(r"already on (\w+)", str(r.get("message", "")))
            if not already:
                return False
            msg = f"It's already on {already.group(1)}."
        else:
            self._app_set(r)
            lvl = (r.get("patch") or {}).get(key) if key else None
            if lvl is not None:
                from ..mcp_servers.live import LEVEL_NAMES
                msg = f"Done, it's on {LEVEL_NAMES.get(round(float(lvl)), 'normal')} now."
            else:
                msg = ("Done, it's easier now: slower, with more room." if want in ("easier", "easy")
                       else "Done, it's harder now: faster, with less room.")
        self.context.add_message({"role": "user", "content": said})
        self.context.add_message({"role": "assistant", "content": msg})
        await self._announce(msg)
        return True

    async def _direct_tweak(self, text: str) -> bool:
        """'Change the bird to a butterfly' while a game is on screen: change it at once, without the LLM
        (a small model routes this badly). Returns True when handled."""
        if not self.app_keys:
            return False
        from ..mcp_servers import live
        said = strip_name(text, self.cfg.name)
        if await self._difficulty_direct(said) or await self._correction_direct(said, text):
            return True
        mf = re.match(r"^(?:please |can you |could you )?(?:flip|mirror|reverse|turn)(?: around)? (?:the )?(\w+)"
                      r"(?: around| over)?\W*$|^(?:make|have|let) (?:the )?(\w+) (?:face|look|point|go)(?: to)?(?: the)? "
                      r"(?:right|left|other way|forward|backward)s?\W*$|^(?:the )?(\w+) (?:is|are) facing the wrong way\W*$",
                      said, re.I)
        m = mf or _TWEAK.match(text.strip()) or _TWEAK_SHORT.match(text.strip())
        if not m:
            return False
        if mf:
            what, value = next(g for g in mf.groups() if g), "flip"
            side = re.search(r"\b(right|left|forwards?|backwards?)\b", said, re.I)
            if side:                                 # "make the bird face right": that way, not a toggle
                value = f"face {side.group(1).lower()}"
        else:
            what, value = m.group(1), m.group(2).strip(" .!?")
        return await self._apply_tweak(what, value, text)

    async def _correction_direct(self, said: str, text: str) -> bool:
        """'Make the bird green. Wait, no. I want it to be black instead.': only the last one is done (green never
        shows), and Elkit says so: "Okay, not green: black instead. The bird is black now." """
        c = _CORRECTION.match(said.strip())
        first = c and (_TWEAK.match(c["first"].strip()) or _TWEAK_SHORT.match(c["first"].strip()))
        if first:
            what1, v1 = first.group(1), first.group(2).strip(" .!?")
            sec = c["second"]
        else:
            # said on its own, just after a change ("Make the bird green." ... "Wait, no. Black instead."):
            # it corrects that change
            alone = _CORRECTION_ALONE.match(said.strip())
            last = getattr(self, "_last_tweak", None)
            if not (alone and last and time.monotonic() - last[2] < 60):
                return False
            # "Actually, what time is it?" is a new question, not a correction; a plain "No, black." needs a
            # clear cue (wait no / I meant / instead)
            if re.match(r"(?:what|who|where|when|why|how|is|are|can|could|do|does|will|tell)\b", alone["second"], re.I) \
                    or said.strip().endswith("?") or not re.search(r"\binstead\b|^\W*(?:wait|hold on)\W+no|^\W*no\W+wait|"
                                                                     r"\bi meant\b", said, re.I):
                return False
            what1, v1, sec = last[0], last[1], alone["second"]
        sec = sec.strip(" .!?")
        m2 = _TWEAK.match(sec) or _TWEAK_SHORT.match(sec)
        if m2:
            what2, v2 = m2.group(1), m2.group(2).strip(" .!?")
        else:
            what2 = "it"
            v2 = re.sub(r"^(?:i want (?:it|that|them)(?: to be)?|i meant|i'd like (?:it )?|make (?:it|that|them)|"
                        r"let'?s (?:do|go with|have)|go with|it should be|(?:it'?s|that'?s))\s+", "", sec, flags=re.I)
        v2 = re.sub(r"\s+instead(?: of \w+)?$", "", v2, flags=re.I).strip(" .!?")
        if what2.lower() in ("it", "that", "this", "them"):
            what2 = what1
        if not v2 or len(v2.split()) > 4:
            return False
        if re.sub(r"^(?:the|my)\s+", "", v2.lower()).rstrip("s") in {k.lower().rstrip("s") for k in self.app_keys}:
            return False                             # "No, I meant the pipes": which thing, not a new value: the LLM
        logger.info(f"change of mind: {what1} {v1!r} -> {what2} {v2!r}")
        self._drop_held()                            # green was still held: it is never made
        return await self._apply_tweak(what2, v2, text, lead=f"Okay, not {v1}: {v2} instead.")

    async def _apply_tweak(self, what: str, value: str, text: str, lead: str = "") -> bool:
        from ..mcp_servers import live
        if re.fullmatch(r"(a (lot|bit|little) )?(faster|slower|quicker)", value, re.I):
            # "make the snake faster" -> speed; "make the ball faster" -> ballSpeed when the game has it
            what = next((k for k in self.app_keys if k.lower() == f"{what.lower()}speed"), "speed")
        if what.lower() in ("it", "him", "her", "them", "this", "that", "player", "character"):
            # "change it to a dog" mid-game: the player (first thing that is not background/text/a number)
            what = next((k for k in self.app_keys if k.lower() not in
                         ("sky", "background", "bg", "text", "score", "speed", "gap", "gravity")), what)
        key = live.match_key(what, self.app_keys)
        if key is None or not value or len(value.split()) > 6 or \
                value.lower() in ("on", "off", "up", "down", "back", "over", "around", "it", "that"):
            return False
        if not lead and getattr(self, "_lead", None):  # a change of mind for anything: made now, not held
            lead, self._lead = self._lead, None
        await self._go_first(self._app_tab())          # the app's own tab first, then the change is seen
        logger.info(f"live change by voice: {key} -> {value!r}")
        if re.search(r"photo|picture|image", value, re.I):
            await self._announce("Finding a picture.")
        elif getattr(self, "tweak_hold", 0) and not lead and value != "flip" and not value.startswith("face "):
            # "Green, okay." now, the change a moment later: "wait, no, black instead" can still cancel it
            self._drop_held()
            self._last_tweak = (what, value, time.monotonic())
            self._held = {"now": False, "task": asyncio.create_task(self._hold_then_tweak(key, what, value, text))}
            await self._announce(f"{value[0].upper()}{value[1:]}, okay.")
            return True
        return await self._do_tweak(key, what, value, text, lead)

    def _drop_held(self):
        held, self._held = getattr(self, "_held", None), None
        if held and not held["task"].done():
            logger.info("held change dropped")
            held["task"].cancel()

    async def _flush_held(self):
        held = getattr(self, "_held", None)
        if held and not held["task"].done():
            held["now"] = True
            try:
                await held["task"]
            except asyncio.CancelledError:
                pass

    async def _hold_then_tweak(self, key: str, what: str, value: str, text: str):
        """The held change is made tweak_hold seconds after "Green, okay." has been said (2 x tweak_hold if nothing is
        heard of it). If the person starts talking meanwhile, it waits for what they say: a "wait, no..." drops it,
        anything else lets it go first (3 s after they stop, at most)."""
        t0 = last_voice = time.monotonic()
        spoke = bot_seen = False
        while True:
            held = self._held
            if held is None or held["task"] is not asyncio.current_task():
                return
            now = time.monotonic()
            if getattr(self.io, "user_speaking", False):
                spoke, last_voice = True, now
            talking = bool(getattr(self.io, "bot_speaking", False))
            bot_seen = bot_seen or talking
            said_ok = (bot_seen and not talking and now - max(t0, getattr(self.io, "t_bot_stopped", t0)) >= self.tweak_hold) \
                or (not bot_seen and now - t0 >= 2 * self.tweak_hold)
            if held["now"] or (not spoke and said_ok) or (spoke and now - last_voice >= 3.0) or now - t0 >= 12.0:
                break
            await asyncio.sleep(0.05)
        self._held = None
        if not await self._do_tweak(key, what, value, text, quiet=True):
            await self._announce(f"Sorry, I couldn't change the {what}.")

    async def _do_tweak(self, key: str, what: str, value: str, text: str, lead: str = "", quiet: bool = False) -> bool:
        try:
            r = await self.tools_conn.call("tweak_app", {"changes": {key: value}}, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tweak_app: {e}")
            return False
        if not isinstance(r, dict):
            return False
        if not r.get("ok"):
            already = re.search(r"already faces (\w+)", str(r.get("message", "")))
            if not already:
                return False                         # e.g. "bigger" with no size setting: the LLM (edit_app)
            msg = f"The {what} already faces {already.group(1)}."
            self.context.add_message({"role": "user", "content": text})
            self.context.add_message({"role": "assistant", "content": msg})
            await self._announce(msg)
            return True
        self._app_set(r)
        self._last_tweak = (what, value, time.monotonic())    # "wait, no, black instead" corrects this one
        flipped = any("flip" in str(c) for c in (r.get("changed") or {}).values()) if isinstance(r.get("changed"), dict) \
            else bool(re.search(r"flip|turned back", str(r.get("changed", "")), re.I))
        are = "are" if what.lower().endswith("s") and not what.lower().endswith(("ss", "us", "is")) else "is"
        verb = re.match(r"(?:(?:a )?(?:lot|bit|little)\s+)?(jump|fly|flap|bounce|shoot|fire|move|run|fall|spawn|come|grow)s?"
                        r"\s+(\w+(?: often)?)$", value, re.I)
        who = "it" if what.lower() in ("it", "this", "that") else f"the {what}"
        msg = f"Done, {who} {verb.group(1).lower() + ('' if are == 'are' else 'es' if verb.group(1).lower().endswith('sh') else 's')} " \
              f"{verb.group(2).lower()} now." if verb else \
            "Done, it's faster now." if re.search(r"faster|quicker", value, re.I) else \
            "Done, it's slower now." if re.search(r"slower", value, re.I) else \
            f"Done, the {what} {'face' if are == 'are' else 'faces'} {value[5:]} now." if value.startswith("face ") else \
            f"Done, the {what} {'face' if are == 'are' else 'faces'} the other way now." if value == "flip" or flipped \
            else f"Done, the {what} {are} {value} now."
        if lead:                                     # "Okay, not green: black instead. The bird is black now."
            msg = f"{lead} {msg[6].upper() + msg[7:] if msg.startswith('Done, ') else msg}"
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": msg})
        if not quiet:                                # a held change: "Green, okay." was said already
            await self._announce(msg)
        return True

    def _open_app_view(self, url: str, title: str) -> dict:
        base = url.split("?")[0]
        kw = _app_kind(title)
        return self._open_view("app", {"url": base, "title": title}, [title, base.rsplit("/", 1)[-1].replace("-", " ")],
                               match=("url", base), reverse=False, kindword=kw)

    def _show_app(self, output):
        try:
            d = json.loads(output) if isinstance(output, str) else output
            self._remember_app(d)
            if isinstance(d, dict) and d.get("url"):
                self._open_app_view(d["url"], d.get("title", ""))
        except (ValueError, TypeError):
            pass
        return output

    async def _watch_apps(self):
        """When a background build finishes: show it and say so (waits until nobody is talking)."""
        while True:
            try:
                await self._watch_apps_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("app watcher")

    async def _watch_apps_once(self):
        while True:
            await asyncio.sleep(1.5)
            try:
                st = await self.tools_conn.call("app_status", {}, timeout=10)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"app_status: {e}")
                continue
            if isinstance(st, dict):
                self._building = bool(st.get("building"))
            if self.ui and isinstance(st, dict):
                self.ui.send({"type": "build", **{k: st.get(k) for k in (
                    "building", "title", "elapsed_s", "chars", "expect", "code", "step", "eta_s", "kind", "change")}})
            finished = (st.get("finished") or []) if isinstance(st, dict) else []
            for n, job in enumerate(finished):
                while self.io.user_speaking or self.io.bot_speaking:
                    await asyncio.sleep(0.3)
                if job.get("state") == "ready":
                    self._remember_app(job)
                    logger.info(f"app ready: {job.get('title')} -> {job.get('url')} ({job.get('seconds')} s)")
                    v = self._open_app_view(job["url"], job.get("title", ""))
                    what = "changed" if job.get("kind") == "edit" else "ready"
                    msg = f"{job.get('title', 'Your app')} is {what}, it's on the {v['label']} tab."
                    if job.get("problems"):
                        # not said any more (user 2026-11-06): only the log keeps it
                        # msg += " It may have a bug; tell me if something doesn't work."
                        logger.warning(f"app ready with problems (not said): {str(job.get('problems'))[:200]}")
                else:
                    msg = f"Sorry, building {job.get('title', 'the app')} failed."
                self._event("app_" + job.get("state", "?"), **{k: job.get(k) for k in ("title", "url", "seconds")})
                if job.get("settings"):
                    self.context.add_message({"role": "system", "content":
                        f"{job.get('title', 'The app')} live settings (change with tweak_app): {job['settings']}"})
                # several edits of the same app finishing together: say it once, not once per edit
                last_msg, last_t = getattr(self, "_last_build_msg", ("", 0.0))
                if n < len(finished) - 1 or (msg == last_msg and time.monotonic() - last_t < 30):
                    logger.info(f"build note not repeated: {msg[:60]!r}")
                    continue
                self._last_build_msg = (msg, time.monotonic())
                self.context.add_message({"role": "assistant", "content": msg})
                await self._announce(msg)

    async def close(self):
        for t in self.tasks:
            t.cancel()
        for c in (self.asr, self.tts_conn, self.tools_conn):
            await c.close()
        try:
            await self.mcp_tools.close()
        except Exception:  # noqa: BLE001
            pass
        if getattr(self, "room_hub", None):
            await self.room_hub.cleanup()
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            p = self.log_dir / f"fda-pc-events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            stats = {"kind": "stats", **self.guard.stats, "not_for_me": self.nfm.ignored,
                     "own_voice_blocks": getattr(self.io, "own_blocks", 0)}
            p.write_text("\n".join(json.dumps(e, default=str) for e in self.events + [stats]) + "\n",
                         encoding="utf-8")
