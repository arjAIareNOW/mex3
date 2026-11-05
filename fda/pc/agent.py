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
from . import fallback_llm
from ..mcp_servers import llm as llmlib
from ..room import Room
from ..config import ROOT
from .processors import (CONTINUE, MEDIA_CMD, DuplexGuard, HeardTap, Mark, NotForMe, SpokenTap, ToolFiller, TurnClock,
                         is_bare_stop,
                         strip_name)
from .quick import QuickAnswers
from .services import McpConnection, McpSTTService, McpTTSService

SYSTEM = """You are {name}, a voice assistant on a small screen. You speak out loud in real time: warm, relaxed and
natural, usually one or two short sentences. Everything you write is spoken: no lists, no markdown, no emoji,
numbers as words people say.
How you work:
- Questions get spoken answers. "What options are there for the bird", "what could I use", "any ideas": say two
  or three choices out loud in one sentence and wait for them to pick. Do not build, change, open or show anything
  until they choose. create_app only when they ask you to make something new; a change to what is on the screen
  is tweak_app or edit_app, never a new app.
- Do what the person asks, nothing more. Do not steer the talk to any one thing, do not ask what to do next or what
  to build, and do not list what you can do unless they ask. After a task, a short confirmation is enough.
- Things you can do when asked: build websites, apps and small games live on the screen; the weather and the time
  (get_weather, get_time); the room on the screen: lights, thermostat, timers, alarms, reminders, music and volume.
- Building: when they ask for an app, a website or a game and the idea is clear enough, start at once with create_app
  and a rich, specific description (purpose, sections or gameplay, layout, colours, fonts, mood, controls). If it is
  very vague, ask one quick question, or suggest a direction and start. While it builds you can keep talking; you
  are told when it is ready, then say one short thing about it.
- Remember what was built and what they liked or disliked, and build on it: "it", "that", "the button", "make it
  pop" mean the app on screen. Look changes to a game (colours, swap the bird for a butterfly, a photo for the
  pipes, faster): tweak_app. Any other change, a new section or feature: edit_app with a clear instruction.
  "Undo that": undo_app_change. "The earlier version": restore_app_version (app_history lists them).
  "Show me the snake game": open_app.
- Looking things up: for prices, flights, schedules, news, people or anything current or that you are not sure of,
  use web_search (get_news for headlines, wikipedia for background). If the snippets lack a detail they asked for
  (the airline, the exact price, the date), read_page the best link. Then say the answer in one or two sentences:
  the number, the name, and which site it is from. The links show on the screen by themselves: say "the link is on
  the screen", never read a web address out loud. If the results do not say it, say so; never make up a price.
- Videos: only when they ask for a video and say what it is, play_video with exactly what they asked; never pick
  one they did not ask for. "Play a video" with no subject: ask which. While a video is on the screen, pause, play,
  mute, louder, quieter, skip, restart or close mean the video: video_control (not the room volume or music).
  While a video plays you only hear requests that say your name.
- The person has no mouse or keyboard: everything is by voice. To use the app on the screen (go to a tab, page or
  section, click a button, scroll, type into a field, press a key in a game, read what it says), use app_control.
  If it says not found, choose from the options it lists, or tell them in a few words what is there.
- The screen has two tabs: Interface and Room. "Show me the room" or "the lights" is show_screen room; "back to the
  interface" is show_screen interface. Interface has a sub-tab for every app, game, video, photo set and search,
  each with its own name ("Snake game", "Davao links"); the tool results say the name. Switch or close one with
  interface_tab (action go / close / list); when you open something new, you may say which tab it is on.
- Talk like a person: short acknowledgements, a little enthusiasm, an opinion when it helps. Do not repeat what they
  said, do not list options unless they asked for options, do not explain your tools.
- Stop when the answer is said: never add offers like "let me know if you need anything else" or "what would
  you like next".
- Everything you write is spoken aloud to them. Think silently; write only the words you say TO them. Never
  write "the user wants", "we need to", or plans about which tool to call. No stage directions or asides in
  brackets like "(no further response)" or "(pause)": when you have nothing more to say, just stop.
- Never say something is done unless the tool said so. If something fails, say so and offer to try again.
- If they interrupt you, drop what you were saying and answer what they just said.
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

# MCP tools the agent gets (the server has more; `wikipedia` is left out: general knowledge is answered directly,
# which is ~5-8 s faster with a local model; current facts still go through web_search)
# the tools the agent is given (the tools server has more: news, web search, documents, device check...)
AGENT_MCP_TOOLS = ["create_app", "edit_app", "tweak_app", "undo_app_change", "app_history", "restore_app_version",
                   "list_apps", "open_app", "app_settings", "get_weather", "get_time",
                   "web_search", "read_page", "get_news", "wikipedia", "play_video"]
# a short filler is spoken at once for these (internet, 1-3 s; photos for tweak_app are handled too)
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


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9']+", (text or "").lower().replace("'s", ""))
            if len(w) > 2 and w not in _VIEW_STOP]


def _app_kind(title: str) -> str:
    w = set(re.findall(r"[a-z]+", (title or "").lower()))
    return "game" if w & _GAME_WORDS else "site" if w & _SITE_WORDS else "app"


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
    "music": "Play, pause, resume, stop or skip the music.",
    "volume": "Change the speaker volume (up/down or a level 0-100).",
}


def room_tools(room: Room, on_use=None) -> list[FunctionSchema]:
    out = []
    for name, desc in ROOM_TOOLS.items():
        props = brain.TOOLS[name]

        async def handler(params, _name=name):
            if on_use:
                on_use()
            r = await room.execute({"tool": _name, **(params.arguments or {})})
            await params.result_callback({"ok": r.get("ok", True), "result": r.get("note", ""),
                                          **({"say": r["speak"]} if r.get("speak") else {})})

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
        self.active_view: int | None = None
        self._view_n = 0
        self.context: LLMContext | None = None
        self.quick: QuickAnswers | None = None      # predefined answers (quick_answers.txt), set in build()
        self.tasks: list[asyncio.Task] = []
        self.events: list[dict] = []

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
            name="show_screen", description="Switch the screen tab: 'room' shows the room (lights, thermostat, "
            "timers, alarms, reminders, music); 'interface' shows apps, videos, links and photos.",
            properties={"screen": {"type": "string", "enum": ["interface", "room"]}}, required=["screen"],
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
            name="interface_tab", description="The Interface sub-tabs (one per app, game, video, photos, search). "
            "go: show the tab called `name`; close: close it (no name: the one showing); list: the tabs.",
            properties={"action": {"type": "string", "enum": ["go", "close", "list"]},
                        "name": {"type": "string", "description": "the tab's name or a word of it"}},
            required=["action"], handler=self._interface_tab)
        tools = ToolsSchema(standard_tools=[*remote.standard_tools,
                                            *room_tools(self.room, on_use=lambda: self._tab("room")), show, video,
                                            nav, subtab])
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
        stt = McpSTTService(self.asr, far_db=cfg.far_speech_db, on_ignored=self._resume_soon,
                             own_name=cfg.name)
        self.guard.asr, self.guard.stt = self.asr, stt
        self.guard.bot_text = lambda: self.nfm.reply_full
        self.guard.echo_text = lambda: self.nfm.recent_text()      # its voice can come back late (Pi + PipeWire)
        self.clock = TurnClock()          # 'turn timing (ms)' line in the log, per turn
        pipeline = Pipeline([
            self.transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.6, start_secs=0.15,
                                                                         stop_secs=0.2))),
            self.guard,
            Mark(self.clock, {VADUserStoppedSpeakingFrame: "speech_end"}),
            stt,
            HeardTap(self.io, self.ui, name=cfg.name, on_heard=self._heard, on_ignored=self._resume_soon,
                     intercept=self._intercept, echo_text=lambda: self.nfm.recent_text()),
            Mark(self.clock, {TranscriptionFrame: "heard"}),
            agg.user(),
            Mark(self.clock, {LLMContextFrame: "sent"}),
            llm,
            Mark(self.clock, {LLMTextFrame: "first_word", FunctionCallsStartedFrame: "first_word"}),
            ToolFiller(slow=SLOW_TOOLS),
            self.nfm,
            McpTTSService(self.tts_conn),
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
        return self.worker

    def _tab(self, tab: str):
        if self.ui:
            self.ui.send({"type": "tab", "tab": tab})

    async def _show_screen(self, params):
        tab = (params.arguments or {}).get("screen", "interface")
        self._tab("room" if tab == "room" else "design")
        await params.result_callback({"ok": True, "result": f"the {tab} screen is showing"})

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
            self._tab("design")

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
        if re.fullmatch(r"(?:go|switch|change|jump|move|head|take me|flip)(?: back)?(?: over)? to (?:the )?room(?: tab)?|"
                        r"(?:show me|open|show) the room(?: tab)?|the room tab|room tab", t):
            self._tab("room")
            return self._noted(text, "showing the Room tab")
        if re.fullmatch(r"(?:go|switch|change|jump|move|head|take me|flip)(?: back)?(?: over)? to (?:the )?interface"
                        r"(?: tab)?|(?:show me|open|show) the interface(?: tab)?|(?:the )?interface tab", t):
            self._tab("design")
            return self._noted(text, "showing the Interface tab")
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
        if self.worker is not None:
            await self.worker.queue_frames([TTSSpeakFrame(text)])

    def _app_set(self, output):
        """tweak_app changed live settings: push them into the running app on the display."""
        try:
            d = json.loads(output) if isinstance(output, str) else output
            if isinstance(d, dict) and d.get("ok") and d.get("patch") and self.ui:
                self.app_url = d.get("url") or self.app_url
                if any("|flip" in str(x) for x in d["patch"].values()):
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
        return (await self._continue(text) or await self._view_direct(text) or await self._media_direct(text)
                or await self._nav_direct(text) or await self._room_direct(text) or await self._quick(text)
                or await self._direct_tweak(text))

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
        t = time.perf_counter()
        for a in actions:
            r = await room.execute(a)
            if not r.get("ok", True):
                say = r.get("speak") or "Sorry, I couldn't do that."
                break
            say = r.get("speak") or say
        logger.info(f"room (direct, {1000 * (time.perf_counter() - t):.0f} ms): {text!r} -> {actions} {say!r}")
        self.context.add_message({"role": "user", "content": text})          # the LLM still knows it happened
        self.context.add_message({"role": "assistant", "content": say or "(done)"})
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

    async def _direct_tweak(self, text: str) -> bool:
        """'Change the bird to a butterfly' while a game is on screen: change it at once, without the LLM
        (a small model routes this badly). Returns True when handled."""
        if not self.app_keys:
            return False
        from ..mcp_servers import live
        said = strip_name(text, self.cfg.name)
        mf = re.match(r"^(?:please |can you |could you )?(?:flip|mirror|reverse|turn)(?: around)? (?:the )?(\w+)"
                      r"(?: around| over)?\W*$|^(?:make|have|let) (?:the )?(\w+) (?:face|look|point|go)(?: to)?(?: the)? "
                      r"(?:right|left|other way|forward|backward)s?\W*$|^(?:the )?(\w+) (?:is|are) facing the wrong way\W*$",
                      said, re.I)
        m = mf or _TWEAK.match(text.strip()) or _TWEAK_SHORT.match(text.strip())
        if not m:
            return False
        if mf:
            what, value = next(g for g in mf.groups() if g), "flip"
        else:
            what, value = m.group(1), m.group(2).strip(" .!?")
        if re.fullmatch(r"(a (lot|bit|little) )?(faster|slower|quicker)", value, re.I):
            what = "speed"                          # "make the snake faster" -> the speed setting
        if what.lower() in ("it", "him", "her", "them", "this", "that", "player", "character"):
            # "change it to a dog" mid-game: the player (first thing that is not background/text/a number)
            what = next((k for k in self.app_keys if k.lower() not in
                         ("sky", "background", "bg", "text", "score", "speed", "gap", "gravity")), what)
        key = live.match_key(what, self.app_keys)
        if key is None or not value or len(value.split()) > 6 or \
                value.lower() in ("on", "off", "up", "down", "back", "over", "around", "it", "that"):
            return False
        logger.info(f"live change by voice: {key} -> {value!r}")
        if re.search(r"photo|picture|image", value, re.I):
            await self._announce("Finding a picture.")
        try:
            r = await self.tools_conn.call("tweak_app", {"changes": {key: value}}, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tweak_app: {e}")
            return False
        if not (isinstance(r, dict) and r.get("ok")):
            return False
        self._app_set(r)
        flipped = any("flip" in str(c) for c in (r.get("changed") or {}).values()) if isinstance(r.get("changed"), dict) \
            else bool(re.search(r"flip|turned back", str(r.get("changed", "")), re.I))
        msg = "Done, it's faster now." if re.search(r"faster|quicker", value, re.I) else \
            "Done, it's slower now." if re.search(r"slower", value, re.I) else \
            f"Done, the {what} faces the other way now." if value == "flip" or flipped else f"Done, the {what} is {value} now."
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": msg})
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
                        msg += " It may have a bug; tell me if something doesn't work."
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
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            p = self.log_dir / f"fda-pc-events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            stats = {"kind": "stats", **self.guard.stats, "not_for_me": self.nfm.ignored,
                     "own_voice_blocks": getattr(self.io, "own_blocks", 0)}
            p.write_text("\n".join(json.dumps(e, default=str) for e in self.events + [stats]) + "\n",
                         encoding="utf-8")
