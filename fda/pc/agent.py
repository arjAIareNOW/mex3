"""FDA on Pipecat: the agent harness with MCP.

    transport.input()                      FDA audio engine: mic + speaker, AEC3, own-voice gate  (or fdbench)
    -> VADProcessor(Silero)                voice activity, on the Pi
    -> DuplexGuard                         duck on overlap, stop on real interruptions, resume after backchannels
    -> McpSTTService ---- MCP ---->        asr server (Moonshine offline / Groq Whisper)
    -> HeardTap                            display; "uh huh" while talking is not a turn
    -> user aggregator                     Smart Turn v3 (end of turn), >= 3 words to interrupt the bot
    -> LLM (OpenAI-compatible)             local llama.cpp (free) or Groq/Gemini free tier, tool calling
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
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame
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

from .. import brain
from . import fallback_llm
from ..mcp_servers import llm as llmlib
from ..room import Room
from .processors import CONTINUE, DuplexGuard, HeardTap, NotForMe, SpokenTap, ToolFiller
from .services import McpConnection, McpSTTService, McpTTSService

SYSTEM = """You are {name}, a friendly web designer. You build websites, apps and small games live on the screen
while you talk with the person, like a designer sitting next to a client. You speak out loud in real time: warm,
relaxed and natural, usually one or two short sentences. Everything you write is spoken: no lists, no markdown,
no emoji, numbers as words people say.
How you work:
- This is one continuous design session. Remember what you are building, what they liked and disliked, colours,
  names and ideas from earlier, and build on them. "It", "that", "the button", "make it pop" mean the current design.
- When they want something new and the idea is clear enough, start at once with create_app and a rich, specific
  description (purpose, sections or gameplay, layout, colours, fonts, mood, controls). If it is very vague, ask one
  quick question (who it is for, or the feel they want), or suggest a direction and start.
- While it builds, keep talking naturally: say what you are going for, or ask about the next detail. You are told
  when it is ready; then say one short thing about it and invite feedback.
- Look changes to a game (colours, swap the bird for a butterfly, a photo for the pipes, faster): tweak_app.
  Any other change, a new section or feature: edit_app with a clear instruction. "Undo that": undo_app_change.
  "The earlier version": restore_app_version (app_history lists them). "Show me the snake game": open_app.
- Talk like a person: short acknowledgements, a little enthusiasm, a design opinion when it helps. Do not repeat
  what they said, do not list options, do not explain your tools.
- You can also tell the weather or the time when asked (get_weather, get_time), then carry on designing.
- You can also run the room when asked: lights, thermostat, timers, alarms, reminders, music and volume. They show
  on a separate Room tab; "show me the room" or "the lights panel" is show_screen room, "back to the design" is
  show_screen design. Do this only when asked, then go back to designing.
- Never say something is done unless the tool said so. If a build fails, say so and offer to try again.
- If they interrupt you, drop what you were saying and answer what they just said.
- Only when speech is clearly not meant for you (a TV, or someone talking to another person by name), reply with
  exactly (silent) and nothing else."""

FEW_SHOT = [   # how "(silent)" works; short so they cost little prompt
    {"role": "user", "content": "Stocks fell sharply in afternoon trading, analysts said, as investors weighed the "
                                "latest inflation figures."},
    {"role": "assistant", "content": "(silent)"},
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hey! What would you like to make today?"},
]

_TWEAK = re.compile(r"^\W*(?:(?:can|could) you\s+|please\s+|now\s+|and\s+|okay,?\s+)*"
                    r"(?:change|make|turn|replace|swap|set|switch|paint|colou?r)\s+(?:the\s+|my\s+|its\s+)?"
                    r"([a-z]+(?:\s[a-z]+)?)\s+(?:to|into|with|for|as|be)\s+(.+)$", re.I)

_TWEAK_SHORT = re.compile(r"^\W*(?:(?:can|could) you\s+|please\s+|now\s+|okay,?\s+)*(?:make|turn|paint|colou?r)\s+"
                          r"(?:the\s+|my\s+|(?=it\s|him\s|her\s))([a-z]+)\s+((?:a (?:lot|bit|little) )?[a-z]+(?:\s[a-z]+)?)\W*$", re.I)

# MCP tools the agent gets (the server has more; `wikipedia` is left out: general knowledge is answered directly,
# which is ~5-8 s faster with a local model; current facts still go through web_search)
# Elkit is a designer: only the tools for making things (the server has more: weather, news, documents, device...)
AGENT_MCP_TOOLS = ["create_app", "edit_app", "tweak_app", "undo_app_change", "app_history", "restore_app_version",
                   "list_apps", "open_app", "app_settings", "get_weather", "get_time"]   # server name: get_weather
SLOW_TOOLS: set[str] = set()      # a short filler is spoken at once for these (photos for tweak_app are handled too)

ROOM_TOOLS = {
    "lights": "Turn the room lights on/off, set brightness (0-100) or colour.",
    "thermostat": "Set the room temperature in degrees Celsius.",
    "timer": "Start a countdown timer (seconds).",
    "cancel_timer": "Cancel the running timer.",
    "alarm": "Set an alarm at a time like 06:30.",
    "add_reminder": "Remember something for the user, optionally at a time like 15:00.",
    "list_reminders": "List the user's reminders.",
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
        self.context: LLMContext | None = None
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
        mcp_tools = MCPClient(server_params=StreamableHttpParameters(url=self.urls["tools"], timeout=60,
                                                                    sse_read_timeout=600),
                              tools_filter=AGENT_MCP_TOOLS,
                              tools_output_filters={**{n: self._show_app for n in ("open_app", "undo_app_change", "restore_app_version")},
                                                    "tweak_app": self._app_set})
        self.mcp_tools = mcp_tools
        remote = await mcp_tools.tools()
        show = FunctionSchema(
            name="show_screen", description="Switch the screen: 'room' shows the lights, thermostat, timers, "
            "reminders and music; 'design' shows what you are building.",
            properties={"screen": {"type": "string", "enum": ["design", "room"]}}, required=["screen"],
            handler=self._show_screen)
        tools = ToolsSchema(standard_tools=[*remote.standard_tools,
                                            *room_tools(self.room, on_use=lambda: self._tab("room")), show])
        logger.info(f"tools: {[t.name for t in tools.standard_tools]}")

        self.context = LLMContext(messages=[*FEW_SHOT], tools=tools)
        agg = LLMContextAggregatorPair(self.context, user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]),
            user_turn_stop_timeout=3.0))

        self.guard = DuplexGuard(self.io, cfg)
        self.guard.on_bot = lambda talking: self._status("speaking" if talking else "listening")
        self.nfm = NotForMe(on_ignored=lambda: self._event("not_for_me"), io=self.io)
        self.nfm.own_name = cfg.name
        stt = McpSTTService(self.asr, far_db=cfg.far_speech_db, on_ignored=self._resume_soon,
                             own_name=cfg.name)
        self.guard.asr, self.guard.stt = self.asr, stt
        self.guard.bot_text = lambda: self.nfm.reply_full
        pipeline = Pipeline([
            self.transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.6, start_secs=0.15,
                                                                         stop_secs=0.2))),
            self.guard,
            stt,
            HeardTap(self.io, self.ui, name=cfg.name, on_heard=self._heard, on_ignored=self._resume_soon,
                     intercept=self._intercept),
            agg.user(),
            llm,
            ToolFiller(slow=SLOW_TOOLS),
            self.nfm,
            McpTTSService(self.tts_conn),
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
        tab = (params.arguments or {}).get("screen", "design")
        self._tab("room" if tab == "room" else "design")
        await params.result_callback({"ok": True, "result": f"the {tab} screen is showing"})

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
        self.tasks.append(asyncio.create_task(self.nfm._maybe_resume()))

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
        """Requests handled without the LLM: "continue" after an interruption, and live app tweaks."""
        return await self._continue(text) or await self._direct_tweak(text)

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
        m = _TWEAK.match(text.strip()) or _TWEAK_SHORT.match(text.strip())
        if not m:
            return False
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
        msg = "Done, it's faster now." if re.search(r"faster|quicker", value, re.I) else \
            "Done, it's slower now." if re.search(r"slower", value, re.I) else f"Done, the {what} is {value} now."
        self.context.add_message({"role": "user", "content": text})
        self.context.add_message({"role": "assistant", "content": msg})
        await self._announce(msg)
        return True

    def _show_app(self, output):
        try:
            d = json.loads(output) if isinstance(output, str) else output
            self._remember_app(d)
            if isinstance(d, dict) and d.get("url") and self.ui:
                self.ui.send({"type": "app", "url": d["url"], "title": d.get("title", "")})
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
                    if self.ui:
                        self.ui.send({"type": "app", "url": job["url"], "title": job.get("title", "")})
                    what = "changed" if job.get("kind") == "edit" else "ready"
                    msg = f"{job.get('title', 'Your app')} is {what}, it's on the screen."
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
