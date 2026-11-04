"""The talker: listens all the time, takes turns like a person, and hands work to the task agents.

Per 32 ms of echo-cancelled audio (Silero VAD):

  idle ──speech──► in turn ──200 ms silence──► SPECULATIVE: transcribe now (Qwen2-Audio) ──500 ms silence──► COMMIT
                    │                              user keeps talking → speculative request cancelled
                    └─ if FDA is talking: duck its voice at once; after 300 ms of real speech fade out and stop

After COMMIT:  echo of FDA's own words? → drop (and resume if we stopped)     backchannel ("yeah", "mm-hmm")? → keep talking
               fast path rules (lights off, timers, stop…) → act now, no LLM
               otherwise planner (Qwen2-Audio, text): for_me? → do (device agent, search agent) → say (streamed TTS)
Instant path: your VCM intent Conformer runs on the audio itself (~20 ms); if it is very sure about a simple command
(lights on/off, pause, next, volume), the action happens before the transcript, and is undone if the transcript disagrees.
"""
from __future__ import annotations

import asyncio
import collections
import difflib
import logging
import re
import time

import numpy as np

from . import brain, rules, selfvoice
from .audio import AudioEngine, level_db, wav_bytes
from .room import Room
from .search import SearchAgent
from .servers import make_servers
from .speaker import Speaker
from .vad import SileroVAD

log = logging.getLogger("fda.agent")
WIN_S = 0.032


class Agent:
    def __init__(self, cfg, ui=None, audio_factory=None):
        self.cfg = cfg
        self.ui = ui
        self.servers = make_servers(cfg)
        self.audio = (audio_factory or AudioEngine)(cfg, self._on_audio_thread)
        self.mixer = self.audio.mixer
        self.speaker = Speaker(cfg, self.servers, self.mixer, on_text=self._on_fda_text)
        self.room = Room(cfg, self.mixer, on_change=lambda s: self._ui({"type": "room", "state": _public(s)}),
                         on_ring=self._announce)
        self.search = SearchAgent(cfg, self.servers)
        self.vad = SileroVAD()
        self.own = selfvoice.OwnVoiceDetector(cfg.selfvoice_margin_db) if cfg.selfvoice else None
        self.own_windows = 0                  # windows FDA recognised as its own voice (tests / display)
        self.vcm = None
        if cfg.vcm_fast_path:
            try:
                from .vcm import VCMIntent
                self.vcm = VCMIntent()
            except Exception as e:
                log.warning("VCM intent model unavailable: %s", e)
        self.history: list[dict] = []
        self._vcm_pending: dict | None = None   # room snapshot to undo an unconfirmed VCM action
        self.events: list[dict] = []          # timeline for tests / debugging
        # turn state
        self.in_turn = False
        self.preroll = collections.deque(maxlen=max(1, int(cfg.preroll_s / WIN_S)))
        self.turn_audio: list[np.ndarray] = []
        self.recent = collections.deque(maxlen=3)
        self.residual = collections.deque(maxlen=90)   # echo-cancelled level while FDA talks (~3 s)
        self.noise = collections.deque(maxlen=150)     # room noise level while idle (~5 s)
        self.onsets = collections.deque(maxlen=4)
        self.speech_s = self.silence_s = 0.0
        self.spec_task: asyncio.Task | None = None
        self.commit: asyncio.Event | None = None
        self.turn_ctx: dict = {}
        self.turn_seq = 0
        self.reply_msg: dict | None = None     # history entry of the reply being spoken
        self.reply_task: asyncio.Task | None = None
        self.last_speech_t = 0.0
        self.level = -120.0
        self.status = "starting"
        self.loop: asyncio.AbstractEventLoop | None = None
        self.inq: asyncio.Queue | None = None

    # ------------------------------------------------------------------ plumbing
    def _ui(self, msg: dict):
        if self.ui is not None:
            self.ui.send(msg)

    def _event(self, kind: str, **kw):
        e = {"t": time.time(), "kind": kind, **kw}
        self.events.append(e)
        log.info("%s %s", kind, {k: v for k, v in kw.items() if k != "audio"})

    def _set_status(self, s: str):
        if s != self.status:
            self.status = s
            self._ui({"type": "status", "status": s})

    def _on_fda_text(self, text: str, final: bool):
        self._ui({"type": "fda", "text": text, "final": final})

    def _on_audio_thread(self, clean: np.ndarray, ref: np.ndarray):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.inq.put_nowait, (clean, ref))

    async def run(self, stop: asyncio.Event | None = None):
        self.loop = asyncio.get_running_loop()
        self.inq = asyncio.Queue()
        self.audio.start()
        tasks = [asyncio.create_task(self.room.tick_forever()), asyncio.create_task(self._status_loop())]
        health = await self.servers.check()
        self._ui({"type": "health", **health})
        if not all(health.values()):
            log.error("server check failed: %s", health)
        await self.servers.warm_up()
        self._ui({"type": "room", "state": _public(self.room.state)})
        if self.cfg.say_ready:
            await self.speaker.say(f"{self.cfg.name} is ready.")
        self._set_status("listening")
        try:
            while stop is None or not stop.is_set():
                try:
                    block, ref = await asyncio.wait_for(self.inq.get(), 0.5)
                except asyncio.TimeoutError:
                    continue
                if self.own is not None:   # learn the leak only while FDA plays and nobody is talking
                    self.own.push(block, ref, learn=self.speaker.active and not self.in_turn)
                for w, p in self.vad.push(block):
                    self._on_window(w, p)
        finally:
            for t in tasks:
                t.cancel()
            self.audio.stop()
            await self.servers.close()

    async def _status_loop(self):
        while True:
            await asyncio.sleep(0.1)
            if not self.in_turn and self.status not in ("thinking", "starting"):
                self._set_status("speaking" if self.speaker.active else "listening")
            self._ui({"type": "level", "db": round(self.level, 1)})
            if self.speaker.active or self.in_turn:
                self.room.music_duck(True)
            else:
                self.room.music_duck(False)

    # ------------------------------------------------------------------ listening (every 32 ms)
    def _on_window(self, w: np.ndarray, p: float):
        cfg = self.cfg
        db = level_db(w)
        self.level = db
        speaking = self.speaker.active
        # Energy detector for double talk: AEC leaves only a low residual while FDA talks, so the user's voice
        # stands out by level even when the VAD (slow to ramp on short words like "stop") is unsure.
        floor = float(np.median(self.residual)) if len(self.residual) >= 10 else -70.0
        loud = len(self.residual) >= 30 and db >= max(cfg.bargein_min_db, floor + cfg.bargein_margin_db)
        # FDA's own playback explains this sound (predicted leak through the AEC): never the user
        own = self.own is not None and self.own.is_own() and self.own.ref_active()
        if own:
            self.own_windows += 1
            loud = False
        if speaking and not self.in_turn:
            self.residual.append(db)
        if speaking:
            self.recent.append(not own and ((p >= cfg.bargein_vad and db >= cfg.bargein_min_db) or loud))
            start = sum(self.recent) >= 2
        else:
            # Silero sometimes lags ~1 s on a speech onset; a clear level jump over the room noise also starts a turn
            # (false starts from a door slam just give an empty transcript and are dropped)
            nfloor = float(np.median(self.noise)) if len(self.noise) >= 20 else -60.0
            onset = not own and db >= max(cfg.onset_min_db, nfloor + cfg.onset_margin_db)
            self.onsets.append(onset)
            if not self.in_turn and not onset:
                self.noise.append(db)
            self.recent.append(p >= cfg.vad_start and not own)
            start = (sum(list(self.recent)[-cfg.start_frames:]) >= cfg.start_frames) or sum(self.onsets) >= 3
            loud = loud or (self.in_turn and onset)
        if not self.in_turn:
            self.preroll.append(w)
            if start:
                self._begin_turn(speaking)
            return

        self.turn_audio.append(w)
        if self.turn_ctx["while_speaking"]:
            is_speech = (p >= cfg.vad_end and db >= cfg.bargein_min_db - 6) or loud
        else:
            is_speech = p >= cfg.vad_end or loud
        is_speech = is_speech and not own      # FDA's own voice neither extends the turn nor stops FDA
        if is_speech:
            self.speech_s += WIN_S; self.silence_s = 0.0; self.last_speech_t = time.time()
            if self.spec_task is not None and not self.commit.is_set():
                self.spec_task.cancel(); self.spec_task = None       # they kept talking
                self.turn_ctx.pop("early_ok", None)
                self.turn_ctx.pop("incomplete", None)
                self._event("speculative_cancelled")
        else:
            self.silence_s += WIN_S
        if self.turn_ctx.get("bargein_pending") and self.speech_s >= cfg.bargein_confirm_s and self.speaker.active:
            self._confirm_bargein()
        total = len(self.turn_audio) * WIN_S
        if (self.spec_task is None and self.silence_s >= cfg.speculative_silence_s
                and self.speech_s >= cfg.min_speech_s):
            self._launch_turn()
        early = self.turn_ctx.get("early_ok") and self.silence_s >= cfg.early_endpoint_s
        # "search the net for..." (pause) "...a recipe": an unfinished sentence waits longer for the rest
        need = cfg.incomplete_endpoint_s if self.turn_ctx.get("incomplete") else cfg.endpoint_silence_s
        if early or self.silence_s >= need or total >= cfg.max_turn_s:
            self._end_turn()

    def _begin_turn(self, speaking: bool):
        self.in_turn = True
        self.turn_audio = list(self.preroll)
        self.preroll.clear()
        self.speech_s = self.cfg.start_frames * WIN_S
        self.silence_s = 0.0
        self.spec_task = None
        self.turn_ctx = {"while_speaking": speaking, "bargein_pending": speaking, "stopped": None,
                         "t_start": time.time() - len(self.turn_audio) * WIN_S}
        if speaking:
            self.speaker.duck()
            self._event("bargein_duck")
        self._set_status("hearing")

    def _confirm_bargein(self):
        ctx = self.turn_ctx
        ctx["bargein_pending"] = False
        self._event("bargein_stop", after_s=round(self.speech_s, 3))
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        ctx["stopped"] = asyncio.create_task(self._stop_talking())

    async def _stop_talking(self) -> tuple[str, str]:
        t = time.time()
        heard, rest = await self.speaker.interrupt()
        self._event("fda_stopped", heard_until=heard[-60:], t_fade_start=t)
        if self.reply_msg is not None:
            self.reply_msg["content"] = (heard + " [interrupted]").strip()
            self.reply_msg = None
        return heard, rest

    def _launch_turn(self):
        self.commit = asyncio.Event()
        audio = np.concatenate(self.turn_audio)
        self.spec_task = asyncio.create_task(self._turn(audio, self.commit, self.turn_ctx))
        self._event("speculative_start", speech_s=round(self.speech_s, 2))

    def _end_turn(self):
        self.in_turn = False
        self.recent.clear(); self.onsets.clear()
        ctx = self.turn_ctx
        if self.speech_s < self.cfg.min_speech_s:
            if self.spec_task:
                self.spec_task.cancel()
            if ctx.get("bargein_pending"):
                self.speaker.unduck()
            self._event("blip_ignored", speech_s=round(self.speech_s, 2))
            self._set_status("speaking" if self.speaker.active else "listening")
            return
        if self.spec_task is None:
            self._launch_turn()
        ctx["t_commit"] = time.time()
        ctx["t_speech_end"] = self.last_speech_t
        self.turn_seq += 1
        ctx["seq"] = self.turn_seq
        self.commit.set()
        self._event("endpoint", speech_s=round(self.speech_s, 2))
        self._set_status("thinking")

    # ------------------------------------------------------------------ one user turn
    async def _turn(self, audio: np.ndarray, commit: asyncio.Event, ctx: dict):
        try:
            await self._turn_inner(audio, commit, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("turn failed")
            self._ui({"type": "error", "message": str(e)})
            await self._resume(ctx)
        finally:
            if self.status == "thinking" and not self.in_turn:
                self._set_status("speaking" if self.speaker.active else "listening")

    async def _turn_inner(self, audio: np.ndarray, commit: asyncio.Event, ctx: dict):
        """Everything that only *thinks* runs before the endpoint (speculatively): VCM guess, transcript, echo check,
        rules, and the planner. Everything that *acts* or *speaks* waits for the commit."""
        cfg = self.cfg
        plan_task: asyncio.Task | None = None
        try:
            # 1) instant guess from your VCM model (audio only, ~20 ms)
            vcm_guess, snap = None, None
            if self.vcm is not None and not ctx["while_speaking"] and len(audio) <= 2.5 * 16000:
                intent, conf = await asyncio.to_thread(self.vcm.predict, audio)
                music_on = self.room.state["music"]["state"] == "playing"
                if (intent in rules.VCM_ACTIONS and conf >= cfg.vcm_min_conf
                        and (intent not in ("PAUSE", "NEXT") or music_on)):
                    vcm_guess = (intent, conf)
            # 2) transcript
            heard = await self.servers.hear(brain.hear_messages(self.servers.audio_part(wav_bytes(audio))),
                                            brain.HEAR_SCHEMA)
            if heard and self.cfg.strip_own_words:
                # take out the words FDA itself was playing while this audio was recorded
                played = self.speaker.words_between(ctx["t_start"], ctx.get("t_speech_end") or time.time())
                kept, removed = selfvoice.strip_own_words(heard, played)
                if removed:
                    self._event("own_words_removed", heard=heard, kept=kept, removed=removed)
                    heard = kept
                    ctx["own_words_only"] = not kept
            echo = not heard or self._is_echo(heard, ctx)
            backchannel = (not echo and ctx["while_speaking"] and rules.is_backchannel(heard))
            fast = rules.match(heard, self.room.state) if (cfg.fast_path and not echo) else None
            if not echo and fast is None and rules.sounds_unfinished(heard):
                ctx["incomplete"] = True        # "...search the net for?": wait longer for the rest
            elif fast is not None or echo or backchannel or heard.rstrip().endswith("?"):
                ctx["early_ok"] = True          # complete: the endpoint may come sooner
            # real words over FDA's voice (not echo, not "uh huh"): stop talking now, don't wait for the endpoint
            if (ctx["while_speaking"] and ctx.get("bargein_pending") and not echo and not backchannel
                    and (fast is not None or len(heard.split()) >= 2) and self.speaker.active):
                self._event("bargein_by_words", text=heard)
                self._confirm_bargein()
            # 3) planner, also speculative (its events wait in a queue until the commit)
            plan_q: asyncio.Queue = asyncio.Queue()
            if fast is None and not echo and not backchannel:
                msgs = brain.plan_messages(cfg, self.room.brief(), self.history, heard,
                                           static_prefix=self.servers.static_prefix)

                async def pump():
                    try:
                        async for ev in self.servers.plan(msgs, brain.plan_schema(cfg.planner_say_first)):
                            plan_q.put_nowait(ev)
                    except Exception as e:        # never leave _act waiting
                        log.warning("planner failed: %s", e)
                        plan_q.put_nowait(("for_me", True))
                        plan_q.put_nowait(("do", []))
                        plan_q.put_nowait(("done", {"say": "Sorry, I lost my train of thought. Could you say that again?"}))
                plan_task = asyncio.create_task(pump())
            await commit.wait()
            await self._act(ctx, heard, echo, backchannel, fast, vcm_guess, plan_q)
            if plan_task:
                await plan_task
        finally:
            if plan_task and not plan_task.done():
                plan_task.cancel()

    async def _act(self, ctx, heard, echo, backchannel, fast, vcm_guess, plan_q: asyncio.Queue):
        self._vcm_pending = None
        try:
            await self._act_inner(ctx, heard, echo, backchannel, fast, vcm_guess, plan_q)
        finally:
            # an instant VCM action that the turn never confirmed (cancelled, echo, error...) is undone
            if self._vcm_pending is not None:
                self.room.restore(self._vcm_pending)
                self._event("vcm_undone")
                self._vcm_pending = None

    async def _act_inner(self, ctx, heard, echo, backchannel, fast, vcm_guess, plan_q: asyncio.Queue):
        snap = None
        if vcm_guess and not echo:
            snap = self.room.snapshot()
            self._vcm_pending = snap
            for a in rules.VCM_ACTIONS[vcm_guess[0]]:
                await self.room.execute(a)
            self._event("vcm_instant", intent=vcm_guess[0], conf=round(vcm_guess[1], 3),
                        ms_after_end=_ms(ctx["t_speech_end"]))
        self._event("heard", text=heard, ms_after_end=_ms(ctx["t_speech_end"]))
        seq = ctx["seq"]

        # is it FDA's own voice coming back?
        if echo:
            self._event("echo_ignored" if heard else "empty", text=heard)
            await self._resume(ctx)
            return
        self._ui({"type": "user", "text": heard})
        # a backchannel while FDA talks ("yeah", "mm-hmm") is not an interruption
        if backchannel and ctx["stopped"] is None:
            self._event("backchannel", text=heard)
            ctx["bargein_pending"] = False
            self.speaker.unduck()
            return
        if backchannel:          # it already stopped (long "uh huh"): carry on where it was
            await self._resume(ctx)
            return

        # fast path
        if fast is not None:
            actions, say = fast
            await self._take_floor(ctx)
            if snap and actions != rules.VCM_ACTIONS.get(vcm_guess[0]):
                self.room.restore(snap); snap = None
            done_by_vcm = snap is not None
            self._vcm_pending = None             # settled: kept (same command) or undone above
            self._event("fast_path", actions=actions, say=say, ms_after_end=_ms(ctx["t_speech_end"]))
            msg = {"role": "assistant", "content": say, "do": actions}
            self.history += [{"role": "user", "content": heard}, msg]
            self.reply_msg = msg
            if say and seq == self.turn_seq:
                await self.speaker.say(say)
            spoken = [] if done_by_vcm else await self._run_actions(actions, heard)
            await self._speak_results(spoken, msg, seq)
            return

        # planner (already running since before the endpoint)
        started = finished = False
        bridge = False
        actions: list = []
        msg = {"role": "assistant", "content": "", "do": []}
        spoken_results: list = []
        while True:
            kind, v = await plan_q.get()
            if kind == "for_me":
                self._event("for_me", value=v, ms_after_end=_ms(ctx["t_speech_end"]))
                if not v:
                    self._ui({"type": "user", "text": heard, "ignored": True})
                    if snap:
                        self.room.restore(snap)
                        self._vcm_pending = None
                    await self._resume(ctx)
                    return
                await self._take_floor(ctx)
                self.history += [{"role": "user", "content": heard}, msg]
                self.reply_msg = msg
            elif kind == "do":
                actions = [a for a in v if a.get("tool")]
                dropped = [a for a in actions if not rules.plausible(a, heard)]
                if dropped:
                    self._event("action_dropped", actions=dropped, heard=heard)
                    actions = [a for a in actions if a not in dropped]
                msg["do"] = actions
                self._event("plan_do", actions=actions, ms_after_end=_ms(ctx["t_speech_end"]))
                if snap:
                    vcm_actions = rules.VCM_ACTIONS[vcm_guess[0]]
                    if all(any(_same(a, b) for a in actions) for b in vcm_actions):
                        actions = [a for a in actions if not any(_same(a, b) for b in vcm_actions)]
                    else:
                        self.room.restore(snap)
                    self._vcm_pending = None         # settled: confirmed by the planner or undone
                spoken_results = await self._run_actions(actions, heard)
                if (any(a["tool"] in ("web_search", "weather") for a in actions) and seq == self.turn_seq
                        and not started):
                    bridge = True     # our own short bridge; the model's "say" may guess the answer
                    line = "Let me check the weather." if actions[0]["tool"] == "weather" else "Let me check."
                    msg["content"] = line
                    self._on_fda_text(line, True)
                    await self.speaker.say(line)
                    self._event("say_start", ms_after_end=_ms(ctx["t_speech_end"]), bridge=True)
            elif kind == "say":
                v = _PLACEHOLDER.sub("", v)       # "[insert name]" style guesses are never spoken
                if seq != self.turn_seq or bridge:
                    continue          # a newer turn has the floor (still do the actions), or bridge said
                if not started:
                    started = True
                    await self.speaker.start()
                    self._event("say_start", ms_after_end=_ms(ctx["t_speech_end"]))
                self.speaker.feed(v)
                msg["content"] = v
            elif kind == "say_end":
                # the reply is complete: speak its last phrase now, don't wait for the "do" list after it
                if started and not finished and seq == self.turn_seq and not bridge:
                    finished = True
                    msg["content"] = _PLACEHOLDER.sub("", v)
                    self.speaker.finish(msg["content"])
            elif kind == "done":
                if not bridge and not finished:
                    msg["content"] = _PLACEHOLDER.sub("", v.get("say", msg["content"]))
                if started and not finished:
                    self.speaker.finish(msg["content"])
                self._event("plan_done", ms=round(v.get("_ms", 0)), say=v.get("say", ""))
                break
        await self._speak_results(spoken_results, msg, seq)

    async def _take_floor(self, ctx: dict):
        """The user's turn is real: stop whatever FDA was saying (if the barge-in didn't already)."""
        self.reply_task = asyncio.current_task()
        if ctx.get("stopped") is not None:
            await ctx["stopped"]
        elif self.speaker.active:
            ctx["bargein_pending"] = False
            await self._stop_talking()

    async def _resume(self, ctx: dict):
        """Not a real interruption: carry on talking (unduck, or re-say what was cut)."""
        if not ctx.get("while_speaking"):
            return
        if ctx.get("stopped") is None:
            ctx["bargein_pending"] = False
            self.speaker.unduck()
            self._event("resume_unduck")
        elif self.cfg.resume_after_false_bargein:
            heard, rest = await ctx["stopped"]
            if rest:
                self._event("resume_rest", text=rest)
                if self.reply_msg is None and self.history and self.history[-1]["role"] == "assistant":
                    self.history[-1]["content"] = (heard + " " + rest).strip()
                await self.speaker.say(rest)

    async def _run_actions(self, actions: list, heard: str) -> list:
        """Device agent (room), search agent (web, weather). Returns things to say afterwards, in order."""
        out = []
        for a in actions:
            tool = a.get("tool")
            if tool == "web_search":
                out.append(("search", a.get("query") or heard))
            elif tool == "weather":
                out.append(("weather", a.get("place")))
            else:
                r = await self.room.execute(a)
                if r.get("speak"):
                    out.append(("text", r["speak"]))
        return out

    async def _speak_results(self, results: list, msg: dict, seq: int):
        for kind, arg in results:
            if kind == "search":
                stream, sources, fallback = await self.search.web(arg)
                if sources:
                    self._ui({"type": "sources", "query": arg, "items": sources})
                text = await self._speak_after(stream, fallback, seq)
            elif kind == "weather":
                text = await self._speak_after(None, await self.search.weather(arg), seq)
            else:
                text = await self._speak_after(None, arg, seq)
            if text:
                msg["content"] = (msg["content"] + " " + text).strip()

    async def _speak_after(self, stream, text: str, seq: int) -> str:
        """Say a result once FDA has finished the bridge sentence and the user isn't talking."""
        await self.speaker.wait_done()
        for _ in range(100):
            if not self.in_turn:
                break
            await asyncio.sleep(0.1)
        if seq != self.turn_seq:
            return ""
        if stream is None:
            if text:
                await self.speaker.say(text)
                await self.speaker.wait_done()
            return text
        await self.speaker.start()
        out = ""
        async for piece in stream:
            out += piece
            self.speaker.feed(out)
        self.speaker.finish(out)
        await self.speaker.wait_done()
        return out.strip()

    async def _announce(self, text: str):
        """Timers, alarms, reminders, calls: say it when FDA isn't busy."""
        for _ in range(50):
            if not self.in_turn and not self.speaker.active:
                break
            await asyncio.sleep(0.2)
        self.history.append({"role": "assistant", "content": text, "do": []})
        self._ui({"type": "fda", "text": text, "final": True})
        await self.speaker.say(text)

    # ------------------------------------------------------------------ echo filter
    def _is_echo(self, heard: str, ctx: dict) -> bool:
        spoken = self.speaker.recent_spoken(since=ctx["t_start"] - 1.0)
        if not spoken:
            return False
        hw, sw = _words(heard), _words(spoken)
        if not hw:
            return True
        inside = sum(w in set(sw) for w in hw) / len(hw)
        seq = difflib.SequenceMatcher(None, " ".join(hw), " ".join(sw)).find_longest_match()
        longest = seq.size / max(1, len(" ".join(hw)))
        return inside >= 0.8 and (longest >= 0.6 or len(hw) <= 3)


_PLACEHOLDER = re.compile(r"[^.!?\[]*\[[^\]]*\]?[^.!?]*[.!?]?\s*")   # whole sentence with a [placeholder]


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def _same(a: dict, b: dict) -> bool:
    return all(a.get(k) == v for k, v in b.items())


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000) if t0 else -1


def _public(state: dict) -> dict:
    s = dict(state)
    s["timers"] = [{"label": t["label"], "left": max(0, int(t["ends"] - time.time())), "seconds": t["seconds"]}
                   for t in state["timers"]]
    return s
