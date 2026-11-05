"""The virtual room FDA controls (shown on the 7" display), plus timers, alarms, reminders, music and the phone.

Every change goes through Room.execute(action) - from the voice agent, the fast path, or a tap on the screen - and is
broadcast to the display through on_change.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import itertools
import logging
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

from . import sounds
from .audio import read_wav

log = logging.getLogger("fda.room")
_ids = itertools.count(1)


GAME_DEFAULTS = {"plane_color": "white", "tower_color": "pink", "sky": "sunset", "speed": 1.0, "gap": 1.0,
                 "gravity": 1.0, "size": 1.0, "paused": False, "restart": 0, "title": "Sky Hop"}
_LEVELS = {"slow": 0.7, "low": 0.7, "narrow": 0.8, "small": 0.7, "normal": 1.0, "fast": 1.4, "high": 1.4,
           "wide": 1.3, "big": 1.5}
_RELATIVE = {"slower": 1 / 1.25, "lower": 1 / 1.25, "narrower": 1 / 1.15, "smaller": 1 / 1.25,
             "faster": 1.25, "higher": 1.25, "wider": 1.15, "bigger": 1.25}


def _step(current: float, word: str) -> float:
    """'faster' / 'wider' change the current value a step; 'fast' / 'normal' set it."""
    v = current * _RELATIVE[word] if word in _RELATIVE else _LEVELS.get(word, current)
    return round(min(2.5, max(0.4, v)), 2)


def _level_name(v: float) -> str:
    return "normal" if abs(v - 1) < 0.08 else (f"{v:.1f}x")


class Room:
    def __init__(self, cfg, mixer, on_change: Callable[[dict], None], on_ring: Callable[[str], Awaitable[None]]):
        self.cfg, self.mixer = cfg, mixer
        self.on_change, self.on_ring = on_change, on_ring
        self.state = {
            "lights": {"power": "off", "brightness": 80, "color": "warm white"},
            "thermostat_c": 24.0,
            "music": {"state": "stopped", "track": "", "index": 0},
            "volume": int(mixer.volume * 100),
            "timers": [], "alarms": [], "reminders": [],
            "ringing": None,
            "phone": {"state": "idle", "contact": ""},
            "messages": [],
            "game": dict(GAME_DEFAULTS, open=False),
        }
        self.tracks = self._load_tracks()
        self._ring_task: asyncio.Task | None = None
        self._call_task: asyncio.Task | None = None

    # ---------- state for the planner (short) and the display (full) ----------
    def brief(self) -> dict:
        s = self.state
        b = {"lights": s["lights"], "thermostat_c": s["thermostat_c"], "music": s["music"]["state"],
             "volume": s["volume"]}
        if s["timers"]:
            b["timers"] = [f'{t["label"] or "timer"} {max(0, int(t["ends"] - time.time()))}s left' for t in s["timers"]]
        if s["alarms"]:
            b["alarms"] = [a["time"] for a in s["alarms"]]
        if s["reminders"]:
            b["reminders"] = [f'{r["text"]}' + (f' at {r["time"]}' if r["time"] else "") for r in s["reminders"]]
        if s["ringing"]:
            b["ringing"] = s["ringing"]
        g = s["game"]
        if g["open"]:
            # only what differs from the defaults: short (faster on the Pi), and nothing for the model to copy back
            # ("normal" values echoed into a change would undo it)
            b["game"] = {"open": True, **{k: (_level_name(v) if isinstance(v, float) else v) for k, v in g.items()
                                          if k in GAME_DEFAULTS and k != "restart" and v != GAME_DEFAULTS[k]}}
        return b

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def restore(self, snap: dict):
        """Undo (used when the instant VCM guess turns out wrong)."""
        music_before = self.state["music"]["state"]
        self.state = snap
        self.mixer.volume = self.state["volume"] / 100
        if self.state["music"]["state"] != music_before:
            self._apply_music()
        self._changed()

    def _changed(self):
        self.on_change(self.state)

    # ---------- actions ----------
    async def execute(self, a: dict) -> dict:
        """Run one action. Returns {"ok": bool, "note": short text for history, "speak": optional text}."""
        tool = a.get("tool")
        try:
            fn = getattr(self, f"_do_{tool}")
        except AttributeError:
            return {"ok": False, "note": f"unknown tool {tool}"}
        res = fn(a)
        if asyncio.iscoroutine(res):
            res = await res
        self._changed()
        log.info("action %s -> %s", a, res.get("note"))
        return res

    def _do_lights(self, a):
        L = self.state["lights"]
        if "power" in a:
            L["power"] = a["power"]
        if a.get("brightness") is not None:
            L["brightness"] = int(a["brightness"]); L["power"] = "on" if a["brightness"] > 0 else "off"
        if a.get("color"):
            L["color"] = str(a["color"]).lower(); L["power"] = "on"
        return {"ok": True, "note": f'lights {L["power"]} {L["brightness"]}% {L["color"]}'}

    def _do_thermostat(self, a):
        c = float(a["celsius"])
        if not 10 <= c <= 35:
            return {"ok": False, "note": f"{c} C is out of range (10-35)", "speak": "I can only set between 10 and 35 degrees."}
        self.state["thermostat_c"] = round(c, 1)
        return {"ok": True, "note": f"thermostat {c} C"}

    def _do_timer(self, a):
        secs = int(a["seconds"])
        t = {"id": next(_ids), "label": a.get("label", ""), "seconds": secs, "ends": time.time() + secs}
        self.state["timers"].append(t)
        return {"ok": True, "note": f"timer {secs}s"}

    def _do_cancel_timer(self, a):
        n = len(self.state["timers"]); self.state["timers"].clear()
        return {"ok": True, "note": f"cancelled {n} timers"}

    def _do_alarm(self, a):
        hhmm = _parse_hhmm(a.get("time", ""))
        if not hhmm:
            return {"ok": False, "note": f"bad time {a.get('time')}", "speak": "Sorry, what time should I set it for?"}
        self.state["alarms"].append({"id": next(_ids), "time": hhmm, "fired_on": ""})
        return {"ok": True, "note": f"alarm {hhmm}"}

    def _do_cancel_alarm(self, a):
        al = self.state["alarms"]
        if not al:
            return {"ok": False, "note": "no alarms", "speak": "You don't have any alarms set."}
        want = (a.get("time") or "").strip().lower()
        if want in ("all", "every", "everything", "all alarms") or (not want and len(al) == 1):
            n = len(al); al.clear()
            return {"ok": True, "note": f"cancelled {n} alarm{'s' if n > 1 else ''}"}
        hhmm = _parse_hhmm(want) if want else None
        hit = [x for x in al if hhmm and x["time"] == hhmm]
        if not hit and hhmm and ":" not in want:              # "6" -> 06:00 or 18:00, whichever is set
            h = int(hhmm[:2]) % 12
            hit = [x for x in al if int(x["time"][:2]) % 12 == h and x["time"][3:] == hhmm[3:]]
        if not hit:
            times = ", ".join(_say_time(x["time"]) for x in al)
            return {"ok": False, "note": f"which alarm? set: {times}"}
        for x in hit:
            al.remove(x)
        return {"ok": True, "note": f"cancelled the {_say_time(hit[0]['time'])} alarm, {len(al)} left"}

    def _do_add_reminder(self, a):
        hhmm = _parse_hhmm(a.get("time", "") or "")
        self.state["reminders"].append({"id": next(_ids), "text": a["text"], "time": hhmm or "", "fired_on": ""})
        return {"ok": True, "note": f"reminder {a['text']} {hhmm or ''}".strip()}

    def _find_reminders(self, which: str) -> list[dict]:
        """Reminders meant by `which`: "all", a number ("2", "number two", "the first"), "last", or words of it."""
        r = self.state["reminders"]
        w = re.sub(r"[^a-z0-9: ]", " ", (which or "").lower()).split()
        if not r:
            return []
        if not w or set(w) & {"all", "everything", "every"}:
            return list(r) if w else []
        if set(w) & {"last", "latest", "newest", "recent"}:
            return [r[-1]]
        nums = {"one": 1, "first": 1, "two": 2, "second": 2, "three": 3, "third": 3, "four": 4, "fourth": 4,
                "five": 5, "fifth": 5, "six": 6, "sixth": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        if len(w) <= 3:                                   # "2", "number two", "the first one"
            for x in w:
                n = int(x) if x.isdigit() else nums.get(x)
                if n and (x.isdigit() or x in nums) and 1 <= n <= len(r) and not any(
                        y not in ("number", "the", "one", "reminder", "no", x) for y in w):
                    return [r[n - 1]]
        skip = {"the", "a", "an", "to", "my", "reminder", "reminders", "about", "at", "for", "of", "one", "that"}
        words = {x for x in w if x not in skip}
        when = _parse_hhmm(which or "")

        def score(x):
            s = len(words & set(re.sub(r"[^a-z0-9 ]", " ", x["text"].lower()).split()))
            return s + (1 if when and x["time"] == when else 0)
        best = max((score(x) for x in r), default=0)
        return [x for x in r if score(x) == best] if best else []

    def _do_delete_reminder(self, a):
        r = self.state["reminders"]
        if not r:
            return {"ok": False, "note": "no reminders", "speak": "You don't have any reminders."}
        hit = self._find_reminders(a.get("which", ""))
        if not hit:
            names = ", ".join(f"{i + 1}: {x['text']}" for i, x in enumerate(r))
            return {"ok": False, "note": f"no reminder matches {a.get('which')!r}; there are: {names}"}
        for x in hit:
            r.remove(x)
        if len(hit) > 1:
            return {"ok": True, "note": f"deleted {len(hit)} reminders, {len(r)} left"}
        return {"ok": True, "note": f"deleted reminder {hit[0]['text']}, {len(r)} left"}

    def _do_edit_reminder(self, a):
        r = self.state["reminders"]
        if not r:
            return {"ok": False, "note": "no reminders", "speak": "You don't have any reminders."}
        hit = self._find_reminders(a.get("which", ""))
        if len(hit) != 1:
            names = ", ".join(f"{i + 1}: {x['text']}" for i, x in enumerate(r))
            return {"ok": False, "note": f"which reminder? there are: {names}"}
        x = hit[0]
        if a.get("text"):
            x["text"] = a["text"]
        if a.get("time"):
            hhmm = _parse_hhmm(a["time"])
            if not hhmm:
                return {"ok": False, "note": f"bad time {a['time']}", "speak": "Sorry, what time should it be?"}
            x["time"], x["fired_on"] = hhmm, ""
        return {"ok": True, "note": f"reminder now: {x['text']} {x['time']}".strip()}

    def _do_list_reminders(self, a):
        r = self.state["reminders"]
        if not r:
            return {"ok": True, "note": "no reminders", "speak": "You don't have any reminders."}
        items = [x["text"] + (f" at {_say_time(x['time'])}" if x["time"] else "") for x in r]
        say = ("You have one reminder: " if len(items) == 1 else f"You have {len(items)} reminders: ") + \
              ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] + "."
        return {"ok": True, "note": "listed reminders", "speak": say}

    def _do_game(self, a):
        """Sky Hop on the display, changed live: every setting is room state the display pushes into the game."""
        g = self.state["game"]; act = a.get("action", "set")
        if act == "open":
            g["open"] = True; g["paused"] = False
        elif act == "close":
            g["open"] = False
            return {"ok": True, "note": "game closed"}
        elif not g["open"]:
            return {"ok": False, "note": "no game is open"}    # changes only apply to an open game
        if act == "restart":
            g["restart"] += 1; g["paused"] = False
        elif act in ("pause", "resume"):
            g["paused"] = act == "pause"
        for k in ("plane_color", "tower_color", "title"):
            if a.get(k):
                g[k] = str(a[k]).strip()[:40]
        if a.get("sky") in ("sunset", "day", "night", "space"):
            g["sky"] = a["sky"]
        for k in ("speed", "gap", "gravity", "size"):
            if a.get(k):
                g[k] = _step(g[k], a[k])
        changed = ", ".join(f"{k} {a[k]}" for k in a if k not in ("tool", "action"))
        return {"ok": True, "note": f"game {act}" + (f": {changed}" if changed else "")}

    def _do_stop_ringing(self, a):
        self.stop_ringing()
        return {"ok": True, "note": "stopped ringing"}

    def _do_music(self, a):
        m = self.state["music"]; act = a["action"]
        if act == "resume" and m["state"] == "stopped":
            return {"ok": False, "note": "no music is playing"}      # resume only continues paused music
        if act in ("play", "resume"):
            if act == "play":
                m["track"] = self.tracks[m["index"] % len(self.tracks)][0]
            m["state"] = "playing"
        elif act == "pause":
            m["state"] = "paused" if m["state"] == "playing" else m["state"]
        elif act == "stop":
            m["state"] = "stopped"
        elif act == "next":
            if m["state"] != "playing":
                return {"ok": False, "note": "no music is playing"}  # "next" never starts the music
            m["index"] = (m["index"] + 1) % len(self.tracks)
            m["track"] = self.tracks[m["index"]][0]
        self._apply_music(restart=act in ("play", "next"))
        return {"ok": True, "note": f"music {m['state']} {m['track']}"}

    def _do_volume(self, a):
        v = self.state["volume"]
        if a.get("level") is not None:
            v = int(a["level"])
        elif a.get("change") == "up":
            v += 10
        elif a.get("change") == "down":
            v -= 10
        v = max(0, min(100, v))
        self.state["volume"] = v; self.mixer.volume = v / 100
        return {"ok": True, "note": f"volume {v}"}

    def _do_call(self, a):
        contact = a["contact"].strip().title()
        self.state["phone"] = {"state": "calling", "contact": contact}
        if self._call_task:
            self._call_task.cancel()
        self._call_task = asyncio.create_task(self._simulate_call(contact))
        return {"ok": True, "note": f"calling {contact} (simulated)"}

    async def _simulate_call(self, contact):
        try:
            self.mixer.fx.set_loop(sounds.ringback(self.mixer.rate))
            await asyncio.sleep(9)
        finally:
            self.mixer.fx.set_loop(None)
        self.state["phone"] = {"state": "no answer", "contact": contact}; self._changed()
        await self.on_ring(f"{contact} didn't pick up.")
        await asyncio.sleep(4)
        self.state["phone"] = {"state": "idle", "contact": ""}; self._changed()

    def _do_message(self, a):
        self.state["messages"].append({"to": a["contact"].strip().title(), "text": a["text"],
                                       "at": time.strftime("%H:%M")})
        self.state["messages"] = self.state["messages"][-6:]
        return {"ok": True, "note": f"message to {a['contact']}: {a['text']}"}

    # weather / web_search are handled by the agent (they need the network and speak later)

    # ---------- music ----------
    def _load_tracks(self) -> list[tuple[str, object]]:
        d = Path(self.cfg.music_dir)
        files = sorted(d.glob("*.wav")) if d.exists() else []
        tracks: list[tuple[str, object]] = [(f.stem.replace("_", " "), f) for f in files]
        return tracks or [("FDA lo-fi loop", None), ("FDA lo-fi loop 2", 1)]

    def _apply_music(self, restart: bool = False):
        m = self.state["music"]
        if m["state"] == "playing":
            if restart or self.mixer.music.loop is None:
                name, src = self.tracks[m["index"] % len(self.tracks)]
                if isinstance(src, Path):
                    x = read_wav(src, self.mixer.rate)
                else:
                    x = sounds.music_loop(self.mixer.rate, seed=int(src or 0))
                self.mixer.music.set_loop(x)
            self.mixer.music.ramp(0.6, 300)
        elif m["state"] == "paused":
            self.mixer.music.ramp(0.0, 200)
        else:
            self.mixer.music.ramp(0.0, 200)
            asyncio.get_running_loop().call_later(0.3, lambda: self.state["music"]["state"] == "stopped"
                                                  and self.mixer.music.set_loop(None))

    def music_duck(self, duck: bool):
        if self.state["music"]["state"] == "playing":
            self.mixer.music.ramp(0.12 if duck else 0.6, 150 if duck else 600)

    # ---------- ringing ----------
    def ring(self, what: str):
        self.state["ringing"] = what
        self.mixer.fx.set_loop(sounds.alarm(self.mixer.rate))
        self._changed()
        if self._ring_task:
            self._ring_task.cancel()
        self._ring_task = asyncio.create_task(self._auto_stop())

    async def _auto_stop(self):
        await asyncio.sleep(60)
        self.stop_ringing()

    def stop_ringing(self):
        self.state["ringing"] = None
        self.mixer.fx.set_loop(None)
        self._changed()

    async def tick_forever(self):
        """Timers, alarms and reminders (checked twice a second)."""
        while True:
            await asyncio.sleep(0.5)
            now = time.time(); hhmm = time.strftime("%H:%M"); today = time.strftime("%Y-%m-%d")
            for t in [t for t in self.state["timers"] if t["ends"] <= now]:
                self.state["timers"].remove(t)
                self.ring(f'{t["label"] or _say_duration(t["seconds"])} timer')
                await self.on_ring(f'Your {t["label"] or _say_duration(t["seconds"])} timer is done.')
            for a in self.state["alarms"]:
                if a["time"] == hhmm and a["fired_on"] != today:
                    a["fired_on"] = today
                    self.ring(f'{a["time"]} alarm')
                    await self.on_ring(f"It's {_say_time(a['time'])}. Time to wake up!")
            for r in list(self.state["reminders"]):
                if r["time"] == hhmm and r["fired_on"] != today:
                    r["fired_on"] = today
                    self.mixer.fx.push(sounds.chime(self.mixer.rate))
                    await self.on_ring(f"Reminder: {r['text']}.")
                    self.state["reminders"].remove(r)
                    self._changed()
            if self.state["timers"]:
                self._changed()      # countdowns on the display


def _parse_hhmm(s: str) -> str | None:
    s = (s or "").strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}" if 0 <= h < 24 and 0 <= mi < 60 else None


def _say_time(hhmm: str) -> str:
    return dt.datetime.strptime(hhmm, "%H:%M").strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")


def _say_duration(s: int) -> str:
    if s % 3600 == 0:
        return f"{s // 3600} hour"
    if s % 60 == 0:
        return f"{s // 60} minute"
    return f"{s} second"
