"""One-off: turn the agent into Elkit, a voice web designer (only the building tools, a designer persona, longer
memory, status for the speaking pulse) and fix the own-voice gate learning the user's voice as echo."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# ---------------- agent ----------------
p = ROOT / "fda/pc/agent.py"
s = p.read_text(encoding="utf-8")
a, b = s.index('SYSTEM = """'), s.index("_TWEAK = re.compile(")
s = s[:a] + '''SYSTEM = """You are {name}, a friendly web designer. You build websites, apps and small games live on the screen
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

''' + s[b:]

old = s[s.index("AGENT_MCP_TOOLS = ["):s.index("ROOM_TOOLS = {")]
s = s.replace(old, '''# Elkit is a designer: only the tools for making things (the server has more: weather, news, documents, device...)
AGENT_MCP_TOOLS = ["create_app", "edit_app", "tweak_app", "undo_app_change", "app_history", "restore_app_version",
                   "list_apps", "open_app", "app_settings"]
SLOW_TOOLS: set[str] = set()      # a short filler is spoken at once for these (photos for tweak_app are handled too)

''')
old = '''        tools = ToolsSchema(standard_tools=[*room_tools(self.room), *remote.standard_tools])'''
assert old in s, "tools"
s = s.replace(old, '''        tools = ToolsSchema(standard_tools=[*remote.standard_tools])        # no lights/timers/music: a designer''')
old = '''        self.tasks.append(asyncio.create_task(self.room.tick_forever()))
'''
assert old in s, "tick"
s = s.replace(old, "")
old = '''        self.guard = DuplexGuard(self.io, cfg)'''
assert old in s, "guard"
s = s.replace(old, '''        self.guard = DuplexGuard(self.io, cfg)
        self.guard.on_bot = lambda talking: self._status("speaking" if talking else "listening")''')
old = '''            HeardTap(self.io, self.ui, name=cfg.name, on_heard=self._trim_context, on_ignored=self._resume_soon,'''
assert old in s, "heard"
s = s.replace(old, '''            HeardTap(self.io, self.ui, name=cfg.name, on_heard=self._heard, on_ignored=self._resume_soon,''')
old = '''    def _trim_context(self, keep_turns: int = 6):
        """Keep the few-shot plus the last `keep_turns` user turns: short prompts = fast local LLM.'''
assert old in s, "trim"
s = s.replace(old, '''    def _status(self, status: str):
        """The speaking pulse on the display: listening | thinking | speaking."""
        if self.ui:
            self.ui.send({"type": "status", "status": status})

    def _heard(self):
        self._trim_context()
        self._status("thinking")

    def _trim_context(self, keep_turns: int = 24):
        """Keep the few-shot plus the last `keep_turns` user turns (a whole design session; a cloud LLM reads this
        in a blink, a local one would want fewer).''')
p.write_text(s, encoding="utf-8")

# ---------------- duplex guard: tell the display when the bot starts / stops talking ----------------
p = ROOT / "fda/pc/processors.py"
s = p.read_text(encoding="utf-8")
old = '''        self.bot_text = lambda: ""       # what the bot is saying now (echo check)'''
assert old in s, "bot_text"
s = s.replace(old, old + '''
        self.on_bot = None               # callback(talking: bool) for the display''')
old = '''        elif isinstance(frame, BotStartedSpeakingFrame):
            io.bot_speaking = True'''
assert old in s, "start"
s = s.replace(old, '''        elif isinstance(frame, BotStartedSpeakingFrame):
            if not io.bot_speaking and self.on_bot:
                self.on_bot(True)
            io.bot_speaking = True''')
old = '''        elif isinstance(frame, BotStoppedSpeakingFrame):
            io.bot_speaking = False'''
assert old in s, "stop"
s = s.replace(old, '''        elif isinstance(frame, BotStoppedSpeakingFrame):
            if io.bot_speaking and self.on_bot:
                self.on_bot(False)
            io.bot_speaking = False''')
p.write_text(s, encoding="utf-8")

# ---------------- transport: the voice level, for the pulse ----------------
p = ROOT / "fda/pc/transport.py"
s = p.read_text(encoding="utf-8")
old = '''        voice = eng.mixer.voice
        voice.push(x)'''
assert old in s, "voice"
s = s.replace(old, '''        voice = eng.mixer.voice
        voice.push(x)
        cb = self.owner.on_level
        if cb is not None and len(x):
            now = time.monotonic()
            if now - self._t_level >= 0.05:                  # ~20 updates a second for the speaking pulse
                self._t_level = now
                cb(float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)))''')
old = '''        self.owner = owner
        self._rs: dict[int, soxr.ResampleStream] = {}'''
assert old in s, "init"
s = s.replace(old, '''        self.owner = owner
        self._rs: dict[int, soxr.ResampleStream] = {}
        self._t_level = 0.0''')
old = '''        self.bot_speaking = False
        self.user_speaking = False'''
assert old in s, "owner"
s = s.replace(old, '''        self.bot_speaking = False
        self.user_speaking = False
        self.on_level = None             # callback(dB) with the voice level while the bot speaks (the display pulse)''', 1)
if "\nimport time" not in s:
    s = s.replace("import asyncio\n", "import asyncio\nimport time\n", 1)
p.write_text(s, encoding="utf-8")

# ---------------- main: wire the level to the display ----------------
p = ROOT / "fda/pc/__main__.py"
s = p.read_text(encoding="utf-8")
old = '''    agent = Agent(cfg, transport, transport, urls, ui=ui, out_rate=out_rate)'''
assert old in s, "main"
s = s.replace(old, old + '''
    transport.on_level = lambda db: ui.send({"type": "level", "db": round(db, 1)})''')
p.write_text(s, encoding="utf-8")

# ---------------- own-voice gate: never learn the user's voice as echo ----------------
p = ROOT / "fda/selfvoice.py"
s = p.read_text(encoding="utf-8")
old = '''        obs = 10 * np.log10(mb) - 10 * np.log10(E)             # mic over reference, per band
        if learn:'''
assert old in s, "gate"
s = s.replace(old, '''        obs = 10 * np.log10(mb) - 10 * np.log10(E)             # mic over reference, per band
        # learn only sound that looks like echo: a person talking over FDA before the VAD has noticed them would
        # otherwise be learned as "leak", raise the bound, and from then on be silenced as FDA's own voice
        if learn and self.bound is not None and float(np.clip(obs - self.bound, 0, None).mean()) > self.margin + 6.0:
            learn = False
        if learn:''')
p.write_text(s, encoding="utf-8")

# ---------------- name ----------------
p = ROOT / "fda.toml"
s = p.read_text(encoding="utf-8")
if re.search(r"^name\s*=", s, re.M):
    s = re.sub(r"^name\s*=.*$", 'name = "Elkit"', s, flags=re.M)
else:
    s = 'name = "Elkit"            # what the agent is called (and answers to)\n' + s
p.write_text(s, encoding="utf-8")
print("ok")
