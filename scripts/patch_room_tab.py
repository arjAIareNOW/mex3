"""One-off: the room (lights, thermostat, timers, music...) comes back as a separate Room tab that appears only when
asked for or used; design stays the main screen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "fda/pc/agent.py"
s = p.read_text(encoding="utf-8")

old = '''    "game": "Open, close or change Sky Hop, the built-in tap-to-fly game (colours, sky, speed, gap, size).",
}'''
assert old in s, "game"
s = s.replace(old, "}")   # Elkit builds its own games

old = '''def room_tools(room: Room) -> list[FunctionSchema]:
    out = []
    for name, desc in ROOM_TOOLS.items():
        props = brain.TOOLS[name]

        async def handler(params, _name=name):
            r = await room.execute({"tool": _name, **(params.arguments or {})})'''
assert old in s, "room_tools"
s = s.replace(old, '''def room_tools(room: Room, on_use=None) -> list[FunctionSchema]:
    out = []
    for name, desc in ROOM_TOOLS.items():
        props = brain.TOOLS[name]

        async def handler(params, _name=name):
            if on_use:
                on_use()
            r = await room.execute({"tool": _name, **(params.arguments or {})})''')

old = '''        tools = ToolsSchema(standard_tools=[*remote.standard_tools])        # no lights/timers/music: a designer'''
assert old in s, "tools"
s = s.replace(old, '''        show = FunctionSchema(
            name="show_screen", description="Switch the screen: 'room' shows the lights, thermostat, timers, "
            "reminders and music; 'design' shows what you are building.",
            properties={"screen": {"type": "string", "enum": ["design", "room"]}}, required=["screen"],
            handler=self._show_screen)
        tools = ToolsSchema(standard_tools=[*remote.standard_tools,
                                            *room_tools(self.room, on_use=lambda: self._tab("room")), show])''')

old = '''    def _status(self, status: str):'''
s = s.replace(old, '''    def _tab(self, tab: str):
        if self.ui:
            self.ui.send({"type": "tab", "tab": tab})

    async def _show_screen(self, params):
        tab = (params.arguments or {}).get("screen", "design")
        self._tab("room" if tab == "room" else "design")
        await params.result_callback({"ok": True, "result": f"the {tab} screen is showing"})

''' + old, 1)

# timers / alarms tick again
old = '''        self.guard = DuplexGuard(self.io, cfg)
'''
assert "tick_forever" not in s
s = s.replace("        self.room = Room(cfg, self.io.mixer, on_change=self._room_changed, on_ring=self._announce)\n",
              "        self.room = Room(cfg, self.io.mixer, on_change=self._room_changed, on_ring=self._announce)\n", 1)
p.write_text(s, encoding="utf-8")

# the persona: knows about the room
s = p.read_text(encoding="utf-8")
old = '''- Never say something is done unless the tool said so.'''
assert old in s, "prompt"
s = s.replace(old, '''- You can also run the room when asked: lights, thermostat, timers, alarms, reminders, music and volume. They show
  on a separate Room tab; "show me the room" or "the lights panel" is show_screen room, "back to the design" is
  show_screen design. Do this only when asked, then go back to designing.
''' + old, 1)
p.write_text(s, encoding="utf-8")

# ui cache: a new browser opens on the tab in use
p = ROOT / "fda/ui.py"
s = p.read_text(encoding="utf-8")
old = '''        if t in ("room", "status", "health", "sources", "app", "build"):'''
assert old in s, "ui"
s = s.replace(old, '''        if t in ("room", "status", "health", "sources", "app", "build", "tab"):''')
p.write_text(s, encoding="utf-8")
print("ok")
