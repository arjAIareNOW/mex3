"""One-off: add the Room tab (lights, thermostat, music, timers, reminders) to Elkit's display. Hidden until asked
for; a build or a new app switches back to Design."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "fda/web/index.html"
s = p.read_text(encoding="utf-8")
if 'id="tabs"' in s:
    raise SystemExit("already patched")

css = """  /* ---- tabs: Design (always) and Room (appears once asked for) ---- */
  #right { display: flex; flex-direction: column; gap: 8px; min-width: 0; min-height: 0; }
  #tabs { display: flex; gap: 6px; }
  #tabs[hidden] { display: none; }
  #tabs button { background: transparent; color: var(--muted); border: 1px solid var(--line); border-radius: 999px;
                 padding: 5px 14px; font: inherit; font-size: 13px; cursor: pointer; }
  #tabs button.on { color: var(--text); border-color: var(--accent); background: rgba(122,162,255,.12); }
  #stage { flex: 1; }
  #stage[hidden] { display: none; }
  #room { flex: 1; overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 10px; align-content: start; }
  #room[hidden] { display: none; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; display: flex;
          flex-direction: column; gap: 8px; min-width: 0; }
  .card h3 { margin: 0; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .big { font-size: 24px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .card button { background: #1c2230; color: var(--text); border: 1px solid var(--line); border-radius: 8px;
                 padding: 6px 12px; font: inherit; cursor: pointer; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .lamp { display: flex; align-items: center; gap: 10px; cursor: pointer; }
  .bulb { width: 46px; height: 46px; border-radius: 50%; flex: none; transition: all .4s; border: 2px solid #333; background: #20262e; }
  .list { margin: 0; padding: 0; list-style: none; font-size: 14px; display: flex; flex-direction: column; gap: 3px; }
  .list li { display: flex; justify-content: space-between; gap: 8px; }
  .muted { color: var(--muted); }
  #ring { position: fixed; inset: 0; background: rgba(13,17,23,.92); display: flex; flex-direction: column; align-items: center;
          justify-content: center; gap: 18px; z-index: 9; animation: flash 1.2s ease-in-out infinite; }
  #ring[hidden] { display: none; }
  #ring button { font-size: 22px; padding: 14px 44px; background: #d9534f; color: #fff; border: none; border-radius: 10px; }
  @keyframes flash { 50% { background: rgba(120,20,20,.92); } }
"""
s = s.replace("  @media (max-width: 640px)", css + "  @media (max-width: 640px)", 1)
s = s.replace("grid-template-columns: minmax(220px, 32%) 1fr;", "grid-template-columns: minmax(220px, 32%) minmax(0, 1fr);")

old = """<section id="stage">
  <div id="bar"></div>
  <div id="empty">Tell Elkit what to make: a website, an app or a game.</div>
  <div id="note"></div>
</section>"""
assert old in s, "stage"
s = s.replace(old, """<section id="right">
  <nav id="tabs" hidden><button id="tDesign" class="on">Design</button><button id="tRoom">Room</button></nav>
  <section id="stage">
    <div id="bar"></div>
    <div id="empty">Tell Elkit what to make: a website, an app or a game.</div>
    <div id="note"></div>
  </section>
  <section id="room" hidden>
    <div class="card"><h3>Lights</h3>
      <div class="lamp" id="lamp"><div class="bulb" id="bulb"></div><div><div class="big" id="lightTxt">Off</div><div class="muted" id="lightSub"></div></div></div>
      <input type="range" min="0" max="100" id="bright" aria-label="Brightness"></div>
    <div class="card"><h3>Thermostat</h3>
      <div class="row"><button id="tDown">&minus;</button><div class="big" id="temp">--</div><button id="tUp">+</button></div></div>
    <div class="card"><h3>Music</h3><div id="track" class="muted">Stopped</div>
      <div class="row"><button id="mPlay">Play</button><button id="mNext">Next</button></div>
      <input type="range" min="0" max="100" id="vol" aria-label="Volume"></div>
    <div class="card"><h3>Timers &amp; alarms</h3><ul class="list" id="timers"></ul></div>
    <div class="card"><h3>Reminders</h3><ul class="list" id="reminders"></ul></div>
  </section>
</section>
<div id="ring" hidden><div class="big" id="ringTxt"></div><button id="stopRing">Stop</button></div>""")

js = r"""
// ---- Room tab: hidden until asked for ("show me the lights") or a room tool is used ----
const COLORS = {"warm white":"#ffd59a","cool white":"#e6f0ff","white":"#ffffff","red":"#ff4d4d","green":"#3ddc84","blue":"#4d8bff",
  "yellow":"#ffe14d","orange":"#ff9f40","purple":"#a66bff","pink":"#ff7ac6","cyan":"#4de1ff","magenta":"#ff4dff","violet":"#9a6bff","teal":"#2bc4b4"};
let room = null;
function act(action){ if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:"action", action})); }
function tab(t){
  const r = t === "room";
  if (r) $("tabs").hidden = false;
  $("room").hidden = !r; $("stage").hidden = r;
  $("tRoom").classList.toggle("on", r); $("tDesign").classList.toggle("on", !r);
}
function fmt(s){ const m = Math.floor(s/60), r = s%60; return `${m}:${String(r).padStart(2,"0")}`; }
function li(a, b){ const l = document.createElement("li"); l.innerHTML = "<span></span><span class='muted'></span>";
  l.children[0].textContent = a; l.children[1].textContent = b || ""; return l; }
function renderRoom(s){
  if (!s || !s.lights) return;
  room = s; const L = s.lights, on = L.power === "on", c = COLORS[L.color] || L.color || "#ffd59a";
  $("bulb").style.background = on ? c : "#20262e";
  $("bulb").style.boxShadow = on ? `0 0 ${10 + L.brightness/2.5}px ${L.brightness/6}px ${c}` : "none";
  $("lightTxt").textContent = on ? `${L.brightness}%` : "Off"; $("lightSub").textContent = L.color;
  if (document.activeElement !== $("bright")) $("bright").value = L.brightness;
  $("temp").textContent = `${s.thermostat_c}°C`;
  const m = s.music || {state:"stopped"};
  $("track").textContent = m.state === "stopped" ? "Stopped" : `${m.state === "playing" ? "▶" : "❚❚"} ${m.track}`;
  $("mPlay").textContent = m.state === "playing" ? "Pause" : "Play";
  if (document.activeElement !== $("vol")) $("vol").value = s.volume;
  const T = $("timers"); T.replaceChildren();
  (s.timers || []).forEach(t => T.append(li(t.label || "Timer", fmt(t.left))));
  (s.alarms || []).forEach(a => T.append(li("Alarm", a.time)));
  if (!T.children.length) T.append(li("None"));
  const R = $("reminders"); R.replaceChildren();
  (s.reminders || []).forEach(r => R.append(li(r.text, r.time)));
  if (!R.children.length) R.append(li("None"));
  $("ring").hidden = !s.ringing; if (s.ringing) $("ringTxt").textContent = s.ringing;
}
$("tDesign").onclick = () => tab("design");
$("tRoom").onclick = () => tab("room");
$("lamp").onclick = () => room && act({tool:"lights", power: room.lights.power === "on" ? "off" : "on"});
$("bright").onchange = e => act({tool:"lights", brightness: +e.target.value});
$("tDown").onclick = () => room && act({tool:"thermostat", celsius: room.thermostat_c - 0.5});
$("tUp").onclick = () => room && act({tool:"thermostat", celsius: room.thermostat_c + 0.5});
$("mPlay").onclick = () => room && act({tool:"music", action: room.music.state === "playing" ? "pause" : (room.music.state === "paused" ? "resume" : "play")});
$("mNext").onclick = () => act({tool:"music", action:"next"});
$("vol").onchange = e => act({tool:"volume", level: +e.target.value});
$("stopRing").onclick = () => { $("ring").hidden = true; act({tool:"stop_ringing"}); };
"""
s = s.replace("function onMsg(m){", js + "function onMsg(m){", 1)
old = """    case "build": showBuild(m); break;
  }"""
assert old in s, "build"
s = s.replace(old, """    case "build": if (m.building) tab("design"); showBuild(m); break;
    case "room": renderRoom(m.state); if (m.state && m.state.ringing) tab("room"); break;
    case "tab": tab(m.tab); break;
  }""")
s = s.replace("""    case "app": showApp(m.url, m.title); break;""", """    case "app": tab("design"); showApp(m.url, m.title); break;""")
p.write_text(s, encoding="utf-8")
print("ok")
