// Checks Elkit's display without a browser: fake WebSocket, feed server messages, look at the tabs and pulse.
//   node scripts/check_ui.js
const fs = require("fs"), path = require("path");
const { JSDOM } = require(path.join(__dirname, "../fda/mcp_servers/node_modules/jsdom"));
const html = fs.readFileSync(path.join(__dirname, "../fda/web/index.html"), "utf8");
let sock = null; const sent = [];
class FakeWS { constructor(){ sock = this; this.readyState = 1; setTimeout(() => this.onopen && this.onopen(), 0); }
  send(x){ sent.push(JSON.parse(x)); } }
const dom = new JSDOM(html, { runScripts: "dangerously", url: "http://localhost:8765/",
  beforeParse(w){ w.WebSocket = FakeWS; } });
const d = dom.window.document, $ = id => d.getElementById(id);
const msg = m => sock.onmessage({ data: JSON.stringify(m) });
const ok = (c, t) => { console.log((c ? "PASS " : "FAIL ") + t); if (!c) process.exitCode = 1; };
setTimeout(() => {
  ok($("tabs").hidden, "room tab hidden at start");
  ok(!d.querySelector("#chat"), "no conversation transcript");
  msg({ type: "status", status: "speaking" }); msg({ type: "level", db: -20 });
  ok(d.body.dataset.s === "speaking" && /scale/.test(d.querySelector("#pulse .core").style.transform), "pulse moves while speaking");
  msg({ type: "status", status: "listening" });
  ok(d.querySelector("#pulse .core").style.transform === "", "pulse still when listening");
  msg({ type: "room", state: { lights: { power: "on", brightness: 70, color: "blue" }, thermostat_c: 21, music: { state: "stopped" },
        volume: 50, timers: [{ label: "Tea", left: 125 }], alarms: [], reminders: [] } });
  ok($("tabs").hidden, "room update alone does not open the tab");
  msg({ type: "tab", tab: "room" });
  ok(!$("tabs").hidden && !$("room").hidden && $("stage").hidden, "asked: room tab shows");
  ok($("lightTxt").textContent === "70%" && /Tea/.test($("timers").textContent), "room shows lights and timer");
  $("lamp").click();
  ok(sent.length && sent[0].action.tool === "lights" && sent[0].action.power === "off", "tap the lamp turns it off");
  msg({ type: "build", building: true, title: "Coffee page", chars: 1000, expect: 4000 });
  ok($("room").hidden && !$("stage").hidden, "a build switches back to Design");
  ok(/Coffee page/.test($("note").textContent), "build note shows");
  msg({ type: "app", url: "/apps/coffee.html", title: "Coffee page" });
  ok(!!d.querySelector("#stage iframe"), "app shows in the build box");
  $("tRoom").click();
  ok(!$("room").hidden, "Room tab button works");
  $("tDesign").click();
  ok(!$("stage").hidden, "Design tab button works");
  dom.window.close();
}, 30);
