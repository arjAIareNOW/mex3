// Checks Elkit's display without a browser: fake WebSocket, feed server messages, look at the tabs, pulse, room
// picture and the pick pop-up.
//   node scripts/check_ui.js
const fs = require("fs"), path = require("path");
const { JSDOM } = require(path.join(__dirname, "../fda/mcp_servers/node_modules/jsdom"));
const html = fs.readFileSync(path.join(__dirname, "../fda/web/index.html"), "utf8");
let sock = null; const sent = [], errors = [];
class FakeWS { constructor(){ sock = this; this.readyState = 1; setTimeout(() => this.onopen && this.onopen(), 0); }
  send(x){ sent.push(JSON.parse(x)); } }
const dom = new JSDOM(html, { runScripts: "dangerously", url: "http://localhost:8765/",
  beforeParse(w){ w.WebSocket = FakeWS; w.addEventListener("error", e => errors.push(e.message)); } });
const d = dom.window.document, $ = id => d.getElementById(id);
const msg = m => sock.onmessage({ data: JSON.stringify(m) });
const ok = (c, t) => { console.log((c ? "PASS " : "FAIL ") + t); if (!c) process.exitCode = 1; };
setTimeout(() => {
  ok(!errors.length, "page script runs without errors" + (errors.length ? ": " + errors[0] : ""));
  ok(!$("stage").hidden && $("room").hidden, "starts on Interface");
  ok(!d.querySelector("#chat"), "no conversation transcript");
  msg({ type: "status", status: "speaking" }); msg({ type: "level", db: -20 });
  ok(d.body.dataset.s === "speaking", "status: speaking");
  msg({ type: "status", status: "listening" });
  ok(d.body.dataset.s === "listening", "status: listening");
  const room = { lights: { power: "on", brightness: 70, color: "blue" }, thermostat_c: 21, music: { state: "stopped" },
                 volume: 50, timers: [{ label: "Tea", left: 125, seconds: 300 }], alarms: [],
                 reminders: [{ text: "Call mom", time: "18:00" }] };
  msg({ type: "room", state: room });
  ok(!$("stage").hidden, "a room update alone does not switch tabs");
  msg({ type: "tab", tab: "room" });
  ok(!$("room").hidden && $("stage").hidden, "asked: Room shows");
  ok(/70%/.test($("lampTxt").textContent) && /2:0[45]/.test($("tmTxt").textContent), "room shows lights and timer");
  ok(/Call mom/.test($("notes").textContent), "room shows reminders");
  msg({ type: "room", state: { ...room, ringing: "Alarm 06:30" } });
  ok(!$("ring").hidden && /06:30/.test($("ringTxt").textContent), "ringing: the stop screen shows");
  $("stopRing").click();
  ok(sent.some(s => s.type === "action" && s.action.tool === "stop_ringing"), "tap Stop stops the ringing");
  msg({ type: "build", building: true, title: "Coffee page", chars: 1000, expect: 4000 });
  ok($("room").hidden && !$("stage").hidden, "a build switches back to Interface");
  ok(/Coffee page/.test($("note").textContent), "build note shows");
  $("tRoom").click();
  ok(!$("room").hidden, "Room tab button works");
  $("tDesign").click();
  ok(!$("stage").hidden, "Interface tab button works");
  // the pick pop-up
  ok($("choices").hidden, "no pop-up at start");
  msg({ type: "choices", title: "Pick a bird", options: [
    { n: 1, label: "Parrot", img: "/assets/sprites/parrot.svg" }, { n: 2, label: "Owl", img: "/assets/sprites/owl.svg" },
    { n: 3, label: "Penguin", img: "/assets/sprites/penguin.svg" }, { n: 4, label: "Duck", img: "/assets/sprites/duck.svg" }] });
  const cards = d.querySelectorAll("#chGrid .ch");
  ok(!$("choices").hidden && cards.length === 4 && $("chTitle").textContent === "Pick a bird", "pop-up: 4 cards and a title");
  ok(cards[1].querySelector(".n").textContent === "2" && cards[1].querySelector("img").getAttribute("src") === "/assets/sprites/owl.svg"
     && /Owl/.test(cards[1].textContent), "pop-up: number, picture and name on each card");
  cards[2].click();
  ok(sent.some(s => s.type === "choice" && s.n === 3), "pop-up: a tap sends the card's number");
  msg({ type: "choices", title: "Colour schemes", options: [
    { n: 1, label: "Ocean", colors: { background: "#f0f7ff", surface: "#ffffff", primary: "#1e6fd9", accent: "#00b3a4", text: "#0f2a44" } }] });
  const mock = d.querySelector("#chGrid .mock");
  ok(!!mock && mock.querySelector(".hd").style.background !== "" && d.querySelectorAll("#chGrid .ch").length === 1,
     "pop-up: a colour scheme shows as a small page in its colours (old cards replaced)");
  msg({ type: "choices", close: true });
  ok($("choices").hidden, "pop-up closes");
  dom.window.close();
}, 30);
