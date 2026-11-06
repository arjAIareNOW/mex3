// Runs a built app in jsdom (a real DOM in Node: ids, selectors, events that bubble) with a stand-in canvas,
// then runs the app's own behaviour tests. Used instead of runcheck.js when jsdom is installed (npm install here).
// Usage: node runcheck_dom.js live.js app.js page.html [tests.json]
//   -> one JSON line {error, line, draws, dom, settings, drawn, tests}
// tests.json: [{"name", "setup": [...], "before": "<js>", "do": [...], "expect": "<js; `before` = earlier value>"}]
// steps: "run <js>", "tap", "space", "key ArrowLeft", "click <selector>", "set <selector> <value>", "wait N" (frames)
const fs = require("fs");
const vm = require("vm");
const { JSDOM, VirtualConsole } = require("jsdom");
const [liveFile, appFile, pageFile, testFile] = process.argv.slice(2);
const out = { error: null, line: null, draws: 0, dom: 0, settings: {} };
const drawn = {};
let rafs = [];
let firstErr = null;
function fail(e) {
  if (firstErr) return;
  firstErr = e || new Error("error");
  out.error = String((e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : e));
  const m = /app\.js:(\d+)/.exec((e && e.stack) || "");
  if (m) out.line = +m[1];
}
const vc = new VirtualConsole();
vc.on("jsdomError", e => {
  if (/Not implemented/.test(e.message)) return;         // navigation, media...: not the app's fault
  fail(e.detail instanceof Error ? e.detail : e);
});
const dom = new JSDOM(fs.readFileSync(pageFile, "utf8"), { url: "http://localhost/", runScripts: "outside-only",
  pretendToBeVisual: true, virtualConsole: vc });
const w = dom.window, document = w.document;
const ctxVM = dom.getInternalVMContext();

// --- the stand-ins jsdom does not have: canvas drawing, a fixed 800x480 screen, frames we step ourselves
const DRAW = new Set(["fillRect", "strokeRect", "fillText", "strokeText", "drawImage", "fill", "stroke",
  "putImageData", "arc", "rect", "lineTo", "ellipse", "roundRect"]);
function makeCtx(canvas) {
  const base = {
    canvas, measureText: t => ({ width: String(t).length * 8, actualBoundingBoxAscent: 8, actualBoundingBoxDescent: 2 }),
    createLinearGradient: () => ({ addColorStop() {} }), createRadialGradient: () => ({ addColorStop() {} }),
    createConicGradient: () => ({ addColorStop() {} }),
    createPattern: () => ({}), getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(4, (w * h * 4) | 0)) }),
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(Math.max(4, (w * h * 4) | 0)) }),
    getTransform: () => ({ a: 1, d: 1, e: 0, f: 0 }), isPointInPath: () => false, setLineDash() {}, getLineDash: () => [],
  };
  return new Proxy(base, {
    get(t, k) {
      if (k in t) return t[k];
      if (typeof k !== "string") return undefined;
      return (...a) => { if (DRAW.has(k)) out.draws++; };
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}
w.HTMLCanvasElement.prototype.getContext = function () { return this.__ctx || (this.__ctx = makeCtx(this)); };
w.HTMLCanvasElement.prototype.toDataURL = () => "data:image/png;base64,";
for (const [k, v] of Object.entries({ innerWidth: 800, innerHeight: 480, devicePixelRatio: 1 }))
  Object.defineProperty(w, k, { value: v, configurable: true, writable: true });
w.requestAnimationFrame = fn => (rafs.push(fn), rafs.length);
w.cancelAnimationFrame = () => {};
if (!w.PointerEvent) w.PointerEvent = w.MouseEvent;     // jsdom has no PointerEvent; apps and tests may use it
// the app's clock follows the frames we step (16.7 ms each), not the real clock: frames here run far faster than
// real time, so a game timing its shots with performance.now() would otherwise never fire twice
let simT = 0;
try { Object.defineProperty(w.performance, "now", { value: () => 1000 + simT, configurable: true }); } catch (e) {}
w.alert = () => {}; w.confirm = () => true; w.prompt = () => "";
w.scrollTo = () => {}; w.focus = () => {}; w.open = () => null;
w.matchMedia = w.matchMedia || (() => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
w.Audio = class { play() { return Promise.resolve(); } pause() {} };
w.AudioContext = w.webkitAudioContext = undefined;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.fetch = () => new Promise(() => {});
w.HTMLMediaElement.prototype.play = () => Promise.resolve();
w.HTMLMediaElement.prototype.pause = () => {};
w.Element.prototype.scrollIntoView = () => {};
w.Element.prototype.requestFullscreen = () => Promise.resolve();
w.Element.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, x: 0, y: 0, width: 800, height: 480, right: 800, bottom: 480 };
};
w.addEventListener("error", e => fail(e.error || e.message));

function run(code, filename) {
  try { vm.runInContext(code, ctxVM, { filename }); } catch (e) { fail(e); }
}
run(fs.readFileSync(liveFile, "utf8"), "live.js");
try {
  const realDraw = w.FDA.draw;
  w.FDA.draw = function (ctx, k, x, y, wd, h) {
    const cfg = w.FDA.cfg || {};
    const name = typeof k === "string" && k in cfg ? k : Object.keys(cfg).find(n => cfg[n] === k) || String(k).slice(0, 20);
    const seen = drawn[name] || (drawn[name] = new Set());
    if (seen.size < 50) seen.add(Math.round(x) + "," + Math.round(y));
    return realDraw.apply(this, arguments);
  };
} catch (e) {}
run(fs.readFileSync(appFile, "utf8"), "app.js");
for (const t of ["DOMContentLoaded", "load"]) {
  try { (t === "load" ? w : document).dispatchEvent(new w.Event(t, { bubbles: t !== "load" })); } catch (e) { fail(e); }
}

// --- input and frames
let frame = 0, ts = 0;
const pause = () => new Promise(r => setImmediate(r));
async function runFrame() {
  frame++; ts += 16.7; simT = ts;
  const batch = rafs; rafs = [];
  for (const fn of batch) { try { fn(ts); } catch (e) { fail(e); } }
  await pause();
}
const canvas = () => document.querySelector("canvas") || document.body;
function send(target, type, extra) {
  let e;
  const pos = { clientX: 400, clientY: 240, screenX: 400, screenY: 240, button: 0, buttons: 1, bubbles: true, cancelable: true };
  if (/^key/.test(type)) e = new w.KeyboardEvent(type, Object.assign({ bubbles: true, cancelable: true }, extra));
  else if (/^(mouse|click|pointer)/.test(type)) e = new w.MouseEvent(type, pos);
  else e = new w.Event(type, { bubbles: true, cancelable: true });
  const props = Object.assign({ offsetX: 400, offsetY: 240, pageX: 400, pageY: 240, pointerId: 1, pointerType: "mouse" }, extra || {});
  if (/^touch/.test(type)) {
    const t = [{ clientX: 400, clientY: 240, pageX: 400, pageY: 240, identifier: 1, target }];
    Object.assign(props, { touches: type === "touchend" ? [] : t, changedTouches: t, targetTouches: t });
  }
  for (const [k, v] of Object.entries(props)) {
    try { if (!(k in e) || e[k] === undefined || k === "keyCode" || k === "which") Object.defineProperty(e, k, { value: v }); } catch (er) {}
  }
  try { target.dispatchEvent(e); } catch (er) { fail(er); }
  const h = target["on" + type];          // jsdom runs on* handlers itself; nothing else to do
  return h;
}
function find(sel) {
  const el = document.querySelector(sel) || document.getElementById(sel.replace(/^#/, ""));
  if (!el) throw new Error("no element matches " + sel);
  return el;
}
async function doStep(step) {
  const str = String(step).trim();
  const [verb, ...restParts] = str.split(/\s+/);
  const rest = str.slice(verb.length).trim();
  if (verb === "run") { vm.runInContext(rest, ctxVM); await runFrame(); return; }   // set up a situation directly
  if (verb === "wait") { for (let i = 0; i < Math.min(+rest || 1, 300); i++) await runFrame(); return; }
  if (verb === "tap") {
    const c = canvas();
    for (const t of ["pointerdown", "mousedown", "touchstart"]) send(c, t);
    await runFrame();
    for (const t of ["pointerup", "mouseup", "touchend", "click"]) send(c, t);
  } else if (verb === "space" || verb === "key") {
    const key = verb === "space" ? " " : rest, code = verb === "space" ? "Space" : rest;
    const kc = { " ": 32, ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Enter: 13 }[key] || 0;
    const tgt = document.activeElement || document.body;
    send(tgt, "keydown", { key, code, keyCode: kc, which: kc });
    for (let i = 0; i < 6; i++) await runFrame();       // held about 0.1 s, like a real key press
    send(tgt, "keyup", { key, code, keyCode: kc, which: kc });
  } else if (verb === "click") {
    const el = find(rest);
    for (const t of ["pointerdown", "mousedown", "pointerup", "mouseup"]) send(el, t);
    try { el.click(); } catch (e) { fail(e); }
  } else if (verb === "set") {
    // "set #bill 100" / "set input[name=\"a b\"] 100": the value is the last word
    const m = /^(.*\S)\s+(\S+)$/.exec(rest);
    if (!m) throw new Error("bad step: " + str);
    const el = find(m[1]);
    el.value = m[2];
    send(el, "input"); send(el, "change");
  } else throw new Error("unknown test step: " + str);
  await runFrame();
}
async function runTests() {
  let tests;
  try { tests = JSON.parse(fs.readFileSync(testFile, "utf8")); } catch (e) { return [{ name: "tests", ok: false, error: "the tests are not valid JSON" }]; }
  const res = [];
  for (const t of (Array.isArray(tests) ? tests : []).slice(0, 6)) {
    const r = { name: String(t.name || "test").slice(0, 80), ok: false };
    try {
      // each test starts from a fresh game (the app's own reset()), not from where the test before it left
      // things: a blink after a lost life, a snake that already turned, a game that is over
      try { vm.runInContext("if (typeof reset === 'function' && typeof state !== 'undefined') reset();", ctxVM); }
      catch (e) {}
      for (const s of t.setup || []) await doStep(s);
      w.__before = t.before ? vm.runInContext(String(t.before), ctxVM) : undefined;
      if (w.__before && typeof w.__before === "object") w.__before = JSON.parse(JSON.stringify(w.__before));
      for (const s of t.do || []) await doStep(s);
      if (firstErr) { r.error = "the app crashed: " + out.error; res.push(r); break; }
      const expect = "(before => (" + String(t.expect) + "))(__before)";
      r.ok = !!vm.runInContext(expect, ctxVM);
      // not yet: things may still be on their way (a fruit falling off the screen): up to 1.5 s more, as a
      // person watching would wait. Something that never happens still fails.
      for (let i = 0; i < 90 && !r.ok && !firstErr; i++) {
        await runFrame();
        try { r.ok = !!vm.runInContext(expect, ctxVM); } catch (e) { break; }
      }
      if (!r.ok) {
        r.error = "expected " + t.expect;
        try {
          if (t.before) r.error += " (before: " + JSON.stringify(w.__before) + ", now: " +
            JSON.stringify(vm.runInContext(String(t.before), ctxVM)) + ")";
        } catch (e) {}
      }
    } catch (e) { r.error = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : e); }
    res.push(r);
  }
  return res;
}
const WARMUP = { 8: "keydown", 9: "keyup", 20: "click", 30: "pointerdown", 31: "pointerup", 40: "mousedown",
  41: "mouseup", 50: "touchstart", 51: "touchend", 60: "keydown", 61: "keyup", 75: "click" };
(async () => {
  // 4 s of play (a tap now and then after the first 1.5 s): some things only come after a while (the first
  // cactus of a runner), and "the game never draws X" must not be said of them
  for (let f = 105; f < 240; f += 45) { WARMUP[f] = "pointerdown"; WARMUP[f + 1] = "pointerup"; }
  while (frame < 240 && !firstErr) {
    const t = WARMUP[frame + 1];
    if (t) send(/^key/.test(t) ? document.body : canvas(), t, /^key/.test(t) ? { key: " ", code: "Space", keyCode: 32, which: 32 } : {});
    await runFrame();
  }
  try {
    const c = w.FDA ? w.FDA.cfg : {};
    for (const k in c) {
      const v = c[k];
      if (typeof v === "string" && v.startsWith("data:")) out.settings[k] = "(photo)";
      else if (["string", "number", "boolean"].includes(typeof v)) out.settings[k] = v;
    }
  } catch (e) {}
  out.dom = document.body ? document.body.querySelectorAll("*").length : 0;
  out.drawn = {};
  for (const k in drawn) out.drawn[k] = drawn[k].size;
  if (testFile && !firstErr) {
    // tests start from a fresh copy of the page? No: from the state after the warm-up, like a person who played
    // a little. Games restart with "tap" in setup.
    out.tests = await runTests();
  }
  process.stdout.write(JSON.stringify(out) + "\n");
  process.exit(0);
})();
