// Runs a built app's script for a moment in Node, with a stand-in page (canvas, document, events),
// to catch errors that only show up when it runs ("Assignment to constant variable") and apps that draw nothing.
// Usage: node runcheck.js live.js app.js [tests.json]  ->  one JSON line {error, line, draws, dom, settings, drawn, tests}
// tests.json: the app's own behaviour tests, [{"name", "do": ["tap", "space", "key ArrowLeft", "click #id",
//   "set #id 100", "wait 10"], "before": "<js value>", "expect": "<js true/false; `before` = value before 'do'>"}]
const fs = require("fs");
const vm = require("vm");
const [liveFile, appFile, testFile] = process.argv.slice(2);
const out = { error: null, line: null, draws: 0, dom: 0, settings: {} };
const listeners = [];
const drawn = {};
const rafs = [];
let firstErr = null;
function fail(e) {
  if (firstErr) return;
  firstErr = e;
  out.error = String((e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : e));
  const m = /app\.js:(\d+)/.exec((e && e.stack) || "");
  if (m) out.line = +m[1];
}
const DRAW = new Set(["fillRect", "strokeRect", "fillText", "strokeText", "drawImage", "fill", "stroke",
  "putImageData", "arc", "rect", "lineTo", "ellipse", "roundRect"]);
function makeCtx(canvas) {
  const base = {
    canvas, measureText: t => ({ width: String(t).length * 8 }),
    createLinearGradient: () => ({ addColorStop() {} }), createRadialGradient: () => ({ addColorStop() {} }),
    createPattern: () => ({}), getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(4, (w * h * 4) | 0)) }),
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(Math.max(4, (w * h * 4) | 0)) }),
    getTransform: () => ({ a: 1, d: 1 }), isPointInPath: () => false, setLineDash() {}, getLineDash: () => [],
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
function makeEl(tag) {
  out.dom++;
  const store = { tagName: String(tag || "div").toUpperCase(), style: {}, dataset: {}, children: [], width: 800, height: 480,
    value: "", textContent: "", innerHTML: "", innerText: "", className: "", id: "", disabled: false, checked: false,
    offsetWidth: 800, offsetHeight: 480, clientWidth: 800, clientHeight: 480 };
  let ctx = null;
  const el = new Proxy(store, {
    get(t, k) {
      if (k in t) return t[k];
      switch (k) {
        case "getContext": return () => ctx || (ctx = makeCtx(el));
        case "addEventListener": return (type, fn) => listeners.push([type, fn, el]);
        case "removeEventListener": return () => {};
        case "getBoundingClientRect": return () => ({ left: 0, top: 0, x: 0, y: 0, width: 800, height: 480, right: 800, bottom: 480 });
        case "appendChild": case "append": case "prepend": case "insertBefore": case "replaceChildren":
          return (...c) => { t.children.push(...c); return c[0]; };
        case "removeChild": case "remove": return () => {};
        case "querySelector": return () => makeEl("div");
        case "querySelectorAll": case "getElementsByTagName": case "getElementsByClassName": return () => [];
        case "classList": return { add() {}, remove() {}, toggle() {}, contains: () => false };
        case "setAttribute": case "removeAttribute": case "focus": case "blur": case "click": case "scrollIntoView":
        case "requestFullscreen": case "play": case "pause": return () => {};
        case "getAttribute": return () => null;
        case "parentNode": case "parentElement": return body;
        case "firstChild": case "lastChild": return null;
        case "toDataURL": return () => "data:image/png;base64,";
      }
      return undefined;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return el;
}
const body = makeEl("body");
const ids = {};
const document = {
  body, documentElement: makeEl("html"), head: makeEl("head"), readyState: "complete",
  getElementById: id => ids[id] || (ids[id] = makeEl(/canvas/i.test(id) ? "canvas" : "div")),
  querySelector: s => ids[s] || (ids[s] = makeEl(/canvas/i.test(s) ? "canvas" : "div")),
  querySelectorAll: () => [], getElementsByTagName: t => [makeEl(t)], getElementsByClassName: () => [],
  createElement: makeEl, createTextNode: () => makeEl("text"),
  addEventListener: (type, fn) => listeners.push([type, fn, document]), removeEventListener() {},
  hidden: false, fullscreenElement: null, exitFullscreen() {},
};
const storage = {};
const g = {
  document, console: { log() {}, warn() {}, error() {}, info() {} },
  innerWidth: 800, innerHeight: 480, devicePixelRatio: 1,
  addEventListener: (type, fn) => listeners.push([type, fn, null]), removeEventListener() {},
  dispatchEvent() { return true; },
  requestAnimationFrame: fn => (rafs.push(fn), rafs.length), cancelAnimationFrame() {},
  setTimeout: (fn, ms) => setTimeout(() => { try { fn(); } catch (e) { fail(e); } }, Math.min(ms || 0, 300)),
  setInterval: (fn, ms) => setInterval(() => { try { fn(); } catch (e) { fail(e); } }, Math.max(16, Math.min(ms || 16, 300))),
  clearTimeout, clearInterval,
  alert() {}, confirm: () => true, prompt: () => "", focus() {}, blur() {}, scrollTo() {}, open() {},
  localStorage: { getItem: k => (k in storage ? storage[k] : null), setItem: (k, v) => { storage[k] = String(v); },
    removeItem: k => { delete storage[k]; }, clear() {} },
  performance: { now: () => Date.now() }, Date, Math, JSON, CustomEvent: class { constructor(t, o) { this.type = t; Object.assign(this, o); } },
  Image: class { constructor() { this.complete = true; this.naturalWidth = 64; this.width = 64; this.height = 64; } },
  Audio: class { play() { return Promise.resolve(); } pause() {} },
  AudioContext: undefined, navigator: { userAgent: "node", vibrate() {} }, location: { reload() {}, href: "" },
  CSS: { supports: () => true }, parent: null, matchMedia: () => ({ matches: false, addEventListener() {} }),
  screen: { width: 800, height: 480 }, getComputedStyle: () => ({ getPropertyValue: () => "" }),
};
Object.assign(g, { URL, URLSearchParams, TextEncoder, TextDecoder, structuredClone, queueMicrotask,
  fetch: () => new Promise(() => {}), Event: class { constructor(t) { this.type = t; } },
  KeyboardEvent: class {}, MouseEvent: class {}, TouchEvent: class {}, HTMLCanvasElement: class {},
  ResizeObserver: class { observe() {} disconnect() {} }, speechSynthesis: { speak() {}, cancel() {} } });
g.window = g; g.self = g; g.globalThis = g; g.parent = g; g.top = g;
g.postMessage = () => {};
vm.createContext(g);
const ev = (type) => ({
  type, key: " ", code: "Space", keyCode: 32, which: 32, button: 0, clientX: 400, clientY: 240, pageX: 400, pageY: 240,
  offsetX: 400, offsetY: 240, touches: [{ clientX: 400, clientY: 240, pageX: 400, pageY: 240 }],
  changedTouches: [{ clientX: 400, clientY: 240, pageX: 400, pageY: 240 }],
  target: ids[Object.keys(ids)[0]] || body, currentTarget: body,
  preventDefault() {}, stopPropagation() {},
});
function fire(type) {
  for (const [t, fn] of listeners.slice()) if (t === type) { try { fn(ev(type)); } catch (e) { fail(e); } }
  for (const el of [body, document, g, ...Object.values(ids)]) {
    const h = el["on" + type];
    if (typeof h === "function") { try { h(ev(type)); } catch (e) { fail(e); } }
  }
}
try {
  vm.runInContext(fs.readFileSync(liveFile, "utf8"), g, { filename: "live.js" });
  // which settings get drawn, and where (does anything move?)
  const realDraw = g.FDA.draw;
  g.FDA.draw = function (ctx, k, x, y, w, h) {
    const name = typeof k === "string" && k in g.FDA.cfg ? k :
      Object.keys(g.FDA.cfg).find(n => g.FDA.cfg[n] === k) || String(k).slice(0, 20);
    const pos = Math.round(x) + "," + Math.round(y);
    const seen = drawn[name] || (drawn[name] = new Set());
    if (seen.size < 50) seen.add(pos);
    return realDraw.apply(this, arguments);
  };
  vm.runInContext(fs.readFileSync(appFile, "utf8"), g, { filename: "app.js" });
} catch (e) { fail(e); }
for (const t of ["DOMContentLoaded", "load"]) fire(t);
let frame = 0, ts = 0;
const INPUT = { 8: "keydown", 9: "keyup", 20: "click", 30: "pointerdown", 31: "pointerup", 40: "mousedown", 41: "mouseup",
  50: "touchstart", 51: "touchend", 60: "keydown", 61: "keyup", 75: "click" };
const pause = () => new Promise(r => setImmediate(r));
async function runFrame() {
  frame++; ts += 16.7;
  const batch = rafs.splice(0);
  for (const fn of batch) { try { fn(ts); } catch (e) { fail(e); } }
  await pause();
}
function evt(type, extra) { return Object.assign(ev(type), extra || {}); }
function fireOn(el, type, extra) {           // an event on one element, bubbling to document and window
  const e = evt(type, Object.assign({ target: el, currentTarget: el }, extra));
  for (const [t, fn, where] of listeners.slice())
    if (t === type && (where === el || where === document || where === null || where === body)) fn(e);
  for (const h of [el["on" + type], document["on" + type], g["on" + type]]) if (typeof h === "function") h(e);
}
function el(sel) {
  const id = sel.replace(/^#/, "");
  return ids[id] || ids[sel] || (ids[id] = makeEl("div"));
}
async function doStep(step) {
  const [verb, arg, ...rest] = String(step).trim().split(/\s+/);
  const canvas = ids[Object.keys(ids).find(k => ids[k].tagName === "CANVAS")] || body;
  if (verb === "wait") { for (let i = 0; i < Math.min(+arg || 1, 300); i++) await runFrame(); return; }
  if (verb === "tap") { for (const t of ["pointerdown", "mousedown", "touchstart"]) fireOn(canvas, t);
    await runFrame(); for (const t of ["pointerup", "mouseup", "touchend", "click"]) fireOn(canvas, t); }
  else if (verb === "space" || verb === "key") {
    const k = verb === "space" ? " " : arg, code = verb === "space" ? "Space" : arg;
    fireOn(body, "keydown", { key: k, code, keyCode: k === " " ? 32 : 0 }); await runFrame();
    fireOn(body, "keyup", { key: k, code, keyCode: k === " " ? 32 : 0 });
  }
  else if (verb === "click") { const e = el(arg); for (const t of ["pointerdown", "mousedown", "pointerup", "mouseup", "click"]) fireOn(e, t); }
  else if (verb === "set") { const e = el(arg); e.value = rest.join(" "); fireOn(e, "input"); fireOn(e, "change"); }
  else throw new Error("unknown test step: " + step);
  await runFrame();
}
async function runTests() {
  let tests = [];
  try { tests = JSON.parse(fs.readFileSync(testFile, "utf8")); } catch (e) { return [{ name: "tests", ok: false, error: "the tests are not valid JSON" }]; }
  const res = [];
  for (const t of (Array.isArray(tests) ? tests : []).slice(0, 6)) {
    const r = { name: String(t.name || "test").slice(0, 80), ok: false };
    try {
      for (const s of t.setup || []) await doStep(s);
      g.__before = t.before ? vm.runInContext(String(t.before), g) : undefined;
      if (g.__before && typeof g.__before === "object") g.__before = JSON.parse(JSON.stringify(g.__before));
      for (const s of t.do || []) await doStep(s);
      if (firstErr) { r.error = "the app crashed: " + out.error; res.push(r); break; }
      r.ok = !!vm.runInContext("(before => (" + String(t.expect) + "))(__before)", g);
      if (!r.ok) {
        r.error = "expected " + t.expect;
        if (t.before) try { r.error += " (before: " + JSON.stringify(g.__before) + ", now: " +
          JSON.stringify(vm.runInContext(String(t.before), g)) + ")"; } catch (e) {}
      }
    } catch (e) { r.error = (e && e.name ? e.name + ": " : "") + (e && e.message ? e.message : e); }
    res.push(r);
  }
  return res;
}
(async () => {
  while (frame < 90 && !firstErr) {
    if (INPUT[frame + 1]) fire(INPUT[frame + 1]);
    await runFrame();
  }
  try {
    const c = g.FDA ? g.FDA.cfg : {};
    for (const k in c) {
      const v = c[k];
      if (typeof v === "string" && v.startsWith("data:")) out.settings[k] = "(photo)";
      else if (["string", "number", "boolean"].includes(typeof v)) out.settings[k] = v;
    }
  } catch (e) {}
  out.drawn = {};
  for (const k in drawn) out.drawn[k] = drawn[k].size;     // setting -> number of different places drawn
  if (testFile && !firstErr) out.tests = await runTests();
  process.stdout.write(JSON.stringify(out) + "\n");
  process.exit(0);
})();
