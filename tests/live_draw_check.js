// node t_live.js path/to/live.js : FDA.draw with library pictures, tints and flips (fake canvas + Image)
const fs = require("fs");
const calls = [];
const ctx2 = () => ({
  save() {}, restore() {}, translate() {}, scale() {}, fillRect(...a) { calls.push(["fillRect", this.fillStyle]); },
  fillText() {}, drawImage(im) { calls.push(["drawImage", im.tag || im._src]); },
  createPattern(im) { calls.push(["pattern", im.tag || im._src]); return { p: im.tag || im._src }; },
  set globalCompositeOperation(v) {}, set globalAlpha(v) {},
});
global.window = { addEventListener() {}, dispatchEvent() {} };
global.CustomEvent = class {};
global.Image = class { constructor() { this.complete = true; this.naturalWidth = 64; } };
global.document = { createElement: () => ({ tag: "tinted", getContext: () => ctx2() }) };
eval(fs.readFileSync(process.argv[2], "utf8"));
const FDA = window.FDA;
let fail = 0;
const ok = (name, got, want) => { const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { fail++; console.log("FAIL", name, g, "!=", w); } else console.log("ok  ", name); };
const ctx = ctx2();
FDA.defaults({ bird: "/assets/sprites/bird.svg", pipe: "/assets/tiles/brick.svg", e: "🐦@red" });
calls.length = 0; FDA.draw(ctx, "bird", 0, 0, 40, 40);
ok("plain picture drawn", calls, [["drawImage", "/assets/sprites/bird.svg"]]);
FDA.set({ bird: "/assets/sprites/bird.svg@black" });
calls.length = 0; FDA.draw(ctx, "bird", 0, 0, 40, 40);
ok("tinted picture: drawn through a tinted copy", calls.map(c => c[1]), ["/assets/sprites/bird.svg", "black", "tinted"]);
FDA.set({ bird: "/assets/sprites/bird.svg|flip@red" });
calls.length = 0; FDA.draw(ctx, "bird", 0, 0, 40, 40);
ok("flip + tint", calls.map(c => c[1]), ["/assets/sprites/bird.svg", "red", "tinted"]);
calls.length = 0; FDA.draw(ctx, "pipe", 0, 0, 40, 300);
ok("long thing: tiled", calls.map(c => c[0]), ["pattern", "fillRect"]);
FDA.set({ pipe: "/assets/tiles/brick.svg@blue" });
calls.length = 0; FDA.draw(ctx, "pipe", 0, 0, 40, 300);
ok("tinted tiles: a new pattern", calls[calls.length - 2], ["pattern", "tinted"]);
FDA.set({ pipe: "/assets/tiles/stone.svg" });
calls.length = 0; FDA.draw(ctx, "pipe", 0, 0, 40, 300);
ok("new picture: new pattern", calls[0], ["pattern", "/assets/tiles/stone.svg"]);
calls.length = 0; FDA.draw(ctx, "e", 0, 0, 40, 40);
ok("emoji tint still works", calls.map(c => c[1]), ["red", "tinted"]);
ok("a colour is not a picture", FDA.isImg("#ff0000@x"), false);
process.exit(fail ? 1 : 0);
