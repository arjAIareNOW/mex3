/* FDA live settings: put into every app FDA builds. The app reads its look from named settings
   (const C = FDA.defaults({bird: "🐦", pipe: "#3a3", sky: "#7cf", speed: 3})) and draws things with
   FDA.draw(ctx, "bird", x, y, w, h). A setting is a colour, an emoji/text or an image (photo), and FDA
   changes it while the app runs (by voice: "make the bird a butterfly"), without rebuilding it. */
window.FDA = (function () {
  const cfg = {}, imgs = {}, pats = {}, tinted = {}, saved = window.FDA_SAVED || {};
  const isImg = v => typeof v === "string" && /^(data:image|https?:|blob:)|\.(png|jpe?g|gif|webp|svg)$/i.test(v);
  const isColor = v => typeof v === "string" && /^(#|rgb|hsl)|^[a-z]+$/i.test(v) && v.indexOf("@") < 0 &&
    (typeof CSS === "undefined" || !CSS.supports || CSS.supports("color", v));
  // "/assets/sprites/bird.svg@red" = that picture tinted red (the tint is cut off before loading it)
  function splitTint(v) {
    v = String(v).replace("|flip", "");
    const i = v.lastIndexOf("@");
    if (i > 0 && /^[#a-z0-9(),.% ]+$/i.test(v.slice(i + 1)) && isImg(v.slice(0, i))) return [v.slice(0, i), v.slice(i + 1)];
    return [v, ""];
  }
  function load(k, v) {
    if (typeof v !== "string") return;
    const src = splitTint(v)[0];
    if (!isImg(src) || (imgs[k] && imgs[k]._src === src)) return;   // (.src would be the full URL)
    const im = new Image(); im.src = src; im._src = src; imgs[k] = im;
    for (const p in pats) if (p.indexOf(k + "@") === 0) delete pats[p];
  }
  function tintImg(im, tint, w, h) {           // the picture, coloured over (its shape and shading stay)
    const key = im._src + "@" + tint + "|" + Math.round(w) + "x" + Math.round(h);
    if (tinted[key]) return tinted[key];
    if (typeof document === "undefined" || !document.createElement) return im;
    const off = document.createElement("canvas");
    off.width = Math.max(1, Math.ceil(w)); off.height = Math.max(1, Math.ceil(h));
    const o = off.getContext("2d");
    o.drawImage(im, 0, 0, off.width, off.height);
    o.globalCompositeOperation = "source-atop"; o.globalAlpha = 0.6; o.fillStyle = tint;
    o.fillRect(0, 0, off.width, off.height);
    return (tinted[key] = off);
  }
  function defaults(d) {
    for (const k in d) if (!(k in cfg)) cfg[k] = (k in saved) ? saved[k] : d[k];
    for (const k in cfg) load(k, cfg[k]);
    return cfg;
  }
  function set(patch) {
    for (const k in patch) { cfg[k] = patch[k]; load(k, patch[k]); }
    try { window.dispatchEvent(new CustomEvent("fda-change", { detail: patch })); } catch (e) {}
  }
  function draw(ctx, k, x, y, w, h) {
    // a setting's name ("bird"), or its value (FDA.draw(ctx, C.bird, ...) works too: C.bird is live)
    let v = (typeof k === "string" && k in cfg) ? cfg[k] : k;
    if (v === undefined || v === null || v === "") return;
    if (!(typeof k === "string" && k in cfg)) { k = String(v).slice(0, 200); load(k, v); }
    // "🐦|flip" (or a photo + "|flip") = drawn mirrored: emoji face one fixed way, the game may need the other
    const flip = typeof v === "string" && v.indexOf("|flip") >= 0;
    if (flip) v = v.replace("|flip", "");
    ctx.save();
    if (flip) { ctx.translate(2 * x + w, 0); ctx.scale(-1, 1); }
    const [src, itint] = splitTint(v);
    if (isImg(src)) {
      const im = imgs[k];
      if (im && im.complete && im.naturalWidth) {
        if (w > 2.2 * h || h > 2.2 * w) {          // long thing (pipe, ground): tile the photo
          const pk = k + "@" + itint;               // a new tint = a new pattern
          if (!pats[pk]) {
            const s = Math.min(w, h);               // tinted: one tile = the short side
            pats[pk] = ctx.createPattern(itint ? tintImg(im, itint, s, s) : im, "repeat");
          }
          ctx.translate(x, y); ctx.fillStyle = pats[pk]; ctx.fillRect(0, 0, w, h);
        } else ctx.drawImage(itint ? tintImg(im, itint, w, h) : im, x, y, w, h);
      } else { ctx.fillStyle = "#999"; ctx.fillRect(x, y, w, h); }
    } else if (isColor(v)) {
      ctx.fillStyle = v; ctx.fillRect(x, y, w, h);
    } else {                                       // emoji or text ("🐦", or "🐦@red" = tinted red)
      const [glyph, tint] = String(v).split("@");
      const s = Math.max(4, Math.min(w, h));
      if (tint) {
        const key = v + "|" + Math.round(s);
        let off = tinted[key];
        if (!off && typeof document !== "undefined" && document.createElement) {
          off = document.createElement("canvas"); off.width = off.height = Math.ceil(s * 1.2);
          const o = off.getContext("2d");
          o.font = Math.round(s * 0.9) + "px serif"; o.textAlign = "center"; o.textBaseline = "middle";
          o.fillText(glyph, off.width / 2, off.height / 2 + s * 0.05);
          o.globalCompositeOperation = "source-atop"; o.globalAlpha = 0.6; o.fillStyle = tint;
          o.fillRect(0, 0, off.width, off.height);
          tinted[key] = off;
        }
        if (off) ctx.drawImage(off, x + w / 2 - off.width / 2, y + h / 2 - off.height / 2);
      } else {
        ctx.font = Math.round(s * 0.9) + "px serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(glyph, x + w / 2, y + h / 2 + s * 0.05);
      }
    }
    ctx.restore();
  }
  function color(k, fallback) { const v = cfg[k]; return isColor(v) ? v : (fallback || "#888"); }
  window.addEventListener("message", e => {
    if (e.data && e.data.type === "fda_set" && e.data.patch) set(e.data.patch);
    if (e.data && e.data.type === "fda_pause") { try { document.querySelectorAll("audio,video").forEach(a => a.pause()); } catch (_) {} }
  });
  return { cfg, defaults, set, draw, color, isImg, isColor };
})();
