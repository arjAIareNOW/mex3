/* FDA live settings: put into every app FDA builds. The app reads its look from named settings
   (const C = FDA.defaults({bird: "🐦", pipe: "#3a3", sky: "#7cf", speed: 3})) and draws things with
   FDA.draw(ctx, "bird", x, y, w, h). A setting is a colour, an emoji/text or an image (photo), and FDA
   changes it while the app runs (by voice: "make the bird a butterfly"), without rebuilding it. */
window.FDA = (function () {
  const cfg = {}, imgs = {}, pats = {}, tinted = {}, saved = window.FDA_SAVED || {};
  const isImg = v => typeof v === "string" && /^(data:image|https?:|blob:)|\.(png|jpe?g|gif|webp|svg)$/i.test(v);
  const isColor = v => typeof v === "string" && /^(#|rgb|hsl)|^[a-z]+$/i.test(v) && v.indexOf("@") < 0 &&
    (typeof CSS === "undefined" || !CSS.supports || CSS.supports("color", v));
  function load(k, v) {
    if (!isImg(v) || (imgs[k] && imgs[k].src === v)) return;
    const im = new Image(); im.src = v; imgs[k] = im; delete pats[k];
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
    const v = (typeof k === "string" && k in cfg) ? cfg[k] : k;
    if (v === undefined || v === null || v === "") return;
    if (!(typeof k === "string" && k in cfg)) { k = String(v).slice(0, 200); load(k, v); }
    ctx.save();
    if (isImg(v)) {
      const im = imgs[k];
      if (im && im.complete && im.naturalWidth) {
        if (w > 2.2 * h || h > 2.2 * w) {          // long thing (pipe, ground): tile the photo
          if (!pats[k]) pats[k] = ctx.createPattern(im, "repeat");
          ctx.translate(x, y); ctx.fillStyle = pats[k]; ctx.fillRect(0, 0, w, h);
        } else ctx.drawImage(im, x, y, w, h);
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
