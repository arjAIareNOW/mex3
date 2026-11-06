"""Real photos for websites, from Unsplash (approved by the user 2026-11-06; key UNSPLASH_ACCESS_KEY in .env).

The website builder writes  <img data-photo="summer dress" alt="...">  with no src; fill() puts in real photos:
  - at most 3 photos per request (a new site, or "add photos"), and one search returns 3 photos (per_page=3): the
    user's choice, "always 3 at a time, or less, so that I can maximize it" (free demo key: 50 searches an hour).
    3 images of the same thing = 1 search; images past the third get the library / emoji picture
  - results are cached on disk (logs/photo_cache.json, 7 days): rebuilding or editing a site costs nothing again
  - Unsplash's rules: their own photo links (hotlinked, not copied), "Photo by NAME on Unsplash" credits on the page,
    and a call to the photo's download_location when it is used
  - no key, no internet, no match, or the hourly limit reached: the art library's picture or an emoji instead (never
    an empty grey box, never a random photo from elsewhere)
Games never get photos (live.py: cut into pipes and birds they looked wrong).
"""
from __future__ import annotations

import html as htmlmod
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from . import live

log = logging.getLogger("fda.mcp.photos")

API = "https://api.unsplash.com/search/photos"
PER_SEARCH = 3                     # photos per search (the user's choice)
MAX_PER_REQUEST = 3                # photos per build or change, at most ("only 3 at a time, or less"): more pictures
                                   # come with the next "add more photos"; the rest of this request: library / emoji
CACHE = Path(__file__).resolve().parents[2] / "logs" / "photo_cache.json"
CACHE_DAYS = 7
UTM = "utm_source=elkit&utm_medium=referral"
_SHAPES = {"portrait": "portrait", "tall": "portrait", "landscape": "landscape", "wide": "landscape",
           "square": "squarish", "squarish": "squarish"}
_IMG = re.compile(r"<img\b[^>]*\bdata-photo\s*=\s*([\"'])(.*?)\1[^>]*>", re.I | re.S)
_lock = threading.Lock()
remaining: int | None = None       # searches left this hour (Unsplash's X-Ratelimit-Remaining)


def key() -> str:
    return os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()


def _norm(q: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", q.lower()).split())[:60]


def _load_cache() -> dict:
    try:
        c = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {k: v for k, v in c.items() if now - v.get("t", 0) < CACHE_DAYS * 86400}


def _save_cache(c: dict):
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(c), encoding="utf-8")
    except OSError as e:
        log.warning("photo cache not saved: %s", e)


def search(query: str, page: int = 1, shape: str = "", client: httpx.Client | None = None) -> list[dict] | None:
    """Up to 3 photos for the words (one API call, or none if cached). None = could not ask (no key, limit, error)."""
    global remaining
    q = _norm(query)
    if not q:
        return []
    ck = f"{q}|{page}|{shape}"
    with _lock:
        cache = _load_cache()
        if ck in cache:
            return cache[ck]["photos"]
    k = key()
    if not k:
        return None
    if remaining is not None and remaining <= 0:
        log.warning("Unsplash hourly limit reached: no photo search for %r", q)
        return None
    params = {"query": q, "per_page": PER_SEARCH, "page": page, "content_filter": "high"}
    if shape:
        params["orientation"] = shape
    try:
        c = client or httpx.Client(timeout=10)
        r = c.get(API, params=params, headers={"Authorization": f"Client-ID {k}", "Accept-Version": "v1"})
        if r.headers.get("x-ratelimit-remaining", "").isdigit():
            remaining = int(r.headers["x-ratelimit-remaining"])
        if r.status_code != 200:
            log.warning("Unsplash search %r: HTTP %s %s", q, r.status_code, r.text[:120])
            return None
        photos = [{"id": p["id"], "raw": p["urls"]["raw"], "alt": p.get("alt_description") or "",
                   "name": (p.get("user") or {}).get("name", ""), "profile": ((p.get("user") or {}).get("links") or {}).get("html", ""),
                   "dl": (p.get("links") or {}).get("download_location", "")}
                  for p in r.json().get("results", [])]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("Unsplash search %r failed: %s", q, e)
        return None
    log.info("Unsplash %r page %d: %d photos (%s searches left this hour)", q, page, len(photos), remaining)
    with _lock:
        cache = _load_cache()
        cache[ck] = {"t": time.time(), "photos": photos}
        _save_cache(cache)
    return photos


def _track(photos: list[dict]):
    """Unsplash asks apps to call download_location when a photo is used (counts the photographer's downloads)."""
    k = key()
    if not k or not photos:
        return

    def run():
        with httpx.Client(timeout=10) as c:
            for p in photos:
                if p.get("dl"):
                    try:
                        c.get(p["dl"], headers={"Authorization": f"Client-ID {k}"})
                    except httpx.HTTPError:
                        pass
    threading.Thread(target=run, daemon=True).start()


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*([\"'])(.*?)\1", tag, re.I | re.S)
    return m.group(2) if m else None


def _set_attr(tag: str, name: str, value: str) -> str:
    v = htmlmod.escape(value, quote=True)
    if _attr(tag, name) is not None:
        return re.sub(rf"\b{name}\s*=\s*([\"']).*?\1", f'{name}="{v}"', tag, count=1, flags=re.I | re.S)
    return re.sub(r"\s*/?>$", f' {name}="{v}">', tag, count=1)


def fallback(query: str) -> str:
    """No photo: the art library's picture, an emoji drawn as a little SVG, or a soft gradient."""
    words = _norm(query).split()
    for cand in [" ".join(words)] + words[::-1]:
        art = live.asset_for(cand)
        if art:
            return art
    emoji = getattr(live, "EMOJI", {})
    for w in words[::-1]:
        e = emoji.get(w) or emoji.get(w.rstrip("s"))
        if e:
            return ("data:image/svg+xml," + quote(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                    f'<rect width="100" height="100" fill="#eef2f7"/><text x="50" y="62" font-size="52" '
                    f'text-anchor="middle">{e}</text></svg>'))
    return ("data:image/svg+xml," + quote('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 3"><defs>'
            '<linearGradient id="g" x2="1" y2="1"><stop offset="0" stop-color="#dfe6ee"/><stop offset="1" '
            'stop-color="#b8c4d2"/></linearGradient></defs><rect width="4" height="3" fill="url(#g)"/></svg>'))


def _is_photo(src: str | None) -> bool:
    return bool(src) and "images.unsplash.com" in src


def _page_of(page: str, q: str) -> int:
    """How many photos of these words the page has already (they came 3 a search): the next search is the next page."""
    have = sum(1 for m in _IMG.finditer(page) if _norm(m.group(2)) == q and
               _is_photo(_attr(m.group(0), "src")))
    return math.ceil(have / PER_SEARCH)


def fill(page: str) -> tuple[str, dict]:
    """Put photos into every <img data-photo="..."> that has no real src yet. Returns (html, info)."""
    tags = list(_IMG.finditer(page))
    # every data-photo image without an Unsplash photo yet (no src, or a stand-in: library art, emoji, gradient)
    todo = [m for m in tags if not _is_photo(_attr(m.group(0), "src"))]
    info = {"images": len(todo), "photos": 0, "fallback": 0, "searches": 0, "cached": 0}
    if not todo:
        return page, info
    # the ones just asked for (no src at all) first; stand-ins from an earlier request get what is left of the 3
    todo.sort(key=lambda m: bool(_attr(m.group(0), "src")))
    groups: dict[tuple[str, str], list[re.Match]] = {}
    for m in todo:
        shape = _SHAPES.get((_attr(m.group(0), "data-shape") or "").lower(), "")
        groups.setdefault((_norm(m.group(2)), shape), []).append(m)
    new: dict[int, str] = {}                  # match start -> new tag
    used: list[dict] = []
    seen_ids: set[str] = set()
    with httpx.Client(timeout=10) as client:
        for (q, shape), ms in groups.items():
            pool: list[dict] = []
            if info["photos"] < MAX_PER_REQUEST:        # the request's 3 photos are used up: no more searches
                pg = 1 + _page_of(page, q)               # "add more photos of dresses": the next 3, not the same
                ck_before = f"{q}|{pg}|{shape}" in _load_cache()
                got = search(q, pg, shape, client)
                if got is not None:
                    info["cached" if ck_before else "searches"] += 1
                    pool = [p for p in got if p["id"] not in seen_ids and p["id"] not in page]
            pool = pool[:max(0, MAX_PER_REQUEST - info["photos"])]
            for i, m in enumerate(ms):
                tag = m.group(0)
                width = re.sub(r"\D", "", _attr(tag, "width") or "") or "800"
                if i < len(pool):
                    p = pool[i]
                    seen_ids.add(p["id"])
                    used.append(p)
                    sep = "&" if "?" in p["raw"] else "?"
                    tag = _set_attr(tag, "src", f"{p['raw']}{sep}w={min(1600, max(200, int(width) * 2))}&q=75&fit=crop&auto=format")
                    tag = _set_attr(tag, "data-photo-by", p["name"])
                    tag = _set_attr(tag, "data-photo-link", p["profile"] or "https://unsplash.com/")
                    tag = _set_attr(tag, "title", f"Photo by {p['name']} on Unsplash")
                    if not (_attr(tag, "alt") or "").strip():
                        tag = _set_attr(tag, "alt", p["alt"] or q)
                    info["photos"] += 1
                else:
                    tag = _set_attr(tag, "src", fallback(q))
                    info["fallback"] += 1
                if not _attr(tag, "loading"):
                    tag = _set_attr(tag, "loading", "lazy")
                new[m.start()] = tag
    out, last = [], 0
    for m in tags:
        if m.start() in new:
            out += [page[last:m.start()], new[m.start()]]
            last = m.end()
    page = "".join(out) + page[last:]
    page = credits(page)
    _track(used)
    log.info("photos: %s", info)
    return page, info


def credits(page: str) -> str:
    """'Photos by A, B and C on Unsplash' at the bottom of the page (Unsplash's attribution rule), from every photo
    on the page now (an edit keeps the earlier photos' credits)."""
    page = re.sub(r"\s*<div id=\"fda-photo-credits\".*?</div>", "", page, flags=re.S)
    names, seen = [], set()
    for m in re.finditer(r"<img\b[^>]*>", page, re.I | re.S):
        name = _attr(m.group(0), "data-photo-by")
        if not name or htmlmod.unescape(name) in seen:
            continue
        name = htmlmod.unescape(name)
        seen.add(name)
        link = htmlmod.unescape(_attr(m.group(0), "data-photo-link") or "https://unsplash.com/")
        link += ("&" if "?" in link else "?") + UTM
        names.append(f'<a href="{htmlmod.escape(link)}" target="_blank" rel="noopener">{htmlmod.escape(name)}</a>')
    if not names:
        return page
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    div = (f'<div id="fda-photo-credits" style="font:12px system-ui,sans-serif;opacity:.7;text-align:center;'
           f'padding:10px 12px">Photos by {who} on <a href="https://unsplash.com/?{UTM}" target="_blank" '
           f'rel="noopener">Unsplash</a></div>')
    i = page.lower().rfind("</body>")
    return page[:i] + div + "\n" + page[i:] if i >= 0 else page + div
