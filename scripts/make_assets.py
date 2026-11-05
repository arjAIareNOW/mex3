"""Builds Elkit's own art library: icons, game sprites, tiles and backgrounds as SVG files, drawn here in code.

    python scripts/make_assets.py          -> fda/assets/{icons,sprites,tiles,backgrounds}/*.svg, manifest.json, index.html

Everything is made by this script (no downloads, no licences to check), so it works offline and on the Pi.
The display serves it at /assets/ (gallery: http://localhost:8765/assets/index.html). Apps and games use it as
"/assets/sprites/bird.svg"; FDA.draw draws such a path, tiles it on long things, mirrors it with "|flip" and
tints it with "@red". Sprites face RIGHT (unlike most emoji, which face left).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fda" / "assets"
INK, WHITE = "#1f2937", "#ffffff"


def svg(w, h, body, extra=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"{extra}>'
            f'{body}</svg>\n')


# ------------------------------------------------------------------ icons (24 x 24, line style)
ICONS = {
    "home": '<path d="M3 11 12 3l9 8M5 10v10h5v-6h4v6h5V10"/>',
    "play": '<path d="M8 5v14l11-7z" fill="C"/>',
    "pause": '<path d="M7 5h3v14H7zM14 5h3v14h-3z" fill="C"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="C"/>',
    "next": '<path d="M6 5l9 7-9 7z" fill="C"/><path d="M18 5v14"/>',
    "previous": '<path d="M18 5 9 12l9 7z" fill="C"/><path d="M6 5v14"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "close": '<path d="M6 6l12 12M18 6 6 18"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7"/>',
    "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "grid": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/>'
            '<rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M16.5 16.5 21 21"/>',
    "star": '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    "heart": '<path d="M12 20s-7.5-4.8-7.5-10.2A4.3 4.3 0 0 1 12 7a4.3 4.3 0 0 1 7.5 2.8C19.5 15.2 12 20 12 20z"/>',
    "settings": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5"/>',
    "users": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c0-3.6 2.9-5.8 6.5-5.8s6.5 2.2 6.5 5.8"/>'
             '<path d="M15.5 4.6a3.5 3.5 0 0 1 0 6.8M18 14.6c2.1.7 3.5 2.6 3.5 5.4"/>',
    "music": '<path d="M9 18V5l11-2v13"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
    "volume": '<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12"/>',
    "mute": '<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M17 9l5 6M22 9l-5 6"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
           'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "cloud": '<path d="M7 18h10a4 4 0 0 0 .5-8 6 6 0 0 0-11.6 1.6A3.3 3.3 0 0 0 7 18z"/>',
    "rain": '<path d="M7 14h10a4 4 0 0 0 .5-8 6 6 0 0 0-11.6 1.6A3.3 3.3 0 0 0 7 14z"/><path d="M8 17l-1 3M12 17l-1 3M16 17l-1 3"/>',
    "lightbulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3z"/>',
    "thermometer": '<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/><path d="M12 9v7"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.2 2"/>',
    "alarm": '<circle cx="12" cy="13" r="7.5"/><path d="M12 9v4l2.5 1.5M4 5l3-2.5M20 5l-3-2.5"/>',
    "bell": '<path d="M6 16v-5a6 6 0 0 1 12 0v5l2 2H4z"/><path d="M10 20.5a2 2 0 0 0 4 0"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="15.5" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "cart": '<path d="M3 4h2.2l2.4 11h10.9L21 8H6.4"/><circle cx="9.5" cy="19.5" r="1.5"/><circle cx="17.5" cy="19.5" r="1.5"/>',
    "phone": '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2z"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3.5 7l8.5 6 8.5-6"/>',
    "chat": '<path d="M4 5h16v11H9.5L4 20z"/>',
    "camera": '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.8"/><path d="M3.5 18l5.5-5.5 4 4 3-3 4.5 4.5"/>',
    "trophy": '<path d="M8 4h8v5a4 4 0 0 1-8 0z"/><path d="M8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4M12 13v4M8 21h8M9.5 17h5"/>',
    "gamepad": '<rect x="2.5" y="7" width="19" height="11" rx="5.5"/><path d="M7 10.5v4M5 12.5h4"/>'
               '<circle cx="15.5" cy="11.5" r="1" fill="C"/><circle cx="18" cy="13.5" r="1" fill="C"/>',
    "book": '<path d="M4 4.5h6a2 2 0 0 1 2 2V20a2 2 0 0 0-2-2H4zM20 4.5h-6a2 2 0 0 0-2 2V20a2 2 0 0 1 2-2h6z"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
    "trash": '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"/>',
    "unlock": '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V7a4 4 0 0 1 7.7-1.5"/>',
    "map-pin": '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "warning": '<path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.5M12 17.2v.3"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    "download": '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
    "upload": '<path d="M12 20V9M7 13.5l5-5 5 5M5 4h14"/>',
    "share": '<circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/>'
             '<path d="M8.2 10.8l7.6-3.6M8.2 13.2l7.6 3.6"/>',
    "arrow-left": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "arrow-up": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "arrow-down": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "wifi": '<path d="M2.5 9a14 14 0 0 1 19 0M5.5 12.5a9.5 9.5 0 0 1 13 0M8.7 15.8a5 5 0 0 1 6.6 0"/>'
            '<circle cx="12" cy="19" r="1" fill="C"/>',
    "battery": '<rect x="2.5" y="7" width="17" height="10" rx="2"/><path d="M22 10.5v3"/><rect x="5" y="9.5" width="9" height="5" fill="C" stroke="none"/>',
    "power": '<path d="M12 3v9M6.3 6.3a8 8 0 1 0 11.4 0"/>',
    "fan": '<circle cx="12" cy="12" r="1.8"/><path d="M12 10.2C10 6 11 3 13.5 3.5 16 4 15 8 13.6 10.9M13.8 12.6c4.6.4 6.9 2.5 5.8 4.8'
           '-1.1 2.2-4.6.2-6.4-2.5M10.6 13.4c-2.6 3.8-5.6 4.6-6.8 2.4-1.2-2.2 2.2-4.4 5.4-4.6"/>',
    "tv": '<rect x="3" y="5" width="18" height="12" rx="2"/><path d="M8 21h8M12 17v4"/>',
    "door": '<path d="M6 21V3.5h12V21M3.5 21h17"/><circle cx="14.5" cy="12.5" r="1" fill="C"/>',
    "plant": '<path d="M12 21v-9M12 12c0-4-3-6-7-6 0 4 3 6 7 6zM12 15c0-4 3-6 7-6 0 4-3 6-7 6zM7.5 21h9"/>',
    "coffee": '<path d="M4 9h12v6a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5zM16 11h1.5a2.5 2.5 0 0 1 0 5H16"/><path d="M8 3c0 1.5 1.5 1.5 1.5 3M12 3c0 1.5 1.5 1.5 1.5 3"/>',
}


def icon_svg(body, color):
    body = body.replace('fill="C"', f'fill="{color}"')
    return svg(24, 24, body, f' fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"')


# ------------------------------------------------------------------ sprites (64 x 64, filled, facing right)
def eye(x, y, r=4.5, look=1.2):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            f'<circle cx="{x + look}" cy="{y}" r="{r * 0.45:.1f}" fill="{INK}"/>')


def star_points(cx, cy, r1, r2, n=5, rot=-90):
    pts = []
    for i in range(2 * n):
        r = r1 if i % 2 == 0 else r2
        a = math.radians(rot + i * 180 / n)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)


def crescent(c1, r1, c2, r2):
    """Path of circle 1 with circle 2 cut out of it (a moon)."""
    (x1, y1), (x2, y2) = c1, c2
    d = math.hypot(x2 - x1, y2 - y1)
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, r1 * r1 - a * a))
    mx, my = x1 + a * (x2 - x1) / d, y1 + a * (y2 - y1) / d
    px, py = -(y2 - y1) / d * h, (x2 - x1) / d * h
    p1, p2 = (mx + px, my + py), (mx - px, my - py)
    return (f"M{p1[0]:.1f} {p1[1]:.1f}A{r1} {r1} 0 1 1 {p2[0]:.1f} {p2[1]:.1f}"
            f"A{r2} {r2} 0 0 0 {p1[0]:.1f} {p1[1]:.1f}z")


O = f'stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"'
SPRITES = {
    "bird": (f'<ellipse cx="30" cy="34" rx="20" ry="17" fill="#ffd23f" {O}/>'
             f'<path d="M15 33c5-7 15-7 19 0-5 7-14 8-19 0z" fill="#f4a51c" {O}/>'
             f'<path d="M48 31l11 4-11 5z" fill="#ff7a1a" {O}/>' + eye(40, 27) +
             f'<path d="M10 29l-6-4 2 7" fill="#f4a51c" {O}/>', ["animal", "flappy", "flying"]),
    "fish": (f'<path d="M8 32c8-14 30-16 44 0-14 16-36 14-44 0z" fill="#3fa9f5" {O}/>'
             f'<path d="M8 32 1 22v20z" fill="#2b86c9" {O}/><path d="M28 20c4 4 4 20 0 24" fill="none" stroke="#2b86c9" stroke-width="2.5"/>'
             + eye(42, 29, 3.8), ["animal", "sea", "water"]),
    "cat": (f'<path d="M12 22 14 6l12 10h12L50 6l2 16v14c0 12-9 20-20 20S12 48 12 36z" fill="#f28c38" {O}/>'
            + eye(24, 32, 4, 0.8) + eye(40, 32, 4, 0.8) +
            f'<path d="M30 40h4l-2 3z" fill="#e0507a"/><path d="M32 43c-2 3-6 3-7 1M32 43c2 3 6 3 7 1" fill="none" stroke="{INK}" stroke-width="2"/>'
            f'<path d="M6 38h12M6 44l12-3M58 38H46M58 44l-12-3" stroke="{INK}" stroke-width="1.5"/>', ["animal", "pet"]),
    "frog": (f'<ellipse cx="32" cy="40" rx="24" ry="16" fill="#4caf50" {O}/>'
             f'<circle cx="20" cy="24" r="8" fill="#4caf50" {O}/><circle cx="44" cy="24" r="8" fill="#4caf50" {O}/>'
             + eye(20, 23, 4.2, 0.8) + eye(44, 23, 4.2, 0.8) +
             f'<path d="M20 42c6 6 18 6 24 0" fill="none" stroke="{INK}" stroke-width="2.5" stroke-linecap="round"/>', ["animal", "jump"]),
    "ghost": (f'<path d="M12 58V28a20 20 0 0 1 40 0v30l-7-6-6 6-7-6-6 6-7-6z" fill="#f4f6fb" {O}/>'
              f'<ellipse cx="25" cy="28" rx="4" ry="6" fill="{INK}"/><ellipse cx="40" cy="28" rx="4" ry="6" fill="{INK}"/>'
              f'<ellipse cx="32" cy="42" rx="5" ry="4" fill="{INK}"/>', ["enemy", "spooky", "halloween"]),
    "slime": (f'<path d="M6 54c0-20 10-36 26-36s26 16 26 36z" fill="#7ed957" {O}/>'
              f'<path d="M16 34c2-6 6-9 10-10" fill="none" stroke="#c9f5b0" stroke-width="4" stroke-linecap="round"/>'
              + eye(26, 38, 4.2) + eye(42, 38, 4.2), ["enemy", "blob"]),
    "bat": (f'<path d="M32 26c-6-10-18-12-28-4 6 0 8 6 6 10 4-3 8-2 10 2 2-4 6-6 12-8 6 2 10 4 12 8 2-4 6-5 10-2-2-4 0-10 6-10-10-8-22-6-28 4z" fill="#5b4b8a" {O}/>'
            f'<circle cx="32" cy="32" r="9" fill="#5b4b8a" {O}/><path d="M26 25l2-6 3 5M38 25l-2-6-3 5" fill="#5b4b8a" {O}/>'
            f'<circle cx="29" cy="31" r="1.8" fill="#ffd23f"/><circle cx="35" cy="31" r="1.8" fill="#ffd23f"/>', ["enemy", "flying", "night"]),
    "robot": (f'<rect x="14" y="16" width="36" height="28" rx="6" fill="#9aa9bf" {O}/>'
              f'<path d="M32 16V8" stroke="{INK}" stroke-width="2.5"/><circle cx="32" cy="7" r="3.5" fill="#ff5a5f" {O}/>'
              f'<rect x="20" y="23" width="24" height="11" rx="5" fill="#1f2937"/><circle cx="27" cy="28.5" r="3" fill="#5cf2ff"/>'
              f'<circle cx="38" cy="28.5" r="3" fill="#5cf2ff"/><rect x="18" y="44" width="28" height="12" rx="3" fill="#7d8ca3" {O}/>'
              f'<path d="M14 30H8M50 30h6" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>', ["character", "machine"]),
    "hero": (f'<circle cx="32" cy="16" r="10" fill="#f6c9a0" {O}/><path d="M22 14c2-8 18-8 20 0-6-3-14-3-20 0z" fill="#5a3a22" {O}/>'
             + f'<circle cx="35" cy="16" r="1.8" fill="{INK}"/>' +
             f'<rect x="21" y="26" width="22" height="18" rx="5" fill="#3d7bf2" {O}/>'
             f'<path d="M24 44l-3 14M40 44l3 14" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
             f'<path d="M21 30l-7 9M43 30l7 -9" stroke="#f6c9a0" stroke-width="5" stroke-linecap="round"/>', ["character", "player", "person"]),
    "rocket": (f'<path d="M32 4c10 8 14 22 12 36H20C18 26 22 12 32 4z" fill="#f4f6fb" {O}/>'
               f'<circle cx="32" cy="22" r="5.5" fill="#5cc8ff" {O}/><path d="M20 32 10 46h10zM44 32l10 14H44z" fill="#ff5a5f" {O}/>'
               f'<path d="M24 40h16l-3 6H27z" fill="#9aa9bf" {O}/><path d="M28 46c0 6 4 12 4 12s4-6 4-12z" fill="#ffb02e"/>', ["space", "vehicle", "up"]),
    "spaceship": (f'<path d="M6 32 22 20h20l16 12-16 12H22z" fill="#c9d3e3" {O}/><path d="M30 24h10l8 8H30z" fill="#5cc8ff" {O}/>'
                  f'<path d="M22 20 14 8h8l10 12M22 44l-8 12h8l10-12" fill="#ff5a5f" {O}/>'
                  f'<path d="M6 32H0" stroke="#ffb02e" stroke-width="5" stroke-linecap="round"/>', ["space", "vehicle", "shooter"]),
    "ufo": (f'<ellipse cx="32" cy="36" rx="28" ry="10" fill="#9aa9bf" {O}/><path d="M18 32a14 12 0 0 1 28 0z" fill="#7ee0ff" {O}/>'
            f'<circle cx="16" cy="38" r="2.5" fill="#ffd23f"/><circle cx="32" cy="41" r="2.5" fill="#ffd23f"/><circle cx="48" cy="38" r="2.5" fill="#ffd23f"/>', ["space", "alien", "enemy"]),
    "car": (f'<path d="M4 42v-8l8-2 8-10h20l10 10 8 2v8z" fill="#ff5a5f" {O}/><path d="M23 24h7v8H16zM34 24h5l7 8H34z" fill="#bfe9ff" {O}/>'
            f'<circle cx="17" cy="44" r="7" fill="{INK}"/><circle cx="17" cy="44" r="3" fill="#c9d3e3"/>'
            f'<circle cx="47" cy="44" r="7" fill="{INK}"/><circle cx="47" cy="44" r="3" fill="#c9d3e3"/>'
            f'<rect x="56" y="34" width="4" height="4" fill="#ffd23f"/>', ["vehicle", "racing", "road"]),
    "coin": (f'<circle cx="32" cy="32" r="24" fill="#ffc83d" {O}/><circle cx="32" cy="32" r="17" fill="none" stroke="#e09f12" stroke-width="3"/>'
             f'<path d="M32 22v20" stroke="#e09f12" stroke-width="5" stroke-linecap="round"/>'
             f'<path d="M20 22a14 14 0 0 1 8-6" fill="none" stroke="#fff3c4" stroke-width="3" stroke-linecap="round"/>', ["item", "money", "score"]),
    "gem": (f'<path d="M18 12h28l12 14-26 30L6 26z" fill="#4dd2ff" {O}/><path d="M6 26h52M18 12l8 14 6-14 6 14 8-14M26 26l6 30 6-30" fill="none" stroke="#1a8fc2" stroke-width="2"/>',
            ["item", "treasure", "diamond", "jewel"]),
    "heart": (f'<path d="M32 56S6 40 6 22A13 13 0 0 1 32 15a13 13 0 0 1 26 7c0 18-26 34-26 34z" fill="#ff4d6d" {O}/>'
              f'<path d="M15 21a6 6 0 0 1 6-5" fill="none" stroke="#ffc2cf" stroke-width="3.5" stroke-linecap="round"/>', ["item", "life", "health", "love"]),
    "star": (f'<polygon points="{star_points(32, 34, 27, 12)}" fill="#ffd23f" {O}/>', ["item", "score", "bonus"]),
    "key": (f'<circle cx="18" cy="32" r="11" fill="#ffc83d" {O}/><circle cx="18" cy="32" r="4" fill="#fff8e1" {O}/>'
            f'<path d="M29 29h28v6h-5v7h-6v-7h-4v5h-6v-5h-7z" fill="#ffc83d" {O}/>', ["item", "unlock", "door"]),
    "chest": (f'<path d="M8 28a24 14 0 0 1 48 0v4H8z" fill="#b5651d" {O}/><rect x="8" y="30" width="48" height="24" rx="2" fill="#c97a2b" {O}/>'
              f'<path d="M8 30h48" stroke="#e8b04a" stroke-width="4"/><rect x="27" y="27" width="10" height="13" rx="2" fill="#ffc83d" {O}/>', ["item", "treasure", "loot"]),
    "bomb": (f'<circle cx="28" cy="38" r="20" fill="#2f3542" {O}/><rect x="36" y="14" width="10" height="10" rx="2" transform="rotate(45 41 19)" fill="#57606f" {O}/>'
             f'<path d="M44 14c4-6 10-6 12-2" fill="none" stroke="#a4703a" stroke-width="2.5"/><polygon points="{star_points(57, 9, 7, 3, 6)}" fill="#ffb02e"/>'
             f'<path d="M17 30a12 12 0 0 1 6-6" fill="none" stroke="#a4b0be" stroke-width="3.5" stroke-linecap="round"/>', ["enemy", "danger", "explosive"]),
    "fireball": (f'<path d="M4 32c8-4 16-14 30-14 12 0 22 6 22 14s-10 14-22 14C20 46 12 36 4 32z" fill="#ff7a1a" {O}/>'
                 f'<path d="M20 32c6-2 10-8 16-8 7 0 12 4 12 8s-5 8-12 8c-6 0-10-6-16-8z" fill="#ffd23f"/>', ["shot", "fire", "danger"]),
    "laser": (f'<rect x="4" y="27" width="56" height="10" rx="5" fill="#ff2d75"/><rect x="8" y="30" width="48" height="4" rx="2" fill="#ffd1e1"/>', ["shot", "space", "shooter"]),
    "asteroid": (f'<path d="M14 18 30 8l18 6 10 16-4 18-16 10-20-4L6 38z" fill="#8d7b6a" {O}/>'
                 f'<circle cx="24" cy="26" r="5" fill="#6f5f50"/><circle cx="40" cy="38" r="7" fill="#6f5f50"/><circle cx="42" cy="20" r="3" fill="#6f5f50"/>', ["space", "rock", "obstacle"]),
    "rock": (f'<path d="M6 52 14 30l14-10 16 4 12 14 4 14z" fill="#9e9e9e" {O}/><path d="M20 34l8-6 6 2" fill="none" stroke="#d6d6d6" stroke-width="3" stroke-linecap="round"/>',
             ["obstacle", "stone"]),
    "spike": (f'<path d="M2 58 12 26l10 32 10-32 10 32 10-32 10 32z" fill="#c9d3e3" {O}/>', ["obstacle", "danger", "trap"]),
    "cloud": (f'<path d="M14 46h36a11 11 0 0 0 1-22 15 15 0 0 0-28-4A12 12 0 0 0 14 46z" fill="#ffffff" {O}/>', ["sky", "weather"]),
    "tree": (f'<rect x="27" y="40" width="10" height="20" fill="#8d5a2b" {O}/><circle cx="32" cy="26" r="20" fill="#3ca55c" {O}/>'
             f'<circle cx="24" cy="22" r="5" fill="#5cc87a"/>', ["nature", "forest", "obstacle"]),
    "pine": (f'<rect x="28" y="48" width="8" height="12" fill="#8d5a2b" {O}/><path d="M32 4 50 30H40l12 18H12l12-18H14z" fill="#2e8b57" {O}/>', ["nature", "forest", "snow"]),
    "bush": (f'<path d="M6 54a12 12 0 0 1 8-20 14 14 0 0 1 26-6 12 12 0 0 1 18 14 8 8 0 0 1 0 12z" fill="#4caf50" {O}/>', ["nature", "plant"]),
    "flower": (f'<path d="M32 36v22" stroke="#3ca55c" stroke-width="4"/><path d="M32 50c6-8 14-8 16-6-4 6-10 8-16 6z" fill="#3ca55c"/>'
               + "".join(f'<circle cx="{32 + 9 * math.cos(math.radians(a)):.1f}" cy="{22 + 9 * math.sin(math.radians(a)):.1f}" r="7" fill="#ff8fab" {O}/>'
                         for a in range(0, 360, 72)) + f'<circle cx="32" cy="22" r="6" fill="#ffd23f" {O}/>', ["nature", "plant", "garden"]),
    "mushroom": (f'<path d="M24 36h16v18a4 4 0 0 1-4 4h-8a4 4 0 0 1-4-4z" fill="#fbe7c6" {O}/>'
                 f'<path d="M6 38a26 26 0 0 1 52 0z" fill="#e63946" {O}/><circle cx="22" cy="26" r="5" fill="#fff"/><circle cx="40" cy="22" r="4" fill="#fff"/><circle cx="46" cy="32" r="3" fill="#fff"/>', ["item", "power-up", "nature"]),
    "apple": (f'<path d="M32 20c-10-6-24 0-24 16 0 14 10 24 18 22 3-1 9-1 12 0 8 2 18-8 18-22 0-16-14-22-24-16z" fill="#e63946" {O}/>'
              f'<path d="M32 20c0-6 2-10 6-12" fill="none" stroke="#6b3e1f" stroke-width="3"/><path d="M34 14c6-6 12-4 14-2-6 4-10 4-14 2z" fill="#3ca55c"/>', ["food", "fruit", "item"]),
    "cherry": (f'<path d="M22 40C24 26 30 14 44 8M42 42c0-14 0-24 2-34" fill="none" stroke="#3ca55c" stroke-width="3"/>'
               f'<circle cx="20" cy="46" r="12" fill="#d00045" {O}/><circle cx="44" cy="46" r="12" fill="#d00045" {O}/>', ["food", "fruit", "item"]),
    "flag": (f'<path d="M14 6v54" stroke="{INK}" stroke-width="4" stroke-linecap="round"/><path d="M16 8h36l-8 10 8 10H16z" fill="#ff5a5f" {O}/>', ["goal", "finish", "level"]),
    "sun": (f'<circle cx="32" cy="32" r="14" fill="#ffd23f" {O}/>' + "".join(
        f'<path d="M{32 + 19 * math.cos(math.radians(a)):.1f} {32 + 19 * math.sin(math.radians(a)):.1f}L{32 + 28 * math.cos(math.radians(a)):.1f} {32 + 28 * math.sin(math.radians(a)):.1f}" stroke="#ffb02e" stroke-width="4" stroke-linecap="round"/>'
        for a in range(0, 360, 45)), ["sky", "weather", "day"]),
    "moon": (f'<path d="{crescent((28, 33), 23, (42, 24), 19)}" fill="#fff3b0" {O}/>'
             f'<circle cx="17" cy="32" r="3" fill="#e9d98a"/><circle cx="25" cy="46" r="4" fill="#e9d98a"/>',
             ["sky", "night"]),
    "pipe": (f'<rect x="10" y="0" width="44" height="64" fill="#3cb043" stroke="{INK}" stroke-width="2.5"/>'
             f'<rect x="16" y="0" width="7" height="64" fill="#7be07f"/><rect x="44" y="0" width="5" height="64" fill="#2a8a31"/>', ["obstacle", "flappy", "tile"]),
    "crate": (f'<rect x="6" y="6" width="52" height="52" rx="3" fill="#c98b4a" {O}/><path d="M6 6l52 52M58 6 6 58" stroke="#8d5a2b" stroke-width="4"/>'
              f'<rect x="6" y="6" width="52" height="52" rx="3" fill="none" stroke="#8d5a2b" stroke-width="6"/>', ["box", "obstacle", "platform"]),
}

FISH = "M8 32c8-14 30-16 44 0-14 16-36 14-44 0z"
SPRITES.update({
    # ---- more birds
    "parrot": (f'<path d="M17 37 3 53l15-9z" fill="#2b86c9" {O}/><path d="M16 33 2 39l14 3z" fill="#ffd23f" {O}/>'
               f'<ellipse cx="31" cy="34" rx="18" ry="16" fill="#e63946" {O}/>'
               f'<path d="M17 34c4-8 15-8 19 0-5 8-14 9-19 0z" fill="#3ca55c" {O}/>'
               f'<circle cx="40" cy="26" r="6.5" fill="#fff"/><circle cx="41.5" cy="26" r="2.8" fill="{INK}"/>'
               f'<path d="M46 23c8-2 13 4 11 12-3-3-7-4-11-3z" fill="#ffd23f" {O}/>', ["animal", "bird", "flying"]),
    "owl": (f'<path d="M13 22 15 5l11 11zM51 22 49 5 38 16z" fill="#8d6e4e" {O}/>'
            f'<ellipse cx="32" cy="36" rx="20" ry="22" fill="#8d6e4e" {O}/>'
            f'<circle cx="24" cy="28" r="9" fill="#e8d5b5"/><circle cx="40" cy="28" r="9" fill="#e8d5b5"/>'
            + eye(24, 28, 5.5, 1.2) + eye(40, 28, 5.5, 1.2) +
            f'<path d="M30 34h5l-2.5 5z" fill="#ffb02e" {O}/>'
            f'<path d="M23 46l3 3 3-3M35 46l3 3 3-3M29 52l3 3 3-3" fill="none" stroke="#e8d5b5" stroke-width="2.5" stroke-linecap="round"/>'
            f'<path d="M26 58h4M34 58h4" stroke="#ffb02e" stroke-width="4" stroke-linecap="round"/>', ["animal", "bird", "night"]),
    "penguin": (f'<path d="M16 32c-7 6-7 15-2 18 4-6 5-11 2-18z" fill="#263238" {O}/>'
                f'<ellipse cx="32" cy="34" rx="18" ry="24" fill="#263238" {O}/><ellipse cx="35" cy="39" rx="11" ry="17" fill="#fff"/>'
                + eye(38, 21, 4.2, 1.2) +
                f'<path d="M45 23l11 3-11 4z" fill="#ffb02e" {O}/>'
                f'<ellipse cx="26" cy="58" rx="6" ry="3" fill="#ffb02e" {O}/><ellipse cx="39" cy="58" rx="6" ry="3" fill="#ffb02e" {O}/>',
                ["animal", "bird", "snow", "cold"]),
    "duck": (f'<path d="M8 38 2 27l12 6z" fill="#ffffff" {O}/><ellipse cx="27" cy="40" rx="21" ry="14" fill="#ffffff" {O}/>'
             f'<circle cx="43" cy="23" r="11" fill="#ffffff" {O}/>'
             f'<path d="M15 39c5-7 15-7 18 0-5 6-13 7-18 0z" fill="#e6edf5" {O}/>'
             f'<path d="M52 21h10c0 5-4 7-10 6z" fill="#ff8c1a" {O}/>' + eye(46, 20, 3.6, 0.8), ["animal", "bird", "water"]),
    # ---- more fish
    "clownfish": (f'<defs><clipPath id="b"><path d="{FISH}"/></clipPath></defs>'
                  f'<path d="M8 32 1 22v20z" fill="#ff7a1a" {O}/><path d="{FISH}" fill="#ff7a1a"/>'
                  f'<g clip-path="url(#b)" stroke="#ffffff" stroke-width="6"><path d="M22 10v44M38 10v44"/></g>'
                  f'<g clip-path="url(#b)" stroke="{INK}" stroke-width="1.5" fill="none"><path d="M18.5 10v44M25.5 10v44M34.5 10v44M41.5 10v44"/></g>'
                  f'<path d="{FISH}" fill="none" {O}/>' + eye(45, 29, 3.4), ["animal", "fish", "sea"]),
    "pufferfish": ("".join(f'<path d="M{30 + 19 * math.cos(math.radians(a - 8)):.1f} {32 + 19 * math.sin(math.radians(a - 8)):.1f}'
                           f'L{30 + 27 * math.cos(math.radians(a)):.1f} {32 + 27 * math.sin(math.radians(a)):.1f}'
                           f'L{30 + 19 * math.cos(math.radians(a + 8)):.1f} {32 + 19 * math.sin(math.radians(a + 8)):.1f}z" '
                           f'fill="#f4b400" {O}/>' for a in range(15, 360, 30) if not 150 < a < 210) +
                   f'<path d="M12 32 3 24v16z" fill="#f4b400" {O}/>'
                   f'<circle cx="30" cy="32" r="20" fill="#ffd23f" {O}/><path d="M16 40c8 8 20 8 28 0" fill="#fff3c4"/>'
                   + eye(38, 26, 4.5) + f'<circle cx="49" cy="35" r="2.5" fill="none" stroke="{INK}" stroke-width="2"/>',
                   ["animal", "fish", "sea"]),
    "shark": (f'<path d="M26 22 33 6l9 18z" fill="#8fa3b8" {O}/><path d="M8 33 1 20v26z" fill="#8fa3b8" {O}/>'
              f'<path d="M4 33c10-12 34-16 56-2-14 10-40 12-56 2z" fill="#8fa3b8" {O}/>'
              f'<path d="M14 37c14 6 30 6 44-4-12 2-30 6-44 4z" fill="#ffffff"/>'
              f'<path d="M24 41l6 8 5-7z" fill="#8fa3b8" {O}/>'
              f'<circle cx="47" cy="29" r="2.3" fill="{INK}"/><path d="M36 30v6M39 29v7" stroke="#5f7186" stroke-width="2" stroke-linecap="round"/>'
              f'<path d="M48 36l2 2 2-2 2 2" fill="none" stroke="{INK}" stroke-width="1.5"/>', ["animal", "fish", "sea", "enemy"]),
    # ---- more cars
    "race-car": (f'<path d="M3 30h9v8H3z" fill="#1f2937" {O}/>'
                 f'<path d="M3 44v-6l13-4h17l7-7h11l10 9v8z" fill="#3d7bf2" {O}/><path d="M42 29h8l6 6H40z" fill="#bfe9ff" {O}/>'
                 f'<path d="M14 39h40" stroke="#ffffff" stroke-width="3"/><circle cx="28" cy="38" r="5" fill="#fff" {O}/>'
                 f'<circle cx="15" cy="45" r="7" fill="{INK}"/><circle cx="15" cy="45" r="3" fill="#c9d3e3"/>'
                 f'<circle cx="50" cy="45" r="7" fill="{INK}"/><circle cx="50" cy="45" r="3" fill="#c9d3e3"/>', ["vehicle", "car", "racing"]),
    "truck": (f'<rect x="2" y="14" width="35" height="29" rx="2" fill="#ffd23f" {O}/>'
              f'<path d="M38 22h14l9 11v10H38z" fill="#ff5a5f" {O}/><path d="M42 25h9l6 8H42z" fill="#bfe9ff" {O}/>'
              f'<rect x="58" y="36" width="4" height="4" fill="#fff3b0"/>'
              + "".join(f'<circle cx="{x}" cy="46" r="6.5" fill="{INK}"/><circle cx="{x}" cy="46" r="2.8" fill="#c9d3e3"/>' for x in (12, 27, 50)),
              ["vehicle", "car", "road"]),
    "bus": (f'<rect x="2" y="12" width="59" height="34" rx="6" fill="#ffc83d" {O}/>'
            + "".join(f'<rect x="{x}" y="18" width="10" height="10" rx="2" fill="#bfe9ff" {O}/>' for x in (7, 20, 33)) +
            f'<path d="M47 18h7a4 4 0 0 1 4 4v8H47z" fill="#bfe9ff" {O}/><path d="M2 34h59" stroke="#e09f12" stroke-width="3"/>'
            + "".join(f'<circle cx="{x}" cy="47" r="6.5" fill="{INK}"/><circle cx="{x}" cy="47" r="2.8" fill="#c9d3e3"/>' for x in (15, 48)),
            ["vehicle", "car", "road", "city"]),
    # ---- more heroes
    "girl": (f'<path d="M21 17c0-13 22-13 22 0v13c-3 0-4-2-4-4H25c0 2-1 4-4 4z" fill="#5a3a22" {O}/>'
             f'<circle cx="32" cy="16" r="10" fill="#f6c9a0" {O}/><path d="M22 14c2-9 18-9 20 0-6-4-14-4-20 0z" fill="#5a3a22" {O}/>'
             f'<circle cx="35" cy="16" r="1.8" fill="{INK}"/>'
             f'<path d="M24 46l-2 12M40 46l2 12" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
             f'<path d="M25 26h14l7 21H18z" fill="#e0507a" {O}/>'
             f'<path d="M24 30l-8 8M40 30l7-9" stroke="#f6c9a0" stroke-width="5" stroke-linecap="round"/>', ["character", "player", "person"]),
    "knight": (f'<path d="M10 28h12v10c0 7-6 10-6 10s-6-3-6-10z" fill="#3d7bf2" {O}/>'
               f'<path d="M24 45l-2 13M36 45l2 13" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>'
               f'<rect x="20" y="27" width="20" height="19" rx="4" fill="#9aa9bf" {O}/>'
               f'<path d="M26 5c3-5 11-4 12 1" fill="none" stroke="#e63946" stroke-width="4" stroke-linecap="round"/>'
               f'<circle cx="30" cy="16" r="11" fill="#c9d3e3" {O}/><path d="M29 15h11" stroke="{INK}" stroke-width="3.5" stroke-linecap="round"/>'
               f'<rect x="45" y="6" width="5" height="28" rx="1" fill="#e6edf5" {O}/><rect x="40" y="33" width="15" height="4" rx="1" fill="#ffc83d" {O}/>'
               f'<path d="M47.5 37v7" stroke="#8d5a2b" stroke-width="4"/>', ["character", "player", "person", "castle"]),
    "astronaut": (f'<rect x="14" y="28" width="9" height="16" rx="3" fill="#9aa9bf" {O}/>'
                  f'<path d="M26 45l-1 13M38 45l1 13" stroke="{INK}" stroke-width="10" stroke-linecap="round"/>'
                  f'<path d="M26 45l-1 13M38 45l1 13" stroke="#e6edf5" stroke-width="5.5" stroke-linecap="round"/>'
                  f'<rect x="21" y="28" width="22" height="19" rx="5" fill="#f4f6fb" {O}/><rect x="28" y="33" width="8" height="6" rx="1" fill="#ff5a5f"/>'
                  f'<path d="M43 32l7-6" stroke="{INK}" stroke-width="9" stroke-linecap="round"/>'
                  f'<path d="M43 32l7-6" stroke="#f4f6fb" stroke-width="4.5" stroke-linecap="round"/>'
                  f'<circle cx="32" cy="16" r="13" fill="#f4f6fb" {O}/><ellipse cx="35" cy="16" rx="8" ry="7" fill="#5cc8ff" {O}/>'
                  f'<path d="M31 13a4 3 0 0 1 4-2" fill="none" stroke="#e6f7ff" stroke-width="2" stroke-linecap="round"/>', ["character", "player", "space"]),
    # ---- more flyers and pets
    "airplane": (f'<path d="M7 29 4 14h8l9 15z" fill="#ff5a5f" {O}/>'
                 f'<path d="M4 34c0-4 4-6 10-6h34c6 0 12 3 14 6-2 3-8 6-14 6H14c-6 0-10-2-10-6z" fill="#f4f6fb" {O}/>'
                 + "".join(f'<circle cx="{x}" cy="32" r="2" fill="#5cc8ff"/>' for x in (20, 26, 32, 38, 44)) +
                 f'<path d="M51 29c4 0 8 2 10 5H51z" fill="#5cc8ff" {O}/><path d="M24 36l10 15h8l-4-15z" fill="#9aa9bf" {O}/>',
                 ["vehicle", "flying", "sky"]),
    "dog": (f'<ellipse cx="14" cy="28" rx="7" ry="15" transform="rotate(15 14 28)" fill="#8d5a2b" {O}/>'
            f'<ellipse cx="50" cy="28" rx="7" ry="15" transform="rotate(-15 50 28)" fill="#8d5a2b" {O}/>'
            f'<ellipse cx="32" cy="32" rx="18" ry="20" fill="#c98b4a" {O}/><ellipse cx="32" cy="43" rx="10" ry="8" fill="#f6e3c6" {O}/>'
            + eye(25, 28, 4, 0.8) + eye(39, 28, 4, 0.8) +
            f'<ellipse cx="32" cy="38" rx="4" ry="3" fill="{INK}"/><path d="M32 41v3M28 45q4 3 8 0" fill="none" stroke="{INK}" stroke-width="2"/>'
            f'<path d="M30 46h4v4a2 2 0 0 1-4 0z" fill="#ff8fab"/>', ["animal", "pet"]),
    "rabbit": (f'<ellipse cx="24" cy="15" rx="6" ry="14" fill="#f4f6fb" {O}/><ellipse cx="40" cy="15" rx="6" ry="14" fill="#f4f6fb" {O}/>'
               f'<ellipse cx="24" cy="16" rx="2.5" ry="9" fill="#ffc2cf"/><ellipse cx="40" cy="16" rx="2.5" ry="9" fill="#ffc2cf"/>'
               f'<ellipse cx="32" cy="41" rx="18" ry="16" fill="#f4f6fb" {O}/>' + eye(25, 37, 3.8, 0.8) + eye(39, 37, 3.8, 0.8) +
               f'<path d="M30 44h4l-2 2.5z" fill="#ff8fab"/><path d="M30 49h4v4h-4z" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
               f'<path d="M6 44h14M8 50l12-3M58 44H44M56 50l-12-3" stroke="{INK}" stroke-width="1.3"/>', ["animal", "pet"]),
})

# Choice groups: "show me other birds" offers these (the first four the pop-up shows)
GROUPS = {
    "bird": ["bird", "parrot", "owl", "penguin", "duck"],
    "fish": ["fish", "clownfish", "pufferfish", "shark"],
    "car": ["car", "race-car", "truck", "bus"],
    "hero": ["hero", "girl", "knight", "astronaut", "robot"],
    "pet": ["cat", "dog", "rabbit", "frog"],
    "enemy": ["slime", "ghost", "bat", "ufo", "bomb"],
    "flyer": ["airplane", "rocket", "spaceship", "ufo"],
    "treasure": ["coin", "gem", "star", "heart", "key", "chest"],
    "plant": ["tree", "pine", "bush", "flower", "mushroom"],
    "obstacle": ["rock", "spike", "crate", "asteroid", "pipe"],
    "food": ["apple", "cherry", "mushroom"],
    "sky": ["sky-day", "sky-night", "sunset", "space"],
    "scene": ["forest", "desert", "snow", "city", "sea", "underwater", "room"],
    "ground": ["grass", "dirt", "sand", "stone", "brick", "ice", "wood", "water", "lava", "checker"],
}

# How to say a name aloud (default: the name with "-" as a space)
SAY = {"sky-day": "day sky", "sky-night": "night sky", "plain-dark": "dark", "plain-light": "light",
       "arrow-left": "left arrow", "arrow-right": "right arrow", "arrow-up": "up arrow", "arrow-down": "down arrow",
       "map-pin": "map pin", "race-car": "race car"}

# Colour schemes for websites and apps: page background, cards, main colour, accent, text
PALETTES = {
    "ocean": ["#f0f7ff", "#ffffff", "#1e6fd9", "#00b3a4", "#0f2a44"],
    "sunset": ["#fff5ee", "#ffffff", "#e8553d", "#ffb02e", "#3b1f1a"],
    "forest": ["#f2f7f1", "#ffffff", "#2e7d4f", "#c9a227", "#1d2b22"],
    "midnight": ["#0f172a", "#1e293b", "#6366f1", "#22d3ee", "#e2e8f0"],
    "candy": ["#fff0f6", "#ffffff", "#e64980", "#7950f2", "#3b1730"],
    "coffee": ["#f7f1ea", "#fffaf5", "#6f4e37", "#d4a373", "#2d1f16"],
    "mono": ["#f8f9fa", "#ffffff", "#212529", "#868e96", "#212529"],
    "neon": ["#0b0b12", "#161625", "#ff2d75", "#39ff14", "#f5f5ff"],
    "lavender": ["#f6f4fe", "#ffffff", "#7c5cff", "#ff9f68", "#2a2340"],
    "citrus": ["#fffdf0", "#ffffff", "#f59f00", "#37b24d", "#2b2a1f"],
}
PALETTE_KEYS = ["background", "surface", "primary", "accent", "text"]

# ------------------------------------------------------------------ tiles (64 x 64, repeat seamlessly)
TILES = {
    "grass": ('<rect width="64" height="64" fill="#8b5a2b"/><rect width="64" height="18" fill="#4caf50"/>'
              '<path d="M0 18q4 8 8 0t8 0 8 0 8 0 8 0 8 0 8 0 8 0v4H0z" fill="#4caf50"/>'
              '<circle cx="14" cy="38" r="3" fill="#6f4520"/><circle cx="44" cy="50" r="4" fill="#6f4520"/><circle cx="52" cy="32" r="2" fill="#6f4520"/>',
              ["ground", "platform"]),
    "dirt": ('<rect width="64" height="64" fill="#8b5a2b"/><circle cx="14" cy="18" r="3" fill="#6f4520"/><circle cx="44" cy="40" r="4" fill="#6f4520"/>'
             '<circle cx="26" cy="52" r="2.5" fill="#6f4520"/><circle cx="54" cy="12" r="2" fill="#6f4520"/>', ["ground"]),
    "brick": ('<rect width="64" height="64" fill="#b5462e"/><g stroke="#e8c9a8" stroke-width="3">'
              '<path d="M0 16h64M0 32h64M0 48h64M0 64h64"/><path d="M32 0v16M16 16v16M48 16v16M32 32v16M16 48v16M48 48v16"/></g>', ["wall", "platform"]),
    "stone": ('<rect width="64" height="64" fill="#8e9aa6"/><g fill="#a9b4bf" stroke="#6b7682" stroke-width="2">'
              '<path d="M2 2h28v20H2zM34 2h28v28H34zM2 26h28v36H2zM34 34h28v28H34z"/></g>', ["wall", "castle", "platform"]),
    "water": ('<rect width="64" height="64" fill="#2b86c9"/><path d="M0 16q8-6 16 0t16 0 16 0 16 0M0 40q8-6 16 0t16 0 16 0 16 0" fill="none" stroke="#7cc4f5" stroke-width="3"/>',
              ["sea", "liquid"]),
    "sand": ('<rect width="64" height="64" fill="#f2d28b"/><circle cx="10" cy="14" r="1.5" fill="#d9b464"/><circle cx="40" cy="22" r="1.5" fill="#d9b464"/>'
             '<circle cx="24" cy="44" r="1.5" fill="#d9b464"/><circle cx="54" cy="50" r="1.5" fill="#d9b464"/><path d="M4 32q12-6 24 0M36 58q12-6 24 0" fill="none" stroke="#e3bd6c" stroke-width="2"/>',
             ["ground", "desert", "beach"]),
    "ice": ('<rect width="64" height="64" fill="#bfe9ff"/><path d="M8 52 26 12M30 60 48 20M50 30l10-20" stroke="#ffffff" stroke-width="3" opacity=".8"/>'
            '<rect width="64" height="64" fill="none" stroke="#8fd3f5" stroke-width="4"/>', ["snow", "winter", "platform"]),
    "wood": ('<rect width="64" height="64" fill="#c98b4a"/><path d="M0 16h64M0 32h64M0 48h64" stroke="#8d5a2b" stroke-width="3"/>'
             '<path d="M20 0v16M44 16v16M12 32v16M36 48v16" stroke="#8d5a2b" stroke-width="2"/>', ["floor", "platform"]),
    "lava": ('<rect width="64" height="64" fill="#e63e1f"/><path d="M0 20q8-8 16 0t16 0 16 0 16 0" fill="none" stroke="#ffb02e" stroke-width="5"/>'
             '<circle cx="18" cy="44" r="5" fill="#ffb02e"/><circle cx="46" cy="50" r="3.5" fill="#ffd23f"/>', ["danger", "fire"]),
    "checker": ('<rect width="64" height="64" fill="#f4f6fb"/><path d="M0 0h32v32H0zM32 32h32v32H32z" fill="#1f2937"/>', ["finish", "racing", "floor"]),
}

# ------------------------------------------------------------------ backgrounds (800 x 480)
def grad(id_, top, bottom):
    return (f'<defs><linearGradient id="{id_}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{top}"/>'
            f'<stop offset="1" stop-color="{bottom}"/></linearGradient></defs><rect width="800" height="480" fill="url(#{id_})"/>')


def cloud(x, y, s=1.0, fill="#ffffff", op=0.95):
    return (f'<g transform="translate({x} {y}) scale({s})" fill="{fill}" opacity="{op}">'
            '<ellipse cx="0" cy="0" rx="46" ry="22"/><ellipse cx="34" cy="-12" rx="34" ry="26"/>'
            '<ellipse cx="70" cy="2" rx="40" ry="20"/></g>')


def stars(n, seed, h=480, color="#ffffff"):
    import random
    r = random.Random(seed)
    return "".join(f'<circle cx="{r.uniform(0, 800):.0f}" cy="{r.uniform(0, h):.0f}" r="{r.choice([0.8, 1.2, 1.8])}" '
                   f'fill="{color}" opacity="{r.uniform(.5, 1):.2f}"/>' for _ in range(n))


def hills(y, color, amp, seed, n=6):
    import random
    r = random.Random(seed)
    pts = [f"M0 {y}"]
    step = 800 / n
    for i in range(n):
        x1, x2 = i * step + step / 2, (i + 1) * step
        pts.append(f"Q{x1:.0f} {y - r.uniform(.4, 1) * amp:.0f} {x2:.0f} {y:.0f}")
    return f'<path d="{" ".join(pts)} V480 H0z" fill="{color}"/>'


def trees_row(y, color, seed, n=14, size=1.0):
    import random
    r = random.Random(seed)
    out = []
    for _ in range(n):
        x, s = r.uniform(0, 800), size * r.uniform(.7, 1.2)
        out.append(f'<path d="M{x:.0f} {y - 70 * s:.0f}l{26 * s:.0f} {50 * s:.0f}h{-12 * s:.0f}l{18 * s:.0f} {26 * s:.0f}'
                   f'h{-64 * s:.0f}l{18 * s:.0f} {-26 * s:.0f}h{-12 * s:.0f}z" fill="{color}"/>')
    return "".join(out)


def skyline(y, color, seed):
    import random
    r = random.Random(seed)
    x, out = 0, []
    while x < 800:
        w, h = r.uniform(40, 90), r.uniform(60, 220)
        out.append(f'<rect x="{x:.0f}" y="{y - h:.0f}" width="{w:.0f}" height="{h + 2:.0f}" fill="{color}"/>')
        for wy in range(int(y - h + 12), int(y - 10), 20):
            for wx in range(int(x + 8), int(x + w - 10), 16):
                if r.random() < .45:
                    out.append(f'<rect x="{wx}" y="{wy}" width="7" height="9" fill="#ffd96a" opacity=".85"/>')
        x += w + r.uniform(2, 10)
    return "".join(out)


BACKGROUNDS = {
    "sky-day": (grad("g", "#5fb8ff", "#cdeeff") + cloud(110, 110) + cloud(420, 70, .8) + cloud(640, 150, 1.1) +
                hills(430, "#7ccf6a", 70, 1) + '<rect y="430" width="800" height="50" fill="#5bb04f"/>', ["day", "flappy", "outdoor"]),
    "sky-night": (grad("g", "#0b1340", "#2b3a7a") + stars(110, 2, 360) +
                  '<circle cx="640" cy="90" r="38" fill="#fff3b0"/><circle cx="625" cy="80" r="7" fill="#e9d98a"/>' +
                  hills(440, "#1c2a55", 60, 3), ["night", "stars", "outdoor"]),
    "sunset": (grad("g", "#ff7e5f", "#feb47b") + '<circle cx="400" cy="360" r="90" fill="#ffd23f" opacity=".9"/>' +
               cloud(120, 120, .9, "#ffd6c0", .8) + cloud(560, 90, 1, "#ffd6c0", .8) +
               hills(410, "#7a3b5c", 80, 4) + hills(450, "#4f2443", 50, 5), ["evening", "outdoor"]),
    "space": ('<rect width="800" height="480" fill="#070b1e"/>' + stars(160, 6) +
              '<path d="M500 140a120 20 0 0 1 240 0" fill="none" stroke="#c4b5ff" stroke-width="6" opacity=".8"/>'
              '<circle cx="620" cy="140" r="70" fill="#7b5cff"/>'
              '<path d="M500 140a120 20 0 0 0 240 0" fill="none" stroke="#c4b5ff" stroke-width="6"/>'
              '<circle cx="150" cy="380" r="30" fill="#ff8a65" opacity=".9"/>', ["space", "shooter", "stars"]),
    "sea": (grad("g", "#7fd1ff", "#d8f3ff") + cloud(150, 90, .9) + cloud(560, 120) +
            '<rect y="260" width="800" height="220" fill="#1f7fc4"/>' +
            "".join(f'<path d="M0 {y}q50-10 100 0t100 0 100 0 100 0 100 0 100 0 100 0 100 0" fill="none" stroke="#7cc4f5" stroke-width="3"/>'
                    for y in (290, 340, 400, 450)), ["water", "ocean", "beach"]),
    "underwater": (grad("g", "#1aa3c7", "#04395e") +
                   "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="#bff3ff" stroke-width="2" opacity=".6"/>'
                           for x, y, r in ((120, 300, 8), (140, 250, 5), (600, 200, 7), (620, 150, 4), (400, 330, 6))) +
                   hills(460, "#e8c97a", 30, 7) +
                   "".join(f'<path d="M{x} 475c-14-25 14-45 0-70s14-45 0-70" fill="none" stroke="#2e8b57" stroke-width="9" '
                           f'stroke-linecap="round"/>' for x in (80, 260, 520, 700)),
                   ["water", "fish", "ocean"]),
    "forest": (grad("g", "#a8e0ff", "#e6f7ff") + cloud(500, 80, .8) + hills(330, "#9ccf8a", 90, 8) +
               trees_row(360, "#4f9a5a", 9, 16, .9) + hills(400, "#3e8a48", 50, 10) + trees_row(440, "#2f6e3a", 11, 12, 1.2) +
               '<rect y="440" width="800" height="40" fill="#2a5e32"/>', ["nature", "trees", "outdoor"]),
    "desert": (grad("g", "#ffd59a", "#fff1d6") + '<circle cx="640" cy="100" r="50" fill="#fff3b0"/>' +
               hills(380, "#f2c56b", 60, 12) + hills(430, "#e8ad4c", 50, 13) +
               '<path d="M170 430v-80M170 380h-20v-24M170 370h20v-30" stroke="#3ca55c" stroke-width="16" stroke-linecap="round" fill="none"/>', ["sand", "hot", "outdoor"]),
    "snow": (grad("g", "#cfe8ff", "#f4fbff") + hills(360, "#ffffff", 80, 14) + trees_row(400, "#2e6b55", 15, 10, 1) +
             hills(430, "#eef6ff", 40, 16) + stars(90, 17, 480, "#ffffff"), ["winter", "cold", "outdoor"]),
    "city": (grad("g", "#151a3a", "#3d3570") + stars(50, 18, 200) + skyline(430, "#20264a", 19) + skyline(470, "#11152e", 20) +
             '<rect y="460" width="800" height="20" fill="#0a0d1f"/>', ["night", "buildings", "urban"]),
    "room": ('<rect width="800" height="480" fill="#f1e3cf"/><rect y="360" width="800" height="120" fill="#c9915a"/>'
             '<path d="M0 360h800" stroke="#8d5a2b" stroke-width="6"/><rect x="520" y="70" width="190" height="150" rx="6" fill="#9fd8ff" stroke="#8d5a2b" stroke-width="10"/>'
             '<path d="M615 70v150M520 145h190" stroke="#8d5a2b" stroke-width="6"/>', ["indoor", "home"]),
    "plain-dark": ('<rect width="800" height="480" fill="#111827"/>', ["dark", "simple"]),
    "plain-light": ('<rect width="800" height="480" fill="#f8fafc"/>', ["light", "simple"]),
}


def gallery(items):
    cards = "".join(
        f'<figure class="{i["kind"]}"><img src="{i["path"].replace("/assets/", "")}" alt="{i["name"]}" loading="lazy">'
        f'<figcaption>{i["name"]}<small>{i["kind"]}</small></figcaption></figure>' for i in items)
    cards = "".join(
        f'<figure class="palette"><div class="pal" style="background:{c[0]};color:{c[4]}">'
        f'<div style="background:{c[1]}"><b style="color:{c[2]}">Aa</b> <i style="background:{c[2]}"></i>'
        f'<i style="background:{c[3]}"></i></div></div><figcaption>{n}<small>colour scheme</small></figcaption></figure>'
        for n, c in PALETTES.items()) + cards
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Elkit art library</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#f8fafc;--card:#fff;--ink:#1f2937;--soft:#64748b}}
body.dark{{--bg:#0f172a;--card:#1e293b;--ink:#f1f5f9;--soft:#94a3b8}}
body{{margin:0;font:15px system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{position:sticky;top:0;display:flex;gap:10px;align-items:center;padding:12px 16px;background:var(--bg);border-bottom:1px solid #8883}}
h1{{font-size:18px;margin:0 auto 0 0}} button,input{{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid #8886;background:var(--card);color:var(--ink)}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px;padding:16px}}
figure{{margin:0;background:var(--card);border-radius:12px;padding:10px;text-align:center;box-shadow:0 1px 3px #0002}}
figure img{{width:100%;height:84px;object-fit:contain}} figure.background{{grid-column:span 2}} figure.background img{{object-fit:cover;border-radius:8px}}
.pal{{height:84px;border-radius:8px;display:flex;align-items:center;justify-content:center;border:1px solid #8883}}
.pal div{{padding:10px 12px;border-radius:8px;box-shadow:0 1px 4px #0003;font-size:22px;display:flex;gap:6px;align-items:center}}
.pal i{{display:inline-block;width:18px;height:18px;border-radius:50%}}
figcaption{{font-weight:600;margin-top:6px;word-break:break-all}} small{{display:block;color:var(--soft);font-weight:400}}
</style></head><body>
<header><h1>Elkit art library ({len(items)})</h1><input id="q" placeholder="search…"><button id="t">dark / light</button></header>
<main>{cards}</main>
<script>
document.getElementById("t").onclick=()=>document.body.classList.toggle("dark");
document.getElementById("q").oninput=e=>{{const q=e.target.value.toLowerCase();
document.querySelectorAll("figure").forEach(f=>f.hidden=!f.textContent.toLowerCase().includes(q))}};
</script></body></html>
"""


def main():
    items = []

    def put(kind, folder, name, text, w, h, tags):
        p = OUT / folder / f"{name}.svg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        items.append({"name": name, "kind": kind, "path": f"/assets/{folder}/{name}.svg", "w": w, "h": h, "tags": tags,
                      "say": SAY.get(name, name.replace("-", " "))})

    for name, body in ICONS.items():
        put("icon", "icons", name, icon_svg(body, INK), 24, 24, ["icon"])
        p = OUT / "icons" / "white" / f"{name}.svg"         # the same icon for dark backgrounds
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(icon_svg(body, WHITE), encoding="utf-8")
    for name, (body, tags) in SPRITES.items():
        put("sprite", "sprites", name, svg(64, 64, body), 64, 64, tags)
    for name, (body, tags) in TILES.items():
        put("tile", "tiles", name, svg(64, 64, body), 64, 64, tags)
    for name, (body, tags) in BACKGROUNDS.items():
        put("background", "backgrounds", name, svg(800, 480, body, ' preserveAspectRatio="xMidYMid slice"'), 800, 480, tags)

    (OUT / "manifest.json").write_text(json.dumps({
        "about": "Elkit's own art (made by scripts/make_assets.py, no licences). Sprites face right. "
                 "Icons: /assets/icons/NAME.svg (dark) and /assets/icons/white/NAME.svg (for dark backgrounds). "
                 "groups: the choices offered for 'show me other ...'. palettes: " + ", ".join(PALETTE_KEYS) + ".",
        "groups": GROUPS,
        "palettes": {n: dict(zip(PALETTE_KEYS, c)) for n, c in PALETTES.items()},
        "items": items}, indent=1), encoding="utf-8")
    names = {i["name"] for i in items}
    bad = {g: [n for n in m if n not in names] for g, m in GROUPS.items()}
    assert not any(bad.values()), f"groups name missing pieces: {bad}"
    (OUT / "index.html").write_text(gallery(items), encoding="utf-8")
    counts = {k: sum(1 for i in items if i["kind"] == k) for k in ("icon", "sprite", "tile", "background")}
    print(f"{len(items)} pieces in {OUT}: {counts} (+ {counts['icon']} white icons)")


if __name__ == "__main__":
    main()
