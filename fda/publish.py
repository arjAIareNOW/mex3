"""Put what Elkit builds (a website, game or app) on the owner's GitHub Pages site (fda.toml: publish_url,
publish_dir), only when they say "publish it".

Off unless fda.toml has `publish_games = true`. Needs a clone of the repo at `publish_dir` that can push (a GitHub
deploy key with write access on the Pi). Runs in a thread: never holds Elkit up, and a failure only goes to the log.

    publish_dir/docs/index.html            what they published: the home page (https://<user>.github.io/)
    publish_dir/docs/<folder>/index.html   only when they name a folder ("publish it in a folder called games")
"""
from __future__ import annotations

import html
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("fda.publish")
SITE = Path("docs")                            # the site root (GitHub Pages: main, /docs)
GAMES = SITE / "apps"                          # the old list folder (index_page); no longer used


def _title(p: Path) -> str:
    m = re.search(r"<title>([^<]{1,120})</title>", p.read_text(encoding="utf-8", errors="ignore"), re.I)
    return html.unescape(m.group(1)).strip() if m else p.stem.replace("-", " ").title()


def index_page(games: Path) -> str:
    """The games list page (newest first)."""
    files = sorted((f for f in games.glob("*.html") if f.name != "index.html"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    items = "\n".join(f'    <a href="{html.escape(f.name)}"><b>{html.escape(_title(f))}</b>'
                      f'<span>{time.strftime("%B %d, %Y", time.localtime(f.stat().st_mtime))}</span></a>'
                      for f in files)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Made with Elkit</title>
<style>
  :root {{ color-scheme: dark; --bg: #0d1017; --card: #161b26; --line: #283041; --text: #e8ecf2; --muted: #8b95a8; --accent: #4de1ff; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font: 16px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 720px; margin: 0 auto; padding: 40px 16px; }}
  h1 {{ margin: 0 0 4px; font-size: 30px; }} p {{ margin: 0 0 28px; color: var(--muted); }}
  .list {{ display: flex; flex-direction: column; gap: 10px; }}
  a {{ display: flex; justify-content: space-between; gap: 12px; padding: 14px 16px; border-radius: 12px; background: var(--card);
       border: 1px solid var(--line); color: var(--text); text-decoration: none; }}
  a:hover {{ border-color: var(--accent); }} a span {{ color: var(--muted); font-size: 14px; white-space: nowrap; }}
</style></head>
<body><main>
  <h1>Made with Elkit</h1>
  <p>Games, websites and apps made by talking to Elkit, a voice assistant on a Raspberry Pi.</p>
  <div class="list">
{items or '    <p>Nothing here yet. Ask Elkit to make something.</p>'}
  </div>
</main></body></html>
"""


async def wait_live(url: str, limit_s: float = 240, every_s: float = 8, expect: str = "") -> bool:
    """GitHub Pages takes ~30-90 s to put a push online: True once `url` answers 200 (and shows `expect`)."""
    import asyncio

    import httpx
    t = time.monotonic()
    async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
        while time.monotonic() - t < limit_s:
            try:
                r = await c.get(url, params={"t": int(time.time())})
                if r.status_code == 200 and expect in r.text:
                    return True
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(every_s)
    log.warning("not live after %.0f s: %s", limit_s, url)
    return False


ASSETS = Path(__file__).parent / "assets"          # Elkit's art library (served by the Pi at /assets/)
_ASSET = re.compile(r"/assets/([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:svg|png|jpe?g|gif|webp))")


def with_assets(html_text: str, games: Path) -> str:
    """The Pi's /assets/ pictures do not exist on GitHub Pages: copy the ones the game uses to games/assets/ and point
    the game there ("/assets/sprites/bird.svg@red" -> "assets/sprites/bird.svg@red")."""
    found = False
    for rel in set(_ASSET.findall(html_text)):
        src = (ASSETS / rel).resolve()
        if ASSETS.resolve() in src.parents and src.is_file():
            out = games / "assets" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            found = True
    return re.sub(r"(?<=[\"'(=\s])/assets/", "assets/", html_text) if found else html_text


def _git(repo: Path, *args: str, timeout: float = 60) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]}: {(r.stderr or r.stdout).strip()[:300]}")
    return r.stdout


def folder_name(said: str) -> str:
    """ "My Games" -> "my-games" (a safe folder name), "" if nothing usable."""
    return re.sub(r"[^a-z0-9]+", "-", (said or "").lower()).strip("-")[:40]


def publish_app(app_file: Path, repo: Path, title: str = "", folder: str = "") -> tuple[str, str]:
    """Put one website, game or app on the site, commit and push. It becomes the home page (docs/index.html), or
    docs/<folder>/index.html when they named a folder. Returns (its path under the site: "" or "games/", a stamp
    written into the page so wait_live can tell the new version is online)."""
    repo = Path(repo).expanduser()
    if not (repo / ".git").exists():
        raise RuntimeError(f"no git clone at {repo}")
    _git(repo, "pull", "--rebase", "-q")
    sub = folder_name(folder)
    where = repo / SITE / sub if sub else repo / SITE
    where.mkdir(parents=True, exist_ok=True)
    stamp = f"elkit-publish {time.strftime('%Y%m%d-%H%M%S')}"
    page = with_assets(app_file.read_text(encoding="utf-8"), where).rstrip() + f"\n<!-- {stamp} -->\n"
    (where / "index.html").write_text(page, encoding="utf-8")
    _git(repo, "add", str((where / "index.html").relative_to(repo)))
    if (where / "assets").is_dir():
        _git(repo, "add", str((where / "assets").relative_to(repo)))
    _git(repo, "commit", "-q", "-m", f"Publish: {title or app_file.stem}" + (f" ({sub}/)" if sub else " (home page)"))
    _git(repo, "push", "-q", timeout=90)
    return (f"{sub}/" if sub else ""), stamp


publish_game = publish_app                          # the old name
