"""Put the games Elkit builds on the owner's GitHub Pages site (arjaiarenow.github.io/mex3/games/).

Off unless fda.toml has `publish_games = true`. Needs a clone of the repo at `publish_dir` that can push (a GitHub
deploy key with write access on the Pi). Runs in a thread after a game is ready: never holds Elkit up, and a failure
only goes to the log.

    publish_dir/docs/games/<name>.html   the game, as built
    publish_dir/docs/games/index.html    a list of all published games (rebuilt every time)
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
GAMES = Path("docs") / "games"


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
<title>Elkit games</title>
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
  <h1>Elkit games</h1>
  <p>Games made by talking to Elkit, a voice assistant on a Raspberry Pi.</p>
  <div class="list">
{items}
  </div>
</main></body></html>
"""


def _git(repo: Path, *args: str, timeout: float = 60) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]}: {(r.stderr or r.stdout).strip()[:300]}")
    return r.stdout


def publish_game(app_file: Path, repo: Path, title: str = "") -> str:
    """Copy one game into the site clone, rebuild the list, commit and push. Returns the published file name."""
    repo = Path(repo).expanduser()
    if not (repo / ".git").exists():
        raise RuntimeError(f"no git clone at {repo}")
    _git(repo, "pull", "--rebase", "-q")
    games = repo / GAMES
    games.mkdir(parents=True, exist_ok=True)
    dest = games / app_file.name
    shutil.copyfile(app_file, dest)
    (games / "index.html").write_text(index_page(games), encoding="utf-8")
    _git(repo, "add", str(GAMES))
    if not _git(repo, "status", "--porcelain", "--", str(GAMES)).strip():
        return dest.name                                   # the same game again: nothing new to push
    _git(repo, "commit", "-q", "-m", f"Publish game: {title or app_file.stem}")
    _git(repo, "push", "-q", timeout=90)
    return dest.name
