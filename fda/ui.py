"""Display server: serves the 7" screen page and pushes live state over a WebSocket. Taps on the screen come back as
room actions (same path as voice commands)."""
from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import WSMsgType, web

from .config import PKG

log = logging.getLogger("fda.ui")


class UI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clients: set[web.WebSocketResponse] = set()
        self.last: dict[str, dict] = {}       # latest message per type, replayed to new screens
        self.agent = None
        self.runner = None

    def send(self, msg: dict):
        t = msg.get("type")
        if t in ("room", "status", "health", "sources", "app", "build", "tab"):
            self.last[t] = msg
        if not self.clients:
            return
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            if not ws.closed:
                asyncio.ensure_future(_safe_send(ws, data))

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        app.router.add_static("/games/", PKG / "games")     # FDA's games (make_game tool)
        from .config import ROOT
        apps = ROOT / self.cfg.apps_dir
        apps.mkdir(parents=True, exist_ok=True)
        app.router.add_static("/apps/", apps)                # apps the agent builds (create_app MCP tool)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.cfg.ui_host, self.cfg.ui_port)
        await site.start()
        url = f"http://localhost:{self.cfg.ui_port}/"
        log.info("display: %s", url)
        if self.cfg.open_browser:
            import webbrowser
            webbrowser.open(url)

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    async def _index(self, request):
        # never cached: a tab left open from an older FDA would miss new features (it reloads itself, see "hello")
        return web.FileResponse(PKG / "web" / "index.html", headers={"Cache-Control": "no-store"})

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        page_v = int((PKG / "web" / "index.html").stat().st_mtime)
        await ws.send_str(json.dumps({"type": "hello", "v": page_v}))
        for m in self.last.values():
            await ws.send_str(json.dumps(m, default=str))
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT and self.agent is not None:
                    try:
                        msg = json.loads(m.data)
                        if msg.get("type") == "action" and isinstance(msg.get("action"), dict):
                            await self.agent.room.execute(msg["action"])
                    except Exception:
                        log.exception("bad message from display")
        finally:
            self.clients.discard(ws)
        return ws


async def _safe_send(ws, data):
    try:
        await ws.send_str(data)
    except Exception:
        pass
