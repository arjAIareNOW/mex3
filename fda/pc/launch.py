"""Start FDA's three MCP servers (asr, tts, tools) as separate processes on this machine, unless already running.

    python -m fda.pc.launch          # just the MCP servers (e.g. as a service on the Pi); Ctrl+C stops them
"""
from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import sys
import time

from ..config import ROOT, Config

log = logging.getLogger("fda.pc.launch")
SERVERS = [("asr", "asr_mcp_port"), ("tts", "tts_mcp_port"), ("tools", "tools_mcp_port")]
_procs: list[subprocess.Popen] = []


def _listening(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def urls(cfg: Config) -> dict[str, str]:
    return {name: f"http://{cfg.mcp_host}:{getattr(cfg, attr)}/mcp" for name, attr in SERVERS}


def start_mcp_servers(cfg: Config, timeout: float = 120.0) -> dict[str, str]:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    started = []
    for name, attr in SERVERS:
        port = getattr(cfg, attr)
        if _listening(cfg.mcp_host, port):
            log.info("MCP %s already running on port %d", name, port)
            continue
        f = open(logs / f"mcp-{name}.log", "w", encoding="utf-8", errors="replace")
        p = subprocess.Popen([sys.executable, "-m", f"fda.mcp_servers.{name}_server", "--host", cfg.mcp_host,
                              "--port", str(port)], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        _procs.append(p)
        started.append((name, port, p))
        log.info("MCP %s server starting (pid %d, port %d, log logs/mcp-%s.log)", name, p.pid, port, name)
    atexit.register(stop_mcp_servers)
    t_end = time.time() + timeout
    for name, port, p in started:
        while not _listening(cfg.mcp_host, port):
            if p.poll() is not None:
                tail = (logs / f"mcp-{name}.log").read_text(encoding="utf-8", errors="replace")[-800:]
                raise SystemExit(f"MCP {name} server exited:\n{tail}")
            if time.time() > t_end:
                raise SystemExit(f"MCP {name} server did not start in {timeout:.0f} s (logs/mcp-{name}.log)")
            time.sleep(0.2)
    return urls(cfg)


def stop_mcp_servers():
    for p in _procs:
        if p.poll() is None:
            p.terminate()
    for p in _procs:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    _procs.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config.load()
    print(start_mcp_servers(cfg))
    try:
        while all(p.poll() is None for p in _procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_mcp_servers()
