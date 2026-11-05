"""OpenAI-compatible chat endpoints: Groq (the main provider) or the local llama.cpp server (free, offline).

engine   endpoint                                      key (.env)       default model (agent / coder)
groq     https://api.groq.com/openai/v1                GROQ_API_KEY     gpt-oss-120b / qwen3.8-27b
local    llama-server started here (offline_models/)   -                offline_llm / coder model
openai   LLM_BASE_URL (any OpenAI-compatible server)   LLM_API_KEY      llm_model
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import time

import httpx

log = logging.getLogger("fda.llm")

CLOUD = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
             {"agent": "openai/gpt-oss-120b", "coder": "qwen/qwen3.8-27b"}),
    "openai": (None, "LLM_API_KEY", {"agent": "", "coder": ""}),
}
MAX_OUT: dict[str, int] = {}      # per-engine cap on output tokens (none needed for Groq)
FALLBACKS = {"groq": {"agent": ["openai/gpt-oss-20b"],         # one fallback per role, used when the main fails
                      "coder": ["openai/gpt-oss-120b"]}}
CODER_LOCAL = ["qwen2.5-coder-3b", "qwen2.5-3b", "qwen2.5-coder-1.5b", "qwen2.5-1.5b"]   # first one downloaded wins


class Endpoint:
    def __init__(self, base_url: str, model: str, key: str = "", engine: str = "local"):
        self.base_url, self.model, self.key, self.engine = base_url.rstrip("/"), model, key, engine
        self.http = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0),
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
        self.tok_s = 20.0 if engine == "local" else 300.0     # rough generation speed, for eta_s

    fallbacks: list = []          # other models to try when this one is busy (503/429) or retired (404)

    def complete(self, messages: list, max_tokens: int = 2000, temperature: float = 0.2, stop_when=None) -> str:
        """complete_once, moving on to a fallback model when the provider says the model is busy or gone."""
        # every call starts again from the configured model: a fallback is for this call only (it used to stick,
        # so one 429 left all later builds on the smaller model)
        self.primary = getattr(self, "primary", None) or self.model
        models = [self.primary] + [m for m in self.fallbacks if m != self.primary]
        for i, model in enumerate(models):
            self.model = model
            try:
                return self.complete_once(messages, max_tokens, temperature, stop_when)
            except RuntimeError as e:
                busy = any(f"HTTP {c}" in str(e) for c in (404, 429, 500, 503))
                if not busy or i == len(models) - 1:
                    raise
                log.warning("%s %s unavailable (%s); trying %s", self.engine, model, str(e)[:60], models[i + 1])
        raise RuntimeError("no model available")

    def complete_once(self, messages: list, max_tokens: int = 2000, temperature: float = 0.2, stop_when=None) -> str:
        """Streams the answer; stop_when(text_so_far) -> True ends it early (e.g. the HTML is complete, or the
        model is looping)."""
        import json as _json
        t = time.perf_counter()
        max_tokens = min(max_tokens, MAX_OUT.get(self.engine, max_tokens))
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": True}
        if self.engine == "local":
            body["repeat_penalty"] = 1.08          # llama.cpp: fewer repeated lines
        body.update(reasoning_extra(self.model))
        out, pieces = "", 0
        with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"{self.engine} HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                try:
                    d = _json.loads(line[6:])
                except ValueError:
                    continue
                c = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if c:
                    out += c
                    pieces += 1
                    if stop_when is not None and pieces % 20 == 0 and stop_when(out):
                        break
        n = len(out) / 3.5
        dt = time.perf_counter() - t
        if dt > 1:
            self.tok_s = 0.5 * self.tok_s + 0.5 * n / dt
        log.info("%s/%s: ~%d tokens in %.1f s", self.engine, self.model, n, dt)
        return out

    def eta_s(self, kind: str = "build") -> int:
        return int(min(900, (2500 if kind == "build" else 3000) / max(self.tok_s, 1)) + 2)


class LocalEndpoint(Endpoint):
    """A llama-server of its own (started on first use, stopped at exit)."""

    def __init__(self, cfg, model_name: str, port: int, threads: int | None = None):
        super().__init__(f"http://127.0.0.1:{port}/v1", model_name, engine="local")
        self.cfg, self.port, self.threads, self.proc = cfg, port, threads, None

    def _ensure(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        from ..local import DIR, LLM, _free_port, _llama_server_exe, _port_busy
        gguf = DIR / LLM[self.model][1]
        if _port_busy(self.port):
            self.port = _free_port()
            self.base_url = f"http://127.0.0.1:{self.port}/v1"
        n = os.cpu_count() or 4
        threads = self.threads or (max(1, n - 1) if n <= 4 else min(8, n // 2))
        cmd = [_llama_server_exe(), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(self.port), "-c", "12288",
               "-np", "1", "-t", str(threads), "--no-webui", "--jinja"]     # --jinja: tool calling
        logf = open(DIR / f"llama-{self.model}.log", "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        atexit.register(self.stop)
        t_end = time.time() + 180
        while time.time() < t_end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server for {self.model} exited, see {DIR / f'llama-{self.model}.log'}")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    log.info("local %s up on port %d (%d threads)", self.model, self.port, threads)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"llama-server for {self.model} did not start")

    def complete(self, messages: list, max_tokens: int = 2000, temperature: float = 0.2, stop_when=None) -> str:
        self._ensure()
        return super().complete(messages, max_tokens, temperature, stop_when)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def reasoning_extra(model: str) -> dict:
    """Thinking models: think as little as possible (voice needs speed; gpt-oss counts thinking in max_tokens)."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}
    if "qwen3" in model and "coder" not in model:
        return {"reasoning_effort": "none"}
    return {}


def cloud(cfg, engine: str, role: str = "agent") -> Endpoint:
    base, key_env, models = CLOUD[engine]
    base = base or os.environ.get("LLM_BASE_URL", "")
    key = os.environ.get(key_env, "")
    model = (cfg.llm_model if role == "agent" else os.environ.get("CODER_MODEL", "")) or models[role] or cfg.llm_model
    if not base or not key or not model:
        raise SystemExit(f"llm engine {engine}: set {key_env}" + (", LLM_BASE_URL and llm_model" if engine == "openai"
                                                                  else "") + " in .env / fda.toml")
    ep = Endpoint(base, model, key, engine)
    ep.fallbacks = FALLBACKS.get(engine, {}).get(role, [])
    return ep


class ChainEndpoint:
    """Several engines in order ("groq,local"): when one fails (busy, rate limit, down), the next one writes."""

    def __init__(self, endpoints: list):
        self.endpoints = endpoints
        self.current = endpoints[0]

    def __getattr__(self, name):              # engine, model, eta_s, tok_s ... of the one in use
        return getattr(self.current, name)

    def complete(self, messages: list, max_tokens: int = 2000, temperature: float = 0.2, stop_when=None) -> str:
        last = None
        for i, ep in enumerate(self.endpoints):
            self.current = ep
            try:
                return ep.complete(messages, max_tokens, temperature, stop_when)
            except (RuntimeError, httpx.HTTPError) as e:
                last = e
                if i + 1 < len(self.endpoints):
                    log.warning("coder %s failed (%s); trying %s", ep.engine, str(e)[:80], self.endpoints[i + 1].engine)
        self.current = self.endpoints[0]
        raise last


def make(cfg, engine: str, role: str = "agent") -> Endpoint:
    if "," in engine:
        eps = []
        for e in (x.strip() for x in engine.split(",")):
            if e in CLOUD and not os.environ.get(CLOUD[e][1]):
                log.warning("%s %s: %s is not set in .env; skipped", role, e, CLOUD[e][1])
                continue
            eps.append(make(cfg, e, role))
        if not eps:
            return make(cfg, "local", role)
        return eps[0] if len(eps) == 1 else ChainEndpoint(eps)
    if engine in CLOUD:
        if role == "coder" and not os.environ.get(CLOUD[engine][1]):
            log.warning("coder %s: %s is not set in .env; using the local coder instead", engine, CLOUD[engine][1])
            return make(cfg, "local", role)
        return cloud(cfg, engine, role)
    if engine != "local":
        raise SystemExit(f"unknown llm engine {engine!r}: groq, local, openai")
    from ..local import DIR, LLM
    if role == "coder":
        name = next((m for m in CODER_LOCAL if m in LLM and (DIR / LLM[m][1]).exists()), None)
        if name is None:
            raise SystemExit("no local model to write apps: run  python -m fda download --coder")
        n = os.cpu_count() or 4
        # fewer threads than the agent's LLM, so conversation stays responsive while an app is being built
        return LocalEndpoint(cfg, name, port=8094, threads=2 if n <= 4 else min(8, n // 3))
    name = getattr(cfg, "agent_llm", "") or cfg.offline_llm
    if not (DIR / LLM[name][1]).exists():
        name = cfg.offline_llm
    return LocalEndpoint(cfg, name, port=cfg.llm_port + 4)      # 8095: never shares the classic FDA planner
