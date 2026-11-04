"""Text-only check of the Pipecat agent's LLM: same system prompt, few-shot and tools (room + MCP tools server),
one fresh conversation per sentence. Shows which tool it calls (or what it says) and how long it takes.

    .venv\\Scripts\\python tests\\pc_llm_eval.py                     # llm_engine from fda.toml (local: offline_llm)
    .venv\\Scripts\\python tests\\pc_llm_eval.py --model qwen2.5-3b  # another local model
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fda.config import Config  # noqa: E402

CASES = [   # (sentence, expected: tool name | "say" | "silent")
    ("What's the weather like right now?", "get_weather"),
    ("Turn off the lights please.", "lights"),
    ("Set a timer for ten minutes.", "timer"),
    ("Play some music.", "music"),
    ("Can you make me a simple tip calculator app?", "create_app"),
    ("Make me a snake game.", "create_app"),
    ("Build a website for my coffee shop with a menu.", "create_app"),
    ("Make the snake red.", "tweak_app|edit_app"),
    ("Change the bird to a butterfly.", "tweak_app"),
    ("Make the pipes a photo of bamboo.", "tweak_app"),
    ("Add a high score to the game.", "edit_app"),
    ("What's in the news today?", "get_news"),
    ("How do I cook chicken adobo?", "find_recipe"),
    ("What is 15 percent of 240?", "calculate"),
    ("Who is the current president of France?", "web_search"),
    ("Tell me a joke.", "say"),
    ("How are you doing today?", "say"),
    ("Tell me a short story about a brave little robot.", "say"),
    ("What's two plus two?", "say|calculate"),
    ("The central bank raised interest rates by a quarter point on Thursday, citing persistent inflation.", "silent"),
    ("And now the weather for the weekend: sunny skies across the region with highs near thirty.", "silent"),
    ("Let's play a game.", "game"),
    ("Remind me to call the doctor at three pm.", "add_reminder"),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="local model name (default: offline_llm)")
    ap.add_argument("--engine", default=None)
    args = ap.parse_args()
    cfg = Config.load()
    if args.model:
        cfg.agent_llm = args.model
    if args.engine:
        cfg.llm_engine = args.engine
    from fda.mcp_servers import llm as llmlib
    from fda.pc.agent import FEW_SHOT, SYSTEM, room_tools
    from fda.pc.launch import start_mcp_servers
    from fda.room import Room
    from fda.audio import Mixer
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    from pipecat.services.mcp_service import MCPClient
    from pipecat.services.openai.llm import OpenAILLMService
    from mcp.client.session_group import StreamableHttpParameters

    urls = await asyncio.to_thread(start_mcp_servers, cfg)
    ep = llmlib.make(cfg, cfg.llm_engine, role="agent")
    if isinstance(ep, llmlib.LocalEndpoint):
        await asyncio.to_thread(ep._ensure)
    room = Room(cfg, Mixer(16000), on_change=lambda s: None, on_ring=None)
    mcp = MCPClient(server_params=StreamableHttpParameters(url=urls["tools"]))
    remote = await mcp.tools()
    tools = ToolsSchema(standard_tools=[*room_tools(room), *remote.standard_tools])
    llm = OpenAILLMService(api_key="none", base_url=ep.base_url, settings=OpenAILLMService.Settings(model=ep.model))
    otools = llm.get_llm_adapter().to_provider_tools_format(tools)
    system = {"role": "system", "content": SYSTEM.format(name=cfg.name, location=cfg.location)}
    ok = 0
    times = []
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {ep.key}"} if ep.key else {}) as c:
        await c.post(f"{ep.base_url}/chat/completions", json={"model": ep.model, "max_tokens": 1, "tools": otools,
                     "messages": [system, *FEW_SHOT, {"role": "user", "content": "hi"}]})
        for text, want in CASES:
            t = time.perf_counter()
            body = {"model": ep.model, "temperature": 0.3, "tools": otools,
                    "max_tokens": 400 if llmlib.reasoning_extra(ep.model) else 120,
                    "messages": [system, *FEW_SHOT, {"role": "user", "content": text}],
                    **llmlib.reasoning_extra(ep.model)}
            r = await c.post(f"{ep.base_url}/chat/completions", json=body)
            dt = time.perf_counter() - t
            if r.status_code != 200:
                print(f"ERR HTTP {r.status_code} {r.text[:300]}  <- {text[:50]}")
                if r.status_code == 429:
                    await asyncio.sleep(float(r.headers.get("retry-after", 20)))
                continue
            times.append(dt)
            m = r.json()["choices"][0]["message"]
            calls = [(x["function"]["name"], x["function"]["arguments"]) for x in (m.get("tool_calls") or [])]
            content = (m.get("content") or "").strip()
            got = calls[0][0] if calls else ("silent" if content.lower().strip("() .[]") == "silent" else "say")
            good = got in want.split("|")
            ok += good
            print(f"{'ok ' if good else 'BAD'} {dt:5.2f}s  want {want:<14} got {got:<12} "
                  f"{json.dumps(calls[0][1]) if calls else content[:80]!s:.90}  <- {text[:50]}")
    print(f"\n{ep.engine}/{ep.model}: {ok}/{len(CASES)} correct, median {sorted(times)[len(times) // 2]:.2f} s")
    await mcp.close()
    if isinstance(ep, llmlib.LocalEndpoint):
        ep.stop()


if __name__ == "__main__":
    asyncio.run(main())
