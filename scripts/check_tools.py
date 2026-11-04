"""Calls the new tools through the real tools MCP server (in memory), no LLM involved:
ask_documents, list_documents, device_check, app_history.

    .venv\\Scripts\\python scripts\\check_tools.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.client.client import Client  # noqa: E402

from fda.config import Config  # noqa: E402
from fda.mcp_servers import tools_server  # noqa: E402


def val(r):
    sc = r.structured_content
    if sc is not None:
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(r.content[0].text)


async def main():
    srv, _ = tools_server.build(Config.load())
    async with Client(srv) as c:
        names = sorted(t.name for t in (await c.list_tools()).tools)
        print("tools:", ", ".join(names))
        print("list_documents:", [d["file"] for d in val(await c.call_tool("list_documents", {}))["documents"]])
        for q in ["What does the billing policy say about annual plans?", "Where is the first aid kit?"]:
            r = val(await c.call_tool("ask_documents", {"question": q}))
            p = r["passages"][0]
            print(f"ask_documents: {q} -> {p['file']} / {p['section']}")
        r = val(await c.call_tool("device_check", {}))
        print("device_check:", r["summary"], "| report:", r["report"])
        h = val(await c.call_tool("app_history", {"name": "bird flap"}))
        print("app_history:", h.get("title"), len(h.get("versions", [])), "versions")

asyncio.run(main())
