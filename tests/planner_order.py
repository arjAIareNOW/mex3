"""Compare planner JSON orders (accuracy + time to first 'say' token) on text transcripts.
python tests/planner_order.py [--offline | --server]   (default: fda.toml backend)"""
import asyncio, sys, time, json
sys.path.insert(0, ".")
from fda.config import Config
from fda.servers import make_servers
from fda import brain

CASES = [  # (heard, for_me, expected tools)
 ("hey can you turn the lights blue and set a timer for ten minutes", True, {"lights", "timer"}),
 ("make it warmer like twenty four degrees", True, {"thermostat"}), ("what's the weather like today", True, {"weather"}),
 ("who won the last nba finals", True, {"web_search"}), ("remind me to call mom at seven pm", True, {"add_reminder"}),
 ("play some music", True, {"music"}), ("honey did you lock the front door when you came in", False, set()),
 ("and in other news the stock market closed higher today amid strong earnings", False, set()),
 ("how are you doing today", True, set()), ("dim the lights a little bit", True, {"lights"}),
 ("call dad", True, {"call"}), ("tell me a joke", True, set()), ("yeah i'll be there in five minutes okay bye", False, set()),
 ("what's the capital of japan", True, set()), ("set an alarm for six thirty tomorrow morning", True, {"alarm"}),
 ("text maria that dinner is ready", True, {"message"}), ("tell me a story about the moon", True, set()),
 ("is it going to rain this afternoon", True, {"weather"}), ("what's the price of bitcoin right now", True, {"web_search"}),
 ("it's a bit dark in here", True, {"lights"}), ("john can you grab the keys", False, set()),
 ("cancel the timer", True, {"cancel_timer"}), ("i'm feeling a bit cold", True, {"thermostat"}),
]

async def run(s, cfg, order):
    sch = brain.plan_schema()
    sch["properties"] = {k: sch["properties"][k] for k in order}; sch["required"] = list(order)
    ok, firsts = 0, []
    for heard, fm, tools in CASES:
        msgs = brain.plan_messages(cfg, {"lights": {"power": "on", "brightness": 80, "color": "warm white"},
                                         "thermostat_c": 24, "music": "stopped", "volume": 80}, [], heard,
                                   static_prefix=s.static_prefix)
        if order != ("for_me", "do", "say"):  # re-render few-shot answers in this order
            for m in msgs:
                if m["role"] == "assistant":
                    o = json.loads(m["content"]); m["content"] = json.dumps({k: o[k] for k in order})
        t = time.perf_counter(); out = ""; first = None
        async for p in s.chat_stream(msgs, sch, temperature=0.0):
            out += p
            if first is None and '"say"' in out:
                first = (time.perf_counter() - t) * 1000
        o = json.loads(out); got = {a["tool"] for a in o["do"]}
        good = o["for_me"] == fm and got == tools
        ok += good; firsts.append(first or 0)
        if not good:
            print(f"   x {heard[:50]:50s} {json.dumps(o)[:140]}")
    print(f"{order}: {ok}/{len(CASES)} correct, 'say' starts at {sum(firsts)/len(firsts):.0f} ms (mean)")

async def main():
    cfg = Config.load()
    if "--offline" in sys.argv: cfg.backend = "offline"
    if "--server" in sys.argv: cfg.backend = "server"
    s = make_servers(cfg); await s.warm_up(); print("backend:", s.kind)
    for order in (("for_me", "do", "say"), ("for_me", "say", "do")):
        await run(s, cfg, order)
    await s.close()
asyncio.run(main())
