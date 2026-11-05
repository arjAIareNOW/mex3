"""The agent LLM with automatic fallback: when Groq says "rate limit" (429), the model is busy or gone
(404/5xx), or the service does not answer, the same turn is sent to the next model in the chain at once.
A model that failed is skipped for a while (`cooldown_s`, or what the provider's retry-after says).

Chain (Groq only, GROQ_API_KEY in .env): gpt-oss-120b, and one fallback, gpt-oss-20b. Each Groq model has its
own rate limit, so a busy model does not stop the conversation.
"""
from __future__ import annotations

import os
import re
import time

from loguru import logger
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, NotFoundError,
                    RateLimitError)
from pipecat.services.openai.llm import OpenAILLMService

from .. import rules
from ..mcp_servers import llm as llmlib

# (engine, base url, key env, model)
CHAIN = [
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b"),   # the one fallback
]
_REASONING_KEYS = ("reasoning_effort",)


def chain(first_model: str = "") -> list[dict]:
    out = []
    for engine, base, key_env, model in CHAIN:
        key = os.environ.get(key_env, "")
        if key:
            out.append({"engine": engine, "base": base, "key": key, "model": model, "down_until": 0.0,
                        "client": None})
    if first_model:                                     # the configured model goes first
        out.sort(key=lambda e: e["model"] != first_model)
    return out


def clean_messages(msgs: list) -> list:
    """Drop what some providers reject and would then poison every later turn: tool calls without a name (and
    their results), and empty assistant turns."""
    out, keep_ids = [], set()
    for m in msgs:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = [c for c in m["tool_calls"] if ((c.get("function") or {}).get("name") or "").strip()]
            keep_ids |= {c.get("id") for c in calls}
            if not calls and not (m.get("content") or "").strip():
                continue
            m = {**m, "tool_calls": calls} if calls else {k: v for k, v in m.items() if k != "tool_calls"}
        elif m.get("role") == "tool" and m.get("tool_call_id") not in keep_ids:
            continue
        elif m.get("role") == "assistant" and not str(m.get("content") or "").strip(" .\u2026\n"):
            continue
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and rules.is_thinking(m["content"]):
            m = {**m, "content": rules.drop_thinking(m["content"])}   # a leaked "The user wants ..." gets copied
            if not m["content"].strip(" .\u2026\n") and not m.get("tool_calls"):
                continue
        out.append(m)
    return out


class FallbackLLMService(OpenAILLMService):
    """OpenAILLMService that tries each entry of `chain()` in order for every turn."""

    def __init__(self, entries: list[dict], cooldown_s: float = 20.0, **kw):
        super().__init__(api_key=entries[0]["key"], base_url=entries[0]["base"], **kw)
        self.entries = entries
        self.cooldown_s = cooldown_s
        self.current = entries[0]

    def _client_for(self, e: dict) -> AsyncOpenAI:
        if e["client"] is None:
            e["client"] = AsyncOpenAI(api_key=e["key"], base_url=e["base"], max_retries=0, timeout=20)
        return e["client"]

    async def _process_context(self, context):
        """One retry when the model's tool call was malformed mid-answer: gpt-oss on Groq sometimes writes a tool
        name with an invisible character in front ("\\u200blist_items"), Groq rejects it while streaming, and the
        turn would end in silence ("add milk to my list" -> nothing). Asking again almost always gives a clean call."""
        # It can also call a tool that does not exist ("repo_browser.open_file") and repeat it on a plain retry,
        # so the retries tell it what went wrong (a note that is taken out of the context again afterwards).
        note = None
        try:
            for attempt in range(3):
                try:
                    return await super()._process_context(context)
                except Exception as err:  # noqa: BLE001
                    msg = str(err)
                    if "tool call validation failed" not in msg.lower() or attempt == 2:
                        raise
                    logger.warning(f"LLM wrote a malformed tool call ({msg[:120]}); asking again ({attempt + 1}/2)")
                    if note is None:
                        m = re.search(r"call tool '([^']+)'", msg)
                        note = {"role": "system", "content":
                                (f"There is no tool called {m.group(1)!r}. " if m else
                                 "Your last tool call was malformed. ")
                                + "Use only the tools you were given, with their exact names, or just answer in words."}
                        context.add_message(note)
        finally:
            if note is not None:
                context.set_messages([x for x in context.get_messages() if x is not note])

    async def get_chat_completions(self, context):
        adapter = self.get_llm_adapter()
        self._log_llm_response(context)
        p = await adapter.get_llm_invocation_params(context, system_instruction=self._settings.system_instruction,
                                                    convert_developer_to_user=not self.supports_developer_role)
        base = self.build_chat_completion_params(p)
        for k in _REASONING_KEYS:
            base.pop(k, None)
        if isinstance(base.get("messages"), list):
            base["messages"] = clean_messages(base["messages"])
        now = time.monotonic()
        ready = [e for e in self.entries if e["down_until"] <= now] or \
            sorted(self.entries, key=lambda e: e["down_until"])[:1]       # all resting: the one free soonest
        last = None
        for e in ready:
            extra = llmlib.reasoning_extra(e["model"])
            params = {**base, "model": e["model"], **extra}
            for k in ("max_tokens", "max_completion_tokens"):
                params.pop(k, None)
            params["max_tokens"] = 400 if extra else 90       # thinking models count thinking in max_tokens
            try:
                stream = await self._client_for(e).chat.completions.create(**params)
            except (RateLimitError, NotFoundError, APIConnectionError, APITimeoutError, APIStatusError) as err:
                code = getattr(err, "status_code", None)
                if isinstance(err, APIStatusError) and code not in (400, 404, 408, 409, 429, 500, 502, 503, 504):
                    raise
                wait = self.cooldown_s
                try:
                    wait = max(wait, float(err.response.headers.get("retry-after", 0)))
                except Exception:  # noqa: BLE001
                    pass
                e["down_until"] = time.monotonic() + wait
                why = ""
                if code == 400:                     # a 400 is our request, not their load: say what they disliked
                    why = " " + str(getattr(err, "body", "") or err)[:160]
                logger.warning(f"LLM {e['engine']}/{e['model']} unavailable ({code or type(err).__name__}){why}; "
                               f"next model, this one rests {wait:.0f} s")
                last = err
                continue
            if e is not self.current:
                logger.info(f"LLM now: {e['engine']}/{e['model']}")
                self.current = e
            return stream
        raise last or RuntimeError("no LLM available")
