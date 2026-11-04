"""One-off fixes from the 20:14-20:23 live test:
- a tool call with no name (Gemini) poisoned the context: every later request failed with HTTP 400
- the coder and the agent shared Groq gpt-oss-120b's 8000 tokens/min: builds starved the conversation (429s)
- edit_app "3D print portfolio" overwrote the flappy bird game ("a" matched a-flappy-bird...)
- websites had no photos and dead menu links ("I think what you're making is fake")
- the ASR wrote Chinese/Portuguese for mumbles; "strop" did not count as stop
- weather was asked for but no longer available"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(rel, pairs):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in s, f"{rel}: {old[:60]!r}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")


# ---- LLM chain: SambaNova before the other Groq models; clean the context before every request ----
patch("fda/pc/fallback_llm.py", [
    ('''    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b"),
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3.8-27b"),
    ("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "gpt-oss-120b"),''',
     '''    ("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "gpt-oss-120b"),
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b"),
    ("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "Meta-Llama-3.3-70B-Instruct"),'''),
    ('''class FallbackLLMService(OpenAILLMService):''',
     '''def clean_messages(msgs: list) -> list:
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
        elif m.get("role") == "assistant" and not str(m.get("content") or "").strip(" .\\u2026\\n"):
            continue
        out.append(m)
    return out


class FallbackLLMService(OpenAILLMService):'''),
    ('''        for k in _REASONING_KEYS:
            base.pop(k, None)''',
     '''        for k in _REASONING_KEYS:
            base.pop(k, None)
        if isinstance(base.get("messages"), list):
            base["messages"] = clean_messages(base["messages"])'''),
    # a 400 from one provider (its own template rules) should not end the conversation: try the next one
    ('''                if isinstance(err, APIStatusError) and code not in (404, 408, 409, 429, 500, 502, 503, 504):
                    raise''',
     '''                if isinstance(err, APIStatusError) and code not in (400, 404, 408, 409, 429, 500, 502, 503, 504):
                    raise'''),
])

# ---- the coder gets its own Groq model (its own tokens-per-minute) ----
patch("fda/mcp_servers/llm.py", [
    ('''{"agent": "openai/gpt-oss-120b", "coder": "openai/gpt-oss-120b"}''',
     '''{"agent": "openai/gpt-oss-120b", "coder": "qwen/qwen3.8-27b"}'''),
])

# ---- apps: right target, real photos, working pages ----
patch("fda/mcp_servers/tools_server.py", [
    ('''        words = set(_slug(name).split("-"))
        best = max(self.dir.glob("*.html"), key=lambda q: len(words & set(q.stem.split("-"))), default=None)
        return best if best is not None and words & set(best.stem.split("-")) else None''',
     '''        stop = {"a", "an", "the", "my", "app", "page", "site", "website", "game", "of", "for", "and", "to", "type"}
        words = {w for w in _slug(name).split("-") if len(w) > 1 and w not in stop}

        def score(q):
            title = _slug(self._meta(q).get("title", ""))
            return len(words & (set(q.stem.split("-")) | set(title.split("-"))))
        cur = self.path(self.current) if self.current and self.path(self.current).exists() else None
        cands = [q for q in self.dir.glob("*.html") if not q.name.endswith(".prev.html")]
        best = max(cands, key=score, default=None)
        if best is None or score(best) == 0 or (cur is not None and score(cur) >= score(best)):
            return cur                    # no clear other app named: the one on screen
        return best'''),
    ('''- No external files, CDNs, fonts, images or network requests. Draw with canvas or CSS. Emoji are fine.''',
     '''- No external scripts, CDNs, fonts or network requests. Games: draw with canvas or CSS, emoji are fine.'''),
    ('''- Websites/tools: a clean, modern single page (cards, rounded corners, soft shadows), working buttons/forms (save
  data in localStorage if useful).''',
     '''- Websites: look like a real, polished business site. Use the real name, wording and details you are given.
  Pages: put each page (Home, About, Services, Gallery, Contact...) in its own <section class="page" id="...">, show
  one at a time, and make EVERY menu link and button work (switch page with location.hash / hashchange, highlight
  the active link). Write real content: headings, short paragraphs, prices or services, a contact form that shows
  a thank-you message.
  Photos: real photos from https://loremflickr.com/WIDTH/HEIGHT/KEYWORD,KEYWORD?lock=N (N = 1, 2, 3... so each
  image differs; keywords about the subject, e.g. 3dprinting,printer). Every <img> gets alt text, object-fit:cover,
  loading="lazy" and onerror="this.style.background='linear-gradient(135deg,#ccd,#99a)';this.removeAttribute('src')"
  so the layout still works offline. A gallery = a grid of 6+ such photos.
  Scrolling inside the page is fine for websites.
- Tools: a clean, modern single page (cards, rounded corners, soft shadows), working buttons/forms (save data in
  localStorage if useful).'''),
    ('''- Keep it short: about 80-150 lines in total. Stop right after </html>.''',
     '''- Keep it short: games and tools about 80-150 lines, websites up to about 250. Stop right after </html>.'''),
])

# ---- speech: drop foreign-script mumbles, "strop" is stop ----
patch("fda/mcp_servers/chain.py", [
    ('''            text = b(x, rate, language)
            if text is not None:
                self.last = b.name
                return text''',
     '''            text = b(x, rate, language)
            if text is not None:
                self.last = b.name
                if language == "en" and text:
                    letters = [c for c in text if c.isalpha()]
                    if letters and sum(c.isascii() for c in letters) < 0.6 * len(letters):
                        return ""             # a mumble written as Chinese etc.: not words for an English agent
                    if any(w in text.lower().split() for w in ("não", "é", "você")):
                        return ""
                return text'''),
])
patch("fda/pc/processors.py", [
    ('''_STOP_WORDS = re.compile(r"\\b(stop|wait|enough|quiet|shut up|hold on|hang on|never ?mind|cancel)\\b", re.I)''',
     '''_STOP_WORDS = re.compile(r"\\b(stop|stopp?|strop|stahp|wait|enough|quiet|shut up|hold on|hang on|never ?mind|"
                         r"cancel|be quiet)\\b", re.I)'''),
])

# ---- agent: weather and time back (cheap, asked for), shorter memory (fewer tokens per turn) ----
patch("fda/pc/agent.py", [
    ('''                   "list_apps", "open_app", "app_settings"]''',
     '''                   "list_apps", "open_app", "app_settings", "weather", "get_time"]'''),
    ('''- You can also run the room when asked:''',
     '''- You can also tell the weather or the time when asked (weather, get_time), then carry on designing.
- You can also run the room when asked:'''),
    ('''    def _trim_context(self, keep_turns: int = 24):''',
     '''    def _trim_context(self, keep_turns: int = 12):'''),
])
print("ok")
