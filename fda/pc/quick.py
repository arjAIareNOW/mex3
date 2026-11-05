"""Predefined answers from a plain text file (edit it in Notepad while FDA runs): when what was heard matches
one of the questions, the answer is spoken at once, without the LLM (saves the ~0.5-1 s LLM round trip and
costs no tokens). Anything that does not match well goes to the LLM as usual.

File format (quick_answers.txt):
    # comment
    Q: what can you do          <- one or more ways people ask it
    Q: what can you make
    A: I build websites, apps and small games while we talk.   <- what FDA says (one line)
    (blank line between entries)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from loguru import logger

# words that change nothing about the question ("hey Elkit, um, so what can you do?"). Pronouns are NOT here:
# "who made you" and "who made the gallery", "your name" and "my name" are different questions.
_FILLER = {"um", "uh", "er", "hmm", "hey", "hi", "okay", "ok", "so", "well", "please", "just", "actually",
           "the", "a", "an", "and", "quick", "question"}


_PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "he", "him", "his", "she", "her",
             "it", "its", "they", "them", "their", "this", "that"}


def _words(text: str, name: str = "") -> list[str]:
    t = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'"))
    drop = _FILLER | ({name.lower()} if name else set())
    return [w for w in t.split() if w not in drop]


class QuickAnswers:
    def __init__(self, path: Path, name: str = "", threshold: float = 0.85):
        self.path, self.name, self.threshold = path, name, threshold
        self.entries: list[tuple[list[list[str]], str]] = []     # ([question words...], answer)
        self._mtime = None

    def _reload(self):
        """Read the file again when it changed (so edits in Notepad apply at once, no restart)."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self.entries, self._mtime = [], None
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        entries, qs = [], []
        for raw in self.path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tag, _, rest = line.partition(":")
            tag, rest = tag.strip().upper(), rest.strip()
            if tag == "Q" and rest:
                qs.append(_words(rest, self.name))
            elif tag == "A" and rest and qs:
                entries.append(([q for q in qs if q], rest))
                qs = []
        self.entries = entries
        logger.info(f"quick answers: {len(entries)} loaded from {self.path.name}")

    def match(self, heard: str) -> tuple[str, float] | None:
        """(answer, score) for the best matching question, or None if nothing matches well enough."""
        self._reload()
        h = _words(heard, self.name)
        if not h or not self.entries:
            return None
        best, best_score = None, 0.0
        hs = " ".join(h)
        hp = set(h) & _PRONOUNS
        for questions, answer in self.entries:
            for q in questions:
                if set(q) & _PRONOUNS != hp:       # "my name" is not "your name", "made you" not "made it"
                    continue
                score = SequenceMatcher(None, hs, " ".join(q)).ratio()
                # all the question's words said, with at most one extra ("what can you do here")
                if len(q) >= 3 and set(q) <= set(h) and len(h) <= len(q) + 1:
                    score = max(score, 0.9)
                if score > best_score:
                    best, best_score = answer, score
        return (best, best_score) if best is not None and best_score >= self.threshold else None
