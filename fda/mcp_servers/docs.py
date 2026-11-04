"""Question answering over local documents, fully offline: drop PDF, DOCX, XLSX, CSV, Markdown or text files into
the `docs_dir` folder (default fda/docs) and ask. Files are split into passages with a place to cite (page,
heading, sheet, rows), ranked with BM25, and the best passages go back to the agent, which answers in a sentence
and names the file and section. Nothing leaves the machine.

The index is rebuilt only for files that changed (modified time), so asking again is instant.
"""
from __future__ import annotations

import csv
import logging
import math
import re
from collections import Counter
from pathlib import Path

log = logging.getLogger("fda.mcp.docs")
KINDS = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".markdown", ".txt"}
_STOP = set("""a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these they this to was we were what when where which who
why will with would you your about any all also just not no yes please tell say says said""".split())
_PASSAGE = 700            # characters per passage (about one paragraph or two)


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):                 # a light stemmer: "refunds" ~ "refund"
            if len(w) > 4 + len(suf) - 1 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def _chunks(section: str, text: str) -> list[tuple[str, str]]:
    """Split long text into passages at paragraph or sentence ends."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9])", text):
        if len(cur) + len(para) > _PASSAGE and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para.strip() + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return [(section, p) for p in parts]


def _read(p: Path) -> list[tuple[str, str]]:
    """[(section, passage)] for one file."""
    ext = p.suffix.lower()
    out: list[tuple[str, str]] = []
    if ext == ".pdf":
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(str(p)).pages, 1):
            out += _chunks(f"page {i}", page.extract_text() or "")
    elif ext == ".docx":
        import docx
        section, buf = "start", []
        for para in docx.Document(str(p)).paragraphs:
            if para.style is not None and para.style.name.lower().startswith(("heading", "title")) and para.text.strip():
                out += _chunks(section, "\n\n".join(buf))
                section, buf = para.text.strip()[:80], []
            elif para.text.strip():
                buf.append(para.text)
        out += _chunks(section, "\n\n".join(buf))
    elif ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = [[("" if c is None else str(c)) for c in r] for r in ws.iter_rows(values_only=True)]
            out += _table(f"sheet {ws.title}", rows)
    elif ext == ".csv":
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            out += _table("rows", list(csv.reader(f)))
    else:                                                    # markdown / text: sections at headings
        section, buf = "start", []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s{0,3}#{1,6}\s+(.*)", line)
            if m:
                out += _chunks(section, "\n".join(buf))
                section, buf = m.group(1).strip()[:80], []
            else:
                buf.append(line)
        out += _chunks(section, "\n".join(buf))
    return out


def _table(where: str, rows: list[list[str]]) -> list[tuple[str, str]]:
    """Tables: groups of rows, each row written as "header: value" so a question can match the column names."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    head, body = rows[0], rows[1:] or [rows[0]]
    out = []
    for i in range(0, len(body), 8):
        lines = ["; ".join(f"{h}: {v}" for h, v in zip(head, r) if v.strip()) for r in body[i:i + 8]]
        out.append((f"{where} {i + 2}-{i + 1 + len(body[i:i + 8])}", "\n".join(lines)))
    return out


class Docs:
    def __init__(self, folder: Path):
        self.dir = folder
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, tuple[float, list[tuple[str, str]]]] = {}    # name -> (mtime, passages)
        self._index = None

    def refresh(self) -> list[str]:
        """Re-read changed files. Returns files that could not be read."""
        bad, seen, changed = [], set(), False
        for p in sorted(self.dir.rglob("*")):
            if p.suffix.lower() not in KINDS or not p.is_file() or p.name.startswith(("~$", ".")):
                continue
            name = p.relative_to(self.dir).as_posix()
            seen.add(name)
            mt = p.stat().st_mtime
            if name in self.files and self.files[name][0] == mt:
                continue
            try:
                self.files[name] = (mt, _read(p))
                log.info("indexed %s: %d passages", name, len(self.files[name][1]))
            except Exception as e:  # noqa: BLE001
                log.warning("could not read %s: %s", name, e)
                bad.append(name)
                self.files[name] = (mt, [])
            changed = True
        for gone in set(self.files) - seen:
            del self.files[gone]
            changed = True
        if changed or self._index is None:
            self._build()
        return bad

    def _build(self):
        docs = [(f, sec, text, Counter(_tokens(f"{Path(f).stem} {sec} {text}")))
                for f, (_, ps) in self.files.items() for sec, text in ps]
        df = Counter(t for *_, tf in docs for t in tf)
        n = max(1, len(docs))
        avg = sum(sum(tf.values()) for *_, tf in docs) / n
        idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._index = (docs, idf, avg or 1.0)

    def search(self, question: str, k: int = 3) -> list[dict]:
        bad = self.refresh()
        docs, idf, avg = self._index
        q = _tokens(question)
        if not q or not docs:
            return []
        k1, b = 1.5, 0.75
        scored = []
        for f, sec, text, tf in docs:
            dl = sum(tf.values())
            s = sum(idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avg)) for t in q if t in tf)
            if s > 0:
                scored.append((s, f, sec, text))
        scored.sort(reverse=True)
        out = [{"file": f, "section": sec, "text": text[:_PASSAGE], "score": round(s, 2)} for s, f, sec, text in scored[:k]]
        if bad:
            out.append({"note": f"could not read: {', '.join(bad)}"})
        return out

    def list(self) -> list[dict]:
        self.refresh()
        return [{"file": f, "passages": len(ps)} for f, (_, ps) in sorted(self.files.items())]
