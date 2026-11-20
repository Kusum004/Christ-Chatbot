"""LLMOps layer: guardrails, groundedness scoring, semantic cache, tracing and feedback."""
import json
import re
import sqlite3
import time
import uuid

import numpy as np

from . import config

# ---------------------------------------------------------------- guardrails
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "register_no": re.compile(r"(?<!\d)\d{7}(?!\d)"),
}
INJECTION_PATTERNS = re.compile(
    r"ignore (all |the )?(previous|above|prior) (instructions|rules)|system prompt|"
    r"you are now|jailbreak|developer mode|pretend to be|reveal your (prompt|instructions)", re.I)
BLOCKLIST = re.compile(r"\b(hack|crack|leak)\b.*\b(portal|marks|exam paper|question paper|password)\b", re.I)


def input_guardrails(query: str):
    """Returns (clean_query, flags, block_message_or_None)."""
    flags = []
    q = query.strip()
    if not q:
        return q, ["empty"], "Please type a question."
    if len(q) > config.MAX_QUERY_CHARS:
        q = q[:config.MAX_QUERY_CHARS]
        flags.append("truncated")
    for name, pat in PII_PATTERNS.items():
        if pat.search(q):
            q = pat.sub(f"<{name.upper()}>", q)
            flags.append(f"pii_{name}")
    if INJECTION_PATTERNS.search(q):
        flags.append("prompt_injection")
        return q, flags, "I can only help with CHRIST University student-support questions."
    if BLOCKLIST.search(q):
        flags.append("policy_violation")
        return q, flags, "I can't help with that. Please follow the university code of conduct."
    return q, flags, None


# ---------------------------------------------------------------- groundedness
_STOP = set("""a an the and or of to in on for is are was were be been by with as at from that this
it its can will may must should their they you your our we he she his her not no do does did have has
had also any all into than then there which who whom what when where how about if each per through""".split())


def _content_tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9%]+", text.lower()) if t not in _STOP and len(t) > 2]


def groundedness(answer: str, context: str) -> float:
    """Fraction of answer sentences whose content words are mostly (>=60%) found in the context.
    A cheap, model-free hallucination proxy suitable for online monitoring."""
    ctx = set(_content_tokens(context))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(_content_tokens(s)) >= 3]
    if not sentences:
        return 1.0 if ctx else 0.0
    supported = sum(1 for s in sentences
                    if np.mean([t in ctx for t in _content_tokens(s)]) >= 0.6)
    return supported / len(sentences)


_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15",
              "twenty": "20", "thirty": "30"}


def _clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)                          # citation markers [1]
    return re.sub(r"(^|\n)\s*(\d{1,2}[.)]|step \d+:?)\s", r"\1 ", text, flags=re.I)  # list numbering


def _numbers(text: str):
    text = _clean(text)
    nums = {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    nums |= {d for w, d in _NUM_WORDS.items() if re.search(rf"\b{w}\b", text, re.I)}
    return nums


def _percents(text: str):
    return {m for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|per ?cent)", _clean(text), re.I)}


def unsupported_numbers(answer: str, context: str, question: str = "") -> list:
    """Figures (marks, %, fees, days, dates) in the answer that the retrieved context does not
    support. Small LLMs most often hallucinate exactly these. Percentages must appear *as
    percentages* in the context, so '30%' is not excused by an unrelated '70:30'."""
    bad = _numbers(answer) - _numbers(context) - _numbers(question)
    bad |= {f"{p}%" for p in _percents(answer) - _percents(context) - _percents(question)}
    return sorted(n for n in bad if n not in {"0", "1"})


# ---------------------------------------------------------------- semantic cache
class SemanticCache:
    """Returns a stored answer when a new query is near-identical in meaning.
    Cuts latency/compute for frequent helpdesk questions. Keyed by prompt version."""

    def __init__(self, threshold: float = config.CACHE_SIMILARITY):
        self.threshold = threshold
        self.items = []   # (version, embedding, payload)

    def get(self, version: str, emb: np.ndarray):
        best, best_sim = None, 0.0
        for v, e, payload in self.items:
            if v == version:
                sim = float(e @ emb)
                if sim > best_sim:
                    best, best_sim = payload, sim
        return (best, best_sim) if best_sim >= self.threshold else (None, best_sim)

    def put(self, version: str, emb: np.ndarray, payload: dict):
        self.items.append((version, emb, payload))


# ---------------------------------------------------------------- tracing
SCHEMA = """CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY, ts REAL, pipeline TEXT, model TEXT, prompt_version TEXT,
    query TEXT, clean_query TEXT, answer TEXT, sources TEXT, top_relevance REAL,
    retrieval_ms REAL, generation_ms REAL, total_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, groundedness REAL,
    cache_hit INTEGER, flags TEXT, feedback INTEGER, comment TEXT)"""


class Tracer:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def log(self, **row) -> str:
        row.setdefault("id", uuid.uuid4().hex[:12])
        row.setdefault("ts", time.time())
        for key in ("sources", "flags"):
            if key in row and not isinstance(row[key], str):
                row[key] = json.dumps(row[key])
        cols = ",".join(row)
        with self._conn() as c:
            c.execute(f"INSERT INTO traces ({cols}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))
        return row["id"]

    def feedback(self, trace_id: str, score: int, comment: str = ""):
        with self._conn() as c:
            c.execute("UPDATE traces SET feedback=?, comment=? WHERE id=?", (score, comment, trace_id))

    def dataframe(self):
        import pandas as pd
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM traces ORDER BY ts DESC", c)
