"""Offline evaluation of the three models (the 'evaluate' stage of the LLMOps loop).

Usage:  python evaluate.py            # Model 1 vs Model 2 vs Model 3
        python evaluate.py --ab       # also A/B test prompt versions rag_v1 vs rag_v2
Outputs: logs/eval_results.csv, logs/eval_summary.csv, logs/eval_report.md
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from chatbot import config
from chatbot.llmops import groundedness
from chatbot.pipelines import PIPELINES, Helpdesk
from chatbot.prompts import format_context

# Only explicit "I don't know / not in the documents" statements count as a refusal.
REFUSAL_RE = re.compile(
    r"don.t have (that|this|any|access|real-time|specific)|do not have (that|this|any|access)|"
    r"(not|isn.t) (available|provided|mentioned|specified|included|stated) in|no information|"
    r"cannot (find|provide)|can.t provide|couldn.t verify|unable to (find|provide|answer)", re.I)


def keyword_hit(pattern: str, text: str) -> bool:
    """Each expected keyword is a regex, so '3 months|three months' both count."""
    return re.search(pattern, text, re.I) is not None


def retrieval_metrics(hd, items):
    hits, rr = [], []
    for it in items:
        if not it["pages"] or not it["answerable"]:
            continue
        results, _ = hd.retriever.search(it["q"])
        ranks = [i for i, r in enumerate(results, 1) if r["chunk"].page in it["pages"]]
        hits.append(bool(ranks))
        rr.append(1 / ranks[0] if ranks else 0.0)
    return float(np.mean(hits)), float(np.mean(rr))


def run(ab: bool):
    items = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    hd = Helpdesk(log=False)
    systems = {
        "Model 1 - LLM only": lambda q: hd.llm_only(q),
        "Model 2 - RAG": lambda q: hd.rag(q),
        "Model 3 - RAG + LLMOps + PEFT": lambda q: hd.rag_ops(q, use_cache=False, peft=True),
        "Ablation: Model 3 without PEFT": lambda q: hd.rag_ops(q, use_cache=False, peft=False),
    }
    if not hd.llm.has_adapter:
        print("WARNING: no LoRA adapter found - run finetune_peft.py first; "
              "Model 3 will run on the base model.")
    if ab:
        systems["A/B: RAG + prompt rag_v2"] = lambda q: hd.rag(q, prompt_version="rag_v2")

    rows = []
    for name, fn in systems.items():
        for it in items:
            a = fn(it["q"])
            evidence = format_context(hd.retriever.search(it["q"])[0])
            refused = bool(REFUSAL_RE.search(a.text)) or a.text.startswith("I can only help")
            rows.append(dict(
                system=name, question=it["q"], answerable=it["answerable"], answer=a.text,
                keyword_recall=(np.mean([keyword_hit(k, a.text) for k in it["kw"]])
                                if it["kw"] else np.nan),
                # groundedness is only meaningful for actual answers, not refusals
                groundedness=np.nan if refused else groundedness(a.text, evidence),
                refused=refused,
                flags=";".join(a.flags),
                correct_refusal=(refused if not it["answerable"] else np.nan),
                latency_ms=a.total_ms, completion_tokens=a.completion_tokens))
            print(f"[{name}] {it['q'][:60]:<60} {a.total_ms:7.0f} ms")

    df = pd.DataFrame(rows)
    df.to_csv(config.LOG_DIR / "eval_results.csv", index=False)
    ans = df[df.answerable]
    summary = pd.DataFrame({
        "Keyword recall (answerable)": ans.groupby("system").keyword_recall.mean(),
        "Groundedness of answers": df.groupby("system").groundedness.mean(),
        "Refusal accuracy (unanswerable)": df.groupby("system").correct_refusal.mean(),
        "Wrong refusals (answerable)": ans.groupby("system").refused.mean(),
        "Avg latency ms": df.groupby("system").latency_ms.mean(),
        "p95 latency ms": df.groupby("system").latency_ms.quantile(0.95),
        "Avg output tokens": df.groupby("system").completion_tokens.mean(),
    }).reindex(list(systems)).round(3)
    summary.to_csv(config.LOG_DIR / "eval_summary.csv")

    hit, mrr = retrieval_metrics(hd, items)
    report = [
        "# Evaluation report", "",
        f"- LLM: `{hd.llm.model_name}` on `{hd.llm.device}`; "
        f"PEFT adapter for Model 3: `{hd.llm.adapter or 'none'}`",
        f"- Embeddings: `{config.EMBED_MODEL}`, hybrid alpha={config.HYBRID_ALPHA}, top_k={config.TOP_K}",
        f"- Questions: {len(items)} ({int(sum(i['answerable'] for i in items))} answerable, "
        f"{int(sum(not i['answerable'] for i in items))} unanswerable/out-of-scope)",
        f"- Retrieval Hit@{config.TOP_K}: **{hit:.2f}**, MRR: **{mrr:.2f}**", "",
        summary.to_markdown(), "",
    ]
    (config.LOG_DIR / "eval_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="also A/B test prompt rag_v1 vs rag_v2")
    run(ap.parse_args().ab)
