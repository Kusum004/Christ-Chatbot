"""Collect per-model numbers for the PDF report from logs/eval_results.csv.
Input (prompt) tokens are recomputed exactly as each pipeline builds its prompt."""
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from chatbot import config
from chatbot.prompts import build_messages, format_context
from chatbot.retriever import HybridRetriever

SYSTEMS = {"Model 1 - LLM only": "Model 1\nLLM only",
           "Model 2 - RAG": "Model 2\nRAG",
           "Model 3 - RAG + LLMOps + PEFT": "Model 3\nRAG + LLMOps + PEFT",
           "Ablation: Model 3 without PEFT": "Model 3\nwithout PEFT"}
PROMPT = {"Model 1 - LLM only": "llm_only_v1", "Model 2 - RAG": "rag_v1",
          "Model 3 - RAG + LLMOps + PEFT": "rag_v2", "Ablation: Model 3 without PEFT": "rag_v2"}


def collect():
    df = pd.read_csv(config.LOG_DIR / "eval_results.csv")
    df = df[df.system.isin(SYSTEMS)]
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    ret = HybridRetriever()
    ctx = {q: format_context(ret.search(q)[0]) for q in df.question.unique()}

    def prompt_tokens(row):
        flags = str(row["flags"]) if pd.notna(row["flags"]) else ""
        if "low_relevance" in flags or "prompt_injection" in flags:
            return 0                                   # Model 3 stopped before calling the LLM
        msgs = build_messages(PROMPT[row.system], row.question,
                              "" if row.system.startswith("Model 1") else ctx[row.question])
        return len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))["input_ids"])

    df["prompt_tokens"] = df.apply(prompt_tokens, axis=1)
    out = {}
    for s, label in SYSTEMS.items():
        x = df[df.system == s]
        a, u = x[x.answerable], x[~x.answerable]
        out[s] = dict(
            label=label,
            lat_avg=x.latency_ms.mean() / 1000, lat_med=x.latency_ms.median() / 1000,
            lat_p95=x.latency_ms.quantile(.95) / 1000, lat_min=x.latency_ms.min() / 1000,
            lat_max=x.latency_ms.max() / 1000,
            in_tok=x.prompt_tokens.mean(), out_tok=x.completion_tokens.mean(),
            total_tok=(x.prompt_tokens + x.completion_tokens).sum(),
            recall=a.keyword_recall.mean(), grounded=x.groundedness.mean(),
            refusal_acc=u.refused.mean(), wrong_refusal=a.refused.mean(),
            correct=int((a.keyword_recall == 1).sum()),
            partial=int(((a.keyword_recall > 0) & (a.keyword_recall < 1)).sum()),
            wrong=int(((a.keyword_recall == 0) & ~a.refused).sum()),
            refused_ans=int(a.refused.sum()),
            unans_refused=int(u.refused.sum()), unans_madeup=int((~u.refused).sum()),
            n_ans=len(a), n_unans=len(u))
    return out, df


if __name__ == "__main__":
    out, _ = collect()
    for s, v in out.items():
        print(s, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in v.items() if k != "label"})
