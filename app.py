"""CHRIST University Student Helpdesk - Streamlit UI.

Run:  streamlit run app.py
Tabs: Chat (pick a model) | Compare 3 models | LLMOps dashboard | Evaluation
"""
import json

import pandas as pd
import streamlit as st

from chatbot import config
from chatbot.pipelines import PIPELINES, Helpdesk
from chatbot.prompts import PROMPTS

st.set_page_config(page_title="CHRIST Student Helpdesk", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading local LLM, embeddings and the Christite Handbook index...")
def get_helpdesk():
    return Helpdesk()


hd = get_helpdesk()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 CHRIST Helpdesk")
    st.caption("Local RAG + LLMOps chatbot over the Christite Handbook 2026-27")
    mode = st.radio("Model", list(PIPELINES), index=2, format_func=PIPELINES.get)
    prompt_version = st.selectbox("Prompt version (Model 3)", ["rag_v2", "rag_v1"],
                                  help=PROMPTS["rag_v2"]["description"])
    use_peft = st.toggle("PEFT LoRA adapter (Model 3)", value=hd.llm.has_adapter,
                         disabled=not hd.llm.has_adapter,
                         help="Model 3 uses a LoRA adapter fine-tuned to answer from retrieved "
                              "handbook passages. Models 1 and 2 always use the base LLM.")
    if hd.llm.has_adapter:
        card = hd.llm.adapter_card
        with st.expander(f"🧩 Adapter: {hd.llm.adapter}"):
            if card:
                st.markdown(
                    f"**Method:** {card['method']}, r={card['lora']['r']}, alpha={card['lora']['alpha']}  \n"
                    f"**Trainable params:** {card['trainable_params']:,} of {card['total_params']:,} "
                    f"({100 * card['trainable_params'] / card['total_params']:.2f}%)  \n"
                    f"**Data:** {card['train']['examples']} train / {card['train']['val_examples']} val "
                    f"(sha {card['data_sha256']})  \n**Trained:** {card['created']} "
                    f"in {card['minutes']} min")
                hist = pd.DataFrame(card["history"]).set_index("epoch")
                st.line_chart(hist[["train_loss", "val_loss"]])
    else:
        st.caption("No LoRA adapter yet - run `python finetune_peft.py`.")
    st.divider()
    st.markdown(f"**LLM:** `{hd.llm.model_name}` on `{hd.llm.device}`  \n"
                f"**Embeddings:** `{config.EMBED_MODEL.split('/')[-1]}`  \n"
                f"**Chunks indexed:** {len(hd.retriever.chunks)}")
    st.caption("Answers come from the handbook. Always confirm deadlines and fees "
               "on christuniversity.in or with your department.")


def render_meta(a):
    bits = [f"⏱ {a.total_ms:.0f} ms"]
    if a.groundedness is not None:
        bits.append(f"grounded {a.groundedness:.0%}")
    if a.top_relevance is not None:
        bits.append(f"relevance {a.top_relevance:.2f}")
    if a.adapter:
        bits.append(f"🧩 {a.adapter}")
    if a.cache_hit:
        bits.append("⚡ cache hit")
    if a.flags:
        bits.append("🚩 " + ", ".join(a.flags))
    st.caption(" · ".join(bits))
    if a.sources:
        with st.expander(f"Sources ({len(a.sources)})"):
            for i, s in enumerate(a.sources, 1):
                st.markdown(f"[{i}] **{s['citation']}** — score {s['score']}")


chat_tab, compare_tab, ops_tab, eval_tab = st.tabs(
    ["💬 Chat", "⚖️ Compare 3 models", "📊 LLMOps dashboard", "🧪 Evaluation"])

# ------------------------------------------------------------------ chat
with chat_tab:
    st.session_state.setdefault("history", [])
    for i, (q, a) in enumerate(st.session_state.history):
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            st.markdown(f"**{PIPELINES[a.pipeline]}**\n\n{a.text}")
            render_meta(a)
            if a.trace_id:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up{i}"):
                    hd.tracer.feedback(a.trace_id, 1)
                    st.toast("Thanks for the feedback!")
                if c2.button("👎", key=f"down{i}"):
                    hd.tracer.feedback(a.trace_id, -1)
                    st.toast("Logged - this helps us improve.")

    examples = ["What is the minimum attendance to write the ESE?",
                "What is the fine for returning a library book late?",
                "How do I apply for a fee concession?",
                "How many courses can I register for CIA repeat at a time?"]
    st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))
    if q := st.chat_input("Ask a question about CHRIST University..."):
        with st.spinner("Thinking..."):
            a = (hd.rag_ops(q, prompt_version=prompt_version, peft=use_peft) if mode == "ragops"
                 else hd.ask(q, mode))
        st.session_state.history.append((q, a))
        st.rerun()

# ------------------------------------------------------------------ compare
with compare_tab:
    st.subheader("Same question, three models")
    q = st.text_input("Question", "What is the fine for returning a library book late?")
    if st.button("Run all three", type="primary"):
        cols = st.columns(3)
        for col, key in zip(cols, PIPELINES):
            with col, st.spinner(PIPELINES[key]):
                a = (hd.rag_ops(q, use_cache=False, peft=use_peft) if key == "ragops"
                     else hd.ask(q, key))
                st.markdown(f"#### {PIPELINES[key]}")
                st.write(a.text)
                render_meta(a)

# ------------------------------------------------------------------ LLMOps dashboard
with ops_tab:
    df = hd.tracer.dataframe()
    if df.empty:
        st.info("No traces yet - ask a few questions in the Chat tab.")
    else:
        df["time"] = pd.to_datetime(df.ts, unit="s")
        df["model"] = df.pipeline.map(PIPELINES) + df.model.str.contains("lora").map({True: " (LoRA)", False: ""})
        m = st.columns(6)
        m[0].metric("Requests", len(df))
        m[1].metric("Avg latency", f"{df.total_ms.mean():.0f} ms")
        m[2].metric("p95 latency", f"{df.total_ms.quantile(.95):.0f} ms")
        m[3].metric("Avg groundedness", f"{df.groundedness.mean():.0%}")
        m[4].metric("Cache hit rate", f"{df.cache_hit.mean():.0%}")
        fb = df.feedback.dropna()
        m[5].metric("👍 rate", f"{(fb > 0).mean():.0%}" if len(fb) else "–", f"{len(fb)} votes")

        st.markdown("##### Per-model summary")
        st.dataframe(df.groupby("model").agg(
            requests=("id", "count"), avg_latency_ms=("total_ms", "mean"),
            avg_groundedness=("groundedness", "mean"), avg_output_tokens=("completion_tokens", "mean"),
            cache_hits=("cache_hit", "sum")).round(2), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Latency over time (ms)")
            st.line_chart(df.pivot_table(index="time", columns="model", values="total_ms"))
        with c2:
            st.markdown("##### Guardrail / quality flags")
            flags = pd.Series([f for fl in df["flags"].dropna() for f in json.loads(fl)])
            if flags.empty:
                st.caption("No flags raised yet.")
            else:
                st.bar_chart(flags.value_counts())

        st.markdown("##### Recent traces")
        st.dataframe(df[["time", "model", "prompt_version", "query", "answer", "total_ms",
                         "groundedness", "top_relevance", "cache_hit", "flags", "feedback"]].head(50),
                     width="stretch")
        low = df[(df.groundedness < config.MIN_GROUNDEDNESS) | (df.feedback < 0)]
        if not low.empty:
            st.markdown("##### ⚠️ Review queue (low groundedness or 👎)")
            st.dataframe(low[["time", "model", "query", "answer", "groundedness", "feedback"]],
                         width="stretch")

# ------------------------------------------------------------------ evaluation
with eval_tab:
    report = config.LOG_DIR / "eval_report.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
        res = config.LOG_DIR / "eval_results.csv"
        if res.exists():
            with st.expander("Per-question results"):
                st.dataframe(pd.read_csv(res), width="stretch")
    else:
        st.info("Run `python evaluate.py --ab` to generate the offline evaluation report.")
