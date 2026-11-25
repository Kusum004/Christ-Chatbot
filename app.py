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

NAVY, GOLD = "#0b2d5b", "#f2b705"
st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.2rem; }}
  /* header banner */
  .christ-hero {{
    display: flex; align-items: center; gap: 1.1rem;
    background: linear-gradient(120deg, {NAVY} 0%, #173f7a 100%);
    border-bottom: 4px solid {GOLD}; border-radius: 0.8rem;
    padding: 1.1rem 1.5rem; margin-bottom: 1rem; color: #fff;
  }}
  .christ-crest {{
    flex: 0 0 auto; width: 58px; height: 58px; border-radius: 50%;
    background: {GOLD}; color: {NAVY}; display: flex; align-items: center;
    justify-content: center; font: 700 1.9rem Georgia, serif;
    box-shadow: 0 0 0 3px rgba(255,255,255,.25);
  }}
  .christ-hero .uni {{ font: 600 0.8rem sans-serif; letter-spacing: .12em;
    text-transform: uppercase; color: {GOLD}; }}
  .christ-hero h1 {{ color: #fff; font-family: Georgia, serif; font-size: 1.85rem;
    margin: 0.1rem 0 0.15rem; padding: 0; }}
  .christ-hero .tag {{ color: #d6deee; font-size: 0.92rem; }}
  .christ-hero .motto {{ color: {GOLD}; font-style: italic; }}
  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid #dfe4ee; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: #4a5675; }}
  .stTabs [aria-selected="true"] {{ color: {NAVY}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: {GOLD}; height: 3px; }}
  /* chat */
  [data-testid="stChatMessage"] {{ border-radius: 0.8rem; padding: 0.8rem 1rem; }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
  [data-testid="stChatMessage"]:has(.bot-badge) {{
    background: #f3f5fa; border-left: 4px solid {GOLD};
  }}
  .bot-badge {{ display: inline-block; background: {NAVY}; color: #fff;
    font-size: 0.72rem; font-weight: 600; letter-spacing: .03em;
    padding: 0.15rem 0.6rem; border-radius: 999px; margin-bottom: 0.35rem; }}
  .welcome {{ border: 1px solid #dfe4ee; border-top: 4px solid {NAVY};
    border-radius: 0.8rem; padding: 1rem 1.2rem; margin-bottom: 0.8rem; background: #fbfcfe; }}
  .welcome h3 {{ color: {NAVY}; margin: 0 0 0.3rem; font-family: Georgia, serif; }}
  /* metrics */
  [data-testid="stMetric"] {{ background: #f3f5fa; border-left: 4px solid {NAVY};
    border-radius: 0.6rem; padding: 0.6rem 0.9rem; }}
  [data-testid="stMetricValue"] {{ color: {NAVY}; }}
  /* sidebar */
  [data-testid="stSidebar"] {{ border-right: 4px solid {GOLD}; }}
  [data-testid="stSidebar"] h1 {{ color: {GOLD}; font-family: Georgia, serif; }}
  [data-testid="stSidebar"] code {{ background: rgba(255,255,255,.12); color: #fff; }}
  @media (max-width: 640px) {{
    .christ-hero {{ padding: 0.9rem 1rem; }} .christ-hero h1 {{ font-size: 1.4rem; }}
    .christ-crest {{ width: 44px; height: 44px; font-size: 1.4rem; }}
  }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="christ-hero">
  <div class="christ-crest">C</div>
  <div>
    <div class="uni">CHRIST (Deemed to be University) · Bengaluru</div>
    <h1>Student Helpdesk</h1>
    <div class="tag">Answers from the Christite Handbook 2026-27 ·
      <span class="motto">Excellence and Service</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

USER_AVATAR, BOT_AVATAR = "🧑‍🎓", "🎓"


@st.cache_resource(show_spinner="Loading local LLM, embeddings and the Christite Handbook index...")
def get_helpdesk():
    return Helpdesk()


hd = get_helpdesk()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("CHRIST Helpdesk")
    st.caption("Local RAG + LLMOps chatbot over the Christite Handbook 2026-27")
    st.divider()
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
    examples = ["What is the minimum attendance to write the ESE?",
                "What is the fine for returning a library book late?",
                "How do I apply for a fee concession?",
                "How many courses can I register for CIA repeat at a time?"]
    picked = None
    if not st.session_state.history:
        st.markdown(
            '<div class="welcome"><h3>Hello! How can I help you today?</h3>'
            'Ask about attendance, exams, library, scholarships, conduct, international '
            'student rules and more. Every answer cites the handbook page it came from.</div>',
            unsafe_allow_html=True)
        cols = st.columns(len(examples))
        for col, e in zip(cols, examples):
            if col.button(e, width="stretch"):
                picked = e

    for i, (q, a) in enumerate(st.session_state.history):
        st.chat_message("user", avatar=USER_AVATAR).write(q)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(f'<span class="bot-badge">{PIPELINES[a.pipeline]}</span>',
                        unsafe_allow_html=True)
            st.markdown(a.text)
            render_meta(a)
            if a.trace_id:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up{i}"):
                    hd.tracer.feedback(a.trace_id, 1)
                    st.toast("Thanks for the feedback!")
                if c2.button("👎", key=f"down{i}"):
                    hd.tracer.feedback(a.trace_id, -1)
                    st.toast("Logged - this helps us improve.")

    if q := (st.chat_input("Ask a question about CHRIST University...") or picked):
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
                st.markdown(f'<span class="bot-badge">{PIPELINES[key]}</span>',
                            unsafe_allow_html=True)
                st.write(a.text)
                render_meta(a)

# ------------------------------------------------------------------ LLMOps dashboard
with ops_tab:
    df = hd.tracer.dataframe()
    if df.empty:
        st.info("No traces yet - ask a few questions in the Chat tab.")
    else:
        df["time"] = pd.to_datetime(df.ts, unit="s")
        lora = df.model.str.contains("lora")
        df["model"] = df.pipeline.map(PIPELINES) + (
            (df.pipeline == "ragops") & ~lora).map({True: " (adapter off)", False: ""})
        m = st.columns(6)
        m[0].metric("Requests", len(df))
        m[1].metric("Avg latency", f"{df.total_ms.mean() / 1000:.2f} s")
        m[2].metric("p95 latency", f"{df.total_ms.quantile(.95) / 1000:.2f} s")
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
                st.bar_chart(flags.value_counts(), color=NAVY)

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
