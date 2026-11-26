"""CHRIST (Deemed to be University) Student Helpdesk Portal.

Run:  streamlit run app.py
Tabs: Student Helpdesk (Inquiry Portal) | Comparative Evaluation | Service Telemetry | Knowledge Audit
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from chatbot import config
from chatbot.pipelines import PIPELINES, Helpdesk
from chatbot.prompts import PROMPTS

st.set_page_config(
    page_title="Student Knowledge Helpdesk | CHRIST (Deemed to be University)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# University Institutional Palette: Deep Oxford Navy (#002147), Classic Gold (#C59B27), Off-white/Parchment (#F9F9FB)
NAVY_PRIMARY = "#002147"
NAVY_ACCENT = "#0A2F5C"
GOLD_PRIMARY = "#C59B27"
GOLD_LIGHT = "#E8C86A"
SURFACE_BG = "#F8F9FA"
TEXT_MAIN = "#1C2430"
BORDER_COLOR = "#DDE2EC"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

  /* Global root typography and layout */
  html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {TEXT_MAIN};
  }}
  .block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
  }}

  /* Top Institutional Navigation Bar */
  .cu-topbar {{
    background-color: {NAVY_PRIMARY};
    border-bottom: 3.5px solid {GOLD_PRIMARY};
    border-radius: 6px;
    padding: 1rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 14px rgba(0, 33, 71, 0.12);
    margin-bottom: 1.4rem;
  }}
  .cu-crest-group {{
    display: flex;
    align-items: center;
    gap: 1.2rem;
  }}
  .cu-crest-shield {{
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, {GOLD_PRIMARY} 0%, #A88118 100%);
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {NAVY_PRIMARY};
    font-family: 'Cinzel', serif;
    font-size: 28px;
    font-weight: 700;
    box-shadow: inset 0 0 0 2px rgba(255,255,255,0.4), 0 2px 6px rgba(0,0,0,0.25);
  }}
  .cu-title-block {{
    display: flex;
    flex-direction: column;
  }}
  .cu-inst-name {{
    font-family: 'Cinzel', serif;
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #FFFFFF;
    text-transform: uppercase;
    margin: 0;
    line-height: 1.25;
  }}
  .cu-sub-header {{
    font-size: 0.82rem;
    color: {GOLD_LIGHT};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
  }}
  .cu-right-tag {{
    text-align: right;
    display: none;
  }}
  @media (min-width: 768px) {{
    .cu-right-tag {{ display: block; }}
  }}
  .cu-portal-badge {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(197, 155, 39, 0.4);
    color: #F0F4FA;
    font-size: 0.76rem;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.04em;
  }}

  /* Institutional Notice Banner */
  .cu-notice {{
    background: #FFFFFF;
    border: 1px solid {BORDER_COLOR};
    border-left: 4px solid {NAVY_PRIMARY};
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.85rem;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }}
  .cu-notice strong {{
    color: {NAVY_PRIMARY};
  }}

  /* Navigation Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0.6rem;
    border-bottom: 1.5px solid {BORDER_COLOR};
    background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 0.88rem;
    font-weight: 600;
    color: #475569;
    padding: 0.6rem 1.1rem;
    border-radius: 4px 4px 0 0;
    border: none;
    transition: all 0.15s ease-in-out;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {NAVY_PRIMARY};
    background: #F1F4F9;
  }}
  .stTabs [aria-selected="true"] {{
    color: {NAVY_PRIMARY} !important;
    background: #FFFFFF !important;
    border-bottom: 3px solid {GOLD_PRIMARY} !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{
    background-color: {GOLD_PRIMARY} !important;
    height: 3px !important;
  }}

  /* Message Feed: Portal Q&A Design instead of bubbly chat */
  [data-testid="stChatMessage"] {{
    background: #FFFFFF;
    border: 1px solid {BORDER_COLOR};
    border-radius: 5px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
  }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
  [data-testid="stChatMessage"]:has(.cu-response-badge) {{
    background: #FBFBFC;
    border-left: 4px solid {NAVY_PRIMARY};
  }}
  .cu-response-badge {{
    display: inline-block;
    background: {NAVY_PRIMARY};
    color: #FFFFFF;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
  }}

  /* Reference Box */
  .cu-citation-box {{
    background: #F4F6F9;
    border: 1px solid #D5DBE5;
    border-radius: 4px;
    padding: 0.6rem 0.85rem;
    margin-top: 0.6rem;
    font-size: 0.82rem;
  }}
  .cu-citation-box strong {{
    color: {NAVY_PRIMARY};
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
    background-color: #FAFBFC;
    border-right: 1px solid {BORDER_COLOR};
  }}
  .cu-sidebar-header {{
    background: {NAVY_PRIMARY};
    color: #FFFFFF;
    padding: 0.9rem;
    border-radius: 4px;
    border-bottom: 2.5px solid {GOLD_PRIMARY};
    margin-bottom: 1rem;
  }}
  .cu-sidebar-header h3 {{
    font-family: 'Cinzel', serif;
    font-size: 1.05rem;
    margin: 0;
    color: #FFFFFF;
  }}
  .cu-sidebar-header p {{
    font-size: 0.75rem;
    color: {GOLD_LIGHT};
    margin: 0.2rem 0 0 0;
  }}

  /* Metric cards */
  [data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid {BORDER_COLOR};
    border-left: 3.5px solid {NAVY_PRIMARY};
    border-radius: 4px;
    padding: 0.65rem 0.9rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }}
  [data-testid="stMetricValue"] {{
    color: {NAVY_PRIMARY};
    font-weight: 700;
  }}
</style>
""", unsafe_allow_html=True)

# Portal Master Banner with Official Christ University Logo & Crest
st.markdown(f"""
<div class="cu-topbar">
  <div class="cu-crest-group">
    <img src="https://christuniversity.in/images/logo.png" alt="CHRIST (Deemed to be University)" style="height: 56px; background: #FFFFFF; padding: 5px 8px; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);" />
    <div class="cu-title-block">
      <h1 class="cu-inst-name">CHRIST (Deemed to be University)</h1>
      <span class="cu-sub-header">Office of Student Affairs · Student Helpdesk Portal (Handbook 2026-27)</span>
    </div>
  </div>
  <div class="cu-right-tag">
    <span class="cu-portal-badge">Academic Year 2026–2027</span>
  </div>
</div>
<div class="cu-notice">
  <img src="https://christuniversity.in/images/logo.png" style="height: 18px; opacity: 0.85;" />
  <span><strong>Official Institutional Handbook Registry:</strong> Authorized information service verified against the Christite Academic & Regulations Handbook. All answers reference audited sections and printed pages.</span>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Institutional Knowledge Base Index...")
def get_helpdesk():
    return Helpdesk()


hd = get_helpdesk()

# ------------------------------------------------------------------ Administrative Control Panel (Sidebar)
with st.sidebar:
    st.markdown("""
    <div class="cu-sidebar-header" style="text-align: center;">
      <img src="https://christuniversity.in/images/logo.png" style="max-width: 170px; background: #FFFFFF; padding: 4px 8px; border-radius: 4px; margin-bottom: 8px;" />
      <h3>System Configuration</h3>
      <p>Handbook Model & Governance Registry</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Operating Pipeline",
        list(PIPELINES),
        index=2,
        format_func=lambda x: {
            "llm": "Standard LLM (Unassisted)",
            "rag": "Knowledge Retrieved (RAG)",
            "ragops": "Institutional Standard (RAG + Verification + PEFT)"
        }.get(x, PIPELINES.get(x))
    )

    prompt_version = st.selectbox(
        "Prompt Protocol",
        ["rag_v2", "rag_v1"],
        format_func=lambda p: f"{p} (Strict Institutional Verification)" if p == "rag_v2" else f"{p} (Standard RAG Prompt)"
    )

    use_peft = st.toggle(
        "Enable Fine-Tuned PEFT Adapter",
        value=hd.llm.has_adapter,
        disabled=not hd.llm.has_adapter,
        help="Applies the low-rank adaptation trained on handbook passage citation and refusal constraints."
    )

    if hd.llm.has_adapter:
        with st.expander(f"Adapter Specification: {hd.llm.adapter}", expanded=False):
            card = hd.llm.adapter_card
            if card:
                st.caption(
                    f"**Architecture:** {card.get('method', 'LoRA')} (Rank {card.get('lora', {}).get('r', 16)})\n"
                    f"**Parameters:** {card.get('trainable_params', 0):,} ({100 * card.get('trainable_params', 0) / card.get('total_params', 1):.2f}% trainable)\n"
                    f"**Dataset Hash:** `{card.get('data_sha256', 'N/A')}`\n"
                    f"**Target Model:** `{card.get('base_model', 'Qwen2.5-0.5B')}`"
                )
    st.divider()
    st.markdown(
        f"**Knowledge Base Index:** `{len(hd.retriever.chunks)} sections`  \n"
        f"**Embedding Model:** `all-MiniLM-L6-v2`  \n"
        f"**Hardware Acceleration:** `{hd.llm.device.upper()}`"
    )
    st.caption("Office of Academic Regulations · CHRIST (Deemed to be University)")


def render_meta(a):
    col_bits = []
    if a.total_ms:
        col_bits.append(f"Processing: {a.total_ms:.0f} ms")
    if a.groundedness is not None:
        col_bits.append(f"Handbook Grounding: {a.groundedness:.0%}")
    if a.top_relevance is not None:
        col_bits.append(f"Retrieval Match: {a.top_relevance:.2f}")
    if a.cache_hit:
        col_bits.append("⚡ Semantic Cache Direct")
    if a.flags:
        col_bits.append("Status: " + ", ".join(a.flags))

    st.markdown(f"<div style='font-size:0.75rem; color:#64748B; margin-top:0.3rem;'>{' · '.join(col_bits)}</div>", unsafe_allow_html=True)
    if a.sources:
        with st.expander(f"Handbook Citations ({len(a.sources)} verified sections)"):
            for i, s in enumerate(a.sources, 1):
                st.markdown(f"**[{i}] {s['citation']}** (Index score: `{s['score']}`)")


# Navigation Tabs
tab_helpdesk, tab_compare, tab_telemetry, tab_audit = st.tabs([
    "🏛️ Student Helpdesk Portal",
    "⚖️ Comparative Pipeline Analysis",
    "📈 Service Telemetry & Governance",
    "📜 Handbook Knowledge Registry"
])

# ------------------------------------------------------------------ Tab 1: Student Helpdesk
with tab_helpdesk:
    st.session_state.setdefault("history", [])

    quick_inquiries = [
        "What is the minimum attendance required to appear for End Semester Examinations (ESE)?",
        "What is the late fee policy for library book returns?",
        "How do I apply for a fee concession or scholarship?",
        "What are the regulations regarding Continuous Internal Assessment (CIA) repeat?"
    ]

    if not st.session_state.history:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid {BORDER_COLOR}; border-radius:5px; padding:1.2rem; margin-bottom:1.2rem;">
              <h3 style="font-family:'Cinzel', serif; font-size:1.15rem; color:{NAVY_PRIMARY}; margin:0 0 0.5rem 0;">
                Welcome to the Student Assistance & Handbook Registry
              </h3>
              <p style="font-size:0.88rem; color:#475569; margin:0 0 1rem 0;">
                Inquire about academic regulations, examination requirements, attendance thresholds, fee schedules, or student welfare.
                Responses are generated strictly against the official Christite Handbook 2026–27.
              </p>
              <div style="font-size:0.78rem; font-weight:600; color:{NAVY_PRIMARY}; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.5rem;">
                Frequent Regulations & Topics
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        cols = st.columns(len(quick_inquiries))
        picked = None
        for col, query in zip(cols, quick_inquiries):
            if col.button(query, key=f"quick_{query[:15]}", use_container_width=True):
                picked = query
    else:
        picked = None

    # Render History
    for i, (q, a) in enumerate(st.session_state.history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**Student Query:** {q}")

        with st.chat_message("assistant", avatar="🏛️"):
            pipeline_label = {
                "llm": "Baseline Generator (Direct)",
                "rag": "Knowledge Retrieved (RAG)",
                "ragops": "Institutional Standard (Model 3)"
            }.get(a.pipeline, a.pipeline)

            st.markdown(f'<span class="cu-response-badge">{pipeline_label}</span>', unsafe_allow_html=True)
            st.markdown(a.text)
            render_meta(a)

            if a.trace_id:
                c1, c2, _ = st.columns([1, 1, 10])
                if c1.button("👍 Verified", key=f"up{i}"):
                    hd.tracer.feedback(a.trace_id, 1)
                    st.toast("Feedback recorded in Institutional Audit Logs.")
                if c2.button("👎 Inaccurate", key=f"down{i}"):
                    hd.tracer.feedback(a.trace_id, -1)
                    st.toast("Flagged for Administrative Review Queue.")

    if query_input := (st.chat_input("Enter student inquiry regarding handbook rules, attendance, exams, fees...") or picked):
        with st.spinner("Searching official handbook regulations and verifying citations..."):
            if mode == "ragops":
                ans = hd.rag_ops(query_input, prompt_version=prompt_version, peft=use_peft)
            else:
                ans = hd.rag(query_input, prompt_version=prompt_version) if mode == "rag" else hd.llm_only(query_input)

        st.session_state.history.append((query_input, ans))
        st.rerun()

# ------------------------------------------------------------------ Tab 2: Comparative Pipeline Analysis
with tab_compare:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
      <h3 style="font-family:'Cinzel', serif; font-size:1.15rem; color:{NAVY_PRIMARY}; margin:0 0 0.2rem 0;">
        Multi-Pipeline Comparative Evaluation
      </h3>
      <p style="font-size:0.85rem; color:#64748B; margin:0;">
        Execute a single inquiry across all three architectural pipelines simultaneously to evaluate factual accuracy, citations, and hallucination containment.
      </p>
    </div>
    """, unsafe_allow_html=True)

    test_q = st.text_input("Test Regulation Inquiry", "What is the fine for returning a library book late?")
    if st.button("Evaluate All 3 Configurations", type="primary"):
        cols = st.columns(3)
        keys = ["llm", "rag", "ragops"]
        titles = [
            "Model 1: Direct LLM (Unassisted)",
            "Model 2: Standard RAG",
            "Model 3: RAG + LLMOps + PEFT"
        ]

        for col, key, title in zip(cols, keys, titles):
            with col:
                st.markdown(f"**{title}**")
                with st.spinner(f"Evaluating {key}..."):
                    if key == "ragops":
                        res = hd.rag_ops(test_q, use_cache=False, peft=use_peft)
                    elif key == "rag":
                        res = hd.rag(test_q)
                    else:
                        res = hd.llm_only(test_q)

                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid {BORDER_COLOR}; border-radius:4px; padding:0.85rem; min-height:160px;">
                  {res.text}
                </div>
                """, unsafe_allow_html=True)
                render_meta(res)

# ------------------------------------------------------------------ Tab 3: Service Telemetry & Governance
with tab_telemetry:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
      <h3 style="font-family:'Cinzel', serif; font-size:1.15rem; color:{NAVY_PRIMARY}; margin:0 0 0.2rem 0;">
        LLMOps Service Telemetry & Audit Logs
      </h3>
      <p style="font-size:0.85rem; color:#64748B; margin:0;">
        Operational monitoring of latency, token utilization, grounding indices, and student verification feedback recorded in <code>logs/llmops.db</code>.
      </p>
    </div>
    """, unsafe_allow_html=True)

    df = hd.tracer.dataframe()
    if df.empty:
        st.info("No query telemetry recorded yet. Perform an inquiry in the Student Helpdesk tab to populate audit logs.")
    else:
        df["time"] = pd.to_datetime(df.ts, unit="s")
        lora = df.model.str.contains("lora")
        df["pipeline_name"] = df.pipeline.map(PIPELINES)

        m = st.columns(6)
        m[0].metric("Total Queries", len(df))
        m[1].metric("Mean Latency", f"{df.total_ms.mean() / 1000:.2f} s")
        m[2].metric("p95 Latency", f"{df.total_ms.quantile(.95) / 1000:.2f} s")
        m[3].metric("Handbook Grounding", f"{df.groundedness.mean():.0%}")
        m[4].metric("Semantic Cache Hits", f"{df.cache_hit.mean():.0%}")
        fb = df.feedback.dropna()
        m[5].metric("Satisfaction Index", f"{(fb > 0).mean():.0%}" if len(fb) else "–", f"{len(fb)} ratings")

        st.markdown("##### Performance Breakdown by Model")
        summary_table = df.groupby("pipeline_name").agg(
            Inquiries=("id", "count"),
            Mean_Latency_ms=("total_ms", "mean"),
            Mean_Grounding=("groundedness", "mean"),
            Mean_Tokens=("completion_tokens", "mean"),
            Cache_Hits=("cache_hit", "sum")
        ).round(2)
        st.dataframe(summary_table, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Query Latency Timeline (ms)")
            st.line_chart(df.pivot_table(index="time", columns="pipeline_name", values="total_ms"))
        with c2:
            st.markdown("##### Quality & Guardrail Flag Distribution")
            flags = pd.Series([f for fl in df["flags"].dropna() for f in json.loads(fl)])
            if flags.empty:
                st.caption("Zero policy or grounding flags triggered.")
            else:
                st.bar_chart(flags.value_counts(), color=NAVY_PRIMARY)

        st.markdown("##### Complete Audit Trail")
        st.dataframe(
            df[["time", "pipeline_name", "prompt_version", "query", "answer", "total_ms", "groundedness", "cache_hit", "flags", "feedback"]].head(50),
            use_container_width=True
        )

# ------------------------------------------------------------------ Tab 4: Knowledge Registry
with tab_audit:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
      <h3 style="font-family:'Cinzel', serif; font-size:1.15rem; color:{NAVY_PRIMARY}; margin:0 0 0.2rem 0;">
        Christite Handbook 2026–27 Knowledge Index
      </h3>
      <p style="font-size:0.85rem; color:#64748B; margin:0;">
        Inspection of indexed sections, printed handbook pagination, and offline evaluation benchmark reports.
      </p>
    </div>
    """, unsafe_allow_html=True)

    report = config.LOG_DIR / "eval_report.md"
    if report.exists():
        st.markdown("#### Offline Evaluation Report")
        st.markdown(report.read_text(encoding="utf-8"))

    res = config.LOG_DIR / "eval_results.csv"
    if res.exists():
        with st.expander("Per-Question Benchmark Dataset (31 Questions)"):
            st.dataframe(pd.read_csv(res), use_container_width=True)

