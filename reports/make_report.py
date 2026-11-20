"""Build reports/Christ_Helpdesk_Model_Report.pdf from the latest evaluation results.

Run:  python -m reports.make_report        (after python evaluate.py)
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from chatbot import config
from reports.report_data import SYSTEMS, collect

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
PDF = OUT_DIR / "Christ_Helpdesk_Model_Report.pdf"

# Reference categorical palette (validated: adjacent CVD dE >= 9.1). Contrast relief: every bar
# carries a visible value label and every chart has a table with the same numbers.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


# ------------------------------------------------------------------ charts
def _bar_panel(ax, labels, values, fmt, title, xmax=None, percent=False):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=MODEL_COLORS[:len(values)], height=0.62,
            edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt(v), va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(list(y), labels)
    ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlim(0, (xmax or max(values)) * 1.28)
    if percent:
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def chart_panels(stats, specs, path, height=2.6):
    labels = [v["label"] for v in stats.values()]
    fig, axes = plt.subplots(1, len(specs), figsize=(7.2, height), sharey=True)
    for i, (ax, (key, title, fmt, xmax)) in enumerate(zip(axes, specs)):
        _bar_panel(ax, labels, [v[key] for v in stats.values()], fmt, title, xmax, percent=xmax == 1.0)
        if i:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_architecture(path):
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")
    cats = {"all": ("#f1f0ec", INK2, "All 3 models"),
            "rag": ("#fde7dd", MODEL_COLORS[1], "Model 2 and Model 3 (RAG)"),
            "m3": ("#dcf3ea", MODEL_COLORS[2], "Model 3 only (LLMOps + PEFT)")}
    steps = [
        (13.2, "all", "Student asks a question", ""),
        (11.75, "m3", "1. Safety check", "hides personal data (email, phone, register no.), blocks trick prompts"),
        (10.3, "m3", "2. Answer cache", "same question asked before?  reply instantly"),
        (8.85, "rag", "3. Search the handbook", "keyword (BM25) + meaning (MiniLM) search: best 4 of 324 passages"),
        (7.4, "m3", "4. Relevance gate", "passages not related to the question?  politely say \"I don't know\""),
        (5.95, "all", "5. Local AI model writes the answer", "Qwen2.5-0.5B-Instruct  (+ LoRA / PEFT adapter in Model 3)"),
        (4.5, "m3", "6. Answer check", "every number must be in the handbook; answer must match passages"),
        (3.05, "all", "Answer shown to the student", "with handbook page citations (Models 2 and 3)"),
        (1.6, "m3", "7. Logs, dashboard and feedback", "time, tokens, warnings and thumbs up/down saved for review"),
    ]
    x0, w, h = 2.0, 6.0, 1.05
    for i, (y, cat, title, sub) in enumerate(steps):
        fill, edge, _ = cats[cat]
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=fill, edgecolor=edge, linewidth=1.6))
        ax.text(x0 + 0.25, y + (0.17 if sub else 0), title, fontsize=10, fontweight="bold",
                color=INK, va="center")
        if sub:
            ax.text(x0 + 0.25, y - 0.24, sub, fontsize=7.6, color=INK2, va="center")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y - h / 2), (x0 + w / 2, steps[i + 1][0] + h / 2),
                                         arrowstyle="-|>", mutation_scale=11, color=INK2, linewidth=1.1))
    # Model 1 shortcut
    ax.add_patch(FancyArrowPatch((x0, 13.2), (x0, 5.95), connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>", mutation_scale=11, color=MODEL_COLORS[0],
                                 linewidth=1.6, linestyle="--"))
    ax.text(0.05, 9.6, "Model 1:\nstraight to\nthe AI model\n(no handbook,\nno checks)",
            fontsize=7.5, color=INK, ha="left", va="center")
    # knowledge base
    ax.add_patch(FancyBboxPatch((8.25, 8.25), 1.65, 1.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="#ffffff", edgecolor=INK2, linewidth=1.0, linestyle=":"))
    ax.text(9.07, 8.85, "Christite\nHandbook\n2026-27 PDF", fontsize=7.5, ha="center", va="center", color=INK)
    ax.add_patch(FancyArrowPatch((8.25, 8.85), (x0 + w, 8.85), arrowstyle="-|>",
                                 mutation_scale=10, color=INK2, linewidth=1.0))
    # legend
    for j, (fill, edge, name) in enumerate(cats.values()):
        lx = 0.2 + j * 3.3
        ax.add_patch(FancyBboxPatch((lx, 0.12), 0.45, 0.32, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=fill, edgecolor=edge, linewidth=1.4))
        ax.text(lx + 0.6, 0.28, name, fontsize=8, va="center", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ PDF helpers
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor(INK), spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor(INK),
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14.5,
                      textColor=colors.HexColor(INK), spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11.5, textColor=colors.HexColor(INK2))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=H1, fontSize=22, leading=27, alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor(INK2))


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [P(f"&bull;&nbsp; {t}") for t in items]


def table(rows, widths, header=True, highlight_col=None):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if (header and r == 0) else CELL)
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6d5d0")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeeea")))
    if highlight_col is not None:
        style.append(("BACKGROUND", (highlight_col, 1), (highlight_col, -1), colors.HexColor("#eef8f3")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * ih / iw)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(2 * cm, 1.2 * cm, "CHRIST University Student Helpdesk Chatbot - model report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------ build
def build():
    stats, df = collect()
    m1, m2, m3, ab = (stats[k] for k in SYSTEMS)
    card = json.loads((config.active_adapter() / "model_card.json").read_text(encoding="utf-8"))
    reg = json.loads((config.MODELS_DIR / "registry.json").read_text(encoding="utf-8"))
    n_chunks = 324

    pct = lambda v: f"{v * 100:.0f}%"
    sec = lambda v: f"{v:.1f} s"
    num = lambda v: f"{v:,.0f}"

    chart_architecture(FIG_DIR / "architecture.png")
    chart_panels(stats, [("recall", "Correct facts", pct, 1.0),
                         ("grounded", "Backed by handbook", pct, 1.0),
                         ("refusal_acc", "Honest \"I don't know\"", pct, 1.0)],
                 FIG_DIR / "effectiveness.png")
    chart_panels(stats, [("lat_avg", "Average reply time", sec, None),
                         ("lat_p95", "Slowest 5% of replies (p95)", sec, None)],
                 FIG_DIR / "latency.png")
    chart_panels(stats, [("in_tok", "Input tokens per question", num, None),
                         ("out_tok", "Output tokens per answer", num, None)],
                 FIG_DIR / "tokens.png")

    names = ["Model 1 - LLM only", "Model 2 - RAG", "Model 3 - RAG + LLMOps + PEFT",
             "Model 3 without PEFT (test only)"]
    story = []

    # ---------------- page 1: title + summary
    story += [Spacer(1, 1.2 * cm),
              P("CHRIST University Student Helpdesk Chatbot", TITLE),
              P("How three AI models compare: plain LLM, RAG, and RAG + LLMOps + PEFT", SUB),
              P(f"LCA Activity: Experimental and Problem-Based Learning (LLM / RAG / LLMOps) &nbsp;|&nbsp; "
                f"{date.today():%d %B %Y}", SUB),
              Spacer(1, 0.6 * cm),
              P("The result in one paragraph", H2),
              P("We built a chatbot that answers student questions using the official <b>Christite Handbook "
                "2026-27</b>, and built it three ways. A plain AI model on its own (Model 1) got only "
                f"<b>{pct(m1['recall'])}</b> of handbook facts right and confidently made things up. Letting it "
                f"<b>search the handbook first</b> (Model 2, RAG) raised that to <b>{pct(m2['recall'])}</b>. "
                "Adding <b>safety checks, monitoring and a small fine-tuned adapter</b> (Model 3, LLMOps + PEFT) "
                f"made it the <b>safest and fastest</b>: it said \"I don't know\" to {pct(m3['refusal_acc'])} of "
                f"off-topic questions, every answer was backed by the handbook, and it replied in "
                f"<b>{m3['lat_avg']:.1f} seconds</b> on average. The cost: it answered fewer handbook "
                f"questions fully ({pct(m3['recall'])}), because the adapter was trained on only "
                f"{card['train']['examples']} examples."),
              Spacer(1, 0.2 * cm),
              table([["", "Model 1<br/>LLM only", "Model 2<br/>RAG", "Model 3<br/>RAG + LLMOps + PEFT"],
                     ["Correct facts (26 handbook questions)", pct(m1["recall"]), pct(m2["recall"]), pct(m3["recall"])],
                     ["Answers backed by the handbook", pct(m1["grounded"]), pct(m2["grounded"]), pct(m3["grounded"])],
                     ["Says \"I don't know\" to off-topic questions (5)", pct(m1["refusal_acc"]), pct(m2["refusal_acc"]), pct(m3["refusal_acc"])],
                     ["Average reply time", sec(m1["lat_avg"]), sec(m2["lat_avg"]), sec(m3["lat_avg"])],
                     ["Tokens per question (read + written)", num(m1["in_tok"] + m1["out_tok"]),
                      num(m2["in_tok"] + m2["out_tok"]), num(m3["in_tok"] + m3["out_tok"])],
                     ["Best for", "nothing on its own", "most facts answered", "safety, speed, trust"]],
                    [6.4 * cm, 3.4 * cm, 3.4 * cm, 3.8 * cm], highlight_col=3),
              Spacer(1, 0.3 * cm),
              P("Key takeaways", H2)]
    story += bullets([
        "<b>Never use a plain AI model for university questions.</b> It does not know CHRIST's rules and "
        "invents answers (it said Christ College was founded in 1920; it was 1969).",
        "<b>RAG is the biggest single improvement</b>: about 5 times more correct facts, shorter answers, "
        "and every answer can be checked against a handbook page.",
        "<b>LLMOps makes it safe to put in front of students</b>: it blocks trick prompts, hides personal "
        "data, refuses off-topic questions and records everything for review.",
        "<b>PEFT (fine-tuning) changed the style, not the knowledge</b>: shorter, faster, always cited, and "
        "it fixed one error no rule could catch, but it needs more training examples to keep full accuracy.",
    ])
    story.append(PageBreak())

    # ---------------- page 2: what we used + architecture
    story += [P("1. What we built and which models we used", H1),
              table([["Part", "What we used", "In simple words"],
                     ["AI language model (LLM)", f"{config.LLM_MODEL} (0.5 billion parameters)",
                      "A small open-source model from Alibaba's Qwen family that runs on a laptop. It writes the answers."],
                     ["Fine-tuning (PEFT)", f"LoRA adapter <b>{card['adapter']}</b>: rank {card['lora']['r']}, "
                      f"{card['trainable_params'] / 1e6:.1f} M trained parameters "
                      f"({100 * card['trainable_params'] / card['total_params']:.2f}% of the model)",
                      "A small add-on that teaches the model how to answer from handbook passages. The main model is not changed. Used in Model 3 only."],
                     ["Search (retrieval)", "BM25 keyword search + all-MiniLM-L6-v2 meaning search",
                      "Finds the 4 handbook passages most related to the question."],
                     ["Knowledge base", f"Christite Handbook 2026-27 (108-page PDF), split into {n_chunks} passages",
                      "The only source of facts. Each passage keeps its chapter and page number for citations."],
                     ["Computer", "Laptop with NVIDIA RTX 3050 (6 GB) GPU",
                      "Everything runs locally: no internet or paid API needed, so student questions stay private."],
                     ["Software", "Python, PyTorch, Hugging Face Transformers, PEFT, Streamlit, SQLite",
                      "Streamlit gives the chat website and dashboard; SQLite stores the logs."]],
                    [3.6 * cm, 6.0 * cm, 7.4 * cm]),
              Spacer(1, 0.3 * cm),
              P("The three models", H2)]
    story += bullets([
        "<b>Model 1 - LLM only:</b> the question goes straight to the AI model. It answers from what it "
        "learned during its general training. It has never seen the handbook.",
        "<b>Model 2 - RAG (Retrieval-Augmented Generation):</b> the chatbot first searches the handbook, then "
        "gives the AI model the 4 best passages and asks it to answer from them.",
        "<b>Model 3 - RAG + LLMOps + PEFT (hybrid):</b> Model 2 plus safety checks before and after the AI "
        "model, an answer cache, logging and a dashboard (LLMOps), and the LoRA fine-tuned adapter (PEFT).",
    ])
    story += [PageBreak(), P("2. Architecture: how a question becomes an answer", H1),
              P("Boxes are coloured by which model uses that step. Model 1 skips the handbook and all checks "
                "(dashed blue arrow).", SMALL),
              img(FIG_DIR / "architecture.png", 16.2), PageBreak()]

    # ---------------- effectiveness
    story += [P("3. How effectively each model responded", H1),
              P("We tested every model on the <b>same 31 questions</b>: 26 that the handbook can answer (for "
                "example \"What is the fine for returning a library book late?\") and 5 it cannot (for example "
                "\"Who won the FIFA World Cup in 2022?\" or \"What is on the cafeteria menu today?\"). "
                "A good helpdesk answers the first kind correctly and says \"I don't know\" to the second."),
              img(FIG_DIR / "effectiveness.png", 17.0),
              P("<b>Correct facts</b> = share of the key facts present in the answer. <b>Backed by the handbook</b> "
                "= share of answer sentences that can be found in the retrieved passages. <b>Says \"I don't "
                "know\"</b> = share of the 5 off-topic questions answered honestly instead of with a made-up "
                "answer. The fourth bar (Model 3 without PEFT) is a test run that shows what the PEFT adapter "
                "alone changes.", SMALL),
              Spacer(1, 0.2 * cm),
              P("What happened to each question", H2),
              table([["Model", "Fully correct", "Partly correct", "Wrong answer", "Said \"I don't know\" (but could answer)",
                      "Off-topic: refused", "Off-topic: made up"]] +
                    [[n, f"{v['correct']} / 26", f"{v['partial']}", f"{v['wrong']}", f"{v['refused_ans']}",
                      f"{v['unans_refused']} / 5", f"{v['unans_madeup']}"] for n, v in zip(names, stats.values())],
                    [4.6 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm, 2.9 * cm, 2.0 * cm, 2.2 * cm]),
              Spacer(1, 0.25 * cm),
              P("In simple words", H2)]
    story += bullets([
        f"<b>Model 1</b> got only {m1['correct']} of 26 fully right and invented answers to "
        f"{m1['unans_madeup']} of 5 off-topic questions (for example \"South Africa won the 2022 World Cup\").",
        f"<b>Model 2</b> got {m2['correct']} of 26 right, but still made up answers to "
        f"{m2['unans_madeup']} of 5 off-topic questions because it always tries to answer.",
        f"<b>Model 3</b> refused {m3['unans_refused']} of 5 off-topic questions and every answer was backed by "
        f"the handbook. It got {m3['correct']} of 26 fully right: the adapter sometimes picked the wrong "
        f"passage (for example it gave the library fine when asked about the visa fine) or played safe and "
        f"said \"I don't know\" ({m3['refused_ans']} times).",
    ])
    story.append(PageBreak())

    # ---------------- latency + tokens
    story += [P("4. Speed (latency) of each model", H1),
              P("Latency is the time from sending a question to getting the full answer, measured on the "
                "laptop GPU for all 31 questions."),
              img(FIG_DIR / "latency.png", 17.0),
              table([["Model", "Average", "Typical (median)", "Fastest", "Slowest 5% (p95)", "Slowest"]] +
                    [[n, sec(v["lat_avg"]), sec(v["lat_med"]), f"{v['lat_min']:.2f} s", sec(v["lat_p95"]),
                      sec(v["lat_max"])] for n, v in zip(names, stats.values())],
                    [5.6 * cm, 2.0 * cm, 2.5 * cm, 2.0 * cm, 2.8 * cm, 2.1 * cm]),
              Spacer(1, 0.2 * cm)]
    story += bullets([
        f"<b>Model 1 is the slowest</b> ({sec(m1['lat_avg'])} on average): without the handbook it writes "
        f"long, padded answers (about {m1['out_tok']:.0f} tokens each).",
        f"<b>Model 3 is the fastest</b> ({sec(m3['lat_avg'])} average, {sec(m3['lat_p95'])} for the slowest 5%). "
        f"Its fine-tuned answers are short (about {m3['out_tok']:.0f} tokens), off-topic questions are refused "
        "in about 0.02 s without calling the AI model, and the adapter is merged into the model so it adds no delay.",
        "Writing words (output tokens) is what takes time; reading the handbook passages (input tokens) is fast.",
    ])
    story += [PageBreak(), P("5. How many tokens each model used", H1),
              P("A <b>token</b> is a piece of a word (roughly 3/4 of an English word). <b>Input tokens</b> are "
                "what the model reads (the question plus handbook passages); <b>output tokens</b> are what it "
                "writes. With a paid cloud AI service you pay per token, so this is also a cost measure."),
              img(FIG_DIR / "tokens.png", 17.0),
              table([["Model", "Input tokens per question", "Output tokens per answer",
                      "Total tokens for all 31 questions"]] +
                    [[n, num(v["in_tok"]), num(v["out_tok"]), num(v["total_tok"])]
                     for n, v in zip(names, stats.values())],
                    [5.6 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm])]
    story += bullets([
        f"RAG models read about {m2['in_tok']:.0f} tokens per question because they include 4 handbook "
        f"passages; Model 1 reads only about {m1['in_tok']:.0f}. Reading is cheap and fast.",
        f"Model 3 writes the fewest tokens ({m3['out_tok']:.0f} per answer) and uses <b>zero</b> tokens on "
        "questions it blocks or refuses at the relevance gate, so it used the fewest tokens in total among "
        "the RAG models.",
    ])
    story.append(PageBreak())

    # ---------------- examples
    def ans(system, q):
        r = df[(df.system == system) & (df.question == q)]
        if not len(r):
            return "-"
        a = r.answer.iloc[0].replace("**", "").replace("\n", " ")
        return a if len(a) <= 260 else a[:260].rsplit(" ", 1)[0] + " ..."

    story += [P("6. Same question, three answers", H1),
              P("Real answers from the test run (shortened where long).")]
    for q, note in [("What is the fine for returning a library book late?", "Correct answer: Rs 5 per day (handbook p.15)."),
                    ("What is the reduction in weightage for a repeat Mid Semester Examination?",
                     "Correct answer: 15% (handbook p.51)."),
                    ("Who won the FIFA World Cup in 2022?", "Not a university question: the right reply is \"I don't know\"."),
                    ("When was Christ College established?", "Correct answer: 1969 (handbook p.1).")]:
        rows = [["Question", f"<b>{q}</b><br/><font color='{INK2}'>{note}</font>"]]
        for n, key in zip(["Model 1 - LLM only", "Model 2 - RAG", "Model 3 - RAG + LLMOps + PEFT"], SYSTEMS):
            rows.append([n, ans(key, q)])
        story += [KeepTogether([table(rows, [4.3 * cm, 12.7 * cm], header=False)]), Spacer(1, 0.3 * cm)]

    # ---------------- PEFT experiment
    v1 = next(v for v in reg["versions"] if v["adapter"] == "lora-helpdesk-v1")
    story += [PageBreak(), P("7. The PEFT experiment (fine-tuning the hybrid model)", H1),
              P("<b>PEFT (Parameter-Efficient Fine-Tuning)</b> trains a small add-on (a LoRA adapter) instead of "
                f"the whole model. We trained only {card['trainable_params'] / 1e6:.1f} million of "
                f"{card['total_params'] / 1e6:.0f} million parameters, in about {card['minutes']:.0f} minutes on the "
                f"laptop, using {card['train']['examples']} handbook question-answer examples. The adapter learns "
                "a <b>behaviour</b>: answer briefly from the given passages, cite them, and say \"I don't know\" "
                "when the answer is not there. The facts still come from the handbook search."),
              table([["Version", "What changed", "Correct facts", "Wrong \"I don't know\"", "Decision"],
                     ["No adapter", "Model 3 without PEFT", pct(ab["recall"]), pct(ab["wrong_refusal"]), "baseline"],
                     ["v1", "3 training rounds, 25% \"say I don't know\" examples", pct(m3["recall"]),
                      pct(m3["wrong_refusal"]), "<b>in use</b>"],
                     ["v2", "fewer \"I don't know\" examples (10%), stop early", "62%", "4%",
                      "rolled back"]],
                    [2.2 * cm, 6.4 * cm, 2.4 * cm, 3.0 * cm, 3.0 * cm]),
              Spacer(1, 0.2 * cm),
              P("v2 said \"I don't know\" less often, but those answers turned into <b>confident wrong answers</b>. "
                "For students, \"please check with the office\" is much safer than a wrong fine or deadline, so we "
                "went back to v1. Keeping every version with its test results (a <b>model registry</b>) made this a "
                "one-line, documented decision. This is the LLMOps cycle: train, test, compare, roll back.", BODY)]

    # ---------------- ranking + recommendations
    story += [P("8. Overall comparison and recommendation", H1),
              table([["", "Model 1", "Model 2", "Model 3"],
                     ["Accuracy on handbook questions", "Poor", "<b>Best</b>", "Good"],
                     ["Honesty (no made-up answers)", "Poor", "Fair", "<b>Best</b>"],
                     ["Speed", "Slowest", "Fast", "<b>Fastest</b>"],
                     ["Shows sources (page citations)", "No", "Yes", "Yes"],
                     ["Safety (trick prompts, personal data)", "No", "No", "<b>Yes</b>"],
                     ["Monitoring and feedback", "No", "No", "<b>Yes</b>"],
                     ["Ready for students?", "No", "Not without checks", "<b>Yes, after more fine-tuning data</b>"]],
                    [6.2 * cm, 3.2 * cm, 3.4 * cm, 4.2 * cm], highlight_col=3),
              Spacer(1, 0.25 * cm),
              P("<b>Recommendation:</b> use <b>Model 3 (the hybrid)</b> for a real student helpdesk, because it is "
                "the only one that is safe, honest, traceable and fast. To close its accuracy gap with Model 2:"),
             ]
    story += bullets([
        "grow the fine-tuning set from 72 to 300+ checked question-answer examples (taken from real student "
        "questions marked thumbs-down in the dashboard);",
        "try a larger local model (1.5B to 7B parameters), which is one setting change;",
        "keep testing every new version against the same questions before switching it on.",
    ])
    story += [P("Limitations", H2)]
    story += bullets([
        "Only 31 test questions, all written by us; a larger test set would give firmer numbers.",
        "Times are from one laptop GPU and can vary by about 30% between runs.",
        "The chatbot only knows the 2026-27 handbook. Fees, timetables and new circulars are not in it, so it "
        "should send those questions to the right office.",
    ])

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="CHRIST Helpdesk Chatbot - Model Report", author="LCA Activity")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    build()
