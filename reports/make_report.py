"""Build reports/Christ_Helpdesk_Model_Report.pdf (technical report) from the latest evaluation results.

Run:  python -m reports.make_report        (after python evaluate.py --ab)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from chatbot import config
from reports.report_data import SYSTEMS, collect

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
PDF = OUT_DIR / "Christ_Helpdesk_Model_Report.pdf"

# Categorical palette for the four configurations (validated: adjacent CVD dE >= 9.1).
# Every bar carries a value label and every chart has a matching table.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#111111", "#55544f", "#e4e3df", "#ffffff"
NAVY, GOLD = "#0b2d5b", "#c99a06"          # institutional accent, used sparingly
RUNNING_TITLE = "Student Helpdesk Chatbot: LLM vs RAG vs RAG + LLMOps + PEFT"
CONFIG_NAMES = ["M1 LLM only", "M2 RAG", "M3 RAG + LLMOps + PEFT", "M3 without PEFT"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


# ------------------------------------------------------------------ figures
def _bar_panel(ax, labels, values, fmt, title, xmax=None, percent=False):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=MODEL_COLORS[:len(values)], height=0.6, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt(v), va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks(list(y), labels)
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    ax.set_xlim(0, (xmax or max(values)) * 1.28)
    if percent:
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def chart_panels(stats, specs, path, height=2.4):
    labels = [v["label"] for v in stats.values()]
    fig, axes = plt.subplots(1, len(specs), figsize=(7.2, height), sharey=True)
    for i, (ax, (key, title, fmt, xmax)) in enumerate(zip(axes, specs)):
        _bar_panel(ax, labels, [v[key] for v in stats.values()], fmt, title, xmax, percent=xmax == 1.0)
        if i:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def chart_loss(versions, path):
    fig, axes = plt.subplots(1, len(versions), figsize=(7.2, 2.3), sharey=True)
    for ax, v in zip(axes, versions):
        h = pd.DataFrame(v["history"])
        tr = h.dropna(subset=["train_loss"])
        ax.plot(tr.epoch, tr.train_loss, color=MODEL_COLORS[0], lw=2, marker="o", ms=4, label="train")
        ax.plot(h.epoch, h.val_loss, color=MODEL_COLORS[1], lw=2, marker="o", ms=4, label="validation")
        ax.set_title(v["adapter"], loc="left", fontsize=9, color=INK)
        ax.set_xlabel("epoch")
        ax.set_xticks(list(h.epoch))
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def chart_architecture_m1(path):
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.axis("off")
    blue_fill, blue_edge = "#dce9f9", MODEL_COLORS[0]

    # Query
    ax.add_patch(FancyBboxPatch((0.4, 0.9), 2.2, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(1.5, 1.65, "User Query", fontsize=9.2, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(1.5, 1.25, "Raw student question", fontsize=7.0, ha="center", va="center", color=INK2)

    # Arrow 1
    ax.add_patch(FancyArrowPatch((2.6, 1.5), (3.6, 1.5), arrowstyle="-|>", mutation_scale=11, color=blue_edge, linewidth=1.4))
    ax.text(3.1, 1.75, "prompt:\nllm_only_v1", fontsize=6.8, ha="center", va="bottom", color=blue_edge, fontweight="bold")

    # Base LLM
    ax.add_patch(FancyBboxPatch((3.6, 0.7), 2.8, 1.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=blue_fill, edgecolor=blue_edge, linewidth=1.4))
    ax.text(5.0, 1.75, "Base LLM Generator", fontsize=9.2, fontweight="bold", ha="center", va="center", color=NAVY)
    ax.text(5.0, 1.4, "Qwen2.5-0.5B-Instruct", fontsize=7.6, ha="center", va="center", color=INK)
    ax.text(5.0, 1.1, "Parametric pre-training only", fontsize=7.0, ha="center", va="center", color=INK2)
    ax.text(5.0, 0.85, "(No retrieval • No guardrails)", fontsize=6.8, ha="center", va="center", color="#b3261e")

    # Arrow 2
    ax.add_patch(FancyArrowPatch((6.4, 1.5), (7.4, 1.5), arrowstyle="-|>", mutation_scale=11, color=blue_edge, linewidth=1.4))

    # Output
    ax.add_patch(FancyBboxPatch((7.4, 0.9), 2.2, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(8.5, 1.65, "Unverified Answer", fontsize=9.0, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(8.5, 1.25, "High hallucination risk\nNo page citations", fontsize=6.8, ha="center", va="center", color=INK2)

    ax.text(0.4, 2.7, "Model 1: Direct LLM Only (Baseline Architecture)", fontsize=10, fontweight="bold", color=blue_edge)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def chart_architecture_m2(path):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    orange_fill, orange_edge = "#fdebe3", MODEL_COLORS[1]

    # Offline KB branch (top right)
    ax.add_patch(FancyBboxPatch((6.8, 3.1), 2.8, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#ffffff", edgecolor=INK2, linewidth=1.0, linestyle="--"))
    ax.text(8.2, 3.8, "Christite Handbook PDF", fontsize=8.2, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(8.2, 3.4, "TOC parse • 324 cited chunks", fontsize=7.0, ha="center", va="center", color=INK2)

    # User Query
    ax.add_patch(FancyBboxPatch((0.4, 1.5), 1.9, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(1.35, 2.3, "User Query", fontsize=9.0, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(1.35, 1.85, "Raw question", fontsize=7.0, ha="center", va="center", color=INK2)

    # Arrow to Retriever
    ax.add_patch(FancyArrowPatch((2.3, 2.15), (2.9, 2.15), arrowstyle="-|>", mutation_scale=10, color=orange_edge, linewidth=1.3))

    # Hybrid Retriever
    ax.add_patch(FancyBboxPatch((2.9, 1.3), 2.3, 1.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=orange_fill, edgecolor=orange_edge, linewidth=1.3))
    ax.text(4.05, 2.65, "Hybrid Retriever", fontsize=8.8, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(4.05, 2.25, "Dense MiniLM (0.6)", fontsize=7.0, ha="center", va="center", color=INK2)
    ax.text(4.05, 1.95, "+ BM25 lexical (0.4)", fontsize=7.0, ha="center", va="center", color=INK2)
    ax.text(4.05, 1.6, "Top-4 Chunks (Hit@4=1.0)", fontsize=6.8, ha="center", va="center", color=orange_edge, fontweight="bold")

    # Arrow KB -> Retriever
    ax.add_patch(FancyArrowPatch((6.8, 3.3), (4.5, 3.0), arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=1.0, linestyle=":"))

    # Arrow Retriever -> Prompt/LLM
    ax.add_patch(FancyArrowPatch((5.2, 2.15), (5.7, 2.15), arrowstyle="-|>", mutation_scale=10, color=orange_edge, linewidth=1.3))
    ax.text(5.45, 2.35, "rag_v1", fontsize=6.6, ha="center", va="bottom", color=orange_edge, fontweight="bold")

    # Base Generator
    ax.add_patch(FancyBboxPatch((5.7, 1.3), 2.1, 1.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f8f9fa", edgecolor=NAVY, linewidth=1.3))
    ax.text(6.75, 2.65, "Base Generator", fontsize=8.8, fontweight="bold", ha="center", va="center", color=NAVY)
    ax.text(6.75, 2.25, "Qwen2.5-0.5B", fontsize=7.4, ha="center", va="center", color=INK)
    ax.text(6.75, 1.9, "Answers with context", fontsize=7.0, ha="center", va="center", color=INK2)
    ax.text(6.75, 1.6, "(No guardrails / gate)", fontsize=6.8, ha="center", va="center", color="#b3261e")

    # Arrow LLM -> Output
    ax.add_patch(FancyArrowPatch((7.8, 2.15), (8.3, 2.15), arrowstyle="-|>", mutation_scale=10, color=orange_edge, linewidth=1.3))

    # Output
    ax.add_patch(FancyBboxPatch((8.3, 1.5), 1.5, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(9.05, 2.3, "RAG Answer", fontsize=8.6, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(9.05, 1.85, "Cited [n] text\nPartial refusals", fontsize=6.6, ha="center", va="center", color=INK2)

    ax.text(0.4, 4.25, "Model 2: Standard RAG Pipeline Architecture", fontsize=10, fontweight="bold", color=orange_edge)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def chart_architecture_m3(path):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.8); ax.axis("off")
    green_fill, green_edge = "#e2f4ec", MODEL_COLORS[2]

    # Query
    ax.add_patch(FancyBboxPatch((0.2, 4.6), 1.7, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(1.05, 5.45, "User Query", fontsize=8.6, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(1.05, 5.0, "Student query", fontsize=6.8, ha="center", va="center", color=INK2)

    # 1. Guardrails
    ax.add_patch(FancyArrowPatch((1.9, 5.25), (2.3, 5.25), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((2.3, 4.6), 1.8, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=green_fill, edgecolor=green_edge, linewidth=1.2))
    ax.text(3.2, 5.5, "1. Guardrails", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(3.2, 5.15, "PII redaction", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(3.2, 4.85, "Injection block", fontsize=6.8, ha="center", va="center", color=INK2)

    # 2. Semantic Cache
    ax.add_patch(FancyArrowPatch((4.1, 5.25), (4.5, 5.25), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((4.5, 4.6), 2.0, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=green_fill, edgecolor=green_edge, linewidth=1.2))
    ax.text(5.5, 5.5, "2. Semantic Cache", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(5.5, 5.15, "Cosine sim >= 0.95", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(5.5, 4.85, "Scoped cache-key", fontsize=6.8, ha="center", va="center", color=INK2)

    # Cache hit path to response
    ax.add_patch(FancyArrowPatch((5.5, 5.9), (5.5, 6.3), arrowstyle="-", color=green_edge, linewidth=1.1, linestyle=":"))
    ax.add_patch(FancyArrowPatch((5.5, 6.3), (8.9, 6.3), arrowstyle="-", color=green_edge, linewidth=1.1, linestyle=":"))
    ax.add_patch(FancyArrowPatch((8.9, 6.3), (8.9, 3.4), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.1, linestyle=":"))
    ax.text(7.2, 6.45, "Cache Hit (<50ms fast path)", fontsize=6.8, ha="center", va="bottom", color=green_edge, fontweight="bold")

    # 3. Hybrid Retriever
    ax.add_patch(FancyArrowPatch((5.5, 4.6), (5.5, 3.9), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.text(5.7, 4.25, "miss", fontsize=6.6, color=INK2, fontstyle="italic")
    ax.add_patch(FancyBboxPatch((4.3, 2.6), 2.4, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#fdebe3", edgecolor=MODEL_COLORS[1], linewidth=1.2))
    ax.text(5.5, 3.5, "3. Hybrid Retriever", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(5.5, 3.15, "Dense MiniLM + BM25", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(5.5, 2.85, "Top-4 cited chunks", fontsize=6.8, ha="center", va="center", color=MODEL_COLORS[1], fontweight="bold")

    # 4. Relevance Gate
    ax.add_patch(FancyArrowPatch((4.3, 3.25), (3.6, 3.25), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((1.6, 2.6), 2.0, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=green_fill, edgecolor=green_edge, linewidth=1.2))
    ax.text(2.6, 3.5, "4. Relevance Gate", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(2.6, 3.15, "Dense sim < 0.40?", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(2.6, 2.85, "Refuse if out-of-scope", fontsize=6.8, ha="center", va="center", color=green_edge, fontweight="bold")

    # Low relevance refusal arrow down to response
    ax.add_patch(FancyArrowPatch((1.8, 2.6), (1.8, 1.6), arrowstyle="-", color=INK2, linewidth=1.0, linestyle="-."))
    ax.add_patch(FancyArrowPatch((1.8, 1.6), (7.8, 1.6), arrowstyle="-|>", mutation_scale=8, color=INK2, linewidth=1.0, linestyle="-."))
    ax.text(2.1, 1.75, "Low relevance -> templated refusal (no LLM call)", fontsize=6.4, color=INK2, va="bottom")

    # 5. Generator + LoRA Adapter
    ax.add_patch(FancyArrowPatch((2.6, 2.6), (2.6, 1.2), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.text(2.8, 2.0, "Pass: rag_v2", fontsize=6.6, color=green_edge, fontweight="bold")
    ax.add_patch(FancyBboxPatch((2.6, 0.4), 2.7, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#f8f9fa", edgecolor=NAVY, linewidth=1.4))
    ax.text(3.95, 1.35, "5. Generator + LoRA (PEFT)", fontsize=8.5, fontweight="bold", ha="center", va="center", color=NAVY)
    ax.text(3.95, 1.05, "Qwen2.5-0.5B + lora-helpdesk-v1", fontsize=7.0, ha="center", va="center", color=INK)
    ax.text(3.95, 0.75, "r=16, 1.75% trained, merged serving", fontsize=6.8, ha="center", va="center", color=green_edge, fontweight="bold")

    # 6. Output Guardrails
    ax.add_patch(FancyArrowPatch((5.3, 1.05), (5.9, 1.05), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((5.9, 0.4), 1.9, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=green_fill, edgecolor=green_edge, linewidth=1.2))
    ax.text(6.85, 1.35, "6. Output Verifier", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(6.85, 1.05, "Number check", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(6.85, 0.75, "Groundedness >= 0.5", fontsize=6.8, ha="center", va="center", color=INK2)

    # 7. Final Response
    ax.add_patch(FancyArrowPatch((7.8, 1.05), (8.2, 1.05), arrowstyle="-|>", mutation_scale=9, color=green_edge, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((8.2, 0.4), 1.6, 2.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#f5f5f4", edgecolor=INK2, linewidth=1.2))
    ax.text(9.0, 2.9, "7. Response", fontsize=8.5, fontweight="bold", ha="center", va="center", color=INK)
    ax.text(9.0, 2.5, "Verified citation\nHigh grounding", fontsize=6.8, ha="center", va="center", color=INK2)
    ax.text(9.0, 1.7, "8. Telemetry", fontsize=8.2, fontweight="bold", ha="center", va="center", color=green_edge)
    ax.text(9.0, 1.3, "logs/llmops.db\nTraces, Latency\nFeedback rating", fontsize=6.6, ha="center", va="center", color=INK2)
    ax.text(9.0, 0.7, "-> Dashboard", fontsize=6.8, ha="center", va="center", color=NAVY, fontweight="bold")

    ax.text(0.2, 6.55, "Model 3: Hybrid Architecture (RAG + LLMOps + PEFT)", fontsize=10, fontweight="bold", color=green_edge)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def chart_architecture(path):
    # Combined multi-panel overview
    chart_architecture_m1(path.parent / "architecture_m1.png")
    chart_architecture_m2(path.parent / "architecture_m2.png")
    chart_architecture_m3(path.parent / "architecture_m3.png")

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.8), gridspec_kw={"height_ratios": [1.0, 1.4, 2.2]})
    for ax in axes:
        ax.axis("off")

    import matplotlib.image as mpimg
    img1 = mpimg.imread(str(path.parent / "architecture_m1.png"))
    img2 = mpimg.imread(str(path.parent / "architecture_m2.png"))
    img3 = mpimg.imread(str(path.parent / "architecture_m3.png"))

    axes[0].imshow(img1)
    axes[1].imshow(img2)
    axes[2].imshow(img3)

    fig.tight_layout(h_pad=0.8)
    fig.savefig(path, dpi=240)
    plt.close(fig)


# ------------------------------------------------------------------ document styles
BODY = ParagraphStyle("Body", fontName="Times-Roman", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY,
                      textColor=colors.HexColor(INK), spaceAfter=6)
H1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor(NAVY),
                    spaceBefore=12, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor(INK),
                    spaceBefore=8, spaceAfter=3, keepWithNext=1)
CAP = ParagraphStyle("Cap", fontName="Times-Roman", fontSize=9, leading=11.5, alignment=TA_LEFT,
                     textColor=colors.HexColor(INK2), spaceBefore=3, spaceAfter=10)
CAPT = ParagraphStyle("CapT", parent=CAP, spaceBefore=6, spaceAfter=4, keepWithNext=1)
CELL = ParagraphStyle("Cell", fontName="Times-Roman", fontSize=9, leading=11.2)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Times-Bold")
LIST = ParagraphStyle("List", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=8.8, leading=11.5, leftIndent=12,
                      textColor=colors.HexColor(INK), spaceAfter=2)


def P(text, style=BODY):
    return Paragraph(text, style)


def items(rows, numbered=False):
    return [Paragraph(t, LIST, bulletText=(f"{i}." if numbered else "•"))
            for i, t in enumerate(rows, 1)]


class Counter:
    def __init__(self):
        self.fig = self.tab = 0

    def figure(self, text):
        self.fig += 1
        return P(f"<b>Figure {self.fig}.</b> {text}", CAP)

    def table(self, text):
        self.tab += 1
        return P(f"<b>Table {self.tab}.</b> {text}", CAPT)


def table(rows, widths, bold_col=None):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if r == 0 else CELL)
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.HexColor(INK)),
             ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK)),
             ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.HexColor(INK)),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if bold_col is not None:
        style.append(("BACKGROUND", (bold_col, 0), (bold_col, -1), colors.HexColor("#f2f4f8")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * ih / iw)


def framed(path, width_cm):
    t = Table([[img(path, width_cm)]], colWidths=[width_cm * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9a9a96")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def page_decor(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(2.2 * cm, h - 1.35 * cm, RUNNING_TITLE)
    canvas.drawRightString(w - 2.2 * cm, h - 1.35 * cm, "Technical report")
    canvas.setStrokeColor(colors.HexColor(NAVY))
    canvas.setLineWidth(0.6)
    canvas.line(2.2 * cm, h - 1.5 * cm, w - 2.2 * cm, h - 1.5 * cm)
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor(INK))
    canvas.drawCentredString(w / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def title_decor(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor(NAVY))
    canvas.rect(2.2 * cm, h - 2.35 * cm, w - 4.4 * cm, 0.09 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor(GOLD))
    canvas.rect(2.2 * cm, h - 2.47 * cm, w - 4.4 * cm, 0.04 * cm, stroke=0, fill=1)
    canvas.restoreState()


# ------------------------------------------------------------------ build
def build():
    stats, df = collect()
    m1, m2, m3, ab = (stats[k] for k in SYSTEMS)
    card = json.loads((config.active_adapter() / "model_card.json").read_text(encoding="utf-8"))
    reg = json.loads((config.MODELS_DIR / "registry.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(config.LOG_DIR / "eval_summary.csv").set_index("system")
    abp = summary.loc[[i for i in summary.index if i.startswith("A/B")][0]]
    rep_md = (config.LOG_DIR / "eval_report.md").read_text(encoding="utf-8")
    hit = re.search(r"Hit@\d+: \*\*([\d.]+)", rep_md).group(1)
    mrr = re.search(r"MRR: \*\*([\d.]+)", rep_md).group(1)
    n_chunks = 324
    n_q = m1["n_ans"] + m1["n_unans"]
    tp, tot = card["trainable_params"], card["total_params"]
    kinds = card["train"]["kinds"]
    configs = list(zip(CONFIG_NAMES, (m1, m2, m3, ab)))

    f2 = lambda v: f"{v:.2f}"
    sec = lambda v: f"{v:.2f} s"
    num = lambda v: f"{v:,.0f}"

    chart_architecture(FIG_DIR / "architecture.png")
    chart_panels(stats, [("recall", "Fact recall", f2, 1.0),
                         ("grounded", "Groundedness", f2, 1.0),
                         ("refusal_acc", "Refusal accuracy (OOS)", f2, 1.0)],
                 FIG_DIR / "effectiveness.png")
    chart_panels(stats, [("lat_avg", "Mean latency (s)", f2, None),
                         ("lat_p95", "p95 latency (s)", f2, None)],
                 FIG_DIR / "latency.png")
    chart_panels(stats, [("in_tok", "Prompt tokens / query", num, None),
                         ("out_tok", "Completion tokens / query", num, None)],
                 FIG_DIR / "tokens.png")
    chart_loss(reg["versions"], FIG_DIR / "loss.png")

    C = Counter()
    s = []

    # ================================================================ title page
    s += [Spacer(1, 0.6 * cm),
          P("CHRIST (Deemed to be University), Bengaluru",
            ParagraphStyle("inst", fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor(NAVY),
                           alignment=TA_CENTER)),
          Spacer(1, 1.6 * cm),
          P("A Retrieval-Augmented, LLMOps-Governed<br/>Student Helpdesk Chatbot",
            ParagraphStyle("t", fontName="Times-Bold", fontSize=21, leading=26, alignment=TA_CENTER,
                           textColor=colors.HexColor(INK))),
          Spacer(1, 0.35 * cm),
          P("Comparing a standalone LLM, retrieval-augmented generation, and RAG with LLMOps controls "
            "and LoRA fine-tuning on the Christite Handbook 2026-27",
            ParagraphStyle("st", fontName="Times-Italic", fontSize=11.5, leading=15, alignment=TA_CENTER,
                           textColor=colors.HexColor(INK2))),
          Spacer(1, 0.6 * cm),
          P("Technical Report &nbsp;&middot;&nbsp; LCA Activity: Experimental and Problem-Based Learning "
            "(LLM / RAG / LLMOps)",
            ParagraphStyle("meta", fontName="Helvetica", fontSize=8.5, alignment=TA_CENTER,
                           textColor=colors.HexColor(INK2))),
          Spacer(1, 1.4 * cm),
          P("Abstract", ParagraphStyle("abs_h", fontName="Helvetica-Bold", fontSize=10, alignment=TA_CENTER,
                                       spaceAfter=6))]
    abstract = (
        "Institutional question answering is a poor fit for a general-purpose language model: the facts are "
        "local, change every academic year, and an incorrect answer about attendance, fines or deadlines has "
        "real consequences for students. We built a fully local student helpdesk for CHRIST University over the "
        f"108-page Christite Handbook 2026-27 and evaluated three configurations on a shared {n_q}-question test "
        f"set: (M1) a standalone 0.5B-parameter instruction-tuned LLM, (M2) the same model with hybrid "
        f"BM25 + dense retrieval, and (M3) retrieval combined with an LLMOps layer (input guardrails, relevance "
        f"gating, output verification, semantic caching, tracing) and a LoRA adapter trained on "
        f"{card['train']['examples']} RAFT-style examples. Retrieval raised fact recall from {f2(m1['recall'])} "
        f"to {f2(m2['recall'])} and cut mean latency by half. The full M3 system was the most reliable "
        f"and fastest configuration (groundedness {f2(m3['grounded'])}, out-of-scope refusal accuracy "
        f"{f2(m3['refusal_acc'])}, mean latency {sec(m3['lat_avg'])}), at the cost of lower fact recall "
        f"({f2(m3['recall'])}). An ablation shows that this recall loss comes from the adapter rather than from "
        "the LLMOps controls. A second adapter iteration that reduced over-refusal introduced confident "
        "wrong answers and was rolled back through a versioned model registry. We conclude that retrieval "
        "supplies the knowledge, fine-tuning shapes the answering behaviour, and operational controls are what "
        "make the system safe to deploy.")
    s += [Table([[P(abstract, ParagraphStyle("abs", parent=BODY, fontSize=10, leading=13.6))]],
                colWidths=[14.2 * cm], style=[("LEFTPADDING", (0, 0), (-1, -1), 0),
                                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]),
          Spacer(1, 0.4 * cm),
          P("<b>Keywords:</b> retrieval-augmented generation, LLMOps, parameter-efficient fine-tuning, LoRA, "
            "hybrid retrieval, guardrails, evaluation, student support",
            ParagraphStyle("kw", parent=BODY, fontSize=9.5, alignment=TA_CENTER)),
          PageBreak()]

    # ================================================================ 1 introduction
    s += [P("1&nbsp;&nbsp;Introduction", H1),
          P("Students at CHRIST University routinely ask the same operational questions: the attendance "
            "required to sit the End Semester Examination, the fine for a late library return, how to apply "
            "for a fee concession, or what happens after a missed assessment. The authoritative answers are in "
            "the Christite Handbook, a long PDF that few students read in full, so the questions are repeatedly "
            "routed to faculty, class coordinators and administrative staff."),
          P("A conversational assistant is an obvious remedy, but a general-purpose LLM has never seen the "
            "handbook. It produces fluent, plausible and frequently incorrect institutional facts, and in this "
            "domain a confident wrong answer is worse than no answer. The engineering question is therefore not "
            "whether an LLM can answer, but which combination of techniques makes its answers correct, "
            "traceable and safe, and at what cost in latency and compute."),
          P("Objectives", H2)]
    s += items([
        "Build three helpdesk configurations of increasing capability on the same base model and data.",
        "Measure each one on answer quality, refusal behaviour, latency and token usage with a fixed test set.",
        "Isolate the contribution of each technique (retrieval, LLMOps controls, PEFT) through ablation.",
        "Keep the whole system local, so that student queries never leave institutional hardware.",
    ], numbered=True)
    s += [P("Contributions", H2)]
    s += items([
        f"A page-cited knowledge base of {n_chunks} chunks built from the handbook PDF, with a hybrid "
        f"BM25 + dense retriever (Hit@4 = {hit}, MRR = {mrr}).",
        "An LLMOps layer covering input guardrails, relevance gating, numeric output verification, a "
        "version-aware semantic cache, SQLite tracing, user feedback and a monitoring dashboard.",
        f"A LoRA adapter ({tp / 1e6:.1f}M trainable parameters, {100 * tp / tot:.2f}% of the model) trained "
        "with RAFT-style data and served as merged weights, plus a versioned registry with a documented rollback.",
        "An offline evaluation harness with an ablation and a prompt A/B test, and a Streamlit interface "
        "exposing all three configurations side by side.",
    ])

    # ================================================================ 2 system overview
    s += [P("2&nbsp;&nbsp;System Overview", H1),
          P("All three configurations share the same generator, tokenizer, decoding settings and hardware, so "
            "observed differences are attributable to the pipeline rather than to the model. Table 1 lists the "
            "components."),
          C.table("Components and technology stack."),
          table([["Component", "Implementation", "Notes"],
                 ["Generator", f"{config.LLM_MODEL}", "0.5B parameters, fp16 on GPU, greedy decoding, "
                  "repetition penalty 1.1"],
                 ["Embeddings", "sentence-transformers/all-MiniLM-L6-v2", "384-d, used for retrieval, gating "
                  "and the semantic cache"],
                 ["Sparse retrieval", "BM25 (rank-bm25)", "fused with dense scores, alpha = 0.6"],
                 ["Fine-tuning", f"PEFT LoRA, r = {card['lora']['r']}, alpha = {card['lora']['alpha']}",
                  "all attention and MLP projections; merged for serving"],
                 ["Knowledge base", "Christite Handbook 2026-27 (PDF)", f"{n_chunks} chunks with chapter and "
                  "page metadata"],
                 ["Observability", "SQLite traces, Streamlit dashboard", "latency, tokens, flags, versions, "
                  "ratings"],
                 ["Hardware", "Laptop, NVIDIA RTX 3050 6 GB", "no external API calls"]],
                [3.0 * cm, 6.2 * cm, 7.4 * cm]),
          P("Configurations under test", H2),
          P("<b>M1, LLM only.</b> The query is sent directly to the generator with a short system prompt. The "
            "model answers from its pre-training knowledge alone."),
          P("<b>M2, RAG.</b> The hybrid retriever selects the four highest-scoring chunks, which are inserted "
            "into the prompt with numbered citations. The prompt (<i>rag_v1</i>) instructs the model to answer "
            "only from the supplied context."),
          P("<b>M3, RAG + LLMOps + PEFT.</b> M2 wrapped in pre- and post-generation controls, using a stricter "
            "prompt (<i>rag_v2</i>) and the LoRA-adapted generator. An ablation run of M3 with the adapter "
            "disabled is reported alongside, to separate the effect of PEFT from that of the controls.")]

    # ================================================================ 3 architecture
    s += [KeepTogether([
          P("3&nbsp;&nbsp;Architecture", H1),
          P("Figure 1 shows the request path. Stages are shaded by the configurations that use them. M1 bypasses "
            "retrieval and all controls; M2 adds the retriever; M3 adds every LLMOps stage and the adapter. "
            "Requests stopped by the guardrails or the relevance gate never reach the generator, so they cost "
            "neither GPU time nor tokens."),
          img(FIG_DIR / "architecture.png", 13.4),
          C.figure("Request path for the three configurations.")])]

    # ================================================================ 4 implementation
    s += [PageBreak(), P("4&nbsp;&nbsp;Implementation", H1),
          P("4.1&nbsp;&nbsp;Knowledge base construction", H2),
          P("The handbook is parsed with pypdf. Its table of contents is used to map page ranges to chapter "
            "titles, and running headers and footers are stripped. Text is then split into windows of "
            f"{config.CHUNK_WORDS} words with a {config.CHUNK_OVERLAP}-word overlap, which keeps most rules and "
            f"their conditions in a single chunk. Each of the resulting {n_chunks} chunks carries a citation "
            "string of the form <i>Section, p.N</i>, using printed page numbers rather than PDF indices so that "
            "students can verify answers in the physical handbook."),
          P("4.2&nbsp;&nbsp;Hybrid retrieval", H2),
          P("Dense cosine similarity and BM25 scores are min-max normalised per query and fused as "
            "<i>0.6 x dense + 0.4 x BM25</i>. Dense retrieval handles paraphrase (\"bunk classes\" against "
            "\"absence\"); BM25 anchors exact terms such as <i>FRRO</i>, <i>CIA</i> or <i>ESE</i>, which small "
            f"embedding models represent poorly. On the {m1['n_ans']} answerable test questions the gold page "
            f"appears in the top four for every query (Hit@4 = {hit}, MRR = {mrr}), so generation errors in "
            "M2 and M3 are not caused by retrieval misses."),
          P("4.3&nbsp;&nbsp;Generation and prompt versioning", H2),
          P("Prompts are stored as versioned templates (<i>llm_only_v1</i>, <i>rag_v1</i>, <i>rag_v2</i>), and "
            "every trace records the version used. <i>rag_v2</i> requires a [n] citation after each claim and an "
            "explicit refusal string when the context is insufficient. Decoding is greedy with a "
            f"{config.MAX_NEW_TOKENS}-token limit, which makes evaluation runs reproducible."),
          P("4.4&nbsp;&nbsp;LLMOps layer", H2)]
    s += items([
        "<b>Input guardrails.</b> Regular expressions redact e-mail addresses, phone numbers and university "
        "register numbers before the query reaches the prompt, the logs or any future training data. "
        "Prompt-injection patterns (\"ignore previous instructions\", role overrides) and out-of-policy requests "
        "are rejected.",
        f"<b>Relevance gate.</b> If the best dense similarity is below {config.MIN_RELEVANCE:.2f}, M3 returns a "
        "templated refusal that points to the relevant office instead of calling the generator.",
        "<b>Output verification.</b> Every number in the answer (amounts, percentages, days) must occur in the "
        "retrieved context or the question; otherwise the answer is replaced with a pointer to the cited page. "
        "A sentence-level groundedness score is computed and logged.",
        f"<b>Semantic cache.</b> Queries with cosine similarity of at least {config.CACHE_SIMILARITY:.2f} to a "
        "cached query return the stored answer. The cache key includes the prompt version and the adapter, so a "
        "deployment change cannot serve stale answers.",
        "<b>Tracing, feedback and dashboard.</b> Each request is logged to SQLite with latency, token counts, "
        "flags, prompt and adapter versions, and the user's rating. The dashboard aggregates these per "
        "configuration and maintains a review queue of low-groundedness or negatively rated answers.",
    ])
    s += [P("4.5&nbsp;&nbsp;Parameter-efficient fine-tuning", H2),
          P(f"The adapter targets all seven attention and MLP projections with rank {card['lora']['r']}, "
            f"alpha {card['lora']['alpha']} and dropout {card['lora']['dropout']}, giving {tp:,} trainable "
            f"parameters out of {tot:,} ({100 * tp / tot:.2f}%). Training data follows the RAFT recipe: each "
            "example contains the question, the gold passage at a random position among retrieved distractors, "
            f"and a short cited answer. The {card['train']['examples']} training examples comprise "
            f"{kinds['answer']} answerable items, {kinds['refuse']} out-of-scope refusals and "
            f"{kinds['gold_removed']} gold-removed examples, in which the gold passage is withheld and the target is "
            "a refusal. Loss is computed on answer tokens only. A leakage guard drops any training question with "
            f"cosine similarity of 0.85 or more to an evaluation question ({len(card['leakage_dropped'])} removed)."),
          P(f"Training used learning rate {card['train']['lr']}, gradient accumulation "
            f"{card['train']['grad_accum']}, bf16 base weights with fp32 adapter weights and gradient "
            f"checkpointing, and took {card['minutes']:.1f} minutes on the laptop GPU. For serving, the adapter "
            "is merged into a separate copy of the base weights (<i>merge_and_unload</i>), which removes the "
            "adapter's inference overhead while M1 and M2 keep using the untouched base model. Each trained "
            "adapter is saved with a model card (hyper-parameters, data hash, loss history, evaluation notes) and "
            "registered in <i>models/registry.json</i>, which records the active version."),
          P("4.6&nbsp;&nbsp;User interface", H2),
          P("The Streamlit application (Figures 2 to 5) provides a chat view, a side-by-side comparison of the "
            "three configurations, the LLMOps dashboard and the latest offline evaluation. A sidebar selects "
            "the configuration, the prompt version and the adapter. Every answer shows its latency, groundedness, "
            "retrieval score, adapter version and expandable sources, with thumbs-up and thumbs-down buttons that "
            "feed the trace store."),
          framed(FIG_DIR / "ui_home.png", 14.0),
          C.figure("Chat view on first load, with suggested queries."),
          framed(FIG_DIR / "ui_chat.png", 14.0),
          C.figure("An M3 answer with citation, latency, groundedness, retrieval score, adapter version, "
                   "sources and feedback controls."),
          framed(FIG_DIR / "ui_compare.png", 15.6),
          C.figure("Comparison view. M1 produces a generic range (\"5-10 rupees per day or per week\"); M2 and M3 "
                   "return the handbook value of Rs 5 per day."),
          framed(FIG_DIR / "ui_dashboard.png", 15.6),
          C.figure("LLMOps dashboard: request volume, latency, groundedness, cache hit rate, feedback and "
                   "per-configuration summary.")]

    # ================================================================ 5 experimental setup
    s += [P("5&nbsp;&nbsp;Experimental Setup", H1),
          P(f"The test set contains {n_q} hand-written questions. {m1['n_ans']} are answerable from the handbook "
            "and are annotated with the gold page(s) and one or more required key facts, written as regular "
            f"expressions to tolerate formatting (\"Rs 5\", \"Rs.5/-\", \"five rupees\"). {m1['n_unans']} are "
            "out of scope (OOS): programme fees, hostel fees, today's cafeteria menu, a bus route and a general "
            "knowledge question. All configurations were run once over the full set with greedy decoding on the "
            "same GPU. Table 2 defines the metrics."),
          C.table("Evaluation metrics."),
          table([["Metric", "Definition"],
                 ["Fact recall", "Mean fraction of required key facts present in the answer (answerable questions)."],
                 ["Groundedness", "Fraction of answer sentences whose content overlaps a retrieved passage; "
                  "refusals are excluded."],
                 ["Refusal accuracy", "Fraction of OOS questions answered with a refusal rather than a fabricated "
                  "answer."],
                 ["Wrong-refusal rate", "Fraction of answerable questions incorrectly refused."],
                 ["Latency", "Wall-clock time from query to complete answer, including retrieval and checks."],
                 ["Tokens", "Prompt tokens (recomputed with the model tokenizer) and generated tokens per query; "
                  "zero for requests stopped before generation."]],
                [3.4 * cm, 13.2 * cm])]

    # ================================================================ 6 results
    hdr = ["", "M1<br/>LLM only", "M2<br/>RAG", "M3<br/>RAG + LLMOps + PEFT", "M3 without<br/>PEFT"]
    W5 = [4.6 * cm, 2.6 * cm, 2.6 * cm, 3.6 * cm, 3.0 * cm]
    row = lambda name, key, fmt: [name] + [fmt(x[key]) for x in (m1, m2, m3, ab)]
    s += [P("6&nbsp;&nbsp;Results", H1),
          P("6.1&nbsp;&nbsp;Answer quality", H2),
          P(f"Table {C.tab + 1} and Figure {C.fig + 1} summarise answer quality. Retrieval is the dominant factor: fact recall rises "
            f"from {f2(m1['recall'])} to {f2(m2['recall'])} when the same model is given handbook context. M1's "
            "answers are fluent but largely invented; for example, it dates the founding of Christ College to "
            "1920 (the handbook gives 1969) and states that South Africa won the 2022 FIFA World Cup."),
          C.table(f"Answer-quality metrics ({m1['n_ans']} answerable and {m1['n_unans']} out-of-scope questions)."),
          table([hdr, row("Fact recall", "recall", f2), row("Groundedness", "grounded", f2),
                 row("Refusal accuracy (OOS)", "refusal_acc", f2),
                 row("Wrong-refusal rate", "wrong_refusal", f2)], W5, bold_col=3),
          Spacer(1, 0.2 * cm),
          img(FIG_DIR / "effectiveness.png", 16.0),
          C.figure("Fact recall, groundedness and out-of-scope refusal accuracy per configuration."),
          P(f"M2 still fabricated answers to {m2['unans_madeup']} of {m2['n_unans']} OOS questions: when retrieval "
            "returns loosely related text, the model tries to answer from it (for example, a bus route number "
            "taken from a page about highways). M3's controls doubled OOS refusal accuracy to "
            f"{f2(m3['refusal_acc'])}. The ablation shows that this gain comes from the LLMOps layer, since M3 "
            f"without PEFT also reaches {f2(ab['refusal_acc'])}. The adapter's contribution is different: "
            f"groundedness rises to {f2(m3['grounded'])}, every answer carries a citation, and answers become "
            f"shorter. Its cost is a drop in fact recall from {f2(ab['recall'])} to {f2(m3['recall'])} and a "
            f"wrong-refusal rate of {f2(m3['wrong_refusal'])}. Table 4 breaks the outcomes down per question."),
          C.table("Per-question outcomes."),
          table([["Configuration", "Fully correct", "Partly correct", "Wrong", "Refused (answerable)",
                  "OOS refused", "OOS fabricated"]] +
                [[n, f"{x['correct']}/{x['n_ans']}", str(x["partial"]), str(x["wrong"]), str(x["refused_ans"]),
                  f"{x['unans_refused']}/{x['n_unans']}", str(x["unans_madeup"])] for n, x in configs],
                [4.2 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm, 2.5 * cm, 2.0 * cm, 2.3 * cm]),
          P("6.2&nbsp;&nbsp;Latency", H2),
          P("Generation dominates latency, and generation time scales with output length. M1 is the slowest "
            f"configuration ({sec(m1['lat_avg'])} mean) because it produces long, hedged answers "
            f"({m1['out_tok']:.0f} tokens on average). M3 is the fastest ({sec(m3['lat_avg'])} mean, "
            f"{sec(m3['lat_p95'])} p95): the adapter's answers average {m3['out_tok']:.0f} tokens, gated requests "
            "return in milliseconds, and merged weights add no adapter overhead. Without the adapter, M3's p95 "
            f"rises to {sec(ab['lat_p95'])}, which makes PEFT the main source of M3's tail-latency improvement."),
          img(FIG_DIR / "latency.png", 15.0),
          C.figure("Mean and 95th-percentile latency per configuration."),
          C.table("Latency distribution (seconds)."),
          table([["Configuration", "Mean", "Median", "p95", "Min", "Max"]] +
                [[n] + [f"{x[k]:.2f}" for k in ("lat_avg", "lat_med", "lat_p95", "lat_min", "lat_max")]
                 for n, x in configs],
                [5.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]),
          P("6.3&nbsp;&nbsp;Token usage", H2),
          P(f"Retrieval increases the prompt from about {m1['in_tok']:.0f} to {m2['in_tok']:.0f} tokens, since four "
            "passages are inserted. Prompt tokens are cheap on a local GPU because they are processed in a single "
            "parallel prefill pass. Completion tokens are generated one at a time and dominate cost. M3 generates "
            f"the fewest ({m3['out_tok']:.0f} per query) and spends no tokens on gated requests, giving it the "
            f"lowest total among the retrieval configurations ({num(m3['total_tok'])} tokens for the full test "
            "set). With a hosted API, the same proportions would carry over to cost."),
          img(FIG_DIR / "tokens.png", 15.0),
          C.figure("Mean prompt and completion tokens per query."),
          C.table("Token usage."),
          table([["Configuration", "Prompt tokens / query", "Completion tokens / query",
                  f"Total over {n_q} queries"]] +
                [[n, f"{x['in_tok']:.0f}", f"{x['out_tok']:.0f}", num(x["total_tok"])] for n, x in configs],
                [5.6 * cm, 3.6 * cm, 3.8 * cm, 3.6 * cm])]

    # ---- qualitative
    def ans(system, q):
        r = df[(df.system == system) & (df.question == q)]
        if not len(r):
            return "-"
        a = r.answer.iloc[0].replace("**", "").replace("\n", " ").replace("<", "&lt;")
        return a if len(a) <= 240 else a[:240].rsplit(" ", 1)[0] + " [...]"

    s += [P("6.4&nbsp;&nbsp;Qualitative examples", H2),
          P(f"Table {C.tab + 1} shows verbatim answers (truncated where long) for four representative "
            "questions.")]
    caption = C.table("Selected answers from the evaluation run.")
    for q, note in [("What is the fine for returning a library book late?", "Reference: Rs 5 per day (p.15)."),
                    ("What is the reduction in weightage for a repeat Mid Semester Examination?",
                     "Reference: 15% (p.51)."),
                    ("Who won the FIFA World Cup in 2022?", "Out of scope; expected behaviour is refusal."),
                    ("When was Christ College established?", "Reference: 1969 (p.1).")]:
        rows = [["", f"{q} <font color='{INK2}'><i>{note}</i></font>"]]
        for n, key in zip(["M1", "M2", "M3"], SYSTEMS):
            rows.append([n, ans(key, q)])
        block = [table(rows, [1.2 * cm, 15.4 * cm])]
        if caption is not None:
            block.insert(0, caption)
            caption = None
        s += [KeepTogether(block), Spacer(1, 0.25 * cm)]
    s += [P("The repeat-examination example is notable. Without the adapter, the model answered \"30%\". This "
            "passed the numeric check because the same page also contains the 70% CIA / 30% ESE split. The "
            "adapter answered the correct 15%. Output verification cannot catch an error that is "
            "numerically supported by the context but semantically wrong, so it has to be fixed in the "
            "generator.")]

    # ---- PEFT iteration
    s += [P("6.5&nbsp;&nbsp;Adapter iteration and rollback", H2),
          P("Adapter v1 over-refused, declining three answerable questions. We trained v2 with the "
            "gold-removed share reduced from 25% to 10% and with early stopping on validation loss "
            f"(Figure {C.fig + 1}). As Table {C.tab + 1} shows, v2 cut wrong refusals as intended, but the questions it no longer refused became "
            "confident answers drawn from the wrong passage. For example, the visa-renewal fine was answered "
            "with the library fine. Because a refusal that redirects the student is a safer failure than a "
            "wrong fine or deadline, v1 was reinstated as the active adapter and the decision was recorded "
            "in the registry."),
          img(FIG_DIR / "loss.png", 15.0),
          C.figure("Training and validation loss per epoch for the two adapter versions."),
          C.table("Adapter versions evaluated within M3."),
          table([["Adapter", "Change", "Fact recall", "Wrong refusals", "Groundedness", "Status"],
                 ["none (ablation)", "base generator", f2(ab["recall"]), f2(ab["wrong_refusal"]),
                  f2(ab["grounded"]), "baseline"],
                 ["lora-helpdesk-v1", "25% gold-removed, 3 epochs", f2(m3["recall"]), f2(m3["wrong_refusal"]),
                  f2(m3["grounded"]), "active"],
                 ["lora-helpdesk-v2", "10% gold-removed, early stopping", "0.62", "0.04", "0.96",
                  "rolled back"]],
                [3.2 * cm, 4.8 * cm, 2.0 * cm, 2.4 * cm, 2.3 * cm, 1.9 * cm]),
          P("6.6&nbsp;&nbsp;Prompt-only control", H2),
          P("To test whether the stricter prompt alone could deliver M3's refusal behaviour, <i>rag_v2</i> was run "
            "on M2 without any controls or adapter. It reached the highest fact recall of any run "
            f"({abp['Keyword recall (answerable)']:.2f}) but a refusal accuracy of "
            f"{abp['Refusal accuracy (unanswerable)']:.2f}: the 0.5B model did not follow the instruction to "
            "decline. At this model scale, a prompt is not a safety mechanism. Refusal has to be enforced by the "
            "system (gating) or learned (fine-tuning).")]

    # ================================================================ 7 discussion
    s += [P("7&nbsp;&nbsp;Discussion", H1),
          P("<b>Retrieval supplies knowledge; fine-tuning shapes behaviour.</b> Retrieval produced a five-fold gain "
            "in fact recall, while the adapter changed how context is used: brevity, citation discipline, "
            f"refusal and speed. The adapter did not add knowledge, and with {card['train']['examples']} examples "
            "it learned the output format faster than passage selection. The two techniques address different "
            "failure modes, which is why the final system uses both."),
          P("<b>Groundedness alone is a misleading quality signal.</b> M3 scored perfect groundedness while its "
            "recall fell. An answer can be faithful to a retrieved passage and still answer a different "
            "question. Evaluation needs both a faithfulness metric and a correctness metric against a reference."),
          P("<b>Ablations are necessary for PEFT.</b> Without the M3-without-PEFT run, the adapter's gains (correct "
            "15%, halved tail latency, full grounding) and its losses (recall, over-refusal) would have been "
            "mixed with the effect of the controls and impossible to attribute."),
          P("<b>Deployment decisions should weigh failure severity, not only aggregate scores.</b> v2 improved "
            "one metric by converting safe failures into harmful ones. A versioned registry with model cards "
            "turned the rollback into a single, auditable configuration change.")]

    # ================================================================ 8 limitations / future work
    s += [P("8&nbsp;&nbsp;Limitations and Future Work", H1)]
    s += items([
        f"<b>Evaluation size.</b> {n_q} questions, of which {m1['n_unans']} are out of scope, give coarse "
        "estimates: one OOS question moves refusal accuracy by 0.20. The set should be grown, ideally from real "
        "negatively rated queries in the review queue.",
        "<b>Keyword-based scoring.</b> Regular-expression key facts are transparent but miss paraphrases and "
        "cannot judge completeness. A small human-graded sample or a calibrated LLM judge would complement them.",
        "<b>Single-run latency.</b> Latency was measured once on a laptop GPU and varies by roughly 30% between "
        "runs. Repeated runs with confidence intervals would firm up the comparison.",
        f"<b>Adapter data.</b> {card['train']['examples']} examples are enough to teach format but not reliable "
        "passage selection. The next step is 300 or more human-verified RAFT examples, followed by a 1.5B to 7B "
        "base model.",
        "<b>Gating and tables.</b> Topic-level intent classification would catch OOS queries that share "
        "vocabulary with the handbook (the bus-route case). Table-aware parsing, with one chunk per table row, "
        "would fix confusions such as UG and PG borrowing limits.",
        "<b>Recall recovery.</b> When the adapter refuses but retrieval relevance is high, falling back to the "
        "base generator under the same output checks would recover most of the lost recall at little cost.",
    ])

    # ================================================================ 9 conclusion
    s += [P("9&nbsp;&nbsp;Conclusion", H1),
          P("A standalone LLM is not suitable for institutional question answering: it lacks the facts and does "
            "not reliably signal uncertainty. Retrieval-augmented generation is the minimum viable architecture "
            "and the largest single improvement. The LLMOps layer is what makes the system deployable: it "
            "refuses out-of-scope queries, verifies numbers, protects personal data and makes every answer "
            f"auditable. LoRA fine-tuning is inexpensive ({100 * tp / tot:.2f}% of parameters, a 35 MB adapter, "
            f"{card['minutes']:.1f} minutes of training) and improves grounding, citation and latency. With the "
            "current data volume, however, it trades away some recall. We recommend M3 as the deployment "
            "architecture, with the adapter retrained on a larger verified dataset and every change evaluated "
            "against a fixed test set and ablation before release."),
          P("Appendix A&nbsp;&nbsp;Reproducibility", H1),
          P("Source code, evaluation data, results and the active adapter are available at "
            "github.com/Kusum004/LLMops-chatbot. The handbook PDF is not redistributed and must be placed in "
            "<i>data/kb/</i>.")]
    s += [P(c.replace(" ", "&nbsp;"), CODE) for c in [
        "pip install -r requirements.txt",
        "streamlit run app.py              # interface",
        "python finetune_peft.py           # train a new adapter version",
        "python evaluate.py --ab           # offline evaluation + prompt A/B",
        "python -m reports.make_report     # rebuild this report"]]

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.1 * cm, bottomMargin=2.0 * cm,
                            title="A Retrieval-Augmented, LLMOps-Governed Student Helpdesk Chatbot",
                            author="", subject="Technical report", creator="", producer="")
    doc.build(s, onFirstPage=title_decor, onLaterPages=page_decor)

    # strip the creation/modification timestamps reportlab writes into the metadata
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(clone_from=PdfReader(str(PDF)))
    w.metadata = None
    w.add_metadata({"/Title": "A Retrieval-Augmented, LLMOps-Governed Student Helpdesk Chatbot",
                    "/Subject": "Technical report"})
    with open(PDF, "wb") as fh:
        w.write(fh)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    build()
