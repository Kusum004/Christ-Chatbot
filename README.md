# CHRIST University Student Helpdesk — LLM vs RAG vs RAG + LLMOps (+ PEFT)

LCA Activity: experiential, problem-based learning in Generative AI (exploring LLM, RAG and LLMOps). This project builds a student-support chatbot for CHRIST (Deemed to be University) in three ways and compares them. Everything runs **locally**: there are no API keys, and no student query leaves the machine.

| | Model 1 — LLM only | Model 2 — RAG | Model 3 — hybrid: RAG + LLMOps + PEFT |
|---|---|---|---|
| LLM weights | Base Qwen2.5-0.5B | Base Qwen2.5-0.5B | Base model + **LoRA adapter** (PEFT, 1.75% of parameters trained) |
| Knowledge source | LLM's pre-training only | Christite Handbook 2026-27 (retrieved) | Same as Model 2 |
| Retrieval | – | Hybrid (BM25 + dense embeddings), top-4 | Same as Model 2, plus a relevance gate |
| Prompt | `llm_only_v1` | `rag_v1` (basic) | `rag_v2` (strict, cited, refusal rule) — versioned |
| Guardrails | – | – | PII redaction, prompt-injection and policy blocking, number check, groundedness check |
| Ops | – | – | Semantic cache, SQLite tracing, feedback, dashboard, offline eval, A/B, adapter registry |

PEFT is applied **only to the hybrid model**. Models 1 and 2 run the untouched base LLM; the same process loads one copy of the base weights and switches the adapter on only for Model 3 (`peft.disable_adapter()` for the others). Any difference in quality therefore comes from the design of each model, not from different base weights.

## Architecture

```
            ┌─────────────── Model 3 only ───────────────┐
question ──►│ input guardrails ─► semantic cache ─► hit? ├──► answer
            └──────────────────────┬─────────────────────┘
                                   ▼ miss
   Christite Handbook PDF ─► section-aware chunks ─► BM25 + MiniLM embeddings
                                   │  hybrid search (top-4)
                                   ▼
               relevance gate (Model 3) ─► versioned prompt + context
                                   ▼
                   Local LLM (Qwen2.5-0.5B-Instruct, GPU/CPU)
                   + LoRA adapter (Model 3 only, from models/registry.json)
                                   ▼
            groundedness check (Model 3) ─► answer + page citations
                                   ▼
             trace → logs/llmops.db  ─► dashboard, feedback, review queue
```

## Project layout

```
chatbot/
  config.py      all tunables (models, top-k, thresholds, prompt version)
  ingest.py      PDF/Markdown loading; chapter detection from the handbook's TOC; page-cited chunks
  retriever.py   hybrid BM25 + dense retriever
  llm.py         local Hugging Face LLM wrapper (greedy, reproducible); loads the LoRA adapter
  prompts.py     versioned prompt registry
  llmops.py      guardrails, groundedness scorer, semantic cache, SQLite tracer, feedback
  pipelines.py   the three models
app.py           Streamlit UI: Chat · Compare 3 models · LLMOps dashboard · Evaluation
evaluate.py      offline evaluation (and prompt A/B with --ab)
finetune_peft.py PEFT/LoRA fine-tuning of the hybrid model (RAFT-style data, leakage guard, registry)
data/kb/         knowledge base (christite_handbook_2026-27.pdf; drop extra .pdf/.md files here)
data/eval_set.json   31 test questions (26 answerable with gold handbook pages, 5 out-of-scope)
data/peft_train.json 68 fine-tuning questions (53 answerable from the handbook + 15 refusals), disjoint from the eval set
models/          versioned LoRA adapters (lora-helpdesk-vN/ with model_card.json) + registry.json
logs/            traces DB and evaluation outputs
```

## PEFT in the LLMOps lifecycle (hybrid model only)

Full fine-tuning of even a 0.5B model means updating 500M parameters and storing a new copy of the model. **PEFT with LoRA** instead freezes the base model and trains small low-rank matrices (r = 16) inside every attention and MLP projection. That is 8.8M trainable parameters (1.75%). The adapter is a ~35 MB file that can be versioned, swapped and rolled back like any other release artifact.

What the adapter is taught is a **behaviour**, not facts. Facts stay in the handbook and are retrieved at run time. The adapter teaches the model to:
1. answer briefly from the retrieved passages and cite them as `[n]`;
2. ignore distractor passages;
3. reply with the exact refusal sentence when the answer is not in the context.

How the training data is built (`finetune_peft.py`, RAFT-style):
- For every training question, the **real retriever** returns its top-4 passages, so the model trains on the same kind of context it sees in production. The correct passage is guaranteed to be included, at a random position, with distractors around it.
- For 25% of answerable questions an extra copy is made with the correct passage **removed**, and the target becomes a refusal. This teaches the model what to do when retrieval misses.
- There are 15 out-of-scope questions (WiFi password, MBA fee, canteen hours, and so on) whose target is a refusal.
- The loss is calculated **only on the answer tokens**; the prompt and context are masked out.

LLMOps controls around the fine-tuning:
- **Leakage guard:** training questions whose embedding similarity to any evaluation question is 0.85 or higher are dropped automatically. Two were dropped, so the evaluation stays honest.
- **Reproducibility:** fixed seed, a 90/10 train/validation split, and a SHA-256 fingerprint of the training data stored in the model card.
- **Model registry:** each run saves `models/lora-helpdesk-vN/` with `model_card.json` (hyperparameters, losses, data hash, leakage report) and updates `models/registry.json → active`. Rolling back means changing one line, or setting `HELPDESK_ADAPTER`.
- **Runtime switch:** `HELPDESK_PEFT=0` or the sidebar toggle turns the adapter off. The semantic cache is keyed by adapter version, and every trace records which adapter produced the answer.
- **Ablation in evaluation:** `evaluate.py` runs Model 3 with and without the adapter on the same questions, so the gain from PEFT is measured, not assumed.
- **Fits a laptop:** bf16 frozen base weights, fp32 LoRA weights and gradient checkpointing let training run on a 6 GB RTX 3050.

## Run it

```bash
pip install -r requirements.txt
python finetune_peft.py        # PEFT: train the LoRA adapter for Model 3 -> models/lora-helpdesk-vN
python evaluate.py --ab        # offline evaluation -> logs/eval_report.md
streamlit run app.py           # chatbot + dashboard at http://localhost:8501
```
The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) and `all-MiniLM-L6-v2` (~90 MB) from Hugging Face. After that, everything works offline. To try a larger local model, run `set HELPDESK_LLM=Qwen/Qwen2.5-1.5B-Instruct` before starting.

## Results

Offline evaluation results (`python evaluate.py --ab`): 31 questions, Qwen2.5-0.5B-Instruct on GPU, greedy decoding. Retrieval Hit@4 = **1.00** and MRR = **0.97**, meaning the correct handbook page was in the top 4 for every answerable question.

| Metric | Model 1 LLM only | Model 2 RAG | Model 3 RAG + LLMOps | A/B: RAG + strict prompt only |
|---|---:|---:|---:|---:|
| Fact recall on 26 answerable questions | 0.17 | 0.90 | 0.90 | 0.94 |
| Groundedness of answers in the handbook | 0.14 | 0.85 | 0.88 | 0.82 |
| Correct refusals on 5 out-of-scope questions | 0.20 | 0.40 | **0.80** | 0.00 |
| Wrong refusals on answerable questions | 0.04 | 0.00 | 0.04 | 0.00 |
| Avg latency (ms) | 3786 | 1731 | 1964 | 2255 |
| Avg output tokens | 102 | 46 | 52 | 60 |

Metric definitions: fact recall checks whether the answer contains the key fact (regex keywords). Groundedness is the share of answer sentences supported by the retrieved handbook text. Refusal accuracy is the share of out-of-scope questions answered with "I don't have that information" rather than an invented answer. Per-question output is in `logs/eval_results.csv`.

### What each model showed

**Model 1 – LLM only.** The model is fluent but cannot be trusted for helpdesk work. It answered only 17% of handbook facts correctly. It gave confident, wrong answers: Christ College was "established in 1920", Kengeri campus "in 1965", the World Cup was "won by South Africa", and a bus route was "102". It cannot know institution-specific rules like the 85% attendance norm or the Rs. 5/day library fine, because they were never in its training data. It was also the slowest, because it writes long, padded answers. The lesson is that an LLM on its own is a language engine, not a knowledge base.

**Model 2 – RAG.** Grounding the same small model in the handbook raised fact recall from 0.17 to 0.90 and halved latency, because the answers were shorter and more focused. Every answer can be traced to a handbook page. Its weaknesses:
- When retrieval returns loosely related text for an out-of-scope question, it still tries to answer. For example, it gave bus route "17" from a page about highways.
- It sometimes misreads tables, giving the PG limit of 4 books where the UG limit is 2.

RAG fixes knowledge but not judgement.

**Model 3 – RAG + LLMOps.** The production controls added safety without losing accuracy:
- Out-of-scope refusals doubled (0.40 → 0.80) because of the retrieval-relevance gate.
- Prompt-injection and PII inputs are blocked or redacted before reaching the model.
- The number check replaced a made-up tuition fee with a pointer to the correct handbook page.
- Every request is traced (latency, tokens, groundedness, flags), feedback is collected, and the dashboard shows a review queue of weak answers.
- Overhead was about 230 ms per request; the semantic cache removes it entirely for repeated questions.

**A/B prompt test.** The stricter prompt (`rag_v2`) on its own produced **zero** refusals on out-of-scope questions. A 0.5B model does not reliably follow "say you don't know" instructions. Safety therefore has to come from deterministic system controls (the relevance gate and the number check), not from prompt wording alone. Without the A/B experiment this would have gone unnoticed.

### Remaining failures (and why they matter)
- **"30%" for the repeat-exam weightage reduction** (the correct value is 15%). The retrieved page also contains "70% CIA / 30% ESE", so the text-matching number check is fooled. A stronger fix would be a larger model, or a check that compares meaning (an entailment model) rather than matching strings.
- **The bus-route question passed the relevance gate.** "Kengeri campus" matched the campus page strongly, and the model turned a distance ("21 Kms") into a route. Topic-level intent classification ("is this question answerable from the handbook at all?") would catch this.
- **Table reading (UG vs PG borrowing limits).** PDF tables flatten into text. Table-aware parsing, for example storing each row as its own chunk, would fix this.

### Recommendations
1. **Never deploy a bare LLM for institutional Q&A.** Use RAG as the minimum. It is the single biggest improvement (5× fact recall) and makes answers auditable through page citations.
2. **Treat LLMOps as a required layer, not an optional extra.** Relevance gating, output verification, tracing and feedback are what turn a demo into something students can rely on. Prompts alone are not a safety mechanism.
3. **Evaluate before and after every change.** The scoring bugs and the A/B finding both surfaced only because a fixed evaluation set was re-run after each change. Grow `eval_set.json` from real 👎-rated questions in the dashboard's review queue.
4. **Scale the model with the risk.** The 0.5B model runs on a laptop but misreads tables and numbers. For deployment, use a 1.5B–7B local model (one config change: `HELPDESK_LLM`) or a hosted model under a data-protection agreement.
5. **Keep the knowledge base current.** Re-index whenever the handbook or circulars change, show the source version in answers, and send fee, deadline and grievance questions to the responsible office rather than letting the bot improvise.
6. **Privacy by design.** Running locally keeps student queries on campus infrastructure. Input PII redaction (emails, phone numbers, register numbers) stops personal data from entering prompts or logs.
#   L L M o p s - c h a t b o t  
 