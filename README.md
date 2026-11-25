# LLMops-chatbot: CHRIST University Student Helpdesk — LLM vs RAG vs RAG + LLMOps (+ PEFT)

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

**Technical report (PDF, 11 pages):** [reports/Christ_Helpdesk_Model_Report.pdf](reports/Christ_Helpdesk_Model_Report.pdf) - abstract, system overview, architecture, implementation (retrieval, LLMOps layer, LoRA), interface, experimental setup, results (quality, latency, tokens, qualitative examples, adapter iteration, prompt-only control), discussion, limitations and conclusion.

## Architecture

```
                                  [ User Query ]
                                        │
           ┌────────────────────────────┼───────────────────────────┐
           │ (Model 1: Direct LLM)      │                           │ (Model 2: Standard RAG)
           │ prompt: llm_only_v1        ▼ (Model 3: Hybrid)         │ prompt: rag_v1
           │                 ┌──────────────────────┐               │
           │                 │ 1. Input Guardrails  │               │
           │                 │ PII / Jailbreak Redact│               │
           │                 └──────────┬───────────┘               │
           │                            ▼                           │
           │                 ┌──────────────────────┐  Cache Hit    │
           │                 │  2. Semantic Cache   ├───────────────┼─────────────┐
           │                 │  MiniLM cosine >=0.95│  (<50ms fast) │             │
           │                 └──────────┬───────────┘               │             │
           │                            ▼ Cache Miss                │             │
           │                 ┌──────────────────────┐               │             │
           │  PDF Ingestion ─► 3. Hybrid Retriever  ◄──────────────┘             │
           │  (324 Chunks)   │ Dense(0.6)+BM25(0.4) │                             │
           │                 └──────────┬───────────┘                             │
           │                            ▼ Top-4 Context                           │
           │                 ┌──────────────────────┐                             │
           │                 │  4. Relevance Gate   │                             │
           │                 │  sim < 0.40 -> Refuse│                             │
           │                 └──────────┬───────────┘                             │
           │                            ▼ prompt: rag_v2                          │
           ▼                            ▼                                         │
┌────────────────────────────────────────────────────────────────────────┐        │
│                5. Local Generator (Qwen2.5-0.5B-Instruct)              │        │
│  M1, M2: Untouched base model          M3: Merged LoRA (lora-helpdesk) │        │
└───────────────────────────────────────┬────────────────────────────────┘        │
                                        ▼                                         │
                             ┌──────────────────────┐                             │
                             │ 6. Output Guardrails │                             │
                             │ Numbers / Grounding  │                             │
                             └──────────┬───────────┘                             │
                                        ▼                                         │
                             ┌──────────────────────┐                             │
                             │  7. Verified Answer  │◄────────────────────────────┘
                             │  + Page Citations [n]│
                             └──────────┬───────────┘
                                        ▼
                             ┌──────────────────────┐
                             │ 8. SQLite Telemetry  │──► Streamlit LLMOps Dashboard
                             │ logs/llmops.db Trace │    Latency, Grounding, Feedback
                             └──────────────────────┘
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
# 1. Download the Christite Handbook 2026-27 PDF from christuniversity.in (Student Life section)
#    and save it as data/kb/christite_handbook_2026-27.pdf  (not included in this repo - (c) CHRIST University)
python finetune_peft.py        # optional: retrain the LoRA adapter (v1 is already included in models/)
python evaluate.py --ab        # offline evaluation -> logs/eval_report.md
python -m reports.make_report  # rebuild the PDF report from the latest evaluation
streamlit run app.py           # chatbot + dashboard at http://localhost:8501
```
The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) and `all-MiniLM-L6-v2` (~90 MB) from Hugging Face. After that, everything works offline. To try a larger local model, run `set HELPDESK_LLM=Qwen/Qwen2.5-1.5B-Instruct` before starting.

## Results

Offline evaluation results (`python evaluate.py --ab`): 31 questions, Qwen2.5-0.5B-Instruct on an RTX 3050 laptop GPU, greedy decoding. Retrieval Hit@4 = **1.00** and MRR = **0.97**, meaning the correct handbook page was in the top 4 for every answerable question. Model 3 uses the deployed adapter `lora-helpdesk-v1`, merged into the base weights for serving.

| Metric | Model 1 LLM only | Model 2 RAG | **Model 3 RAG + LLMOps + PEFT** | Ablation: Model 3 without PEFT | A/B: RAG + strict prompt only |
|---|---:|---:|---:|---:|---:|
| Fact recall on 26 answerable questions | 0.17 | 0.90 | 0.71 | 0.90 | 0.94 |
| Groundedness of answers in the handbook | 0.14 | 0.85 | **1.00** | 0.88 | 0.82 |
| Correct refusals on 5 out-of-scope questions | 0.20 | 0.40 | **0.80** | 0.80 | 0.00 |
| Wrong refusals on answerable questions | 0.04 | 0.00 | 0.12 | 0.04 | 0.00 |
| Avg latency (ms) | 3649 | 1696 | **988** | 1876 | 2928 |
| p95 latency (ms) | 7773 | 4744 | **1826** | 6951 | 10801 |
| Avg output tokens | 102 | 46 | **26** | 52 | 60 |

Metric definitions: fact recall checks whether the answer contains the key fact (regex keywords). Groundedness is the share of answer sentences supported by the retrieved handbook text. Refusal accuracy is the share of out-of-scope questions answered with "I don't have that information" rather than an invented answer. Per-question output is in `logs/eval_results.csv`; earlier runs are kept as `logs/eval_*_peft_v1.*` and `logs/eval_*_peft_v2.*`. Latency on a laptop GPU varies by ±30% between runs, so compare latencies within one run.

### PEFT experiment log (LLMOps iteration)

| Adapter | Change | Val loss (best) | Fact recall | Wrong refusals | Groundedness | Decision |
|---|---|---:|---:|---:|---:|---|
| — (base) | no adapter | – | 0.90 | 0.04 | 0.88 | baseline (ablation) |
| `lora-helpdesk-v1` | r=16, 3 epochs, 25% gold-removed refusal examples | 0.36 (ep. 2) | 0.71 | 0.12 | 1.00 | **deployed** |
| `lora-helpdesk-v2` | 10% gold-removed examples, early stopping (kept ep. 1) | 0.40 (ep. 1) | 0.62 | 0.04 | 0.96 | rolled back |

v2 tested the hypothesis "fewer refusal examples → fewer wrong refusals". It worked: wrong refusals fell from 0.12 to 0.04. But the questions the model used to decline turned into **confident answers from the wrong passage**, for example the visa-renewal fine answered with the library fine, Rs 5/day. For a student helpdesk a refusal ("please check with the office") is a much safer failure than a wrong fine or deadline, so v1 was restored as the active adapter in `models/registry.json`, with the reason recorded. This is the LLMOps loop in practice: version → evaluate → compare → roll back.

### What each model showed

**Model 1 – LLM only.** The model is fluent but cannot be trusted for helpdesk work. It answered only 17% of handbook facts correctly. It gave confident, wrong answers: Christ College was "established in 1920", Kengeri campus "in 1965", the World Cup was "won by South Africa", and a bus route was "102". It cannot know institution-specific rules like the 85% attendance norm or the Rs. 5/day library fine, because they were never in its training data. It was also the slowest, because it writes long, padded answers. The lesson is that an LLM on its own is a language engine, not a knowledge base.

**Model 2 – RAG.** Grounding the same small model in the handbook raised fact recall from 0.17 to 0.90 and halved latency, because the answers were shorter and more focused. Every answer can be traced to a handbook page. Its weaknesses:
- When retrieval returns loosely related text for an out-of-scope question, it still tries to answer. For example, it gave bus route "17" from a page about highways.
- It sometimes misreads tables, giving the PG limit of 4 books where the UG limit is 2.

RAG fixes knowledge but not judgement.

**Model 3 – hybrid: RAG + LLMOps + PEFT.** The LLMOps controls (relevance gate, input guardrails, number check, tracing, feedback, cache) doubled correct out-of-scope refusals (0.40 → 0.80) and blocked prompt injection and PII. The ablation shows what the **PEFT adapter** adds on top of them:
- **Better behaviour.** Every answer is grounded in the retrieved passages (1.00) and carries a `[n]` citation. Answers are half as long (26 tokens vs 52), which makes Model 3 the **fastest model**: 0.99 s on average and 1.8 s at p95, against 1.9 s and 7.0 s without PEFT.
- **Fixed an error no guardrail could catch.** For the repeat-MSE weightage reduction, the base model said "30%" and slipped past the number check, because the same page contains "70% CIA / 30% ESE". The adapter answers the correct **15%**.
- **Lower fact recall (0.90 → 0.71).** With only 72 training examples, LoRA taught the *format* (one short cited sentence) faster than *understanding*. Some answers are faithful to a passage but to the wrong one (visa fine → library fine), or leave out the key fact. Three answerable questions were refused (0.12).

**A/B prompt test.** The stricter prompt (`rag_v2`) on its own produced **zero** refusals on out-of-scope questions. A 0.5B model does not reliably follow "say you don't know" instructions. Reliable safety needs system controls (gating, verification) or training (PEFT), not prompt wording alone.

### Key interpretations
1. **Retrieval provides the knowledge; PEFT shapes the behaviour.** RAG produced the 5× jump in correct facts. PEFT changed *how* the model uses context: brevity, citations, refusals, speed. Neither replaces the other, which is why the hybrid model combines both.
2. **Groundedness alone is a misleading quality metric.** The PEFT model scored a perfect 1.00 groundedness while recall fell. An answer can be faithful to a retrieved passage and still not answer the question. Evaluation needs both "is it supported?" and "is it the right answer?".
3. **Every PEFT change needs an ablation.** Without the "Model 3 without PEFT" row, the gains (15% fix, 2× speed, full grounding) and the losses (recall, over-refusal) would be mixed together and invisible.
4. **Choose which failure is safer, not just the higher score.** v2 had fewer wrong refusals but more wrong answers. A versioned adapter registry made the rollback a one-line, documented decision.

### Remaining failures (and how to fix them)
- **PEFT picks the wrong passage or fact** (visa fine → library fine, NIRF rank, 7-year UG limit). Fix: more training data, several hundred QA pairs drafted from the handbook and checked by a person, plus a larger base model.
- **The bus-route question passed the relevance gate.** "Kengeri campus" matched the campus page strongly. Topic-level intent classification ("is this answerable from the handbook at all?") would catch it.
- **Table reading (UG vs PG borrowing limits).** PDF tables flatten into text. Table-aware parsing, storing each table row as its own chunk, would fix it.

### Recommendations
1. **Never deploy a bare LLM for institutional Q&A.** Use RAG as the minimum. It is the single biggest improvement (5× fact recall) and makes answers auditable through page citations.
2. **Treat LLMOps as a required layer, not an optional extra.** Relevance gating, output verification, tracing, feedback and versioned evaluation are what turn a demo into something students can rely on. Prompts alone are not a safety mechanism.
3. **Use PEFT for behaviour, RAG for facts, and scale the data before the rank.** LoRA is cheap (1.75% of parameters, a ~35 MB adapter, about 10 minutes on a laptop GPU) and clearly improved grounding, citations and speed. To lift recall, grow the training set first (300+ checked, RAFT-style examples), then consider a 1.5B–7B base model (`HELPDESK_LLM`). Keep merged-adapter serving for zero inference overhead.
4. **Evaluate before and after every change, with ablations.** Grow `eval_set.json` from real 👎-rated questions in the dashboard's review queue, and keep the leakage guard between training and evaluation data.
5. **Keep the knowledge base current.** Re-index whenever the handbook or circulars change. Retraining the adapter is only needed when the desired *behaviour* changes, not when the facts change.
6. **Privacy by design.** Running locally keeps student queries on campus infrastructure. Input PII redaction (emails, phone numbers, register numbers) stops personal data from entering prompts, logs or fine-tuning data.
