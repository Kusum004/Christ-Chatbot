"""PEFT (LoRA) fine-tuning of the local LLM for the hybrid model (Model 3: LLM + RAG + LLMOps).

The adapter teaches the small LLM *how to use retrieved handbook passages*: answer briefly from
the context, cite the passage [n], and refuse when the answer is not in the context.
Training data are RAFT-style: each question is paired with the real retriever's top-k passages,
the gold passage is guaranteed to be present (randomly placed), and a share of examples have the
gold passage removed so the target becomes a refusal.

LLMOps practices built in:
  * data-leakage guard (drops training questions too similar to the evaluation set)
  * deterministic seed, train/validation split, loss logging
  * versioned adapter + model card + registry (models/registry.json -> active adapter)

Usage:  python finetune_peft.py [--epochs 3] [--rank 16] [--lr 2e-4]
"""
import argparse
import hashlib
import json
import random
import re
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatbot import config
from chatbot.prompts import REFUSAL, build_messages, format_context
from chatbot.retriever import HybridRetriever

TRAIN_FILE = config.ROOT / "data" / "peft_train.json"
SEED = 42
NEG_SHARE = float(__import__("os").getenv("PEFT_NEG_SHARE", "0.10"))
# share of answerable questions also used as "gold removed -> refuse".
# v1 used 0.25 and over-refused answerable questions (see README); v2 uses 0.10.
LEAK_THRESHOLD = 0.85     # cosine similarity to any eval question above which an item is dropped


def build_examples(retriever, items, rng):
    chunks = retriever.chunks
    examples, skipped = [], []
    for it in items:
        results, _ = retriever.search(it["q"], k=config.TOP_K)
        if it["a"] == "REFUSE":
            examples.append(dict(q=it["q"], results=results, target=REFUSAL, kind="refuse"))
            continue
        pat = re.compile(it["find"], re.I)
        gold = next((c for c in chunks if pat.search(c.text)), None)
        if gold is None:
            skipped.append(it["q"])
            continue
        others = [r for r in results if r["chunk"].id != gold.id][: config.TOP_K - 1]
        ctx = others + [dict(chunk=gold, score=1.0, dense=1.0, bm25=1.0)]
        rng.shuffle(ctx)                                   # gold at a random position
        idx = next(i for i, r in enumerate(ctx, 1) if r["chunk"].id == gold.id)
        examples.append(dict(q=it["q"], results=ctx, target=f"{it['a']} [{idx}]", kind="answer"))
        # RAFT negative: gold removed and no other passage contains the fact -> refuse
        if rng.random() < NEG_SHARE and not any(pat.search(r["chunk"].text) for r in others):
            examples.append(dict(q=it["q"], results=others, target=REFUSAL, kind="gold_removed"))
    return examples, skipped


def leakage_filter(retriever, items):
    evals = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    E = retriever.embedder.encode([e["q"] for e in evals], normalize_embeddings=True)
    T = retriever.embedder.encode([i["q"] for i in items], normalize_embeddings=True)
    sims = (T @ E.T).max(axis=1)
    kept = [it for it, s in zip(items, sims) if s < LEAK_THRESHOLD]
    dropped = [(it["q"], round(float(s), 2)) for it, s in zip(items, sims) if s >= LEAK_THRESHOLD]
    return kept, dropped


def encode(tok, ex, max_len=1280):
    prompt = tok.apply_chat_template(
        build_messages(config.PROMPT_VERSION, ex["q"], format_context(ex["results"])),
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[-max_len:]
    labels = ([-100] * len(p_ids) + a_ids)[-max_len:]      # loss only on the answer tokens
    return torch.tensor([ids]), torch.tensor([labels])


@torch.no_grad()
def val_loss(model, data):
    model.eval()
    losses = [model(input_ids=x.to(model.device), labels=y.to(model.device)).loss.item()
              for x, y in data]
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main(args):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()

    retriever = HybridRetriever()
    items = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
    items, leaked = leakage_filter(retriever, items)
    print(f"leakage guard: dropped {len(leaked)} item(s) {leaked}")
    examples, skipped = build_examples(retriever, items, rng)
    if skipped:
        print(f"WARNING gold passage not found for: {skipped}")
    rng.shuffle(examples)
    n_val = max(4, len(examples) // 10)
    val_ex, train_ex = examples[:n_val], examples[n_val:]
    kinds = {k: sum(e["kind"] == k for e in examples) for k in ("answer", "refuse", "gold_removed")}
    print(f"examples: {len(examples)} {kinds} -> train {len(train_ex)} / val {len(val_ex)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    # bf16 base weights (frozen) keep a 6 GB laptop GPU within memory; LoRA weights stay fp32
    base_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, dtype=base_dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()                         # train adapters in full precision
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_data = [encode(tok, e) for e in train_ex]
    val_data = [encode(tok, e) for e in val_ex]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_total = args.epochs * len(train_data) // args.grad_accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / max(1, steps_total)))

    history = [dict(epoch=0, train_loss=None, val_loss=val_loss(model, val_data))]
    print(f"epoch 0  val_loss {history[0]['val_loss']:.3f}")
    best_val, best_epoch, best_state = float("inf"), 0, None
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)
        losses = []
        for i, (x, y) in enumerate(train_data, 1):
            loss = model(input_ids=x.to(device), labels=y.to(device)).loss
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            if i % 12 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_data)} loss {np.mean(losses[-12:]):.3f}", flush=True)
            if i % args.grad_accum == 0 or i == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        h = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss(model, val_data))
        history.append(h)
        print(f"epoch {epoch}  train_loss {h['train_loss']:.3f}  val_loss {h['val_loss']:.3f}", flush=True)
        if h["val_loss"] < best_val:                         # early stopping: keep the best epoch
            best_val, best_epoch = h["val_loss"], epoch
            best_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

    set_peft_model_state_dict(model, best_state)
    print(f"restored best epoch {best_epoch} (val_loss {best_val:.3f})")

    # ---- versioned save + model card + registry
    models_dir = config.ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    version = 1 + max([int(p.name.rsplit("-v", 1)[1]) for p in models_dir.glob("lora-helpdesk-v*")] or [0])
    out = models_dir / f"lora-helpdesk-v{version}"
    model.save_pretrained(out)
    card = dict(
        adapter=out.name, base_model=config.LLM_MODEL, method="LoRA (PEFT)",
        prompt_version=config.PROMPT_VERSION, created=time.strftime("%Y-%m-%d %H:%M"),
        lora=dict(r=args.rank, alpha=2 * args.rank, dropout=0.05, targets=sorted(lora.target_modules)),
        train=dict(epochs=args.epochs, best_epoch=best_epoch, neg_share=NEG_SHARE,
                   lr=args.lr, grad_accum=args.grad_accum, seed=SEED,
                   examples=len(train_ex), val_examples=len(val_ex), kinds=kinds),
        trainable_params=trainable, total_params=total,
        data_sha256=hashlib.sha256(TRAIN_FILE.read_bytes()).hexdigest()[:16],
        leakage_dropped=leaked, history=history, minutes=round((time.time() - t0) / 60, 1))
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    reg_path = models_dir / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"versions": []}
    reg["versions"].append({k: card[k] for k in ("adapter", "created", "history")})
    reg["active"] = out.name
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"saved {out} and set it active in models/registry.json")

    # ---- quick qualitative check on validation examples
    model.eval()
    model.config.use_cache = True
    for ex in val_ex[:3]:
        x, _ = encode(tok, dict(ex, target=""))
        x = x[:, :-1]  # drop the eos appended for an empty target
        with torch.no_grad():
            g = model.generate(input_ids=x.to(device), max_new_tokens=80, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        print(f"\nQ: {ex['q']}\nTARGET: {ex['target']}\nOUTPUT: "
              f"{tok.decode(g[0][x.shape[1]:], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    main(ap.parse_args())
