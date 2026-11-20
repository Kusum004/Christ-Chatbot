"""Local LLM wrapper (Hugging Face transformers). Runs on GPU if available, else CPU.

Optionally loads a PEFT/LoRA adapter on top of the base model. The adapter is switched on
only when a caller asks for it (Model 3); Models 1 and 2 always run the untouched base model,
so the comparison between the three models stays fair.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


class LocalLLM:
    def __init__(self, model_name: str = config.LLM_MODEL, adapter: Path | None = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.tuned, self.adapter, self.adapter_card = None, None, {}
        if adapter and Path(adapter).exists():
            # Serving best practice: merge the LoRA weights into a separate copy of the base model,
            # so Model 3 has zero adapter overhead and Models 1/2 keep the untouched base weights.
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            self.tuned = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
            self.tuned = self.tuned.to(self.device).eval()
            self.adapter = Path(adapter).name
            card = Path(adapter) / "model_card.json"
            self.adapter_card = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}

    @property
    def has_adapter(self) -> bool:
        return self.tuned is not None

    @torch.inference_mode()
    def chat(self, messages, max_new_tokens: int = config.MAX_NEW_TOKENS, use_adapter: bool = False):
        """messages: list of {'role', 'content'}. Greedy decoding for reproducible answers."""
        t0 = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # base model unless the caller explicitly asks for the LoRA-tuned model
        model = self.tuned if (use_adapter and self.has_adapter) else self.model
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return dict(text=text,
                    prompt_tokens=int(inputs["input_ids"].shape[1]),
                    completion_tokens=int(len(new_tokens)),
                    latency_ms=(time.perf_counter() - t0) * 1000)
