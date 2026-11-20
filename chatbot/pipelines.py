"""The three models compared in this project.

Model 1 - LLM only      : the local LLM answers from its own (pre-trained) knowledge.
Model 2 - RAG           : retrieve relevant KB passages, then let the LLM answer from them.
Model 3 - RAG + LLMOps  : the hybrid model. RAG wrapped in production controls - guardrails,
  + PEFT                  relevance gate, strict versioned prompt, number/groundedness checks,
                          semantic cache, tracing, feedback, offline evaluation - and a LoRA
                          adapter (PEFT) fine-tuned to answer from retrieved passages.
"""
import time
from dataclasses import dataclass, field

from . import config
from .llm import LocalLLM
from .llmops import SemanticCache, Tracer, groundedness, input_guardrails, unsupported_numbers
from .prompts import REFUSAL, build_messages, format_context
from .retriever import HybridRetriever

PIPELINES = {
    "llm": "Model 1 - LLM only",
    "rag": "Model 2 - RAG",
    "ragops": "Model 3 - RAG + LLMOps + PEFT",
}


@dataclass
class Answer:
    pipeline: str
    text: str
    sources: list = field(default_factory=list)   # [{citation, source, score}]
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    groundedness: float | None = None
    top_relevance: float | None = None
    cache_hit: bool = False
    flags: list = field(default_factory=list)
    prompt_version: str = ""
    adapter: str | None = None        # LoRA adapter used (Model 3 with PEFT), else None
    trace_id: str | None = None


class Helpdesk:
    def __init__(self, llm: LocalLLM | None = None, retriever: HybridRetriever | None = None,
                 tracer: Tracer | None = None, log: bool = True):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or LocalLLM(adapter=config.active_adapter())
        self.tracer = tracer or Tracer()
        self.cache = SemanticCache()
        self.log = log

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _sources(results):
        return [dict(citation=r["chunk"].citation, source=r["chunk"].source,
                     chunk_id=r["chunk"].id, score=round(r["score"], 3)) for r in results]

    def _trace(self, ans: Answer, query: str, clean_query: str):
        if not self.log:
            return None
        return self.tracer.log(
            pipeline=ans.pipeline,
            model=self.llm.model_name + (f" + {ans.adapter}" if ans.adapter else ""),
            prompt_version=ans.prompt_version,
            query=query, clean_query=clean_query, answer=ans.text, sources=ans.sources,
            top_relevance=ans.top_relevance, retrieval_ms=ans.retrieval_ms,
            generation_ms=ans.generation_ms, total_ms=ans.total_ms,
            prompt_tokens=ans.prompt_tokens, completion_tokens=ans.completion_tokens,
            groundedness=ans.groundedness, cache_hit=int(ans.cache_hit), flags=ans.flags)

    # ----------------------------------------------------------- Model 1
    def llm_only(self, query: str) -> Answer:
        t0 = time.perf_counter()
        out = self.llm.chat(build_messages("llm_only_v1", query))
        ans = Answer("llm", out["text"], generation_ms=out["latency_ms"],
                     prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                     prompt_version="llm_only_v1")
        ans.total_ms = (time.perf_counter() - t0) * 1000
        ans.trace_id = self._trace(ans, query, query)
        return ans

    # ----------------------------------------------------------- Model 2
    def rag(self, query: str, prompt_version: str = "rag_v1") -> Answer:
        t0 = time.perf_counter()
        results, r_ms = self.retriever.search(query)
        context = format_context(results)
        out = self.llm.chat(build_messages(prompt_version, query, context))
        ans = Answer("rag", out["text"], sources=self._sources(results), retrieval_ms=r_ms,
                     generation_ms=out["latency_ms"], prompt_tokens=out["prompt_tokens"],
                     completion_tokens=out["completion_tokens"],
                     top_relevance=results[0]["dense"], prompt_version=prompt_version,
                     groundedness=groundedness(out["text"], context))
        ans.total_ms = (time.perf_counter() - t0) * 1000
        ans.trace_id = self._trace(ans, query, query)
        return ans

    # ----------------------------------------------------------- Model 3
    def rag_ops(self, query: str, prompt_version: str = config.PROMPT_VERSION,
                use_cache: bool = True, peft: bool | None = None) -> Answer:
        t0 = time.perf_counter()
        use_peft = (config.USE_PEFT if peft is None else peft) and self.llm.has_adapter
        ans = Answer("ragops", "", prompt_version=prompt_version,
                     adapter=self.llm.adapter if use_peft else None)
        cache_key = f"{prompt_version}|{ans.adapter}"     # never mix answers across model versions

        # 1. input guardrails (PII redaction, prompt-injection, policy)
        clean, flags, blocked = input_guardrails(query)
        ans.flags = flags
        if blocked:
            ans.text = blocked
            ans.total_ms = (time.perf_counter() - t0) * 1000
            ans.trace_id = self._trace(ans, query, clean)
            return ans

        # 2. semantic cache
        q_emb = self.retriever.embed(clean)
        if use_cache:
            hit, _ = self.cache.get(cache_key, q_emb)
            if hit:
                ans.text, ans.sources = hit["text"], hit["sources"]
                ans.groundedness, ans.top_relevance = hit["groundedness"], hit["top_relevance"]
                ans.cache_hit = True
                ans.total_ms = (time.perf_counter() - t0) * 1000
                ans.trace_id = self._trace(ans, query, clean)
                return ans

        # 3. retrieval + relevance gate (refuse instead of hallucinating on out-of-scope queries)
        results, ans.retrieval_ms = self.retriever.search(clean)
        ans.top_relevance = results[0]["dense"]
        ans.sources = self._sources(results)
        if ans.top_relevance < config.MIN_RELEVANCE:
            ans.flags.append("low_relevance")
            ans.text, ans.sources, ans.groundedness = REFUSAL, [], 1.0
        else:
            # 4. generation with the strict, versioned prompt
            context = format_context(results)
            out = self.llm.chat(build_messages(prompt_version, clean, context), use_adapter=use_peft)
            ans.text = out["text"]
            ans.generation_ms = out["latency_ms"]
            ans.prompt_tokens, ans.completion_tokens = out["prompt_tokens"], out["completion_tokens"]
            # 5. output guardrail: groundedness check
            ans.groundedness = groundedness(ans.text, context)
            bad_nums = unsupported_numbers(ans.text, context, clean)
            if REFUSAL[:30] in ans.text:
                ans.flags.append("model_refused")
                ans.sources = []
            elif bad_nums:
                # never show a figure the handbook does not contain - point to the source instead
                ans.flags.append("unsupported_numbers:" + ",".join(bad_nums))
                ans.text = (f"I couldn't verify the exact figure in the handbook. Please see "
                            f"**{ans.sources[0]['citation']}** of the Christite Handbook 2026-27, "
                            f"or contact your department / the relevant university office.")
            elif ans.groundedness < config.MIN_GROUNDEDNESS:
                ans.flags.append("low_groundedness")
                ans.text += ("\n\n_Note: parts of this answer could not be verified against the "
                             "knowledge base. Please confirm with the university office._")

        if use_cache and "low_relevance" not in ans.flags:
            self.cache.put(cache_key, q_emb, dict(
                text=ans.text, sources=ans.sources, groundedness=ans.groundedness,
                top_relevance=ans.top_relevance))
        ans.total_ms = (time.perf_counter() - t0) * 1000
        ans.trace_id = self._trace(ans, query, clean)
        return ans

    def ask(self, query: str, pipeline: str = "ragops") -> Answer:
        return {"llm": self.llm_only, "rag": self.rag, "ragops": self.rag_ops}[pipeline](query)
