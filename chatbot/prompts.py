"""Versioned prompt registry (an LLMOps practice: prompts are tracked like code,
every trace records which version produced the answer, and versions can be A/B evaluated)."""

PROMPTS = {
    "llm_only_v1": {
        "description": "Model 1 baseline: no context, the LLM answers from its own parameters.",
        "system": "You are the student helpdesk assistant of CHRIST (Deemed to be University), "
                  "Bengaluru. Answer student questions helpfully and concisely.",
        "user": "{question}",
    },
    "rag_v1": {
        "description": "Model 2: basic RAG prompt, context is pasted before the question.",
        "system": "You are the student helpdesk assistant of CHRIST (Deemed to be University), "
                  "Bengaluru. Use the context to answer the question concisely.",
        "user": "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    },
    "rag_v2": {
        "description": "Model 3: strict grounded prompt with citations and an explicit refusal rule.",
        "system": (
            "You are the official student helpdesk assistant of CHRIST (Deemed to be University), Bengaluru.\n"
            "Rules:\n"
            "1. Answer ONLY using facts in the numbered context passages.\n"
            "2. Cite the passages you used like [1] or [2].\n"
            "3. If the context does not contain the answer, reply exactly: "
            "\"I don't have that information. Please contact the relevant university office or check https://christuniversity.in.\"\n"
            "4. Never invent numbers, dates, fees, names or phone numbers.\n"
            "5. Keep the answer under 120 words, in a friendly tone."
        ),
        "user": "Context:\n{context}\n\nStudent question: {question}\n\nGrounded answer with citations:",
    },
}

REFUSAL = ("I don't have that information. Please contact the relevant university office "
           "or check https://christuniversity.in.")


def build_messages(version: str, question: str, context: str = ""):
    p = PROMPTS[version]
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"].format(question=question, context=context)}]


def format_context(results):
    return "\n\n".join(f"[{i}] ({r['chunk'].citation}) {r['chunk'].text}"
                       for i, r in enumerate(results, 1))
