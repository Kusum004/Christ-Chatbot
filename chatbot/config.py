"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"
EVAL_SET = ROOT / "data" / "eval_set.json"
LOG_DIR = ROOT / "logs"
DB_PATH = LOG_DIR / "llmops.db"
LOG_DIR.mkdir(exist_ok=True)

# Local models (downloaded once from Hugging Face, then run fully offline)
LLM_MODEL = os.getenv("HELPDESK_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL = os.getenv("HELPDESK_EMBED", "sentence-transformers/all-MiniLM-L6-v2")

# Generation
MAX_NEW_TOKENS = int(os.getenv("HELPDESK_MAX_TOKENS", "220"))

# Retrieval
CHUNK_WORDS = 140          # max words per chunk
CHUNK_OVERLAP = 30         # word overlap when a section is split
TOP_K = 4                  # chunks passed to the LLM
HYBRID_ALPHA = 0.6         # weight of dense score vs BM25 in the hybrid ranking

# PEFT (LoRA adapter, used by the hybrid Model 3 only)
MODELS_DIR = ROOT / "models"
USE_PEFT = os.getenv("HELPDESK_PEFT", "1") == "1"


def active_adapter():
    """Adapter path from env HELPDESK_ADAPTER, else the 'active' entry of models/registry.json."""
    import json
    if os.getenv("HELPDESK_ADAPTER"):
        return Path(os.environ["HELPDESK_ADAPTER"])
    reg = MODELS_DIR / "registry.json"
    if reg.exists():
        path = MODELS_DIR / json.loads(reg.read_text(encoding="utf-8"))["active"]
        return path if path.exists() else None
    return None


# LLMOps
PROMPT_VERSION = os.getenv("HELPDESK_PROMPT", "rag_v2")   # prompt used by Model 3
MIN_RELEVANCE = 0.40       # below this dense similarity, Model 3 declines to answer
MIN_GROUNDEDNESS = 0.5      # below this, Model 3 flags / replaces the answer
CACHE_SIMILARITY = 0.95     # semantic-cache hit threshold
MAX_QUERY_CHARS = 600
