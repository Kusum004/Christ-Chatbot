"""Hybrid retriever: dense embeddings (semantic) + BM25 (keyword), fused by weighted score."""
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config
from .ingest import load_chunks


def tokenize(text: str):
    return re.findall(r"[a-z0-9%]+", text.lower())


class HybridRetriever:
    def __init__(self, chunks=None, embed_model: str = config.EMBED_MODEL):
        self.chunks = chunks or load_chunks()
        self.embedder = SentenceTransformer(embed_model)
        self.embeddings = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, convert_to_numpy=True)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def search(self, query: str, k: int = config.TOP_K, alpha: float = config.HYBRID_ALPHA):
        """Returns (results, elapsed_ms). Each result: dict(chunk, score, dense, bm25)."""
        t0 = time.perf_counter()
        dense = self.embeddings @ self.embed(query)
        bm25 = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        bm25_n = bm25 / bm25.max() if bm25.max() > 0 else bm25
        hybrid = alpha * dense + (1 - alpha) * bm25_n
        order = np.argsort(-hybrid)[:k]
        results = [dict(chunk=self.chunks[i], score=float(hybrid[i]),
                        dense=float(dense[i]), bm25=float(bm25_n[i])) for i in order]
        return results, (time.perf_counter() - t0) * 1000
