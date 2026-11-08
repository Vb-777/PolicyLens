"""Hybrid dense + lexical retrieval fused with reciprocal rank fusion.

Neither half is sufficient alone on policy text. Dense retrieval handles the
paraphrase gap — a user asks "how long before my old condition is covered?" and the
clause says "pre-existing disease waiting period" — but it is weak exactly where this
domain is strict, treating "24 months" and "36 months" as near-identical. BM25 is the
opposite. RRF combines them on rank rather than score, which avoids having to
normalise two incomparable score scales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import RetrievalConfig
from ..indexer import RagIndex
from ..schemas import ScoredChunk


@dataclass
class RetrievalResult:
    results: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[int, float]]], k: int = 60
) -> dict[int, dict[str, Any]]:
    """Fuse ranked lists. ``score = sum over lists of 1 / (k + rank)``.

    Rank is 1-based. ``k`` damps the contribution of top positions so a document found
    by both retrievers at middling rank can outrank one found by a single retriever at
    rank 1 — which is the behaviour that makes hybrid retrieval worth the complexity.
    """
    fused: dict[int, dict[str, Any]] = {}
    for source, ranked in ranked_lists.items():
        for rank, (doc_id, raw_score) in enumerate(ranked, start=1):
            entry = fused.setdefault(
                doc_id, {"score": 0.0, "ranks": {}, "raw": {}}
            )
            entry["score"] += 1.0 / (k + rank)
            entry["ranks"][source] = rank
            entry["raw"][source] = raw_score
    return fused


class HybridRetriever:
    def __init__(self, index: RagIndex, config: RetrievalConfig | None = None) -> None:
        self.index = index
        self.cfg = config or RetrievalConfig()
        self._reranker = None

    # -- reranker is built lazily: it may need a model download we want to avoid
    #    paying for when reranking is switched off.
    def _get_reranker(self):
        if self._reranker is None:
            from .rerank import build_reranker

            self._reranker = build_reranker(self.cfg)
        return self._reranker

    def _mode_name(self) -> str:
        if self.cfg.use_dense and self.cfg.use_hybrid:
            return "hybrid"
        return "dense" if self.cfg.use_dense else "lexical"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        policy_filter: list[str] | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.top_k
        do_rerank = self.cfg.rerank if rerank is None else rerank
        timings: dict[str, float] = {}
        ranked_lists: dict[str, list[tuple[int, float]]] = {}

        if self.cfg.use_dense:
            t0 = time.perf_counter()
            query_vector = self.index.embedder.encode([query], is_query=True)[0]
            ranked_lists["dense"] = self.index.store.search(query_vector, self.cfg.dense_k)
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000

        if self.cfg.use_hybrid or not self.cfg.use_dense:
            t1 = time.perf_counter()
            ranked_lists["lexical"] = self.index.bm25.search(query, self.cfg.lexical_k)
            timings["lexical_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion(ranked_lists, k=self.cfg.rrf_k)
        timings["fuse_ms"] = (time.perf_counter() - t2) * 1000

        chunks = self.index.chunks
        candidates: list[ScoredChunk] = []
        for doc_id, entry in fused.items():
            if doc_id >= len(chunks):
                continue
            chunk = chunks[doc_id]
            if policy_filter and chunk.meta.policy_id not in policy_filter:
                continue
            candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(entry["score"]),
                    dense_score=entry["raw"].get("dense"),
                    lexical_score=entry["raw"].get("lexical"),
                    dense_rank=entry["ranks"].get("dense"),
                    lexical_rank=entry["ranks"].get("lexical"),
                    retriever=self._mode_name(),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)

        if do_rerank and candidates:
            t3 = time.perf_counter()
            pool = candidates[: self.cfg.rerank_candidates]
            reranked = self._get_reranker().rerank(query, pool)
            timings["rerank_ms"] = (time.perf_counter() - t3) * 1000
            tail = candidates[self.cfg.rerank_candidates :]
            candidates = reranked + tail

        results = candidates[:top_k]
        timings["retrieve_total_ms"] = sum(
            v for k, v in timings.items() if k.endswith("_ms") and k != "retrieve_total_ms"
        )

        return RetrievalResult(
            results=results,
            timings_ms=timings,
            debug={
                "n_dense": len(ranked_lists.get("dense", [])),
                "n_lexical": len(ranked_lists.get("lexical", [])),
                "n_fused": len(fused),
                "reranked": bool(do_rerank and results),
            },
        )
