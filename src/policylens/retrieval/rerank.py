"""Reranking the fused candidate pool.

Two backends. The cross-encoder is the stronger one where model weights are reachable.
Where they are not, a feature-based reranker takes over; it is not a toy, because the
dominant retrieval error in a multi-policy corpus is *right clause, wrong policy* —
six documents all contain a "grace period" clause — and the single most informative
feature for fixing that is whether the question names the policy the chunk came from.
That is a cheap exact-match signal a bi-encoder throws away.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from ..config import RetrievalConfig
from ..schemas import ScoredChunk
from ..store.bm25 import tokenize

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\b\d[\d,]*\b")


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, config: RetrievalConfig) -> None:
        from sentence_transformers import CrossEncoder

        self.cfg = config
        self.model = CrossEncoder(config.rerank_model)

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.for_embedding()) for c in candidates]
        scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
            candidate.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)


class LexicalFeatureReranker(Reranker):
    """Feature-based reranker that runs anywhere, with no model download.

    Score is a weighted sum of five signals, all in [0, 1]:

    ``fusion``    the retriever's own fused score, min-max normalised in the pool
    ``coverage``  fraction of the query's content tokens present in the chunk
    ``numeric``   fraction of numbers in the query that appear in the chunk
    ``policy``    overlap between the query and the chunk's policy/product name
    ``heading``   overlap between the query and the section + clause heading

    The weights are fixed rather than learned: with 122 evaluation questions, fitting
    them would be fitting the test set.
    """

    name = "lexical-features"

    WEIGHTS = {
        "fusion": 0.40,
        "coverage": 0.20,
        "numeric": 0.10,
        "policy": 0.20,
        "heading": 0.10,
    }

    def __init__(self, config: RetrievalConfig) -> None:
        self.cfg = config

    @staticmethod
    def _overlap(a: set[str], b: set[str]) -> float:
        if not a:
            return 0.0
        return len(a & b) / len(a)

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []

        query_tokens = set(tokenize(query))
        query_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(query)}

        scores = [c.score for c in candidates]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0

        for candidate in candidates:
            meta = candidate.chunk.meta
            chunk_tokens = set(tokenize(candidate.chunk.text))
            chunk_numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(candidate.chunk.text)}
            policy_tokens = set(tokenize(f"{meta.policy_display} {meta.product_type} {meta.insurer}"))
            heading_tokens = set(tokenize(f"{meta.section_title} {meta.clause_title}"))

            features = {
                "fusion": (candidate.score - lo) / span,
                "coverage": self._overlap(query_tokens, chunk_tokens),
                "numeric": self._overlap(query_numbers, chunk_numbers) if query_numbers else 0.0,
                "policy": self._overlap(policy_tokens & query_tokens, query_tokens)
                if policy_tokens
                else 0.0,
                "heading": self._overlap(heading_tokens & query_tokens, query_tokens)
                if heading_tokens
                else 0.0,
            }
            # `policy` and `heading` above measure how much of the *query* is explained
            # by the policy name / heading, which rewards a question that names its
            # policy without penalising one that does not.
            total = sum(self.WEIGHTS[k] * v for k, v in features.items())
            candidate.rerank_score = float(total)
            candidate.score = float(total)

        return sorted(candidates, key=lambda c: c.score, reverse=True)


def build_reranker(config: RetrievalConfig) -> Reranker:
    backend = config.rerank_backend.lower()

    if backend in {"cross-encoder", "ce"}:
        return CrossEncoderReranker(config)
    if backend in {"lexical", "features", "offline"}:
        return LexicalFeatureReranker(config)
    if backend != "auto":
        raise ValueError(f"Unknown rerank backend: {config.rerank_backend}")

    try:
        return CrossEncoderReranker(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cross-encoder reranker unavailable (%s); using the feature-based reranker.",
            type(exc).__name__,
        )
        return LexicalFeatureReranker(config)
