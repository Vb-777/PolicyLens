"""Embedder interface and backend selection.

Three backends exist for one practical reason: the quality backend
(sentence-transformers) needs to download model weights, and plenty of environments —
locked-down CI, air-gapped corporate machines, this project's own container — cannot
reach a model hub. Rather than making the project unrunnable there, the backend is an
interface and ``auto`` degrades to a dependency-free local backend, recording in the
report which one produced the numbers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Maps text to dense vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int: ...

    def fit(self, texts: list[str]) -> None:
        """Corpus-dependent backends override this. Pretrained backends ignore it."""
        return None

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> bool:
        return True

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "dim": self.dim}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    """Construct the configured embedder, falling back when a backend is unavailable."""
    cfg = config or EmbeddingConfig()
    backend = cfg.backend.lower()

    if backend in {"sentence-transformers", "st", "huggingface"}:
        from .st_backend import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(cfg)

    if backend == "openai":
        from .openai_backend import OpenAIEmbedder

        return OpenAIEmbedder(cfg)

    # "tfidf-svd" is the name the backend reports in an index manifest, so reloading
    # an index built with it must resolve here.
    if backend in {"tfidf", "tfidf-svd", "local", "offline"}:
        from .tfidf_backend import TfidfSvdEmbedder

        return TfidfSvdEmbedder(cfg)

    if backend != "auto":
        raise ValueError(f"Unknown embedding backend: {cfg.backend}")

    # auto: prefer pretrained semantics, fall back to the local backend.
    try:
        from .st_backend import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(cfg)
        embedder.ensure_loaded()
        return embedder
    except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
        logger.warning(
            "sentence-transformers backend unavailable (%s); falling back to the "
            "local TF-IDF+SVD backend. Retrieval quality will be lower.",
            type(exc).__name__,
        )
        from .tfidf_backend import TfidfSvdEmbedder

        return TfidfSvdEmbedder(cfg)
