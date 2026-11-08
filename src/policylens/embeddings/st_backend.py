"""sentence-transformers backend. The default when model weights can be fetched."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import EmbeddingConfig
from .base import Embedder, _l2_normalize


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.cfg = config or EmbeddingConfig()
        self._model = None
        self._dim = self.cfg.dim

    def ensure_loaded(self) -> None:
        """Load the model, raising if weights are unavailable.

        Called eagerly by ``build_embedder`` in auto mode so the fallback decision
        happens at construction time rather than halfway through indexing.
        """
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.cfg.model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        self.ensure_loaded()
        return self._dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        self.ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype("float32")
        return _l2_normalize(vectors) if self.cfg.normalize else vectors

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "model": self.cfg.model, "dim": self.dim}

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> bool:
        try:
            self.ensure_loaded()
            return True
        except Exception:  # noqa: BLE001
            return False
