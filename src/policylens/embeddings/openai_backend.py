"""OpenAI embeddings backend, for when hosted embedding quality is wanted."""

from __future__ import annotations

import os

import numpy as np

from ..config import EmbeddingConfig
from .base import Embedder, _l2_normalize


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.cfg = config or EmbeddingConfig()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the openai embedding backend.")
        from openai import OpenAI

        self._client = OpenAI()
        self._dim = 1536 if "3-small" in self.cfg.openai_model else self.cfg.dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vectors: list[list[float]] = []
        batch = max(1, self.cfg.batch_size)
        for start in range(0, len(texts), batch):
            window = texts[start : start + batch]
            response = self._client.embeddings.create(model=self.cfg.openai_model, input=window)
            vectors.extend(item.embedding for item in response.data)
        mat = np.asarray(vectors, dtype="float32")
        self._dim = mat.shape[1]
        return _l2_normalize(mat) if self.cfg.normalize else mat

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "model": self.cfg.openai_model, "dim": self.dim}
