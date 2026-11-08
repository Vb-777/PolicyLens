"""Dependency-free dense backend: TF-IDF followed by truncated SVD (LSA).

This exists so the project runs — and the evaluation harness produces real numbers —
on a machine with no access to a model hub. It is genuinely weaker than a pretrained
sentence encoder on paraphrased questions, which is precisely why the harness reports
the backend alongside every metric instead of quoting a single headline number.

Being corpus-fitted, it must be persisted with the index: a query embedded by a
differently-fitted vectoriser lands in a different space and retrieval silently
collapses.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..config import EmbeddingConfig
from .base import Embedder, _l2_normalize


class TfidfSvdEmbedder(Embedder):
    name = "tfidf-svd"

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.cfg = config or EmbeddingConfig()
        self._vectorizer = None
        self._svd = None
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim or self.cfg.dim

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Word unigrams+bigrams capture phrases such as "waiting period" and
        # "pre-existing disease" that carry most of the signal in policy text.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.9,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w[\w\-]+\b",
        )
        matrix = self._vectorizer.fit_transform(texts)

        max_components = max(2, min(matrix.shape[0] - 1, matrix.shape[1] - 1))
        n_components = int(min(self.cfg.dim, max_components))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)
        self._dim = n_components

    def _require_fitted(self) -> None:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError(
                "TfidfSvdEmbedder must be fitted on the corpus (or loaded from an "
                "existing index) before encoding."
            )

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        self._require_fitted()
        assert self._vectorizer is not None and self._svd is not None
        matrix = self._vectorizer.transform(texts)
        dense = self._svd.transform(matrix).astype("float32")
        return _l2_normalize(dense) if self.cfg.normalize else dense

    def save(self, directory: Path) -> None:
        self._require_fitted()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tfidf_svd.pkl").open("wb") as fh:
            pickle.dump(
                {"vectorizer": self._vectorizer, "svd": self._svd, "dim": self._dim},
                fh,
            )

    def load(self, directory: Path) -> bool:
        path = Path(directory) / "tfidf_svd.pkl"
        if not path.exists():
            return False
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        self._vectorizer = blob["vectorizer"]
        self._svd = blob["svd"]
        self._dim = blob["dim"]
        return True

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "model": "tfidf(1,2)+svd", "dim": self.dim}
