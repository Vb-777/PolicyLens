"""FAISS-backed dense vector store with on-disk persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import Chunk


class FaissStore:
    """Inner-product index over L2-normalised vectors, i.e. cosine similarity.

    Flat (exhaustive) search is the right choice at this corpus size: it is exact, has
    no training step, and removes approximate-search recall loss as a confound when
    measuring retrieval quality. Swapping in IVF or HNSW becomes worthwhile in the
    hundreds-of-thousands-of-chunks range.
    """

    INDEX_FILE = "faiss.index"
    CHUNKS_FILE = "chunks.jsonl"
    META_FILE = "store_meta.json"

    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    # -- build ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(np.ascontiguousarray(vectors.astype("float32")))
        self.chunks.extend(chunks)

    # -- search -----------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query = np.ascontiguousarray(query_vector.reshape(1, -1).astype("float32"))
        scores, indices = self.index.search(query, k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i >= 0]

    # -- persistence ------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        with (directory / self.CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        (directory / self.META_FILE).write_text(
            json.dumps({"dim": self.dim, "n_chunks": len(self.chunks)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        import faiss

        directory = Path(directory)
        meta_path = directory / cls.META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No index found in {directory}. Build one with `python -m policylens.cli index`."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))
        store.index = faiss.read_index(str(directory / cls.INDEX_FILE))
        with (directory / cls.CHUNKS_FILE).open("r", encoding="utf-8") as fh:
            store.chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        return store

    def __len__(self) -> int:
        return len(self.chunks)
