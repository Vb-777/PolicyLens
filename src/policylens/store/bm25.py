"""BM25 Okapi lexical index.

Implemented directly rather than pulled from a library: it is a short algorithm, it
removes a dependency, and the inverted-index structure here is what makes the lexical
half of retrieval fast enough to run on every query.

Lexical matching earns its place in an insurance corpus because the decisive tokens are
often exact: "36 months", "Clause 4.2", "AYUSH", "cataract". A dense encoder will
happily rate "24 months" and "36 months" as near-identical; BM25 will not.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")

# Deliberately short: policy text is formal and most "stopwords" lists would strip
# terms such as "no", "not" and "other" that flip the meaning of an exclusion.
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "with",
    "is", "are", "be", "been", "as", "at", "that", "this", "it", "its", "from",
    "shall", "any", "such", "which", "under", "within", "may", "has", "have",
}


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that keeps numbers and decimal clause references intact."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_dl: float = 0.0
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    def build(self, texts: list[str]) -> None:
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.n_docs = len(self.doc_tokens)
        self.avg_dl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.inverted = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        for doc_id, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            for term, tf in counts.items():
                self.inverted[term].append((doc_id, tf))
                doc_freq[term] += 1

        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it non-negative.
        self.idf = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self.n_docs == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_dl or 1.0))
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:k]]

    # -- persistence ------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "bm25.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "doc_tokens": self.doc_tokens,
                    "doc_len": self.doc_len,
                    "avg_dl": self.avg_dl,
                    "inverted": dict(self.inverted),
                    "idf": self.idf,
                    "n_docs": self.n_docs,
                },
                fh,
            )
        (directory / "bm25_meta.json").write_text(
            json.dumps({"n_docs": self.n_docs, "vocab": len(self.idf)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        path = Path(directory) / "bm25.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found in {directory}")
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        index = cls(k1=blob["k1"], b=blob["b"])
        index.doc_tokens = blob["doc_tokens"]
        index.doc_len = blob["doc_len"]
        index.avg_dl = blob["avg_dl"]
        index.inverted = defaultdict(list, blob["inverted"])
        index.idf = blob["idf"]
        index.n_docs = blob["n_docs"]
        return index

    def __len__(self) -> int:
        return self.n_docs
