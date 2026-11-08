"""Configuration, read from environment variables with sensible defaults.

Everything that changes behaviour between a laptop run and a CI run lives here, so the
evaluation harness can record exactly which configuration produced a given number.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    # src/policylens/config.py -> src/policylens -> src -> repo root
    return Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = field(default_factory=_repo_root)

    @property
    def data(self) -> Path:
        return Path(_env("POLICYLENS_DATA_DIR", str(self.root / "data")))

    @property
    def corpus(self) -> Path:
        """The directory the pipeline ingests.

        Defaults to the rendered PDFs rather than the text sources, because a PDF is
        what an insurer actually publishes and ingesting one is the path that has to
        work. The text corpus remains the editable source of truth and the two are
        held to clause-level parity by ``tests/test_pdf_ingest.py``; point
        ``POLICYLENS_CORPUS_DIR`` at ``data/corpus`` to ingest it instead, or at a
        directory of your own policy PDFs.
        """
        override = _env("POLICYLENS_CORPUS_DIR", "")
        if override:
            return Path(override)
        pdfs = self.data / "corpus_pdf"
        if pdfs.exists() and any(pdfs.glob("*.pdf")):
            return pdfs
        return self.data / "corpus"

    @property
    def index(self) -> Path:
        return self.data / "index"

    @property
    def eval(self) -> Path:
        return self.data / "eval"

    @property
    def reports(self) -> Path:
        return Path(_env("POLICYLENS_REPORT_DIR", str(self.root / "reports")))


@dataclass
class ChunkConfig:
    # Clause-aware chunking means these only bite on unusually long clauses.
    max_chars: int = _env_int("POLICYLENS_CHUNK_MAX_CHARS", 1400)
    overlap_chars: int = _env_int("POLICYLENS_CHUNK_OVERLAP", 180)
    min_chars: int = _env_int("POLICYLENS_CHUNK_MIN_CHARS", 80)


@dataclass
class EmbeddingConfig:
    # "auto" tries sentence-transformers and falls back to the dependency-free backend
    # if the model cannot be downloaded (offline CI, air-gapped machines).
    backend: str = _env("POLICYLENS_EMBED_BACKEND", "auto")
    model: str = _env("POLICYLENS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_model: str = _env("POLICYLENS_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    dim: int = _env_int("POLICYLENS_EMBED_DIM", 384)
    batch_size: int = _env_int("POLICYLENS_EMBED_BATCH", 32)
    normalize: bool = _env_bool("POLICYLENS_EMBED_NORMALIZE", True)


@dataclass
class RetrievalConfig:
    top_k: int = _env_int("POLICYLENS_TOP_K", 5)
    dense_k: int = _env_int("POLICYLENS_DENSE_K", 20)
    lexical_k: int = _env_int("POLICYLENS_LEXICAL_K", 20)
    # Reciprocal rank fusion constant. 60 is the value from the original RRF paper and
    # is deliberately left configurable so the eval harness can sweep it.
    rrf_k: int = _env_int("POLICYLENS_RRF_K", 60)
    use_dense: bool = _env_bool("POLICYLENS_USE_DENSE", True)
    use_hybrid: bool = _env_bool("POLICYLENS_USE_HYBRID", True)
    rerank: bool = _env_bool("POLICYLENS_RERANK", True)
    rerank_backend: str = _env("POLICYLENS_RERANK_BACKEND", "auto")
    rerank_model: str = _env("POLICYLENS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = _env_int("POLICYLENS_RERANK_CANDIDATES", 20)
    # Below this fused score the router treats the corpus as having nothing relevant.
    min_score: float = _env_float("POLICYLENS_MIN_SCORE", 0.0)


@dataclass
class LLMConfig:
    provider: str = _env("POLICYLENS_LLM_PROVIDER", "auto")
    anthropic_model: str = _env("POLICYLENS_ANTHROPIC_MODEL", "claude-sonnet-4-5")
    openai_model: str = _env("POLICYLENS_OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = _env("POLICYLENS_GEMINI_MODEL", "gemini-2.0-flash")
    ollama_model: str = _env("POLICYLENS_OLLAMA_MODEL", "llama3.1")
    ollama_host: str = _env("POLICYLENS_OLLAMA_HOST", "http://localhost:11434")
    max_tokens: int = _env_int("POLICYLENS_LLM_MAX_TOKENS", 900)
    temperature: float = _env_float("POLICYLENS_LLM_TEMPERATURE", 0.0)
    timeout_s: float = _env_float("POLICYLENS_LLM_TIMEOUT", 60.0)


@dataclass
class AgentConfig:
    # The self-check loop retries retrieval with a widened k when groundedness is weak.
    enable_self_check: bool = _env_bool("POLICYLENS_SELF_CHECK", True)
    max_retrieval_retries: int = _env_int("POLICYLENS_MAX_RETRIES", 1)
    groundedness_floor: float = _env_float("POLICYLENS_GROUNDEDNESS_FLOOR", 0.55)
    enable_kg: bool = _env_bool("POLICYLENS_ENABLE_KG", True)
    widen_top_k: int = _env_int("POLICYLENS_WIDEN_TOP_K", 10)
    # When a question names exactly one policy, check whether that policy actually
    # addresses the question's subject. Mere provenance is not enough: a clause from
    # the right policy that happens to rank well says nothing about whether the
    # policy covers the provision being asked about. The guard compares the best
    # topical coverage achievable anywhere *inside* the named policy against the best
    # coverage found *outside* it, and declines when the concept plainly lives
    # elsewhere. Both thresholds are fitted on the dev split, never on test.
    enable_policy_guard: bool = _env_bool("POLICYLENS_POLICY_GUARD", True)
    policy_guard_window: int = _env_int("POLICYLENS_POLICY_GUARD_WINDOW", 3)
    # Fitted by grid search on the dev split: `python scripts/fit_guard.py`, which
    # selects these exact values (precision 1.000, recall 0.750, no false abstentions).
    # Dev holds five unanswerable questions, four of which name a single policy and so
    # reach the guard at all, so the script breaks ties toward the
    # loosest thresholds rather than the best F1, and the test-split figures in the
    # evaluation report are the ones to trust. Re-run it after changing the corpus or
    # the embedding backend.
    policy_guard_margin: float = _env_float("POLICYLENS_POLICY_GUARD_MARGIN", 0.30)
    policy_guard_body_floor: float = _env_float("POLICYLENS_POLICY_GUARD_BODY_FLOOR", 0.55)


@dataclass
class EvalConfig:
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    judge: str = _env("POLICYLENS_JUDGE", "deterministic")  # deterministic | llm | both
    judge_provider: str = _env("POLICYLENS_JUDGE_PROVIDER", "")
    limit: int = _env_int("POLICYLENS_EVAL_LIMIT", 0)  # 0 = all


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def fingerprint(self) -> dict[str, Any]:
        """Configuration snapshot embedded in every evaluation report.

        A metric without the configuration that produced it is not reproducible.
        """
        out = {
            "chunk": asdict(self.chunk),
            "embedding": asdict(self.embedding),
            "retrieval": asdict(self.retrieval),
            "llm": {k: v for k, v in asdict(self.llm).items() if "key" not in k},
            "agent": asdict(self.agent),
        }
        return out


def get_settings() -> Settings:
    return Settings()
