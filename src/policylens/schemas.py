"""Core data structures shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkMeta(BaseModel):
    """Provenance for a chunk, carried all the way through to the citation shown to a user."""

    policy_id: str
    policy_display: str = ""
    product_type: str = ""
    insurer: str = ""
    section_number: str = ""
    section_title: str = ""
    clause_number: str = ""
    clause_title: str = ""
    part: int = 0
    n_parts: int = 1
    source_file: str = ""

    @property
    def clause_id(self) -> str:
        """Stable identity of the clause this chunk came from, e.g. ``HLT-AAROGYA#4.2``."""
        return f"{self.policy_id}#{self.clause_number}"

    def citation_label(self) -> str:
        """Human-readable citation, e.g. ``Aarogya Suraksha Health Insurance, Clause 4.2``."""
        name = self.policy_display or self.policy_id
        if self.clause_number:
            return f"{name}, Clause {self.clause_number} ({self.clause_title})"
        return name


class Chunk(BaseModel):
    """An indexed unit of text."""

    chunk_id: str
    text: str
    meta: ChunkMeta

    @property
    def clause_id(self) -> str:
        return self.meta.clause_id

    def for_embedding(self) -> str:
        """Text sent to the embedder.

        Prefixing the structural path measurably helps retrieval on policy documents:
        many clauses are lexically similar across insurers and the heading is often the
        only thing that disambiguates them.
        """
        m = self.meta
        head = f"{m.policy_display} | {m.product_type} | Section {m.section_number} {m.section_title} | Clause {m.clause_number} {m.clause_title}"
        return f"{head}\n{self.text}"


class ScoredChunk(BaseModel):
    """A chunk with retrieval scores attached."""

    chunk: Chunk
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    retriever: str = "hybrid"


class Citation(BaseModel):
    """A source reference emitted alongside an answer."""

    marker: str  # "[1]"
    clause_id: str
    policy_id: str
    clause_number: str
    label: str
    quote: str = ""


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ERROR = "error"


class Answer(BaseModel):
    """The output of the pipeline for one question."""

    question: str
    answer: str
    status: AnswerStatus = AnswerStatus.ANSWERED
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    # Observability: every stage records its wall-clock cost, so latency is measurable
    # per stage rather than only end to end.
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    llm_provider: str = ""
    self_check: dict[str, Any] = Field(default_factory=dict)

    @property
    def cited_clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.citations]


class PolicyRecord(BaseModel):
    """Metadata for one indexed policy document."""

    policy_id: str
    insurer: str = ""
    product_name: str = ""
    product_type: str = ""
    uin: str = ""
    version: str = ""
    display: str = ""
    source_file: str = ""
    n_clauses: int = 0
