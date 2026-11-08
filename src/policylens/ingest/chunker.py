"""Clause-aware chunking.

The unit of retrieval is the clause, not a fixed window. In policy documents a clause
is already the semantic unit a human would cite, and splitting across clause boundaries
produces chunks that mix an exclusion with an unrelated benefit, which is exactly the
failure mode that causes a grounded-looking but wrong answer.

Long clauses are split on sentence boundaries with overlap, and every part keeps the
parent clause's identity so a citation still points at "Clause 4.2".
"""

from __future__ import annotations

import re

from ..config import ChunkConfig
from ..schemas import Chunk, ChunkMeta, PolicyRecord
from .loaders import ParsedClause

_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedy sentence packing with character overlap carried between windows."""
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for sent in sentences:
        if current and size + len(sent) + 1 > max_chars:
            windows.append(" ".join(current))
            if overlap_chars > 0:
                carry: list[str] = []
                carry_len = 0
                for prev in reversed(current):
                    if carry_len + len(prev) > overlap_chars:
                        break
                    carry.insert(0, prev)
                    carry_len += len(prev) + 1
                current = carry
                size = carry_len
            else:
                current, size = [], 0
        current.append(sent)
        size += len(sent) + 1

    if current:
        windows.append(" ".join(current))
    return windows


def chunk_clauses(
    record: PolicyRecord,
    clauses: list[ParsedClause],
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    cfg = config or ChunkConfig()
    chunks: list[Chunk] = []

    for clause in clauses:
        text = clause.text.strip()
        if len(text) < cfg.min_chars:
            # Too short to stand alone: fold the heading in so it is still retrievable
            # rather than dropping content silently.
            text = f"{clause.clause_title}. {text}".strip()
        if not text:
            continue

        if len(text) <= cfg.max_chars:
            windows = [text]
        else:
            windows = _pack(split_sentences(text), cfg.max_chars, cfg.overlap_chars)

        n_parts = len(windows)
        for i, window in enumerate(windows):
            meta = ChunkMeta(
                policy_id=record.policy_id,
                policy_display=record.display or record.product_name,
                product_type=record.product_type,
                insurer=record.insurer,
                section_number=clause.section_number,
                section_title=clause.section_title,
                clause_number=clause.clause_number,
                clause_title=clause.clause_title,
                part=i,
                n_parts=n_parts,
                source_file=record.source_file,
            )
            suffix = f"::p{i}" if n_parts > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{record.policy_id}#{clause.clause_number}{suffix}",
                    text=window,
                    meta=meta,
                )
            )

    return chunks


def chunk_corpus(
    documents: list[tuple[PolicyRecord, list[ParsedClause]]],
    config: ChunkConfig | None = None,
) -> tuple[list[Chunk], list[PolicyRecord]]:
    all_chunks: list[Chunk] = []
    records: list[PolicyRecord] = []
    for record, clauses in documents:
        all_chunks.extend(chunk_clauses(record, clauses, config))
        records.append(record)
    return all_chunks, records
