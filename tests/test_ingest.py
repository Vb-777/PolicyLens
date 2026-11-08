"""Parsing and chunking: the stage where a mistake silently corrupts every citation."""

from __future__ import annotations

from policylens.config import ChunkConfig
from policylens.ingest.chunker import chunk_clauses, split_sentences
from policylens.ingest.loaders import ParsedClause, parse_fallback, parse_structured
from policylens.schemas import PolicyRecord

SAMPLE = """ACME GENERAL INSURANCE
SAMPLE POLICY
Policy Wording | UIN: TEST123 | Version 1.0
Product Type: Health Indemnity
========================================

SECTION 4. WAITING PERIODS

4.1 Initial Waiting Period
No benefit is payable for Illness contracted within the first thirty (30) days.

4.2 Pre-Existing Disease Waiting Period
Expenses related to a Pre-Existing Disease are excluded until the expiry of
thirty-six (36) months of continuous coverage.

SECTION 5. EXCLUSIONS

5.1 Cosmetic Surgery
Cosmetic surgery is excluded unless medically necessary.
"""


def test_parses_sections_and_clauses():
    clauses = parse_structured(SAMPLE)
    assert [c.clause_number for c in clauses] == ["4.1", "4.2", "5.1"]
    assert clauses[1].clause_title == "Pre-Existing Disease Waiting Period"
    assert clauses[1].section_title == "Waiting Periods"
    assert "thirty-six (36) months" in clauses[1].text


def test_clause_body_is_joined_into_one_line():
    clauses = parse_structured(SAMPLE)
    assert "\n" not in clauses[1].text


def test_body_text_starting_with_a_number_is_not_read_as_a_heading():
    text = SAMPLE + "\n5.2 Reference\nSee 4.2 above, which sets out the applicable waiting period in full detail for the avoidance of doubt.\n"
    clauses = parse_structured(text)
    numbers = [c.clause_number for c in clauses]
    assert numbers == ["4.1", "4.2", "5.1", "5.2"]
    assert "See 4.2 above" in clauses[-1].text


def test_fallback_parser_used_for_unstructured_text():
    clauses = parse_fallback("First paragraph of text.\n\nSecond paragraph of text.")
    assert len(clauses) >= 1
    assert all(c.clause_number.startswith("P") for c in clauses)


def _record() -> PolicyRecord:
    return PolicyRecord(
        policy_id="TEST",
        insurer="Acme General",
        product_name="Sample Policy",
        product_type="Health Indemnity",
        display="Sample Policy",
        source_file="TEST.txt",
    )


def test_short_clause_becomes_one_chunk_with_clause_identity():
    clauses = parse_structured(SAMPLE)
    chunks = chunk_clauses(_record(), clauses, ChunkConfig())
    assert len(chunks) == 3
    assert chunks[1].clause_id == "TEST#4.2"
    assert chunks[1].meta.citation_label().startswith("Sample Policy, Clause 4.2")


def test_long_clause_splits_but_keeps_clause_identity():
    long_text = " ".join(
        f"This is sentence number {i} of a very long clause about coverage limits."
        for i in range(60)
    )
    clause = ParsedClause("9", "Long", "9.1", "Long Clause", long_text)
    chunks = chunk_clauses(_record(), [clause], ChunkConfig(max_chars=400, overlap_chars=80))
    assert len(chunks) > 1
    # Every part still resolves to the same citable clause.
    assert {c.clause_id for c in chunks} == {"TEST#9.1"}
    assert [c.meta.part for c in chunks] == list(range(len(chunks)))
    assert all(c.meta.n_parts == len(chunks) for c in chunks)


def test_chunk_embedding_text_carries_the_heading_path():
    clauses = parse_structured(SAMPLE)
    chunks = chunk_clauses(_record(), clauses, ChunkConfig())
    embedded = chunks[1].for_embedding()
    assert "Waiting Periods" in embedded
    assert "Pre-Existing Disease Waiting Period" in embedded
    assert "Sample Policy" in embedded


def test_sentence_splitter_keeps_parenthesised_numerals_intact():
    sentences = split_sentences(
        "A waiting period of thirty-six (36) months applies. No benefit is payable."
    )
    assert len(sentences) == 2
    assert "(36)" in sentences[0]
