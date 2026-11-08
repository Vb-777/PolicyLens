"""Groundedness scoring, including the cases where it is deliberately strict."""

from __future__ import annotations

from policylens.grounding import (
    citation_validity,
    extract_numbers,
    score_groundedness,
    split_claims,
    strip_citations,
)
from policylens.schemas import Answer, Chunk, ChunkMeta, Citation, ScoredChunk


def _context(text: str, clause_number: str = "4.2") -> ScoredChunk:
    meta = ChunkMeta(
        policy_id="TEST",
        policy_display="Test Policy",
        section_number="4",
        section_title="Waiting Periods",
        clause_number=clause_number,
        clause_title="Pre-Existing Disease Waiting Period",
    )
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"TEST#{clause_number}", text=text, meta=meta), score=1.0
    )


CLAUSE = (
    "Expenses related to the treatment of a Pre-Existing Disease are excluded until "
    "the expiry of thirty-six (36) months of continuous coverage after the date of "
    "inception of the first Policy with the Company."
)


def test_strip_citations_and_number_extraction():
    assert strip_citations("The period is 36 months. [1]") == "The period is 36 months."
    assert "36" in extract_numbers("thirty-six (36) months")
    # Spelled-out numerals fold to digits so an answer may use either form.
    assert "24" in extract_numbers("a twenty-four month wait")


def test_supported_claim_scores_one():
    answer = "The pre-existing disease waiting period is thirty-six (36) months of continuous coverage. [1]"
    report = score_groundedness(answer, [_context(CLAUSE)])
    assert report.score == 1.0
    assert not report.hallucinated


def test_invented_figure_is_caught():
    answer = "The pre-existing disease waiting period is twelve (12) months of continuous coverage. [1]"
    report = score_groundedness(answer, [_context(CLAUSE)])
    assert report.score < 1.0
    assert "12" in report.unsupported_numbers
    assert report.numeric_fidelity < 1.0
    assert report.hallucinated


def test_claim_unrelated_to_context_is_unsupported():
    answer = "Road ambulance charges are reimbursed up to Rs. 5,000 per hospitalisation event. [1]"
    report = score_groundedness(answer, [_context(CLAUSE)])
    assert report.score == 0.0


def test_a_claim_may_not_be_stitched_from_two_contexts():
    # Each half appears in a different clause; neither clause supports the whole.
    answer = "Cataract has a waiting period of thirty-six (36) months and a limit of Rs. 40,000. [1]"
    contexts = [
        _context("Cataract is subject to a waiting period of twenty-four (24) months.", "4.3"),
        _context("Cataract expenses are payable up to Rs. 40,000 per eye.", "5.3"),
    ]
    report = score_groundedness(answer, contexts)
    assert report.score < 1.0


def test_framing_sentences_do_not_count_as_claims():
    claims = split_claims("According to the policy. The waiting period is 36 months of coverage.")
    assert len(claims) == 1


def test_empty_answer_has_no_claims_to_score():
    report = score_groundedness("", [_context(CLAUSE)])
    assert report.total == 0
    assert report.score == 1.0
    assert not report.hallucinated


def test_known_blind_spot_negation_is_not_detected():
    """Documented limitation, pinned by a test so a future change is deliberate.

    A lexical overlap scorer cannot see that this reverses the clause's meaning.
    The LLM judge exists to cover this case.
    """
    answer = "The pre-existing disease waiting period is not thirty-six (36) months. [1]"
    report = score_groundedness(answer, [_context(CLAUSE)])
    assert report.score == 1.0


def test_citation_validity_flags_a_citation_to_an_unretrieved_clause():
    context = _context(CLAUSE)
    answer = Answer(
        question="q",
        answer="The waiting period is 36 months. [1]",
        citations=[
            Citation(
                marker="[1]",
                clause_id="OTHER#9.9",
                policy_id="OTHER",
                clause_number="9.9",
                label="Other",
            )
        ],
    )
    result = citation_validity(answer, [context])
    assert result["citation_precision"] == 0.0


def test_citation_coverage_detects_an_uncited_factual_sentence():
    context = _context(CLAUSE)
    answer = Answer(
        question="q",
        answer="The waiting period is 36 months. [1] It also applies to complications.",
        citations=[
            Citation(
                marker="[1]",
                clause_id="TEST#4.2",
                policy_id="TEST",
                clause_number="4.2",
                label="Test Policy, Clause 4.2",
            )
        ],
    )
    result = citation_validity(answer, [context])
    assert result["citation_precision"] == 1.0
    assert result["citation_coverage"] == 0.5
    assert result["uncited_sentences"] == 1
