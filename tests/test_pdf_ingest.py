"""The PDF path has to produce the same clauses as the text source.

Insurers publish policy wordings as PDFs, so the PDF is the realistic input and the
text corpus is a convenience. If the two disagree, every number in the evaluation
report describes a corpus nobody would actually ingest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from policylens.ingest.loaders import (
    PAGE_NUMBER_RE,
    _titlecase,
    load_documents,
    parse_structured,
    read_document,
    strip_page_furniture,
)

ROOT = Path(__file__).resolve().parents[1]
TEXT_CORPUS = ROOT / "data" / "corpus"
PDF_CORPUS = ROOT / "data" / "corpus_pdf"

pdfs_present = pytest.mark.skipif(
    not PDF_CORPUS.exists() or not any(PDF_CORPUS.glob("*.pdf")),
    reason="PDF corpus not rendered; run scripts/render_pdfs.py",
)


def _normalise(text: str) -> str:
    return " ".join(text.split())


@pdfs_present
def test_every_policy_pdf_parses_to_the_same_clauses_as_its_text_source():
    text_docs = {r.policy_id: clauses for r, clauses in load_documents(TEXT_CORPUS)}
    pdf_docs = {r.policy_id: clauses for r, clauses in load_documents(PDF_CORPUS)}

    assert set(text_docs) == set(pdf_docs)

    for policy_id, text_clauses in text_docs.items():
        pdf_clauses = pdf_docs[policy_id]
        assert [c.clause_number for c in text_clauses] == [
            c.clause_number for c in pdf_clauses
        ], f"{policy_id}: clause numbering diverged between text and PDF"
        for from_text, from_pdf in zip(text_clauses, pdf_clauses, strict=True):
            assert _normalise(from_text.text) == _normalise(
                from_pdf.text
            ), f"{policy_id} clause {from_text.clause_number}: text differs"


@pdfs_present
def test_pdf_corpus_carries_the_same_policy_metadata():
    """Metadata drives policy identification, so a mismatch misroutes questions."""
    text_meta = {r.policy_id: r for r, _ in load_documents(TEXT_CORPUS)}
    pdf_meta = {r.policy_id: r for r, _ in load_documents(PDF_CORPUS)}

    for policy_id, expected in text_meta.items():
        actual = pdf_meta[policy_id]
        assert actual.insurer == expected.insurer
        assert actual.uin == expected.uin
        assert actual.display == expected.display
        assert actual.product_type == expected.product_type


@pdfs_present
def test_running_footer_does_not_leak_into_clause_text():
    """The page footer repeats the insurer and UIN on every page.

    Extraction interleaves it with the body, and page one's footer comes out *before*
    the cover block — which is how it ended up being read as the insurer name.
    """
    pdf = next(PDF_CORPUS.glob("HLT-AAROGYA.pdf"))
    clauses = parse_structured(read_document(pdf))
    assert clauses
    for clause in clauses:
        assert "UIN:" not in clause.text
        assert not PAGE_NUMBER_RE.match(clause.text.strip())


def test_strip_page_furniture_removes_lines_repeated_at_page_edges():
    pages = [
        "Acme Insurance | UIN: X1\nSECTION 1. SCOPE\n1.1 Cover\nWe cover the thing.\nPage 1",
        "Acme Insurance | UIN: X1\n1.2 Limits\nThe limit is ten.\nPage 2",
        "Acme Insurance | UIN: X1\n1.3 Excess\nThe excess is two.\nPage 3",
    ]
    cleaned = "\n".join(strip_page_furniture(pages))
    assert "Acme Insurance" not in cleaned
    assert "Page 1" not in cleaned
    # Content survives.
    assert "We cover the thing." in cleaned
    assert "The excess is two." in cleaned


def test_strip_page_furniture_keeps_body_text_that_merely_resembles_furniture():
    """A single-page document has no recurrence, so only page numbers are dropped."""
    pages = ["Acme Insurance | UIN: X1\n1.1 Cover\nWe cover the thing.\nPage 1"]
    cleaned = "\n".join(strip_page_furniture(pages))
    assert "We cover the thing." in cleaned
    assert "Page 1" not in cleaned
    # With nothing to compare against, the header is not assumed to be furniture.
    assert "Acme Insurance" in cleaned


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GRIHA RAKSHA HOUSEHOLDER'S PACKAGE POLICY", "Griha Raksha Householder's Package Policy"),
        ("DriveShield Private Car Package Policy", "DriveShield Private Car Package Policy"),
        ("MERIDIAN GENERAL INSURANCE COMPANY LIMITED", "Meridian General Insurance Company Limited"),
    ],
)
def test_titlecase_handles_apostrophes_and_leaves_mixed_case_alone(raw, expected):
    assert _titlecase(raw) == expected
