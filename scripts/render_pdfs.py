#!/usr/bin/env python3
"""Render the plain-text corpus into policy-wording PDFs.

The text corpus is the editable source of truth; this produces the artefact the
pipeline is actually meant to consume. Insurers publish policy wordings as PDFs, so
ingesting one is the realistic path and the text files are the convenience.

Fidelity of extraction is the whole point, so the layout is chosen for it rather than
for looks. Clause numbering and section headings carry the document's structure, and
the chunker recovers that structure from line starts; a layout that reflowed a heading
into the end of the preceding paragraph would quietly destroy it. Every heading is
therefore its own flowable with space around it, and body paragraphs are kept as single
blocks so that extracted text breaks where the source breaks.

Run ``python scripts/render_pdfs.py --verify`` to assert that a PDF round-trip yields
the same clauses as the text source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RULE = re.compile(r"^=+$")
SECTION = re.compile(r"^SECTION\s+\d+\.")
CLAUSE = re.compile(r"^\d+\.\d+\s+\S")

BODY = ParagraphStyle(
    "body",
    fontName="Times-Roman",
    fontSize=9.5,
    leading=13,
    alignment=TA_JUSTIFY,
    spaceAfter=5,
)
CLAUSE_HEAD = ParagraphStyle(
    "clause", parent=BODY, fontName="Times-Bold", fontSize=10, spaceBefore=7, spaceAfter=3
)
SECTION_HEAD = ParagraphStyle(
    "section", parent=BODY, fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=13, spaceAfter=6
)
TITLE = ParagraphStyle(
    "title", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=18, alignment=1
)
SUBTITLE = ParagraphStyle(
    "subtitle", parent=BODY, fontName="Helvetica", fontSize=9, leading=12, alignment=1
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class PolicyDoc(BaseDocTemplate):
    """A document template that stamps the UIN and page numbers on every page.

    A real policy wording is footed on every page with its identifier, and carrying
    that through means a chunk's provenance survives even when a reader is looking at
    a single printed page.
    """

    def __init__(self, path: Path, footer: str) -> None:
        super().__init__(
            str(path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=footer,
        )
        self.footer = footer
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates([PageTemplate(id="policy", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillGray(0.35)
        canvas.drawString(20 * mm, 10 * mm, self.footer)
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.setStrokeGray(0.75)
        canvas.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
        canvas.restoreState()


def build_flowables(text: str) -> list:
    """Turn policy text into flowables, preserving which lines are structural."""
    lines = text.splitlines()
    flowables: list = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            flowables.append(Paragraph(_escape(" ".join(paragraph)), BODY))
            paragraph.clear()

    # The first four lines are the cover block: insurer, product, identifiers, type.
    header, body_start = [], 0
    for i, line in enumerate(lines):
        if RULE.match(line.strip()):
            body_start = i + 1
            break
        if line.strip():
            header.append(line.strip())

    if header:
        flowables.append(Paragraph(_escape(header[0]), SUBTITLE))
        if len(header) > 1:
            flowables.append(Spacer(1, 3))
            flowables.append(Paragraph(_escape(header[1]), TITLE))
        for extra in header[2:]:
            flowables.append(Paragraph(_escape(extra), SUBTITLE))
        flowables.append(Spacer(1, 10))

    for raw in lines[body_start:]:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or RULE.match(stripped):
            flush()
            continue
        if SECTION.match(stripped):
            flush()
            flowables.append(Paragraph(_escape(stripped), SECTION_HEAD))
            continue
        if CLAUSE.match(stripped):
            flush()
            flowables.append(Paragraph(_escape(stripped), CLAUSE_HEAD))
            continue
        paragraph.append(stripped)
    flush()
    return flowables


def render(corpus_dir: Path, out_dir: Path) -> list[Path]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    by_file = {p["filename"]: p for p in manifest["policies"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for source in sorted(corpus_dir.glob("*.txt")):
        meta = by_file.get(source.name, {})
        footer = " | ".join(
            part
            for part in (
                meta.get("insurer", ""),
                meta.get("product_name", source.stem),
                f"UIN: {meta['uin']}" if meta.get("uin") else "",
            )
            if part
        )
        target = out_dir / f"{source.stem}.pdf"
        PolicyDoc(target, footer).build(build_flowables(source.read_text()))
        written.append(target)
        print(f"  wrote {target.relative_to(ROOT)} ({target.stat().st_size // 1024} KB)")

    # Each corpus directory describes itself. Without a manifest beside the PDFs the
    # loader falls back to inferring metadata from the cover page, which recovers the
    # insurer and UIN but not the exact product name, so the two corpora would
    # disagree about policy display names for no good reason.
    pdf_manifest = {
        "note": "Generated by scripts/render_pdfs.py from the text corpus. Do not edit by hand.",
        "policies": [
            {**entry, "filename": f"{Path(entry['filename']).stem}.pdf"}
            for entry in manifest["policies"]
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(pdf_manifest, indent=2) + "\n")
    print(f"  wrote {(out_dir / 'manifest.json').relative_to(ROOT)}")
    return written


def verify(corpus_dir: Path, pdf_dir: Path) -> int:
    """Assert the PDF parses into the same clauses as its text source."""
    from policylens.ingest.loaders import parse_structured, read_document

    failures = 0
    for source in sorted(corpus_dir.glob("*.txt")):
        pdf = pdf_dir / f"{source.stem}.pdf"
        if not pdf.exists():
            print(f"  MISSING {pdf.name}")
            failures += 1
            continue
        from_text = parse_structured(source.read_text())
        from_pdf = parse_structured(read_document(pdf))
        text_ids = [c.clause_number for c in from_text]
        pdf_ids = [c.clause_number for c in from_pdf]
        if text_ids == pdf_ids:
            print(f"  OK   {source.stem}: {len(text_ids)} clauses match")
        else:
            failures += 1
            missing = sorted(set(text_ids) - set(pdf_ids))
            extra = sorted(set(pdf_ids) - set(text_ids))
            print(
                f"  FAIL {source.stem}: text={len(text_ids)} pdf={len(pdf_ids)} "
                f"missing={missing[:6]} extra={extra[:6]}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus_pdf")
    parser.add_argument("--verify", action="store_true", help="Check the PDF round-trip.")
    args = parser.parse_args(argv)

    print(f"rendering {args.corpus} -> {args.out}")
    render(args.corpus, args.out)
    if args.verify:
        print("verifying PDF round-trip:")
        failures = verify(args.corpus, args.out)
        if failures:
            print(f"{failures} document(s) did not round-trip")
            return 1
        print("all documents round-trip cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
