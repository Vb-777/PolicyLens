"""Load policy documents from disk and parse them into clauses.

Two paths exist deliberately:

* ``parse_structured`` understands the numbered clause layout that Indian policy
  wordings almost always use (``SECTION 4. WAITING PERIODS`` / ``4.2 Pre-Existing
  Disease``). When it fires, citations resolve to a real clause number, which is what
  makes an answer checkable by a human.
* ``parse_fallback`` handles anything else, including scanned or badly extracted PDFs.
  It degrades to paragraph blocks with synthetic clause numbers rather than failing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..schemas import PolicyRecord

SECTION_RE = re.compile(r"^SECTION\s+([0-9IVXA-Z]+)\.\s+(.+?)\s*$")
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(\S.*?)\s*$")
HEADER_UIN_RE = re.compile(r"UIN:\s*(\S+)")
HEADER_VERSION_RE = re.compile(r"Version\s+([0-9.]+)")
PRODUCT_TYPE_RE = re.compile(r"^Product Type:\s*(.+?)\s*$", re.MULTILINE)
# "Page 4", "Page 4 of 11", "- 4 -", or a line that is nothing but a number.
PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?|[-–—\s]*\d+[-–—\s]*)$", re.IGNORECASE
)


class ParsedClause:
    __slots__ = ("section_number", "section_title", "clause_number", "clause_title", "text")

    def __init__(
        self,
        section_number: str,
        section_title: str,
        clause_number: str,
        clause_title: str,
        text: str,
    ) -> None:
        self.section_number = section_number
        self.section_title = section_title
        self.clause_number = clause_number
        self.clause_title = clause_title
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParsedClause({self.clause_number!r}, {self.clause_title!r}, {len(self.text)} chars)"


def parse_structured(text: str) -> list[ParsedClause]:
    """Parse numbered SECTION / clause layout. Returns [] if the layout is absent."""
    clauses: list[ParsedClause] = []
    section_number = ""
    section_title = ""
    current: ParsedClause | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
            if current.text:
                clauses.append(current)
        current = None
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        m_sec = SECTION_RE.match(stripped)
        if m_sec:
            flush()
            section_number, section_title = m_sec.group(1), m_sec.group(2).title()
            continue

        m_cl = CLAUSE_RE.match(stripped)
        # A clause heading is a short line: body text that happens to start with a
        # number (e.g. "4.2 of the Act provides...") is much longer than a heading.
        if m_cl and len(stripped) <= 90:
            flush()
            number, title = m_cl.group(1), m_cl.group(2)
            if not section_number:
                section_number = number.split(".")[0]
            current = ParsedClause(section_number, section_title, number, title, "")
            continue

        if current is not None:
            buffer.append(stripped)

    flush()
    return clauses


def parse_fallback(text: str, max_block_chars: int = 1200) -> list[ParsedClause]:
    """Paragraph-block parser for documents without recognisable clause numbering."""
    blocks: list[str] = []
    current: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        candidate = (" ".join(current) + " " + para).strip() if current else para
        if current and len(candidate) > max_block_chars:
            blocks.append(" ".join(current).strip())
            current = [para]
        else:
            current.append(para)
    if current:
        blocks.append(" ".join(current).strip())

    out: list[ParsedClause] = []
    for i, block in enumerate(blocks, start=1):
        title = block[:60].rsplit(" ", 1)[0] + "..." if len(block) > 60 else block
        out.append(ParsedClause("1", "Document", f"P{i}", title, block))
    return out


def read_pdf(path: Path) -> str:
    """Extract text from a PDF. Requires pypdf, which is an optional dependency."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading PDFs requires pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(strip_page_furniture(pages))


def strip_page_furniture(pages: list[str]) -> list[str]:
    """Drop running headers, footers and page numbers from extracted pages.

    Every real policy wording foots each page with the insurer, the product and the
    UIN, and extraction interleaves that furniture with the body text. Left in, it
    does direct damage rather than merely adding noise: the footer of page 1 is
    extracted *before* the cover block, so metadata inference reads the footer as the
    insurer name and the page number as the product name, and the repeated line lands
    in the middle of clauses, corrupting the text a citation quotes.

    A line is furniture when it is a page-number line, or when it recurs on most pages
    at the top or bottom. Recurrence is the reliable signal: body text does not repeat
    verbatim across pages, and boilerplate does. Documents of one page have no
    recurrence to measure, so only the page-number rule applies there.
    """
    if not pages:
        return pages

    split = [p.splitlines() for p in pages]
    repeated: set[str] = set()
    if len(split) > 1:
        # Count only the edges of each page; a sentence repeated mid-clause is content.
        edge_counts: dict[str, int] = {}
        for lines in split:
            edges = {ln.strip() for ln in lines[:2] + lines[-3:] if ln.strip()}
            for line in edges:
                edge_counts[line] = edge_counts.get(line, 0) + 1
        threshold = max(2, int(len(split) * 0.6))
        repeated = {line for line, n in edge_counts.items() if n >= threshold}

    cleaned: list[str] = []
    for lines in split:
        keep = [
            line
            for line in lines
            if line.strip() not in repeated and not PAGE_NUMBER_RE.match(line.strip())
        ]
        cleaned.append("\n".join(keep))
    return cleaned


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported document type: {path.suffix} ({path})")


def _titlecase(line: str) -> str:
    """Title-case a shouting header line, leaving mixed-case text alone.

    Policy covers are typeset in capitals, so the inferred product name has to be
    recased to be readable. ``str.title`` is wrong for the job: it capitalises after
    every non-letter, turning "HOUSEHOLDER'S" into "Householder'S" and lower-casing
    deliberate internal capitals, so "DriveShield" survives only because the line it
    sits on is not fully upper-case. Recasing only all-caps lines, and only at word
    starts, keeps both intact.
    """
    if not line.isupper():
        return line
    # Treat a possessive as one word so the "s" in "householder's" is not capitalised.
    return re.sub(
        r"[a-z]+(?:['’][a-z]+)*",
        lambda m: m.group(0)[0].upper() + m.group(0)[1:],
        line.lower(),
    )


def _derive_record(path: Path, text: str, manifest_by_file: dict[str, dict]) -> PolicyRecord:
    # Match on the stem as well as the full name: the manifest records the text source
    # ("HLT-AAROGYA.txt") while the pipeline may be pointed at the rendered PDF of the
    # same policy, and an exact-filename lookup would silently fall through to metadata
    # inference for a document the manifest describes precisely.
    entry = manifest_by_file.get(path.name) or manifest_by_file.get(path.stem)
    if entry:
        return PolicyRecord(
            policy_id=entry["policy_id"],
            insurer=entry.get("insurer", ""),
            product_name=entry.get("product_name", ""),
            product_type=entry.get("product_type", ""),
            uin=entry.get("uin", ""),
            version=entry.get("version", ""),
            display=entry.get("display") or entry.get("product_name", ""),
            source_file=path.name,
        )

    # No manifest entry: infer what we can from the document header. This is the path a
    # real third-party PDF takes.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    insurer = _titlecase(lines[0]) if lines else path.stem
    product = _titlecase(lines[1]) if len(lines) > 1 else path.stem
    uin_match = HEADER_UIN_RE.search(text[:1200])
    ver_match = HEADER_VERSION_RE.search(text[:1200])
    ptype_match = PRODUCT_TYPE_RE.search(text[:1200])
    return PolicyRecord(
        policy_id=path.stem.upper(),
        insurer=insurer,
        product_name=product,
        product_type=ptype_match.group(1) if ptype_match else "Unknown",
        uin=uin_match.group(1) if uin_match else "",
        version=ver_match.group(1) if ver_match else "",
        display=product,
        source_file=path.name,
    )


def load_documents(corpus_dir: Path) -> list[tuple[PolicyRecord, list[ParsedClause]]]:
    """Load and parse every policy document in ``corpus_dir``.

    Honours ``manifest.json`` when present for accurate policy metadata; otherwise it
    infers metadata from the document itself, so dropping a real PDF into the folder
    works with no extra configuration.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. "
            "Run `python scripts/generate_corpus.py` first."
        )

    manifest_by_file: dict[str, dict] = {}
    manifest_path = corpus_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_by_file = {}
        for entry in manifest.get("policies", []):
            filename = entry.get("filename", "")
            if filename:
                manifest_by_file[filename] = entry
                manifest_by_file.setdefault(Path(filename).stem, entry)

    out: list[tuple[PolicyRecord, list[ParsedClause]]] = []
    paths = sorted(
        p for p in corpus_dir.iterdir() if p.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    for path in paths:
        text = read_document(path)
        if not text.strip():
            continue
        clauses = parse_structured(text)
        if len(clauses) < 2:
            clauses = parse_fallback(text)
        record = _derive_record(path, text, manifest_by_file)
        record.n_clauses = len(clauses)
        out.append((record, clauses))

    if not out:
        raise FileNotFoundError(f"No readable documents found in {corpus_dir}")
    return out
