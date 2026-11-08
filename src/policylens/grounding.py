"""Deterministic groundedness scoring.

Used in two places: the agent's self-check node, which can trigger a retrieval retry
before the user ever sees the answer, and the evaluation harness, which reports it as
a metric.

It is a lexical entailment *proxy*, not an NLI model, and the limits are worth stating
plainly because a metric nobody understands is a metric nobody should trust:

* It cannot detect a claim that uses the context's own words to assert the opposite
  ("the waiting period is not 36 months" scores as supported).
* It under-credits a correct paraphrase that shares few tokens with the source.
* It is strict about numbers, which is the right bias here: a figure that does not
  appear in the retrieved clauses is the single most damaging error this system can
  make, and it is caught exactly.

The LLM judge in ``eval/judge.py`` covers the semantic cases this misses. Both are
reported, and where they disagree the disagreement is itself informative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import ScoredChunk
from .store.bm25 import tokenize

_CITATION_RE = re.compile(r"\[\d{1,2}\]")
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# A sentence terminator followed by one or more citation markers, e.g. ". [1] " or "; [2][3] ".
_TRAILING_MARKER_RE = re.compile(r"([.;])\s*((?:\[\d{1,2}\]\s*)+)")

# Sentences that carry no checkable fact should not count against groundedness.
_HEDGE_ONLY = re.compile(
    r"^(based on|according to|the policy states|in summary|note that|however|"
    r"therefore|in short)\b[^.]{0,40}\.?$",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "twelve": "12", "fifteen": "15", "eighteen": "18", "twenty": "20",
    "twenty-four": "24", "twenty-five": "25", "thirty": "30", "thirty-five": "35",
    "thirty-six": "36", "forty": "40", "forty-five": "45", "forty-eight": "48",
    "fifty": "50", "sixty": "60", "seventy-two": "72", "seventy-five": "75",
    "ninety": "90", "hundred": "100", "180": "180",
}


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def extract_numbers(text: str) -> set[str]:
    """Numbers in a comparable form, with spelled-out numerals folded in.

    Policy text writes "thirty-six (36) months"; an answer may say either. Treating
    them as different numbers would produce false hallucination reports.
    """
    found = {n.replace(",", "").rstrip(".0") or n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    lowered = text.lower()
    for word, digits in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            found.add(digits)
    return {n for n in found if n}


def split_sentences_keeping_markers(text: str) -> list[str]:
    """Split into sentences without losing which sentence a citation marker belongs to.

    Answers are written with the marker *after* the full stop ("... is 36 months. [1] It
    also ..."), which is conventional but defeats a naive sentence splitter: the marker
    sits between the terminator and the next capital, so neither the lookbehind nor the
    lookahead matches and two sentences collapse into one. A collapsed sentence contains
    a marker, so every sentence trivially counts as cited and citation coverage reads
    1.0 regardless of the answer — the metric silently stops measuring anything.

    Moving a trailing marker to the inside of the sentence it supports fixes the split
    and keeps the attribution with the claim it belongs to.
    """
    attached = _TRAILING_MARKER_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    return [p.strip() for p in _SENT_SPLIT.split(attached) if p.strip()]


def split_claims(answer: str) -> list[str]:
    """Split an answer into checkable claims, dropping pure framing sentences."""
    body = strip_citations(answer)
    parts = [p.strip() for p in _SENT_SPLIT.split(body) if p.strip()]
    if not parts and body:
        parts = [body]
    return [p for p in parts if len(p) > 12 and not _HEDGE_ONLY.match(p)]


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    best_score: float
    best_context: int | None
    missing_numbers: list[str] = field(default_factory=list)


@dataclass
class GroundednessReport:
    score: float
    supported: int
    total: int
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    numeric_fidelity: float = 1.0
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        """True when any claim is unsupported by the retrieved context."""
        return self.total > 0 and self.supported < self.total

    def to_dict(self) -> dict:
        return {
            "groundedness": round(self.score, 4),
            "supported_claims": self.supported,
            "total_claims": self.total,
            "numeric_fidelity": round(self.numeric_fidelity, 4),
            "unsupported_numbers": self.unsupported_numbers,
            "unsupported_claims": [v.claim for v in self.verdicts if not v.supported],
        }


def _token_recall(claim_tokens: set[str], context_tokens: set[str]) -> float:
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def score_groundedness(
    answer: str,
    contexts: list[ScoredChunk],
    support_threshold: float = 0.60,
) -> GroundednessReport:
    """Fraction of the answer's claims supported by the retrieved clauses.

    A claim counts as supported when some single context both covers enough of its
    content tokens and contains every number it asserts. Requiring one context to do
    both prevents a claim being 'supported' by stitching half of it from one clause
    and half from another.
    """
    claims = split_claims(answer)
    if not claims:
        return GroundednessReport(score=1.0, supported=0, total=0)

    context_data = [
        (set(tokenize(c.chunk.text)), extract_numbers(c.chunk.text)) for c in contexts
    ]

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        claim_numbers = extract_numbers(claim)

        best_score = 0.0
        best_idx: int | None = None
        best_missing: list[str] = sorted(claim_numbers)
        supported = False

        for idx, (ctx_tokens, ctx_numbers) in enumerate(context_data):
            recall = _token_recall(claim_tokens, ctx_tokens)
            missing = sorted(claim_numbers - ctx_numbers)
            ok = recall >= support_threshold and not missing
            if ok and not supported:
                supported = True
            if recall > best_score or (ok and len(missing) < len(best_missing)):
                best_score = max(best_score, recall)
                best_idx = idx
                best_missing = missing

        verdicts.append(
            ClaimVerdict(
                claim=claim,
                supported=supported,
                best_score=round(best_score, 4),
                best_context=best_idx,
                missing_numbers=[] if supported else best_missing,
            )
        )

    supported_count = sum(1 for v in verdicts if v.supported)

    all_answer_numbers = extract_numbers(strip_citations(answer))
    all_context_numbers: set[str] = set()
    for _, nums in context_data:
        all_context_numbers |= nums
    unsupported_numbers = sorted(all_answer_numbers - all_context_numbers)
    numeric_fidelity = (
        1.0
        if not all_answer_numbers
        else 1.0 - len(unsupported_numbers) / len(all_answer_numbers)
    )

    return GroundednessReport(
        score=supported_count / len(claims),
        supported=supported_count,
        total=len(claims),
        verdicts=verdicts,
        numeric_fidelity=numeric_fidelity,
        unsupported_numbers=unsupported_numbers,
    )


def citation_validity(answer, contexts: list[ScoredChunk]) -> dict:
    """Are the answer's citations real, and does every factual sentence carry one?

    ``precision`` counts citations pointing at a clause that was actually retrieved.
    ``coverage`` counts factual sentences carrying at least one marker — the metric
    that catches an answer which is correct but unattributable.
    """
    retrieved_ids = {c.chunk.clause_id for c in contexts}
    cited = answer.citations if hasattr(answer, "citations") else []
    text = answer.answer if hasattr(answer, "answer") else str(answer)

    valid = [c for c in cited if c.clause_id in retrieved_ids]
    precision = (len(valid) / len(cited)) if cited else (1.0 if not cited else 0.0)

    sentences = split_sentences_keeping_markers(text)
    sentences = [s for s in sentences if len(strip_citations(s).strip()) > 12]
    with_marker = [s for s in sentences if _CITATION_RE.search(s)]
    coverage = (len(with_marker) / len(sentences)) if sentences else 1.0

    return {
        "n_citations": len(cited),
        "citation_precision": round(precision, 4),
        "citation_coverage": round(coverage, 4),
        "uncited_sentences": len(sentences) - len(with_marker),
    }
