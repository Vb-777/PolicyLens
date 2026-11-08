"""The evaluation set, and the dev/test split that keeps thresholds honest.

The system has one tunable decision threshold — how much lexical evidence is needed
before it answers rather than abstains. Choosing it by looking at the whole
evaluation set would report a number that cannot be reproduced on unseen questions.
So the set is split deterministically by question id: the threshold is swept on
``dev`` and every headline figure is reported on ``test``.

The split is a hash of the question id, not a shuffle with a seed, so it is stable
when questions are added to the corpus generator. It is also **stratified**: an
unstratified hash put only one unanswerable question in dev, which made the
abstention threshold selected there meaningless and actively harmful on test. The
stratum is mixed into the hash so answerable and unanswerable questions are split in
the same proportion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["dev", "test", "all"]

DEV_FRACTION = 0.30


@dataclass
class EvalQuestion:
    id: str
    question: str
    reference_answer: str
    kind: str
    answerable: bool
    gold_clauses: list[str] = field(default_factory=list)
    gold_policies: list[str] = field(default_factory=list)
    answer_keys: list[str] = field(default_factory=list)
    unanswerable_reason: str = ""

    @property
    def split(self) -> str:
        return assign_split(self.id, stratum="answerable" if self.answerable else "unanswerable")

    @property
    def is_multi_clause(self) -> bool:
        return len(self.gold_clauses) > 1


def assign_split(question_id: str, stratum: str = "") -> str:
    """Deterministic dev/test assignment, stratified by ``stratum``.

    Salting the hash with the stratum makes the two populations split independently,
    so each lands in dev at roughly ``DEV_FRACTION`` instead of by luck.
    """
    digest = hashlib.sha256(f"{stratum}:{question_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < DEV_FRACTION else "test"


def load_questions(path: Path, split: Split = "all", limit: int = 0) -> list[EvalQuestion]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation set not found at {path}. Run `python scripts/generate_corpus.py`."
        )
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        EvalQuestion(
            id=row["id"],
            question=row["question"],
            reference_answer=row["reference_answer"],
            kind=row["kind"],
            answerable=bool(row["answerable"]),
            gold_clauses=row.get("gold_clauses", []),
            gold_policies=row.get("gold_policies", []),
            answer_keys=row.get("answer_keys", []),
            unanswerable_reason=row.get("unanswerable_reason", ""),
        )
        for row in rows
    ]
    if split != "all":
        questions = [q for q in questions if q.split == split]
    if limit:
        questions = questions[:limit]
    return questions


def split_summary(questions: list[EvalQuestion]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for q in questions:
        by_split[q.split] = by_split.get(q.split, 0) + 1
        by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
    return {
        "n_questions": len(questions),
        "answerable": sum(1 for q in questions if q.answerable),
        "unanswerable": sum(1 for q in questions if not q.answerable),
        "multi_clause": sum(1 for q in questions if q.is_multi_clause),
        "by_split": by_split,
        "by_kind": by_kind,
    }
