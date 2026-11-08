"""Metric definitions.

Separated from the runner so each metric can be unit-tested against hand-worked
examples. Several of these are easy to define subtly wrongly — recall@k when a
question has several gold clauses, nDCG when it has one — and a silently wrong metric
is worse than no metric.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from ..grounding import extract_numbers
from ..store.bm25 import tokenize

# --------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Collapse chunk parts back to clauses without disturbing rank order.

    A long clause is indexed as several chunks. Two chunks of the same clause
    occupying ranks 1 and 2 is one retrieved clause, not two, and counting it as two
    would inflate precision@k.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold or k == 0:
        return float("nan")
    top = retrieved[:k]
    if not top:
        return 0.0
    return len([r for r in top if r in set(gold)]) / len(top)


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    The ideal DCG uses ``min(len(gold), k)`` relevant items, so a question with two
    gold clauses is not penalised for k=1 being unable to contain both.
    """
    if not gold:
        return float("nan")
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in gold_set
    )
    ideal_n = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalScores:
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    first_gold_rank: int | None = None


def score_retrieval(
    retrieved_clause_ids: Sequence[str],
    gold: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalScores:
    retrieved = dedupe_preserving_order(retrieved_clause_ids)
    gold_set = set(gold)
    rank = next(
        (i for i, item in enumerate(retrieved, start=1) if item in gold_set), None
    )
    return RetrievalScores(
        recall={k: recall_at_k(retrieved, gold, k) for k in k_values},
        precision={k: precision_at_k(retrieved, gold, k) for k in k_values},
        hit={k: hit_at_k(retrieved, gold, k) for k in k_values},
        ndcg={k: ndcg_at_k(retrieved, gold, k) for k in k_values},
        mrr=reciprocal_rank(retrieved, gold),
        first_gold_rank=rank,
    )


# --------------------------------------------------------------------------------
# Answer correctness
# --------------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    lowered = text.lower().replace(",", "")
    return _WS.sub(" ", lowered).strip()


# "yes" and "no" describe the shape of an answer, not a fact in it. A quoted
# exclusion clause answers "is X covered?" correctly without containing the word
# "no", so requiring the word would penalise a correct answer for its grammar.
_POLARITY_KEYS = {"yes", "no"}


def _matches(key: str, answer: str, answer_numbers: set[str]) -> bool:
    if normalise(key) in normalise(answer):
        return True
    # A numeric key may be written differently: "7,50,000" / "750000", or
    # "thirty-six" / "36".
    key_numbers = extract_numbers(key)
    return bool(key_numbers) and key_numbers <= answer_numbers


def key_fact_recall(answer: str, answer_keys: Sequence[str]) -> float:
    """How much of the reference answer's decisive content the answer reproduces.

    Keys are not all alike, and scoring them uniformly gave misleading results:

    * **Numeric keys are conjunctive.** "Rs. 50,000 for normal delivery and Rs. 75,000
      for caesarean" needs both figures; producing one is not a correct answer.
    * **Wording keys are alternatives.** For "are attendant charges payable?" the keys
      "not payable" and "excluded" are two ways of saying the same thing, and an
      answer is correct if it says either.
    * **Polarity keys are dropped**, per the note above.

    So the score is: every numeric key must appear, and at least one wording key must
    appear. Treating all keys as conjunctive — the first version of this metric —
    marked correct answers "partial" purely because they used one synonym rather
    than three.
    """
    keys = [k for k in answer_keys if normalise(k) not in _POLARITY_KEYS]
    if not keys:
        return float("nan")

    answer_numbers = extract_numbers(answer)
    numeric = [k for k in keys if extract_numbers(k)]
    wording = [k for k in keys if not extract_numbers(k)]

    required = len(numeric) + (1 if wording else 0)
    if required == 0:
        return float("nan")

    satisfied = sum(1 for k in numeric if _matches(k, answer, answer_numbers))
    if wording and any(_matches(k, answer, answer_numbers) for k in wording):
        satisfied += 1
    return satisfied / required


def token_f1(answer: str, reference: str) -> float:
    """Unigram F1 against the reference, for questions with no explicit keys."""
    a = set(tokenize(answer))
    b = set(tokenize(reference))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def classify_correctness(key_recall: float, f1: float) -> str:
    if not math.isnan(key_recall):
        if key_recall >= 0.999:
            return "correct"
        if key_recall > 0.0:
            return "partial"
        return "incorrect"
    if f1 >= 0.55:
        return "correct"
    if f1 >= 0.30:
        return "partial"
    return "incorrect"


# --------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------


def safe_mean(values: Iterable[float]) -> float:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    return mean(cleaned) if cleaned else float("nan")


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Explicit so the reported p95 is unambiguous."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean_ms": round(mean(values), 2),
        "p50_ms": round(percentile(values, 50), 2),
        "p90_ms": round(percentile(values, 90), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "max_ms": round(max(values), 2),
        "n": len(values),
    }


@dataclass
class AbstentionScores:
    """How well the system knows what it does not know.

    Positive class = "should abstain". Reported as precision/recall rather than a
    single accuracy because the two failure modes have very different costs: wrongly
    answering an unanswerable question invents policy terms, while wrongly abstaining
    is merely unhelpful.
    """

    true_abstain: int = 0
    false_abstain: int = 0
    true_answer: int = 0
    false_answer: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_abstain + self.false_abstain
        return self.true_abstain / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_abstain + self.false_answer
        return self.true_abstain / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        total = self.true_abstain + self.false_abstain + self.true_answer + self.false_answer
        return (self.true_abstain + self.true_answer) / total if total else float("nan")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "abstention_precision": round(self.precision, 4) if not math.isnan(self.precision) else None,
            "abstention_recall": round(self.recall, 4) if not math.isnan(self.recall) else None,
            "abstention_f1": round(self.f1, 4) if not math.isnan(self.f1) else None,
            "routing_accuracy": round(self.accuracy, 4) if not math.isnan(self.accuracy) else None,
            "correctly_abstained": self.true_abstain,
            "missed_abstentions": self.false_answer,
            "over_abstentions": self.false_abstain,
            "correctly_answered": self.true_answer,
        }
