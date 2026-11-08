"""Metric definitions, checked against hand-worked examples.

A metric that is wrong in the same direction every run looks like a stable result,
so these are verified against values computed by hand rather than against the
implementation's own output.
"""

from __future__ import annotations

import math

from policylens.eval.metrics import (
    AbstentionScores,
    dedupe_preserving_order,
    key_fact_recall,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_collapses_chunk_parts_but_keeps_order():
    assert dedupe_preserving_order(["A#1", "A#1", "B#2", "A#1", "C#3"]) == ["A#1", "B#2", "C#3"]


def test_recall_with_two_gold_clauses():
    retrieved = ["A", "X", "B", "Y"]
    assert recall_at_k(retrieved, ["A", "B"], 1) == 0.5
    assert recall_at_k(retrieved, ["A", "B"], 3) == 1.0


def test_precision_counts_only_the_window():
    assert precision_at_k(["A", "X", "Y"], ["A"], 1) == 1.0
    assert precision_at_k(["A", "X", "Y"], ["A"], 3) == 1 / 3


def test_reciprocal_rank_is_the_first_hit():
    assert reciprocal_rank(["X", "Y", "A"], ["A"]) == 1 / 3
    assert reciprocal_rank(["X", "Y"], ["A"]) == 0.0


def test_ndcg_is_one_when_all_gold_leads_the_ranking():
    assert ndcg_at_k(["A", "B", "X"], ["A", "B"], 5) == 1.0


def test_ndcg_ideal_accounts_for_more_gold_than_k():
    # Two gold clauses but k=1: retrieving one of them is a perfect result at k=1.
    assert ndcg_at_k(["A", "B"], ["A", "B"], 1) == 1.0


def test_ndcg_penalises_a_later_hit():
    good = ndcg_at_k(["A", "X", "Y"], ["A"], 3)
    worse = ndcg_at_k(["X", "Y", "A"], ["A"], 3)
    assert good == 1.0
    assert 0 < worse < good


def test_recall_is_nan_when_there_is_no_gold():
    assert math.isnan(recall_at_k(["A"], [], 5))


# -- key fact recall -------------------------------------------------------------


def test_numeric_keys_are_all_required():
    answer = "Rs. 50,000 is payable for a normal delivery."
    assert key_fact_recall(answer, ["50,000", "75,000"]) == 0.5


def test_numeric_key_matches_despite_separators():
    assert key_fact_recall("The limit is Rs. 750000.", ["7,50,000"]) == 1.0


def test_spelled_out_numbers_match_digits():
    answer = "The waiting period is thirty-six months."
    assert key_fact_recall(answer, ["36 months"]) == 1.0


def test_wording_keys_are_alternatives_not_all_required():
    answer = "Attendant charges are not payable under this policy."
    # Only one of the two synonyms appears; the answer is still correct.
    assert key_fact_recall(answer, ["not payable", "excluded"]) == 1.0


def test_polarity_keys_are_ignored():
    answer = "Cosmetic surgery is excluded unless reconstruction is required."
    assert key_fact_recall(answer, ["yes", "no", "excluded"]) == 1.0


def test_mixed_keys_require_all_numbers_and_one_wording():
    answer = "Cover runs for twelve (12) months up to ten percent (10%) of the sum insured."
    assert key_fact_recall(answer, ["12 months", "10%", "yes"]) == 1.0


def test_key_recall_is_nan_when_only_polarity_keys_are_given():
    assert math.isnan(key_fact_recall("anything", ["yes"]))


# -- aggregation -----------------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile(values, 100) == 10


def test_abstention_scores():
    scores = AbstentionScores(true_abstain=6, false_abstain=2, true_answer=65, false_answer=10)
    assert scores.precision == 6 / 8
    assert scores.recall == 6 / 16
    assert round(scores.accuracy, 4) == round(71 / 83, 4)
