"""Knowledge graph extraction and the agent's tools."""

from __future__ import annotations

import pytest

from policylens.agent.tools import settlement_calculator
from policylens.kg.graph import extract_amounts, extract_durations, extract_percentages


def test_duration_extraction_prefers_the_parenthesised_numeral():
    assert extract_durations("a period of thirty-six (36) months") == [(36, "month")]


def test_duration_extraction_falls_back_to_bare_digits():
    assert extract_durations("within 48 hours of admission") == [(48, "hour")]


def test_amount_and_percentage_extraction():
    assert ("INR", 50000) in extract_amounts("payable up to Rs. 50,000 per event")
    assert ("USD", 1500) in extract_amounts("reimbursed up to USD 1,500")
    assert extract_percentages("ten percent (10%) of the Sum Insured") == [10]


def test_rupee_pattern_requires_a_digit():
    # A bare "Rs." with no figure must not produce a zero-value amount.
    assert extract_amounts("payable in Rs. as per schedule") == []


def test_graph_has_the_expected_node_types(index):
    stats = index.kg.stats()
    for node_type in ("Policy", "Clause", "WaitingPeriod", "Exclusion", "Benefit"):
        assert stats["node_types"].get(node_type, 0) > 0


def test_waiting_periods_are_normalised_to_months(index):
    rows = index.kg.waiting_periods("HLT-AAROGYA")
    by_label = {r["label"]: r for r in rows}
    ped = by_label.get("Pre-Existing Disease Waiting Period")
    assert ped is not None
    assert ped["months"] == 36.0


def test_comparison_spans_several_policies(index):
    rows = index.kg.compare_across_policies("maternity")
    policies = {r["policy_id"] for r in rows}
    assert {"HLT-AAROGYA", "HLT-NIROGAM"} <= policies


def test_every_graph_fact_points_at_a_real_clause(index):
    clause_ids = {c.clause_id for c in index.chunks}
    for row in index.kg.compare_across_policies("waiting"):
        assert row["clause_id"] in clause_ids


def test_graph_survives_a_save_load_cycle(index, tmp_path):
    from policylens.kg.graph import PolicyKnowledgeGraph

    index.kg.save(tmp_path)
    reloaded = PolicyKnowledgeGraph.load(tmp_path)
    assert reloaded.stats()["nodes"] == index.kg.stats()["nodes"]
    assert reloaded.waiting_periods("HLT-AAROGYA") == index.kg.waiting_periods("HLT-AAROGYA")


# -- settlement calculator -------------------------------------------------------


def test_calculator_returns_the_claim_when_nothing_applies():
    result = settlement_calculator(claim_amount=100000)
    assert result["payable"] == 100000
    assert result["total_deduction"] == 0


def test_proportionate_deduction_then_deductible_then_copay_in_that_order():
    result = settlement_calculator(
        claim_amount=100000,
        entitled_room_rent=5000,
        actual_room_rent=10000,
        deductible=1000,
        co_payment_percent=10,
    )
    # 100000 * 0.5 = 50000 -> minus 1000 deductible = 49000 -> minus 10% = 44100
    assert result["payable"] == pytest.approx(44100.0)
    assert [s["step"] for s in result["steps"]] == [
        "claimed",
        "proportionate_deduction",
        "deductible",
        "co_payment",
    ]


def test_order_matters_copay_before_deductible_would_differ():
    ours = settlement_calculator(claim_amount=100000, deductible=1000, co_payment_percent=10)
    assert ours["payable"] == pytest.approx(89100.0)
    # Co-payment first would give 90000 - 1000 = 89000, a different answer.
    assert ours["payable"] != pytest.approx(89000.0)


def test_no_proportionate_deduction_when_room_is_within_entitlement():
    result = settlement_calculator(
        claim_amount=50000, entitled_room_rent=5000, actual_room_rent=4000
    )
    assert result["payable"] == 50000


def test_sum_insured_caps_the_payout():
    result = settlement_calculator(claim_amount=800000, sum_insured=500000)
    assert result["payable"] == 500000
    assert result["capped_at_sum_insured"] is True


def test_tool_registry_exposes_callable_schemas(agent):
    names = {spec.name for spec in agent.tools.specs()}
    assert names == {
        "search_clauses",
        "query_knowledge_graph",
        "compare_policies",
        "settlement_calculator",
    }
    for schema in agent.tools.anthropic_schemas():
        assert schema["input_schema"]["type"] == "object"
        assert schema["description"]


def test_unknown_tool_raises(agent):
    with pytest.raises(KeyError):
        agent.tools.get("no_such_tool")


def test_search_tool_returns_scored_chunks(agent):
    hits = agent.tools.call("search_clauses", query="grace period", top_k=3)
    assert 0 < len(hits) <= 3
    assert all(hasattr(h, "chunk") for h in hits)


def test_compare_across_policies_reports_one_row_per_provision(index):
    """A clause typed more than once must not appear as duplicate rows.

    A grace period is both a WaitingPeriod (it has a duration) and a Condition (it
    governs renewal), so the same fact was listed twice with the same figure, which
    reads as a data-quality problem in the one view built for comparison.
    """
    rows = index.kg.compare_across_policies("grace period")
    assert rows

    keys = [(r["policy_id"], r["label"], r["clause_id"]) for r in rows]
    assert len(keys) == len(set(keys)), f"duplicate provision rows: {keys}"

    # The types that described the fact are kept rather than one being discarded.
    merged = [r for r in rows if len(r["types"]) > 1]
    assert merged, "expected at least one provision carrying several node types"
    assert all(r["types"] == sorted(r["types"]) for r in rows)


def test_compare_across_policies_fills_values_from_every_type(index):
    """Merging must not lose a figure held by only one of a clause's types.

    The Exclusion node for a waiting-period clause carries no duration; the
    WaitingPeriod node for the same clause does. Whichever is seen first, the merged
    row has to end up with the months.
    """
    rows = index.kg.compare_across_policies("pre-existing")
    waiting = [r for r in rows if "WaitingPeriod" in r["types"]]
    assert waiting
    for row in waiting:
        assert row["value"] is not None, f"{row['label']} lost its duration when merged"
        assert row["unit"]

    # The two health policies state different waiting periods, which is the whole
    # point of the comparison view.
    periods = {r["policy_id"]: r["months"] for r in waiting if r.get("months")}
    assert len(set(periods.values())) > 1, f"expected differing periods, got {periods}"
