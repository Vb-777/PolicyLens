"""End-to-end agent behaviour: routing, citation integrity, abstention."""

from __future__ import annotations

import pytest

from policylens.schemas import AnswerStatus


def _route(answer) -> str:
    return next(
        (step.get("route", "") for step in answer.trace if step.get("node") == "route"), ""
    )


def _nodes(answer) -> list[str]:
    return [step["node"] for step in answer.trace]


def test_simple_lookup_answers_from_the_right_clause(agent):
    answer = agent.answer(
        "What is the pre-existing disease waiting period under the Aarogya Suraksha policy?"
    )
    assert answer.status is AnswerStatus.ANSWERED
    assert "HLT-AAROGYA#4.2" in answer.cited_clause_ids
    assert "36" in answer.answer


def test_a_question_naming_one_policy_is_not_routed_as_a_comparison(agent):
    """Regression: "Aarogya Suraksha" and "Suraksha Kavach" share a word.

    Matching policy names on any long word made a single-policy question look like a
    comparison, which widened retrieval and degraded the answer.
    """
    answer = agent.answer(
        "What is the pre-existing disease waiting period under the Aarogya Suraksha policy?"
    )
    assert _route(answer) == "lookup"


def test_comparison_is_routed_as_such_and_consults_the_graph(agent):
    answer = agent.answer(
        "Which has the shorter pre-existing disease waiting period, Aarogya Suraksha or Nirogam Plus?"
    )
    assert _route(answer) == "comparative"
    assert "kg_lookup" in _nodes(answer)


def test_comparison_cites_clauses_from_both_policies(agent):
    answer = agent.answer(
        "Which has the shorter pre-existing disease waiting period, Aarogya Suraksha or Nirogam Plus?"
    )
    policies = {c.policy_id for c in answer.citations}
    assert {"HLT-AAROGYA", "HLT-NIROGAM"} <= policies


def test_cross_domain_question_is_declined(agent):
    answer = agent.answer(
        "What is the waiting period for dental implants under the DriveShield motor policy?"
    )
    assert answer.status is AnswerStatus.ABSTAINED


def test_policy_guard_declines_a_provision_the_named_policy_lacks(agent):
    """Other policies describe a no claim bonus, so retrieval returns confident
    but wrong clauses. The guard checks the named policy against what was retrieved."""
    answer = agent.answer(
        "What is the no claim bonus under the Griha Raksha householder's policy?"
    )
    assert answer.status is AnswerStatus.ABSTAINED


def test_every_citation_points_at_a_clause_that_was_retrieved(agent):
    answer = agent.answer("What is the free look period under the Aarogya Suraksha policy?")
    retrieved = {c.chunk.clause_id for c in answer.contexts}
    assert answer.citations
    assert all(c.clause_id in retrieved for c in answer.citations)


def test_answered_responses_are_grounded_in_their_context(agent):
    answer = agent.answer("What is the compulsory deductible under the DriveShield policy?")
    assert answer.self_check["groundedness"] >= 0.99
    assert answer.self_check["numeric_fidelity"] >= 0.99


def test_the_graph_runs_the_expected_nodes(agent):
    answer = agent.answer("What is the grace period under the Aarogya Suraksha policy?")
    nodes = _nodes(answer)
    assert nodes[0] == "route"
    assert "retrieve" in nodes
    assert "generate" in nodes
    assert nodes[-1] == "self_check"


def test_timings_are_recorded_per_stage(agent):
    answer = agent.answer("What is the cataract sub-limit under the Aarogya Suraksha policy?")
    assert answer.timings_ms["total_ms"] > 0
    assert "route_ms" in answer.timings_ms
    assert "self_check_ms" in answer.timings_ms


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the maximum no claim bonus under the DriveShield motor policy?", "50"),
        ("What is the free look period under the Suraksha Kavach term plan?", "30"),
        ("What is the trip cancellation limit under the Yatra Safe policy?", "1,500"),
        ("What is the jewellery sub-limit under the Griha Raksha policy?", "2,00,000"),
    ],
)
def test_decisive_figures_appear_in_the_answer(agent, question, expected):
    answer = agent.answer(question)
    assert expected.replace(",", "") in answer.answer.replace(",", "")


def test_mermaid_rendering_is_available_for_documentation(agent):
    diagram = agent.mermaid()
    assert "retrieve" in diagram
    assert "self_check" in diagram
