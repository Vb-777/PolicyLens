"""BM25, rank fusion and end-to-end retrieval quality."""

from __future__ import annotations

import pytest

from policylens.retrieval.hybrid import reciprocal_rank_fusion
from policylens.store.bm25 import BM25Index, tokenize


def test_tokenizer_keeps_numbers_and_hyphenated_terms():
    tokens = tokenize("A waiting period of thirty-six (36) months under Clause 4.2.")
    assert "thirty-six" in tokens
    assert "36" in tokens
    assert "4.2" in tokens
    # Stopwords are stripped, but negations and qualifiers are not.
    assert "the" not in tokens


def test_bm25_ranks_the_document_containing_the_rare_term_first():
    index = BM25Index()
    index.build(
        [
            "The grace period for renewal premium is thirty (30) days.",
            "Cataract surgery is subject to a waiting period of twenty-four (24) months.",
            "Ambulance charges are reimbursed up to Rs. 5,000 per event.",
        ]
    )
    hits = index.search("cataract waiting period", k=3)
    assert hits[0][0] == 1


def test_bm25_distinguishes_different_numbers():
    index = BM25Index()
    index.build(
        [
            "The waiting period is thirty-six (36) months of continuous coverage.",
            "The waiting period is twenty-four (24) months of continuous coverage.",
        ]
    )
    hits = index.search("24 months waiting period", k=2)
    assert hits[0][0] == 1, "lexical search must separate 24 from 36"


def test_bm25_roundtrips_through_disk(tmp_path):
    index = BM25Index()
    index.build(["alpha beta gamma", "beta gamma delta"])
    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    assert reloaded.n_docs == 2
    assert reloaded.search("alpha", k=2) == index.search("alpha", k=2)


def test_rrf_prefers_a_document_both_retrievers_found():
    # Document 7 is rank 1 for one retriever only; document 3 is rank 2 for both.
    fused = reciprocal_rank_fusion(
        {
            "dense": [(7, 0.9), (3, 0.8), (1, 0.1)],
            "lexical": [(9, 5.0), (3, 4.0), (2, 1.0)],
        },
        k=60,
    )
    ranked = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)
    assert ranked[0][0] == 3


def test_rrf_records_provenance_for_each_hit():
    fused = reciprocal_rank_fusion({"dense": [(1, 0.5)], "lexical": [(1, 2.0)]}, k=60)
    assert fused[1]["ranks"] == {"dense": 1, "lexical": 1}
    assert fused[1]["raw"]["lexical"] == 2.0


# -- integration -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_clause"),
    [
        ("pre-existing disease waiting period Aarogya Suraksha", "HLT-AAROGYA#4.2"),
        ("room rent sub-limit Aarogya Suraksha", "HLT-AAROGYA#5.1"),
        ("no claim bonus DriveShield", "MOT-DRIVESHIELD#7.2"),
        ("suicide clause Suraksha Kavach", "LIF-SURAKSHAKAVACH#4.1"),
        ("baggage delay Yatra Safe", "TRV-YATRASAFE#3.4"),
        ("jewellery sub-limit Griha Raksha", "HOM-GRIHARAKSHA#3.2"),
    ],
)
def test_gold_clause_is_retrieved_in_top_5(agent, query, expected_clause):
    results = agent.retriever.retrieve(query, top_k=5).results
    assert expected_clause in {r.chunk.clause_id for r in results}


def test_retrieval_reports_per_stage_timings(agent):
    result = agent.retriever.retrieve("grace period", top_k=5)
    assert "dense_ms" in result.timings_ms
    assert "lexical_ms" in result.timings_ms
    assert result.timings_ms["retrieve_total_ms"] > 0


def test_policy_filter_restricts_results(agent):
    results = agent.retriever.retrieve(
        "waiting period", top_k=5, policy_filter=["HLT-NIROGAM"]
    ).results
    assert results
    assert {r.chunk.meta.policy_id for r in results} == {"HLT-NIROGAM"}
