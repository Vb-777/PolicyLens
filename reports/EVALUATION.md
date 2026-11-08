# PolicyLens evaluation report

Generated 2026-11-08T11:15:38Z · variant `hybrid_rerank`

## Configuration

| Component | Value |
| --- | --- |
| Embedding backend | `tfidf-svd` (tfidf(1,2)+svd, dim 128) |
| Reranker | `lexical-features` |
| Generation | `none` / `extractive` |
| Prompt version | `v3` |
| Router | `rules` |
| Index | 129 chunks, 6 policies |
| Knowledge graph | 239 nodes, 337 edges |
| Abstention threshold | 0.3 |

Dataset: **136 questions** (115 answerable, 21 unanswerable, 8 needing more than one clause).

## Retrieval

| k | Recall@k | Precision@k | Hit@k | nDCG@k |
| --- | --- | --- | --- | --- |
| 1 | 0.843 | 0.866 | 0.866 | 0.866 |
| 3 | 0.970 | 0.338 | 0.970 | 0.926 |
| 5 | 0.985 | 0.206 | 0.985 | 0.932 |
| 10 | 0.985 | 0.194 | 0.985 | 0.932 |

MRR **0.914** over 67 questions with gold clauses.
 Multi-clause questions (3): recall@5 **1.000**.

## Answer quality

| Metric | Value |
| --- | --- |
| Exact accuracy (all key facts present) | 0.851 |
| Lenient accuracy (correct or partial) | 0.910 |
| Key-fact recall | 0.888 |
| Token F1 vs reference | 0.278 |
| Correct / partial / incorrect | 57 / 4 / 6 |

## Faithfulness

| Metric | Value |
| --- | --- |
| Groundedness (claims supported by retrieved context) | 1.000 |
| Numeric fidelity (figures traceable to context) | 1.000 |
| Hallucination rate | 0.000 |
| Citation precision | 1.000 |
| Citation coverage (factual sentences carrying a marker) | 1.000 |
| Answers with no citation | 0 |

## Abstention

| Metric | Value |
| --- | --- |
| Abstention precision | 0.688 |
| Abstention recall | 0.688 |
| Abstention F1 | 0.688 |
| Answer/abstain routing accuracy | 0.879 |
| Correctly declined | 11 |
| Answered when it should not have | 5 |
| Declined when it should have answered | 5 |

## Latency

| Metric | ms |
| --- | --- |
| MEAN | 7.44 |
| P50 | 6.81 |
| P90 | 9.92 |
| P95 | 10.96 |
| MAX | 16.36 |

Mean per stage (ms): `dense_ms` 1.678, `fuse_ms` 0.022, `generate_ms` 0.214, `kg_ms` 0.812, `lexical_ms` 0.107, `rerank_ms` 0.550, `retrieve_node_ms` 2.462, `retrieve_total_ms` 2.357, `route_ms` 0.056, `self_check_ms` 0.889, `total_ms` 7.439

## By question type

| Kind | n | Accuracy | Recall@5 |
| --- | --- | --- | --- |
| boolean | 22 | 0.864 | 0.955 |
| comparative | 3 | 0.333 | 1.000 |
| contradicted_premise | 1 | 1.000 | 1.000 |
| factual | 5 | 1.000 | 1.000 |
| numeric | 34 | 0.853 | 1.000 |
| procedural | 2 | 1.000 | 1.000 |
| unanswerable | 16 | 0.688 | — |

## Ablation

Same questions, same generation, different retrieval configuration.

| Variant | Recall@5 | nDCG@5 | MRR | Answer accuracy | Hallucination | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `lexical_only` | 1.000 | 0.963 | 0.950 | 0.851 | 0.000 | 7.19 |
| `dense_only` | 0.985 | 0.879 | 0.845 | 0.851 | 0.000 | 6.58 |
| `hybrid` | 1.000 | 0.936 | 0.915 | 0.851 | 0.000 | 7.47 |
| `hybrid_rerank` | 0.985 | 0.932 | 0.914 | 0.851 | 0.000 | 8.6 |

## Abstention threshold sweep (dev split)

Threshold selected on dev and applied to test: **0.3**.

| Threshold | Abstention P | Abstention R | Abstention F1 | Answer accuracy | Combined |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 0.750 | 0.600 | 0.667 | 0.833 | 0.811 |
| 0.25 | 0.600 | 0.600 | 0.600 | 0.812 | 0.792 |
| 0.3 ← | 0.667 | 0.800 | 0.727 | 0.812 | 0.811 |
| 0.35 | 0.444 | 0.800 | 0.571 | 0.771 | 0.774 |
| 0.4 | 0.417 | 1.000 | 0.588 | 0.750 | 0.774 |
| 0.45 | 0.312 | 1.000 | 0.476 | 0.688 | 0.717 |
| 0.5 | 0.217 | 1.000 | 0.357 | 0.562 | 0.604 |
| 0.55 | 0.179 | 1.000 | 0.303 | 0.500 | 0.547 |
| 0.6 | 0.179 | 1.000 | 0.303 | 0.500 | 0.547 |

## How to read these numbers

- Generation ran on the **keyless extractive engine**, which composes answers from sentences copied out of retrieved clauses. It is therefore grounded by construction, and the groundedness and numeric-fidelity figures above are close to 1.0 as a property of the method rather than as an achievement. On this engine the informative metrics are retrieval quality, answer accuracy and abstention.
- Run with an LLM provider configured to make groundedness a real measurement of a model that *can* hallucinate.
- The embedding backend is the **offline TF-IDF+SVD fallback**, used because pretrained weights could not be downloaded in this environment. Retrieval numbers with `sentence-transformers/all-MiniLM-L6-v2` are expected to be higher, particularly on paraphrased questions.
- Groundedness is a lexical entailment proxy, not an NLI model: it is strict about figures and blind to negation. The LLM judge covers the semantic cases it misses.
