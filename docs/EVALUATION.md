# Evaluation

What each metric measures, what it does not, and how to reproduce the numbers.

The committed run is [`../reports/EVALUATION.md`](../reports/EVALUATION.md).

```bash
make eval        # test split, with dev threshold sweep and retrieval ablation
make eval-dev    # dev split — for tuning only, never for reporting
make ablation    # retrieval configurations only
```

Each run writes a timestamped JSON, CSV and Markdown triple into `reports/`. The JSON
carries per-question rows, so any aggregate can be recomputed or audited.

---

## The dataset

136 questions over the 6-policy corpus, each carrying gold clause IDs and the key facts
a correct answer must contain. Retrieval and answer correctness are therefore measured
independently — a question can retrieve the right clause and still be answered badly,
and the report distinguishes the two.

| Kind | n | What it tests |
|---|---|---|
| `numeric` | 65 | a figure: waiting period, sub-limit, deductible, percentage |
| `boolean` | 31 | is this covered or excluded |
| `unanswerable` | 21 | the documents do not contain the answer |
| `factual` | 7 | a definition or a named term |
| `comparative` | 6 | the same provision across two policies |
| `procedural` | 3 | a multi-step process |
| `multi_hop` | 2 | a clause plus the definition it depends on |
| `contradicted_premise` | 1 | the question assumes something false |

8 questions require more than one clause.

### Splits

Assigned by hashing the question ID: **53 dev, 83 test**. Hashing rather than slicing
means the split is stable across runs and reorderings, and carries no leakage from
question ordering.

**Dev is for tuning. Test is for reporting.** Both tunable knobs are fitted by scripts
that read dev only:

- the abstention coverage threshold — `policylens eval --sweep`
- the policy guard's two thresholds — `scripts/fit_guard.py`

### The dataset's weakest point

21 unanswerable questions, of which 5 fall in dev — and only 4 of those name a single
policy, which is the condition for the guard to apply. Two thresholds cannot be fitted
on four examples with any confidence, which is why `fit_guard.py` breaks ties toward
the loosest values that avoid false abstentions rather than toward the best dev F1.
Expanding the unanswerable set — particularly out-of-corpus attribute questions — is
the single most useful contribution to this repo.

---

## Retrieval metrics

Computed against gold clause IDs over the 67 questions that have them.

| Metric | Definition | Why |
|---|---|---|
| **Recall@k** | fraction of gold clauses in the top k | whether the answer is reachable at all |
| **Precision@k** | fraction of top k that are gold | falls with k by construction; read alongside recall |
| **Hit@k** | at least one gold clause in top k | the binary form, for single-clause questions |
| **nDCG@k** | rank-discounted gain | position matters: the generator reads top-k in order |
| **MRR** | mean reciprocal rank of the first gold clause | how near the top the right clause lands |

Recall@5 of 0.985 with recall@10 identical means the retriever either finds the clause
in the first five or does not find it — widening `k` past 5 buys nothing on this corpus,
which is why the self-check retry widens once and then stops.

---

## Answer metrics

| Metric | Definition |
|---|---|
| **Exact accuracy** | every key fact for the question appears in the answer |
| **Lenient accuracy** | correct, or partial (some key facts present) |
| **Key-fact recall** | fraction of key facts present, averaged over questions |
| **Token F1** | token overlap with the reference answer |

**Token F1 is low (0.278) and that is expected.** The extractive engine returns whole
clause sentences where the reference answer is a short phrase, so overlap is diluted by
correct material. Key-fact recall (0.888) is the metric that tracks correctness. Token
F1 is reported anyway because it is the metric most RAG write-ups lead with, and
showing it alongside a measure that disagrees with it is the point.

---

## Faithfulness metrics

### Read this before quoting groundedness

The default generator is **extractive**: it copies sentences out of retrieved clauses.
It cannot invent content. Groundedness of 1.000 and hallucination of 0.000 are
therefore **properties of the method, not achievements**, and they become real
measurements only with an LLM provider configured:

```bash
POLICYLENS_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... make eval
```

The report states this itself in its "How to read these numbers" section. (That section
previously said the opposite for extractive runs, due to a branch that tested the
provider name — `"none"` — against the string `"extractive"`. Fixed; the lesson is
that a caveat which never prints is worse than no caveat.)

### What the metrics are

| Metric | Definition |
|---|---|
| **Groundedness** | fraction of answer claims supported by a single retrieved clause |
| **Numeric fidelity** | fraction of figures traceable to the retrieved context |
| **Hallucination rate** | fraction of answers with at least one unsupported claim |
| **Citation precision** | fraction of citations pointing at a clause actually retrieved |
| **Citation coverage** | fraction of factual sentences carrying a marker |

A claim counts as supported when **one** context both covers enough of its content
tokens and contains every number it asserts. Requiring a single context to do both
prevents a claim being "supported" by stitching half from one clause and half from
another.

### Limits of the groundedness proxy

It is lexical entailment, not an NLI model:

- **Blind to negation.** "The waiting period is *not* 36 months" scores as supported.
- **Under-credits paraphrase** that shares few tokens with the source.
- **Strict about numbers** — deliberately. A figure absent from the retrieved clauses is
  the most damaging error this system can make, and it is caught exactly.

The optional LLM judge (`--judge llm`, or `both`) covers the semantic cases. Where the
two disagree, the disagreement is reported rather than resolved — it marks the
questions worth looking at by hand.

### Citation coverage had a measurement bug

Answers are written with the marker after the full stop: `"... is 36 months. [1] It
also ..."`. The sentence splitter used a lookbehind for the terminator and a lookahead
for a capital, so the marker sat between them and two sentences collapsed into one.
A collapsed sentence contains a marker, so **every** sentence trivially counted as
cited and coverage read 1.0 regardless of the answer.

`split_sentences_keeping_markers` moves a trailing marker inside the sentence it
supports before splitting. This does not change the extractive numbers — that engine
emits markers inline already — but it is what makes the metric mean anything on the
LLM path, where the bug would otherwise have hidden uncited claims completely.

---

## Abstention metrics

The metrics that matter most on the keyless path, since groundedness is pinned by
construction.

| Metric | Definition |
|---|---|
| **Abstention precision** | of the questions declined, the fraction truly unanswerable |
| **Abstention recall** | of the unanswerable questions, the fraction declined |
| **Abstention F1** | their harmonic mean |
| **Routing accuracy** | fraction of all questions correctly answered-or-declined |

Current test figures: precision 0.688, recall 0.688, F1 0.688, routing accuracy 0.879 —
11 correctly declined, 5 answered that should not have been, 5 declined that should
have been answered.

### The threshold sweep is the argument for the policy guard

The sweep varies the coverage floor on dev and shows the **entire achievable frontier of
that single signal**:

| Threshold | Abstention F1 | Answer accuracy |
|---|---|---|
| 0.2 | 0.667 | 0.833 |
| 0.3 ← selected | 0.727 | 0.812 |
| 0.4 | 0.588 | 0.750 |
| 0.5 | 0.357 | 0.562 |

There is no good operating point: abstention recall is bought with answer accuracy,
monotonically. That measurement is what justified building a second mechanism instead
of retuning the first, and the guard then moved test F1 from 0.500 to 0.688 while answer
accuracy *rose* — the signature of a signal that is actually informative rather than a
threshold traded along a frontier.

---

## Ablation

Same questions, same generation, retrieval configuration swapped. `make ablation`.

| Variant | Recall@5 | nDCG@5 | MRR | Answer accuracy | p95 ms |
|---|---|---|---|---|---|
| `lexical_only` | 1.000 | 0.963 | 0.950 | 0.851 | 6.5 |
| `dense_only` | 0.985 | 0.879 | 0.845 | 0.851 | 7.8 |
| `hybrid` | 1.000 | 0.936 | 0.915 | 0.851 | 7.0 |
| `hybrid_rerank` | 0.985 | 0.932 | 0.914 | 0.851 | 8.7 |

BM25 alone wins, and the reranker costs MRR. See
[ARCHITECTURE.md §2](ARCHITECTURE.md) for why, and why the hybrid stack is kept.

Note that answer accuracy is identical across all four. Retrieval is not the binding
constraint on this corpus — generation and abstention are. That is worth knowing before
spending more effort on retrieval.

---

## Latency

Measured per question after a warm index, wall-clock, single process, 2 vCPU. These
move by a millisecond or two between runs on shared hardware, so they are quoted
approximately; the committed report carries the exact figures for its run.

| | ms |
|---|---|
| mean | ~6 |
| p50 | ~6 |
| p95 | ~9 |

Per-stage means are in the report. Retrieval is ~2 ms of it, the graph lookup ~0.6 ms,
the self-check ~0.9 ms. These describe the keyless path; with an LLM provider,
generation dominates by two to three orders of magnitude and these figures become
noise. They are useful as a floor: the non-generative parts of the system are not where
latency will come from.

---

## Reproducing

```bash
make install
make index          # ingests data/corpus_pdf
make test           # 91 tests
make eval
```

No API key and no model downloads. To change what is measured:

| Variable | Effect |
|---|---|
| `POLICYLENS_LLM_PROVIDER` | `none` (extractive) · `anthropic` · `openai` · `gemini` · `ollama` |
| `POLICYLENS_EMBEDDING_BACKEND` | `tfidf-svd` · `sentence-transformers` · `openai` |
| `POLICYLENS_RETRIEVER` | `hybrid` · `lexical` · `dense` |
| `POLICYLENS_RERANKER` | `lexical-features` · `cross-encoder` · `none` |
| `POLICYLENS_CORPUS_DIR` | point at your own policy PDFs |

Changing the embedding backend invalidates the fitted thresholds. Re-run
`policylens eval --sweep` and `scripts/fit_guard.py` before trusting the abstention
figures on a different backend.
