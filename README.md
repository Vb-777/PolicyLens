# PolicyLens

Grounded question answering over insurance policy wordings: hybrid retrieval, a typed
knowledge graph of policy provisions, a LangGraph agent that declines questions the
documents cannot answer, and an evaluation harness that measures all of it.

Every number in this README was produced by `make eval` on a held-out test split and
is reproducible from a clone with **no API key and no model downloads**.

```bash
git clone https://github.com/Vb-777/policylens && cd policylens
make install && make index
make ask Q="What is the pre-existing disease waiting period under Aarogya Suraksha?"
```

```
Expenses related to the treatment of a Pre-Existing Disease and its direct
complications shall be excluded until the expiry of thirty-six (36) months of
continuous coverage after the date of inception of the first Policy with the
Company. [1] ...

Sources
  [1] Aarogya Suraksha Health Insurance, Clause 4.2 (Pre-Existing Disease Waiting Period)
  [2] Aarogya Suraksha Health Insurance, Clause 4.3 (Specified Disease and Procedure Waiting Period)
```

---

## Why this exists

A policy wording is a legal document, and an answer drawn from one is only useful if
you can see which clause it came from. Two failure modes matter more than fluency:

1. **Answering from the wrong document.** Ask what the no-claim bonus is under a
   *householder's* policy and a retriever will happily return the motor policy's
   bonus clause, because the words match and the document does not.
2. **Answering a question the documents do not address.** A policy wording states
   what is covered; it does not contain the premium, the insurer's claim settlement
   ratio, or the cashless hospital list. A system that answers anyway is worse than
   one that says it cannot.

PolicyLens treats both as measurable, and reports how often it gets them wrong.

---

## Measured results

Held-out test split, 83 of 136 questions. Full report:
[`reports/EVALUATION.md`](reports/EVALUATION.md).

| | |
|---|---|
| **Retrieval** | recall@5 **0.985** · nDCG@5 **0.932** · MRR **0.914** |
| **Answers** | exact accuracy **0.851** · key-fact recall **0.888** |
| **Abstention** | precision **0.688** · recall **0.688** · F1 **0.688** |
| **Latency** | p50 **~6 ms** · p95 **~9 ms** (retrieval + extractive generation) |

Corpus: 6 policy wordings → 129 clauses → a graph of 239 nodes and 337 edges.

### The ablation changed the default

Same questions, same generation, retrieval swapped:

| Variant | Recall@5 | nDCG@5 | MRR | p95 ms |
|---|---|---|---|---|
| `lexical_only` (BM25) | **1.000** | **0.963** | **0.950** | 6.5 |
| `dense_only` | 0.985 | 0.879 | 0.845 | 7.8 |
| `hybrid` (RRF) | **1.000** | 0.936 | 0.915 | 7.0 |
| `hybrid_rerank` | 0.985 | 0.932 | 0.914 | 8.7 |

**BM25 alone wins on this corpus, and the reranker costs MRR.** That is not the
result the architecture was built expecting, and it is reported rather than buried.
The reason is legible: six formally-worded documents, questions that share vocabulary
with the clauses they target, and a dense side running on the dependency-free
TF-IDF+SVD backend rather than pretrained embeddings. Hybrid retrieval earns its keep
on paraphrase, which this question set under-represents.

`make ablation` re-runs it. Switch retrievers with `POLICYLENS_RETRIEVER=lexical`.

---

## How it works

```
 PDF policy wordings  (data/corpus_pdf/)
          │
          ├─ loader ─────── strips running headers, footers, page numbers
          │                 (page-1 footer otherwise parses as the insurer name)
          │
          └─ chunker ────── clause-aware: splits on the document's own
                            SECTION / n.m numbering, never mid-clause
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
   RETRIEVAL                                          KNOWLEDGE GRAPH
   ├─ BM25            (own implementation)            ├─ Policy ──┐
   ├─ dense           (TF-IDF+SVD | MiniLM)           ├─ Clause   │ typed, with
   ├─ fusion          (reciprocal rank fusion)        ├─ Benefit  │ values:
   └─ rerank          (features | cross-encoder)      ├─ Exclusion│ amount,
                                    │                 ├─ Condition│ percent,
                                    │                 ├─ WaitingPeriod  months
                                    │                 └─ Definition
                                    │                             │
                                    └──────────┬──────────────────┘
                                               ▼
                                   LANGGRAPH AGENT
                                   route → retrieve → kg_lookup
                                        → generate → self_check ⟲ retry
                                               │
                                               ▼
                                 grounded answer + clause citations,
                                        or a refusal
```

### The knowledge graph does real work

It is not a diagram. Each clause the builder can interpret becomes a typed node with
extracted values — `WaitingPeriod(36, "month")`, `Benefit("Road Ambulance", INR 5000)`,
`Benefit("Co-Payment", 10%)` — which gives each policy a **provision inventory**.

An inventory is the one structure that makes *absence* checkable. Full-text search can
only report that a phrase was not found, which is equally true of a provision the
policy lacks and one it paraphrases. So the agent's policy guard asks two questions and
requires both to agree before declining:

- does some **other** policy name this provision far better than the named one?
- does **no clause anywhere** in the named policy discuss it?

Either signal alone misfires, in opposite directions. The provision margin alone
declines *"does the term plan have a maturity value?"* — nothing is labelled "Maturity
Value" but a clause answers it directly. The body floor alone declines *"within how
long must I report a claim?"* — the policy says "intimate", not "report". Requiring
both took abstention F1 from 0.500 to **0.688**, and answer accuracy rose alongside it
rather than being traded away.

Thresholds were fitted by grid search on the **dev** split (`scripts/fit_guard.py`);
the figures above are from test.

---

## Running it without a key

The default path needs no API key, no pretrained weights and no network:

| Stage | Default (offline) | Optional upgrade |
|---|---|---|
| Embeddings | TF-IDF + truncated SVD | `sentence-transformers` · OpenAI |
| Lexical | BM25 (own implementation) | — |
| Reranking | IDF-weighted lexical features | cross-encoder |
| Generation | extractive (copies clause sentences) | Claude · GPT · Gemini · Ollama |
| Vector store | FAISS | — (one adapter; `store/faiss_store.py` is the seam) |

This is deliberate. A reviewer can clone the repo and run `make test && make eval` and
get the published numbers back, which is not true of a project that needs a funded API
account to start. Set a provider in `.env` (see `.env.example`) to use a real model.

```bash
make test        # 91 tests
make eval        # test split + dev sweep + ablation → reports/
make serve       # FastAPI on :8000, OpenAPI docs at /docs
docker compose up --build
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | grounded answer with citations, or a refusal |
| `GET /search` | retrieval only, with per-retriever scores |
| `GET /policies` | indexed policies and metadata |
| `GET /kg/stats` | graph shape |
| `GET /kg/compare?concept=grace+period` | one provision across every policy that states it |
| `GET /metrics` | the latest evaluation report |
| `GET /health` | readiness, including index size |

---

## What the numbers do and do not mean

Stated plainly, because a metric nobody understands is a metric nobody should trust.

- **Groundedness 1.000 and hallucination 0.000 are not achievements.** The default
  generator is *extractive*: it copies sentences out of retrieved clauses, so it cannot
  invent content, and these figures are a property of the method. They become a real
  measurement only with an LLM provider configured. On the offline path the
  informative metrics are retrieval quality, answer accuracy and abstention.
- **Groundedness is a lexical entailment proxy, not an NLI model.** It is strict about
  figures — a number absent from the retrieved clauses is caught exactly — and blind to
  negation: "the waiting period is *not* 36 months" scores as supported. The optional
  LLM judge covers the semantic cases it misses, and disagreement between the two is
  itself informative.
- **Retrieval numbers describe the TF-IDF+SVD backend**, because the environment the
  evaluation ran in could not download pretrained weights. Installing
  `sentence-transformers` changes them; re-run the harness rather than assuming these
  figures still describe your setup.
- **Abstention is the weakest subsystem**, at F1 0.688 — 5 questions answered that
  should have been declined and 5 declined that should have been answered. The dev
  split holds 5 unanswerable questions, only 4 of which name a single policy and so
  reach the guard at all — too thin to tune two thresholds
  on with confidence, so `fit_guard.py` breaks ties toward the loosest values rather
  than fitting until the test numbers looked good.
- **Token F1 against reference answers is 0.278** and low by design: the extractive
  engine returns whole clause sentences where the reference is a short phrase. Key-fact
  recall (0.888) is the metric that tracks whether the answer is right.
- **The corpus is synthetic.** Six policy wordings written for this project in the
  style of Indian regulated products — waiting periods, sub-limits, co-pay, NCB,
  IDV and depreciation grids, free look, grace and revival. Real insurer wordings are
  copyrighted and cannot ship in a public repo; synthetic documents also make a gold
  answer key possible, which is what lets recall@k be measured rather than estimated.
  Invented insurers and UINs throughout — nothing here describes a real product.

### Known limitations

- Comparative questions spanning several policies score worst by type: the agent
  widens retrieval and expands the graph, and retrieves the right clauses, but the
  extractive generator cannot compose a comparison out of two of them. This is the
  clearest case for running the LLM path.
- Out-of-corpus attribute questions still get answered when they should not — premium
  amounts, settlement ratios, GST, hospital network lists. These need an answer-type
  check the current gate does not implement; they are the bulk of the remaining
  abstention error.
- 129 chunks is small. The retrieval design is sound at this size but the
  numbers should not be read as evidence about enterprise-scale corpora.

---

## Layout

```
src/policylens/
  ingest/      PDF and text loaders, page-furniture stripping, clause chunker
  embeddings/  pluggable backends: tfidf-svd, sentence-transformers, openai
  store/       BM25 (own) and FAISS
  retrieval/   hybrid fusion and reranking
  kg/          typed policy graph over networkx
  llm/         anthropic, openai, gemini, ollama, and a keyless base
  agent/       LangGraph state machine and tools
  eval/        metrics, dataset splits, judge, harness, reporting
  api/         FastAPI service
  generation.py  extractive engine and LLM answer assembly
  grounding.py   claim-level groundedness and citation validity

data/corpus/       text sources (editable source of truth)
data/corpus_pdf/   rendered PDFs — what the pipeline ingests by default
data/eval/         136-question gold set with dev/test splits
scripts/           generate_corpus.py  authors the corpus and gold set together,
                                       so retrieval ground truth is exact by
                                       construction rather than hand-labelled
                   render_pdfs.py      text corpus -> PDFs, with a parity check
                   fit_guard.py        fits the policy guard on the dev split
docs/              ARCHITECTURE.md, EVALUATION.md
```

Text and PDF corpora are held to clause-level parity by `tests/test_pdf_ingest.py`;
`make corpus` re-renders and re-verifies.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions and why each
  component is shaped the way it is
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — what each metric measures, how the
  splits work, and how to reproduce
- [`reports/EVALUATION.md`](reports/EVALUATION.md) — the latest committed run

## License

MIT — see [LICENSE](LICENSE).
