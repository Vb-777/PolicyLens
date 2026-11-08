# Architecture

Why each component is shaped the way it is. The README covers what the system does and
what it scores; this covers the decisions, including the ones that turned out wrong.

---

## 1. Ingest: the document's own structure is the chunking strategy

A policy wording is already divided into clauses, each numbered and titled by the
drafter. That structure is not decoration — it is the unit a citation has to point at,
because *"Clause 4.2 (Pre-Existing Disease Waiting Period)"* is checkable and
*"characters 8,400–9,100 of the PDF"* is not.

So chunking splits on the document's `SECTION n.` and `n.m Title` numbering rather than
on a token budget. Consequences:

- **Chunks are uneven**, from one sentence to several paragraphs. That is correct here:
  a waiting-period clause is short and a perils clause is long, and padding the short
  one to a uniform size would mix two provisions into one retrievable unit.
- **A clause is never split mid-provision**, so a retrieved chunk is always a complete
  statement of a rule. Over-long clauses are split with `part`/`n_parts` metadata so
  provenance survives.
- **Headings are retained and indexed.** A clause states its rule in the body but names
  it in the title. "Pre-Existing Disease Waiting Period" never appears in the body of
  the clause that defines it, so a question using the provision's name only matches via
  the heading.

### Page furniture is a correctness problem, not noise

Every real policy PDF foots each page with the insurer, product and UIN. Extraction
interleaves that with body text, and page one's footer comes out *before* the cover
block. Left in, it does active damage: metadata inference read the footer as the
insurer name and the page number as the product name, and the repeated line landed
inside clauses, corrupting the text a citation quotes.

`strip_page_furniture` drops lines that recur at page edges across most pages, plus
page-number lines. Recurrence is the signal that works: body text does not repeat
verbatim across pages, boilerplate does. Single-page documents have no recurrence to
measure, so only the page-number rule applies — a header is not assumed to be furniture
on the evidence of one page.

The text and PDF corpora are held to clause-level parity by `tests/test_pdf_ingest.py`.
That test exists because the two silently disagreed once.

---

## 2. Retrieval: hybrid by design, lexical by measurement

Three retrievers, fused:

- **BM25**, implemented directly (~120 lines) rather than pulled in as a dependency.
  The IDF table it builds is reused by the extractive engine and the policy guard, so
  writing it was cheaper than adding a package and then needing its internals anyway.
- **Dense**, over a pluggable embedder. The default is TF-IDF + truncated SVD: a
  latent-semantic embedding that needs no pretrained weights and no network, so the
  pipeline runs from a bare clone. `sentence-transformers` is a drop-in upgrade.
- **Reciprocal rank fusion** to combine them. RRF over score normalisation because the
  two retrievers' scores are not on comparable scales and rank is what survives a
  change of embedder.

Then reranking: an IDF-weighted feature scorer by default, with a cross-encoder
adapter for when weights are available.

**The ablation says this is more machinery than this corpus needs.** BM25 alone scores
recall@5 1.000 / MRR 0.950 against the full pipeline's 0.985 / 0.914, and it is the
fastest. The honest reading: six formally-worded documents, questions sharing
vocabulary with their target clauses, and a dense side running on a lexical-adjacent
fallback. Dense retrieval and reranking pay off on paraphrase and on corpora large
enough for lexical matching to produce collisions; neither condition holds at 129
chunks. The architecture stays because it is where the system goes next, the finding
is reported because it is what the measurement says, and `POLICYLENS_RETRIEVER=lexical`
is one environment variable away.

---

## 3. The knowledge graph earns its place by making absence checkable

The graph is typed, not a generic entity blob. Each clause the builder can interpret
becomes a node carrying extracted values:

| Node type | Carries | Example |
|---|---|---|
| `Policy` | insurer, product type, UIN | Aarogya Suraksha, Health Indemnity |
| `Clause` | section, number, text | 4.2, Pre-Existing Disease Waiting Period |
| `WaitingPeriod` | value, unit, months | 36 months |
| `Benefit` | amount or percent | Road Ambulance, INR 5,000; Co-Payment, 10% |
| `Exclusion` | — | Cosmetic and Plastic Surgery |
| `Condition` | value, unit | Free Look Period, 15 days |
| `Definition` | defined term | "Hospitalisation" |

This buys three things a vector index cannot:

**A provision inventory per policy.** Full-text search can only report that a phrase
was not found — equally true of a provision the policy lacks and one it paraphrases.
An inventory of canonical provision names makes absence a checkable claim, which is
what the policy guard is built on (§4).

**Cross-policy comparison as a lookup.** "Which policy has the shorter PED waiting
period?" is a graph query over `WaitingPeriod` nodes with normalised `months`, not a
retrieval problem. `GET /kg/compare?concept=grace+period` returns every policy that
states one.

**Definition expansion.** A clause's meaning often depends on a term defined
elsewhere in the same document. The agent's `kg_lookup` node follows `STATED_IN` and
definition edges to pull those in alongside the retrieved clause.

---

## 4. Abstention: the subsystem that took the most work and is still weakest

A policy wording states what is covered. It does not contain the premium, the
insurer's claim settlement ratio, GST treatment, or the cashless hospital list. And a
householder's policy does not have a no-claim bonus, however confidently the motor
policy's bonus clause ranks for that question.

Three mechanisms, in order of how much they contribute:

### Coverage gate (pre-existing)

IDF-weighted coverage of the question's terms by the best retrieved sentence, with
unseen terms weighted at the top of the scale. Below a floor, decline. The floor is
selected on dev by `policylens eval --sweep` and applied to test.

The threshold sweep is the honest part: it shows the **whole achievable frontier of
this one signal**, and it is not good. Best F1 0.727 on dev, and buying abstention
recall costs answer accuracy monotonically. That measurement is what justified
building something better rather than retuning.

### Policy guard (added)

Two signals that must agree before declining:

1. **Provision margin** — some other policy names this provision far better than the
   named one does (`provision_support`, matching against canonical labels).
2. **Body floor** — no clause anywhere in the named policy discusses it
   (`policy_topical_support`, exhaustive scan over the policy's clauses).

Both sides are scanned exhaustively rather than over the retrieved window, because the
claim being established is an *absence* and a retrieved sample cannot establish one.
At 129 chunks that costs well under a millisecond.

Each signal alone misfires, in opposite directions, which is why the conjunction
matters:

| Question | Provision margin | Body floor | Correct |
|---|---|---|---|
| "no claim bonus under the householder's policy" | fires | fires | decline ✓ |
| "does the term plan have a maturity value?" | fires | does not | answer ✓ |
| "within how long must I report a claim?" | does not | fires | answer ✓ |

An earlier version checked only whether any top-ranked clause came *from* the named
policy. That does not work, because provenance is not topicality: some clause of the
right policy almost always ranks well on shared vocabulary ("policy", "period",
"claim"), and its presence was read as evidence the policy covers the provision. It
passed every case it was written to catch.

Thresholds are fitted on dev by `scripts/fit_guard.py`, which breaks ties toward the
loosest values rather than the best F1 — dev holds five unanswerable questions, four of
which name a single policy and so reach the guard, which
is too thin to fit two thresholds on with confidence.

### Self-check loop

After generation, claims are scored for groundedness and citations for validity. Weak
groundedness triggers one retrieval retry at a widened `k` before the answer is
returned. A second graph pass is skipped — it adds nothing the first did not find.

### What is still broken

Out-of-corpus *attribute* questions. "How much does the zero-depreciation add-on
cost?" retrieves the zero-depreciation clause, which is the right clause and does not
contain a price. Both guard signals see a well-covered topic and pass. Catching this
needs an answer-type check — does the context contain a value of the kind being asked
for — which is not implemented. It is most of the remaining abstention error.

---

## 5. Generation: two engines behind one interface

**Extractive (default, keyless).** Selects the clause sentences that best answer the
question and assembles them with citation markers. It cannot hallucinate, because it
only copies. That is why the published groundedness of 1.000 is a property of the
method and not a result — stated in the report itself, since a reader who missed that
would draw exactly the wrong conclusion.

**LLM.** Anthropic, OpenAI, Gemini and Ollama adapters behind a common `LLMClient`,
with a prompt that requires every factual sentence to carry a clause marker and
mandates an explicit refusal token when the context is insufficient.

The default being keyless is a deliberate constraint, not a limitation worked around.
A reviewer can clone the repo and reproduce the published numbers without an API
account, and the test suite runs in CI with no secrets.

### Groundedness is a proxy and says so

`grounding.py` scores a claim as supported when one single context both covers enough
of its content tokens and contains every number it asserts. Requiring *one* context to
do both prevents a claim being "supported" by stitching half from one clause and half
from another.

Its limits are documented in the module because an undocumented metric is worse than
no metric:

- blind to negation — "the waiting period is *not* 36 months" scores as supported;
- under-credits a correct paraphrase sharing few tokens with the source;
- strict about numbers, which is the right bias: a figure absent from the retrieved
  clauses is the most damaging error available here, and it is caught exactly.

The optional LLM judge covers the semantic cases, and disagreement between the two is
itself informative.

---

## 6. Agent: a state machine, not a tool-calling loop

```
route → retrieve → [kg_lookup] → generate → self_check ⟲ retrieve
                                                 ↓
                                                END
```

`route` classifies the question (lookup, comparative, structural) and extracts named
policies; comparative and structural routes widen `k` and enable graph expansion.
`self_check` is the only cycle, capped at one retry.

A LangGraph state machine rather than a free-running agent loop because the control
flow here is known in advance and worth constraining. The failure mode of a free loop
on this task is re-retrieving until something looks good enough to answer with, which
is precisely the behaviour abstention exists to prevent. Tools are exposed through a
registry (`agent/tools.py`) for the LLM path, including a deterministic claim-payout
calculator so arithmetic over sub-limits and co-pay is computed rather than generated.

---

## 7. Evaluation: splits, and not tuning on test

136 questions over 8 kinds: factual, numeric, boolean, procedural, comparative,
multi-hop, contradicted-premise, unanswerable. Each carries gold clause IDs and key
facts, so retrieval and answer correctness are measured independently.

Splits are assigned by hashing the question ID — stable across runs and reorderings,
with no accidental leakage from question ordering. **Dev is for tuning, test is for
reporting**, and the two knobs that could be tuned (the coverage threshold and the
guard thresholds) are both fitted by scripts that read dev only.

The gold set's weakest point is 21 unanswerable questions, of which 4 land in dev.
That is the binding constraint on abstention work, and the most useful next
contribution to this repo.

---

## Component map

| Concern | Module |
|---|---|
| PDF/text loading, furniture stripping | `ingest/loaders.py` |
| Clause chunking | `ingest/chunker.py` |
| Embedding backends | `embeddings/` |
| BM25, FAISS | `store/` |
| Fusion, reranking | `retrieval/` |
| Typed policy graph | `kg/graph.py` |
| LLM adapters | `llm/` |
| Extractive + LLM answering | `generation.py` |
| Groundedness, citation validity | `grounding.py` |
| State machine, tools, policy guard | `agent/` |
| Metrics, splits, judge, harness, report | `eval/` |
| HTTP service | `api/main.py` |
