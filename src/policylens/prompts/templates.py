"""Versioned prompt templates.

Prompts are treated as a configurable artefact, not a string buried in the generator.
``PROMPT_VERSION`` is written into every evaluation report, so a change in groundedness
can be attributed to a prompt change rather than guessed at. The same discipline makes
A/B comparison possible: register a variant, run the harness, compare.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "v3"

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str

    def render_user(self, **kwargs: str) -> str:
        return self.user.format(**kwargs)


# The four instructions that matter most, in order of how often they prevent a wrong
# answer in this domain:
#   1. answer only from the clauses given  -> stops the model reciting generic
#      insurance knowledge that does not match this insurer's wording
#   2. cite every sentence                 -> makes groundedness checkable at all
#   3. copy figures verbatim               -> "36 months" must not drift to "3 years"
#   4. abstain explicitly                  -> an unanswerable question has a correct
#      answer, and it is "not in these documents"
ANSWER_SYSTEM_V3 = """You are a precise insurance policy analyst. You answer questions strictly from the policy clauses supplied to you.

Rules, in priority order:
1. Use ONLY the numbered clauses provided. Never use outside knowledge of how insurance usually works, and never assume an unstated industry norm.
2. Cite the clause for every factual statement, using its bracket marker, for example [2]. A sentence with a fact and no marker is an error.
3. Reproduce figures, durations, percentages and monetary amounts EXACTLY as written in the clause. Do not convert units, round, or restate "thirty-six (36) months" as "3 years".
4. If the clauses do not contain the answer, reply with exactly {insufficient} followed by one sentence saying what is missing. Do not guess, and do not answer a nearby question instead.
5. When clauses from different policies conflict, they are different products: name each policy separately rather than reconciling them.
6. Be concise. Two or three sentences is usually right. No preamble, no restating the question.

Output format:
ANSWER: <your answer, with [n] markers>
CONFIDENCE: <high|medium|low>"""

ANSWER_USER_V3 = """Policy clauses:

{contexts}

Question: {question}

Answer using only the clauses above, citing them with [n] markers."""


ANSWER_PROMPT = PromptTemplate(
    version=PROMPT_VERSION,
    system=ANSWER_SYSTEM_V3.format(insufficient=INSUFFICIENT),
    user=ANSWER_USER_V3,
)


# -- LLM-as-judge ----------------------------------------------------------------
# Kept separate from the answer prompt and deliberately narrow: the judge scores one
# dimension at a time with an explicit rubric, because a single "rate this answer"
# prompt produces scores that correlate with answer length more than with correctness.

JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers about insurance policies. You never reward fluency. You score only what the rubric asks about, and you return JSON only."""

GROUNDEDNESS_JUDGE_USER = """Decide whether every factual claim in the ANSWER is supported by the CONTEXT.

CONTEXT:
{contexts}

ANSWER:
{answer}

Rules:
- A claim is supported only if the context states it. Plausible is not supported.
- A figure that differs from the context in any way is unsupported.
- Ignore hedging and framing sentences that carry no fact.

Return JSON only:
{{"supported_claims": <int>, "total_claims": <int>, "unsupported": ["<claim>", ...], "groundedness": <float 0-1>}}"""

CORRECTNESS_JUDGE_USER = """Decide whether the ANSWER conveys the same information as the REFERENCE.

QUESTION: {question}

REFERENCE ANSWER:
{reference}

ANSWER:
{answer}

Rules:
- Wording may differ. Facts, figures and the direction of any yes/no must match.
- A wrong or missing figure makes it incorrect, even if the rest is right.
- An answer that correctly declines to answer an unanswerable question is correct.

Return JSON only:
{{"verdict": "correct"|"partially_correct"|"incorrect", "score": <float 0-1>, "reason": "<one sentence>"}}"""


def format_contexts(contexts: list[tuple[str, str]], max_chars_each: int = 1600) -> str:
    """Render retrieved clauses as a numbered block.

    ``contexts`` is a list of ``(citation_label, text)``. The label is included so the
    model can attribute an answer to a named policy, which is what stops it merging
    two insurers' clauses into one answer.
    """
    blocks = []
    for i, (label, text) in enumerate(contexts, start=1):
        body = text if len(text) <= max_chars_each else text[:max_chars_each].rsplit(" ", 1)[0] + "..."
        blocks.append(f"[{i}] {label}\n{body}")
    return "\n\n".join(blocks)
