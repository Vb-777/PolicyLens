"""LLM-as-judge scoring, used alongside the deterministic scorer rather than instead.

The deterministic scorer in ``grounding.py`` is cheap, reproducible and strict about
figures, but it cannot tell a correct paraphrase from an unsupported claim that
happens to share vocabulary. The judge covers that gap. Neither is treated as ground
truth: both are reported, and where they disagree the report says so, because
disagreement is the signal that a metric is being gamed by its own blind spot.

The judge is off by default. Running it costs an API call per question and makes the
numbers non-deterministic, which is the wrong default for a regression harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMClient
from ..prompts.templates import (
    CORRECTNESS_JUDGE_USER,
    GROUNDEDNESS_JUDGE_USER,
    JUDGE_SYSTEM,
    format_contexts,
)
from ..schemas import ScoredChunk

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response, tolerating code fences."""
    match = _JSON_RE.search(raw or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class JudgeResult:
    available: bool
    groundedness: float | None = None
    unsupported: list[str] | None = None
    correctness_verdict: str | None = None
    correctness_score: float | None = None
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_available": self.available,
            "judge_groundedness": self.groundedness,
            "judge_unsupported": self.unsupported,
            "judge_verdict": self.correctness_verdict,
            "judge_correctness": self.correctness_score,
            "judge_reason": self.reason,
            "judge_error": self.error,
        }


class NullJudge:
    available = False

    def score(self, **_: Any) -> JudgeResult:
        return JudgeResult(available=False)


class LLMJudge:
    available = True

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def score(
        self,
        question: str,
        answer: str,
        reference: str,
        contexts: list[ScoredChunk],
    ) -> JudgeResult:
        rendered = format_contexts(
            [(c.chunk.meta.citation_label(), c.chunk.text) for c in contexts]
        )
        result = JudgeResult(available=True)

        try:
            grounded_raw = self.llm.complete(
                JUDGE_SYSTEM,
                GROUNDEDNESS_JUDGE_USER.format(contexts=rendered, answer=answer),
            ).text
            parsed = _parse_json(grounded_raw)
            if parsed:
                result.groundedness = _as_float(parsed.get("groundedness"))
                unsupported = parsed.get("unsupported")
                result.unsupported = unsupported if isinstance(unsupported, list) else None
        except Exception as exc:  # noqa: BLE001 - a judge failure must not fail the run
            result.error = f"groundedness: {type(exc).__name__}: {exc}"

        try:
            correctness_raw = self.llm.complete(
                JUDGE_SYSTEM,
                CORRECTNESS_JUDGE_USER.format(
                    question=question, reference=reference, answer=answer
                ),
            ).text
            parsed = _parse_json(correctness_raw)
            if parsed:
                verdict = parsed.get("verdict")
                result.correctness_verdict = str(verdict) if verdict else None
                result.correctness_score = _as_float(parsed.get("score"))
                reason = parsed.get("reason")
                result.reason = str(reason) if reason else None
        except Exception as exc:  # noqa: BLE001
            prefix = f"{result.error}; " if result.error else ""
            result.error = f"{prefix}correctness: {type(exc).__name__}: {exc}"

        return result


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_judge(mode: str, llm: LLMClient | None) -> LLMJudge | NullJudge:
    if mode in {"llm", "both"} and llm is not None and llm.is_generative:
        return LLMJudge(llm)
    return NullJudge()
