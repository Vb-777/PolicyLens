"""The evaluation harness.

Runs the full pipeline over the golden question set and reports, in one place:

* retrieval quality — recall/precision/hit/nDCG at several k, and MRR
* answer correctness — against the decisive facts of the reference answer
* groundedness and hallucination — deterministic, and optionally an LLM judge
* citation integrity — are the citations real, and is every factual sentence cited
* abstention — does the system decline the questions the corpus cannot answer
* latency — per stage, with p50/p95

It also supports two things a one-off script usually lacks and that make the numbers
mean something: an **ablation** mode that re-runs the same questions under different
retrieval configurations so each component's contribution is measured rather than
assumed, and a **threshold sweep** on the dev split so the one tunable decision
threshold is not chosen by peeking at the test numbers.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..agent.graph import PolicyLensAgent
from ..config import Settings, get_settings
from ..generation import Generator
from ..grounding import citation_validity, score_groundedness
from ..indexer import RagIndex, load_or_build_index
from ..prompts.templates import PROMPT_VERSION
from ..schemas import AnswerStatus
from .dataset import EvalQuestion, load_questions, split_summary
from .judge import NullJudge, build_judge
from .metrics import (
    AbstentionScores,
    classify_correctness,
    key_fact_recall,
    latency_summary,
    safe_mean,
    score_retrieval,
    token_f1,
)

logger = logging.getLogger(__name__)

# Each variant isolates one component so its contribution is measurable.
VARIANTS: dict[str, dict[str, Any]] = {
    "lexical_only": {"use_dense": False, "use_hybrid": True, "rerank": False},
    "dense_only": {"use_dense": True, "use_hybrid": False, "rerank": False},
    "hybrid": {"use_dense": True, "use_hybrid": True, "rerank": False},
    "hybrid_rerank": {"use_dense": True, "use_hybrid": True, "rerank": True},
}
DEFAULT_VARIANT = "hybrid_rerank"


@dataclass
class QuestionResult:
    id: str
    question: str
    kind: str
    split: str
    answerable: bool
    gold_clauses: list[str]
    retrieved: list[str]
    first_gold_rank: int | None
    recall: dict[int, float]
    precision: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    status: str
    answer: str
    route: str
    attempts: int
    n_citations: int
    citation_precision: float
    citation_coverage: float
    groundedness: float
    numeric_fidelity: float
    unsupported_numbers: list[str]
    key_fact_recall: float
    token_f1: float
    correctness: str
    latency_ms: float
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)


class EvaluationRunner:
    def __init__(self, settings: Settings | None = None, index: RagIndex | None = None) -> None:
        self.settings = settings or get_settings()
        self.index = index or load_or_build_index(self.settings)

    # -- agent construction ------------------------------------------------------

    def _settings_for(
        self, retrieval_overrides: dict[str, Any] | None = None, min_coverage: float | None = None
    ) -> Settings:
        cfg = self.settings
        retrieval = dataclasses.replace(cfg.retrieval, **(retrieval_overrides or {}))
        return dataclasses.replace(cfg, retrieval=retrieval)

    def _build_agent(
        self, retrieval_overrides: dict[str, Any] | None = None, min_coverage: float | None = None
    ) -> PolicyLensAgent:
        settings = self._settings_for(retrieval_overrides)
        generator = Generator(
            settings=settings,
            idf=self.index.bm25.idf,
            policy_vocab=self.index.policy_vocabulary(),
        )
        if min_coverage is not None:
            generator.extractive.min_coverage = min_coverage
        return PolicyLensAgent(index=self.index, generator=generator, settings=settings)

    # -- single run --------------------------------------------------------------

    def run(
        self,
        questions: Sequence[EvalQuestion],
        variant: str = DEFAULT_VARIANT,
        min_coverage: float | None = None,
        judge_mode: str | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        overrides = VARIANTS.get(variant, VARIANTS[DEFAULT_VARIANT])
        agent = self._build_agent(overrides, min_coverage)

        judge_mode = judge_mode or self.settings.eval.judge
        judge = build_judge(judge_mode, agent.generator.llm)
        if judge_mode in {"llm", "both"} and isinstance(judge, NullJudge):
            logger.warning("LLM judge requested but no generative provider is configured.")

        # The first query pays one-off costs — lazy reranker construction, any model
        # load — that would otherwise land entirely on the p95 of a small sample.
        self._warm_up(agent)

        k_values = list(self.settings.eval.k_values)
        results: list[QuestionResult] = []

        for i, question in enumerate(questions, start=1):
            if progress and i % 20 == 0:
                logger.info("  [%s] %d/%d", variant, i, len(questions))
            results.append(self._run_one(agent, judge, question, k_values))

        return {
            "variant": variant,
            "overrides": overrides,
            "min_coverage": min_coverage
            if min_coverage is not None
            else agent.generator.extractive.min_coverage,
            "results": results,
            "aggregate": aggregate(results, k_values),
            "environment": self._environment(agent),
        }

    def _warm_up(self, agent: PolicyLensAgent) -> None:
        try:
            agent.answer("warm up query about policy coverage")
        except Exception as exc:  # noqa: BLE001
            logger.debug("warm-up failed: %s", exc)

    def _run_one(
        self,
        agent: PolicyLensAgent,
        judge: Any,
        question: EvalQuestion,
        k_values: list[int],
    ) -> QuestionResult:
        start = time.perf_counter()
        answer = agent.answer(question.question)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = [c.chunk.clause_id for c in answer.contexts]
        retrieval = score_retrieval(retrieved, question.gold_clauses, k_values)

        grounding = score_groundedness(answer.answer, answer.contexts)
        citations = citation_validity(answer, answer.contexts)

        abstained = answer.status is AnswerStatus.ABSTAINED
        if not question.answerable:
            # Declining is the correct answer; scoring its prose against the
            # reference would punish a correct abstention for its wording.
            k_recall = 1.0 if abstained else 0.0
            f1 = float("nan")
            correctness = "correct" if abstained else "incorrect"
        elif abstained:
            k_recall = 0.0
            f1 = 0.0
            correctness = "incorrect"
        else:
            k_recall = key_fact_recall(answer.answer, question.answer_keys)
            f1 = token_f1(answer.answer, question.reference_answer)
            correctness = classify_correctness(k_recall, f1)

        judge_result = {}
        if getattr(judge, "available", False) and question.answerable:
            judge_result = judge.score(
                question=question.question,
                answer=answer.answer,
                reference=question.reference_answer,
                contexts=answer.contexts,
            ).to_dict()

        route = next(
            (step.get("route", "") for step in answer.trace if step.get("node") == "route"), ""
        )

        return QuestionResult(
            id=question.id,
            question=question.question,
            kind=question.kind,
            split=question.split,
            answerable=question.answerable,
            gold_clauses=question.gold_clauses,
            retrieved=retrieved[:10],
            first_gold_rank=retrieval.first_gold_rank,
            recall=retrieval.recall,
            precision=retrieval.precision,
            hit=retrieval.hit,
            ndcg=retrieval.ndcg,
            mrr=retrieval.mrr,
            status=answer.status.value,
            answer=answer.answer,
            route=str(route),
            attempts=int(answer.self_check.get("attempts", 1)),
            n_citations=len(answer.citations),
            citation_precision=citations["citation_precision"],
            citation_coverage=citations["citation_coverage"],
            groundedness=grounding.score,
            numeric_fidelity=grounding.numeric_fidelity,
            unsupported_numbers=grounding.unsupported_numbers,
            key_fact_recall=k_recall,
            token_f1=f1,
            correctness=correctness,
            latency_ms=latency_ms,
            stage_latency_ms={
                k: round(v, 2) for k, v in answer.timings_ms.items() if k.endswith("_ms")
            },
            judge=judge_result,
        )

    # -- sweep -------------------------------------------------------------------

    def sweep_abstention_threshold(
        self,
        dev_questions: Sequence[EvalQuestion],
        thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
        variant: str = DEFAULT_VARIANT,
    ) -> dict[str, Any]:
        """Pick the answer/abstain threshold on dev, never on test.

        Reported as a curve rather than a single winner so the trade-off is visible:
        raising the threshold buys abstention recall and costs answered-question
        coverage.
        """
        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            run = self.run(
                dev_questions, variant=variant, min_coverage=threshold, judge_mode="deterministic", progress=False
            )
            agg = run["aggregate"]
            rows.append(
                {
                    "threshold": threshold,
                    "abstention_f1": agg["abstention"]["abstention_f1"],
                    "abstention_precision": agg["abstention"]["abstention_precision"],
                    "abstention_recall": agg["abstention"]["abstention_recall"],
                    "answer_accuracy": agg["answers"]["accuracy"],
                    "overall_score": _combined_score(agg),
                }
            )

        # Ties are common because the curve is flat over a range. Break them toward
        # the higher threshold: at equal overall accuracy the more conservative
        # setting declines more of what it cannot answer.
        best = max(
            rows,
            key=lambda r: (
                r["overall_score"] if r["overall_score"] is not None else -1,
                r["abstention_recall"] if r["abstention_recall"] is not None else -1,
                r["threshold"],
            ),
        )
        return {"curve": rows, "selected_threshold": best["threshold"], "selected": best}

    # -- ablation ----------------------------------------------------------------

    def run_ablation(
        self,
        questions: Sequence[EvalQuestion],
        variants: Sequence[str] = tuple(VARIANTS),
        min_coverage: float | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for variant in variants:
            logger.info("running variant: %s", variant)
            run = self.run(
                questions, variant=variant, min_coverage=min_coverage, judge_mode="deterministic"
            )
            out[variant] = run["aggregate"]
        return out

    # -- environment -------------------------------------------------------------

    def _environment(self, agent: PolicyLensAgent) -> dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "embedder": self.index.embedder.describe(),
            "reranker": getattr(agent.retriever._get_reranker(), "name", "none"),
            "llm_provider": agent.generator.llm.provider,
            "llm_model": agent.generator.llm.model,
            "prompt_version": PROMPT_VERSION,
            "router_mode": agent.router_mode,
            "index": {
                "n_chunks": len(self.index.chunks),
                "n_policies": len(self.index.policies),
                "kg": self.index.kg.stats(),
            },
            "config": self.settings.fingerprint(),
        }


# --------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------


def _combined_score(aggregate_block: dict[str, Any]) -> float | None:
    """Overall accuracy across every dev question, for threshold selection.

    Each question counts once: an answerable one is correct when the decisive facts
    are present, an unanswerable one when the system declines. Two earlier objectives
    were tried and discarded. Abstention F1 is undefined at sweep points that produce
    no abstentions, which silently removed the best thresholds from consideration.
    Balanced accuracy over the two classes gives each of the 21 unanswerable
    questions roughly five times the weight of an answerable one, and selected a
    threshold that abstained on 30 questions to catch 5 — better on the objective,
    clearly worse as a system.
    """
    return aggregate_block["overall_accuracy"]


def _round(value: float) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 4)


def aggregate(results: Sequence[QuestionResult], k_values: Sequence[int]) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable]
    with_gold = [r for r in answerable if r.gold_clauses]
    answered = [r for r in results if r.status == "answered"]
    multi = [r for r in with_gold if len(r.gold_clauses) > 1]

    retrieval = {
        f"recall@{k}": _round(safe_mean([r.recall[k] for r in with_gold])) for k in k_values
    }
    retrieval.update(
        {f"precision@{k}": _round(safe_mean([r.precision[k] for r in with_gold])) for k in k_values}
    )
    retrieval.update(
        {f"hit@{k}": _round(safe_mean([r.hit[k] for r in with_gold])) for k in k_values}
    )
    retrieval.update(
        {f"ndcg@{k}": _round(safe_mean([r.ndcg[k] for r in with_gold])) for k in k_values}
    )
    retrieval["mrr"] = _round(safe_mean([r.mrr for r in with_gold]))
    retrieval["n_evaluated"] = len(with_gold)
    if multi:
        retrieval["multi_clause_recall@5"] = _round(safe_mean([r.recall[5] for r in multi]))
        retrieval["multi_clause_n"] = len(multi)

    correct = sum(1 for r in answerable if r.correctness == "correct")
    partial = sum(1 for r in answerable if r.correctness == "partial")
    answers = {
        "n_answerable": len(answerable),
        "correct": correct,
        "partial": partial,
        "incorrect": len(answerable) - correct - partial,
        "accuracy": _round(correct / len(answerable)) if answerable else None,
        "lenient_accuracy": _round((correct + partial) / len(answerable)) if answerable else None,
        "key_fact_recall": _round(safe_mean([r.key_fact_recall for r in answerable])),
        "token_f1": _round(safe_mean([r.token_f1 for r in answerable])),
    }

    hallucinated = [
        r for r in answered if r.groundedness < 1.0 or r.unsupported_numbers
    ]
    faithfulness = {
        "n_answered": len(answered),
        "groundedness": _round(safe_mean([r.groundedness for r in answered])),
        "numeric_fidelity": _round(safe_mean([r.numeric_fidelity for r in answered])),
        "hallucination_rate": _round(len(hallucinated) / len(answered)) if answered else None,
        "unsupported_number_rate": _round(
            sum(1 for r in answered if r.unsupported_numbers) / len(answered)
        )
        if answered
        else None,
        "citation_precision": _round(safe_mean([r.citation_precision for r in answered])),
        "citation_coverage": _round(safe_mean([r.citation_coverage for r in answered])),
        "answers_without_citation": sum(1 for r in answered if r.n_citations == 0),
    }

    abstention = AbstentionScores()
    for r in results:
        abstained = r.status == "abstained"
        if not r.answerable and abstained:
            abstention.true_abstain += 1
        elif not r.answerable and not abstained:
            abstention.false_answer += 1
        elif r.answerable and abstained:
            abstention.false_abstain += 1
        else:
            abstention.true_answer += 1

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind]
        subset_gold = [r for r in subset if r.gold_clauses]
        by_kind[kind] = {
            "n": len(subset),
            "accuracy": _round(
                sum(1 for r in subset if r.correctness == "correct") / len(subset)
            ),
            "recall@5": _round(safe_mean([r.recall[5] for r in subset_gold]))
            if subset_gold
            else None,
        }

    judged = [r for r in results if r.judge.get("judge_available")]
    judge_block: dict[str, Any] = {"enabled": bool(judged)}
    if judged:
        judge_block.update(
            {
                "n": len(judged),
                "groundedness": _round(
                    safe_mean(
                        [
                            r.judge["judge_groundedness"]
                            for r in judged
                            if r.judge.get("judge_groundedness") is not None
                        ]
                    )
                ),
                "correctness": _round(
                    safe_mean(
                        [
                            r.judge["judge_correctness"]
                            for r in judged
                            if r.judge.get("judge_correctness") is not None
                        ]
                    )
                ),
                "verdict_correct": sum(
                    1 for r in judged if r.judge.get("judge_verdict") == "correct"
                ),
                "errors": sum(1 for r in judged if r.judge.get("judge_error")),
            }
        )

    stage_keys: set[str] = set()
    for r in results:
        stage_keys |= set(r.stage_latency_ms)
    stages = {
        key: _round(safe_mean([r.stage_latency_ms.get(key, float("nan")) for r in results]))
        for key in sorted(stage_keys)
    }

    # Every question counted once: answerable ones scored on their facts,
    # unanswerable ones on whether the system declined.
    overall_accuracy = (
        _round(sum(1 for r in results if r.correctness == "correct") / len(results))
        if results
        else None
    )

    return {
        "overall_accuracy": overall_accuracy,
        "retrieval": retrieval,
        "answers": answers,
        "faithfulness": faithfulness,
        "abstention": abstention.to_dict(),
        "latency": {
            "end_to_end": latency_summary([r.latency_ms for r in results]),
            "stage_mean_ms": stages,
        },
        "by_kind": by_kind,
        "judge": judge_block,
        "retries_triggered": sum(1 for r in results if r.attempts > 1),
    }


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------


def write_reports(
    run: dict[str, Any],
    out_dir: Path,
    dataset_summary: dict[str, Any],
    ablation: dict[str, Any] | None = None,
    sweep: dict[str, Any] | None = None,
    tag: str = "",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""

    results: list[QuestionResult] = run["results"]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": run["variant"],
        "min_coverage": run["min_coverage"],
        "dataset": dataset_summary,
        "environment": run["environment"],
        "aggregate": run["aggregate"],
        "ablation": ablation,
        "threshold_sweep": sweep,
        "results": [asdict(r) for r in results],
    }

    json_path = out_dir / f"eval-{stamp}{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    csv_path = out_dir / f"eval-{stamp}{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "id", "split", "kind", "answerable", "status", "route", "correctness",
                "key_fact_recall", "first_gold_rank", "recall@5", "ndcg@5", "mrr",
                "groundedness", "numeric_fidelity", "citation_precision",
                "latency_ms", "gold_clauses", "top_retrieved", "question", "answer",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.id, r.split, r.kind, r.answerable, r.status, r.route, r.correctness,
                    _round(r.key_fact_recall), r.first_gold_rank, _round(r.recall.get(5, float("nan"))),
                    _round(r.ndcg.get(5, float("nan"))), _round(r.mrr),
                    _round(r.groundedness), _round(r.numeric_fidelity), _round(r.citation_precision),
                    round(r.latency_ms, 2), "|".join(r.gold_clauses), "|".join(r.retrieved[:5]),
                    r.question, r.answer.replace("\n", " ")[:400],
                ]
            )

    md_path = out_dir / f"eval-{stamp}{suffix}.md"
    md_path.write_text(
        render_markdown(payload, ablation=ablation, sweep=sweep), encoding="utf-8"
    )
    (out_dir / "EVALUATION.md").write_text(
        render_markdown(payload, ablation=ablation, sweep=sweep), encoding="utf-8"
    )

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(
    payload: dict[str, Any],
    ablation: dict[str, Any] | None = None,
    sweep: dict[str, Any] | None = None,
) -> str:
    agg = payload["aggregate"]
    env = payload["environment"]
    ds = payload["dataset"]
    lines: list[str] = []

    lines.append("# PolicyLens evaluation report")
    lines.append("")
    lines.append(f"Generated {payload['generated_at']} · variant `{payload['variant']}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Embedding backend | `{env['embedder'].get('backend')}` ({env['embedder'].get('model', '')}, dim {env['embedder'].get('dim')}) |")
    lines.append(f"| Reranker | `{env['reranker']}` |")
    lines.append(f"| Generation | `{env['llm_provider']}` / `{env['llm_model']}` |")
    lines.append(f"| Prompt version | `{env['prompt_version']}` |")
    lines.append(f"| Router | `{env['router_mode']}` |")
    lines.append(f"| Index | {env['index']['n_chunks']} chunks, {env['index']['n_policies']} policies |")
    lines.append(f"| Knowledge graph | {env['index']['kg']['nodes']} nodes, {env['index']['kg']['edges']} edges |")
    lines.append(f"| Abstention threshold | {payload['min_coverage']} |")
    lines.append("")
    lines.append(
        f"Dataset: **{ds['n_questions']} questions** "
        f"({ds['answerable']} answerable, {ds['unanswerable']} unanswerable, "
        f"{ds['multi_clause']} needing more than one clause)."
    )
    lines.append("")

    lines.append("## Retrieval")
    lines.append("")
    r = agg["retrieval"]
    lines.append("| k | Recall@k | Precision@k | Hit@k | nDCG@k |")
    lines.append("| --- | --- | --- | --- | --- |")
    for k in (1, 3, 5, 10):
        if f"recall@{k}" in r:
            lines.append(
                f"| {k} | {_fmt(r[f'recall@{k}'])} | {_fmt(r[f'precision@{k}'])} | "
                f"{_fmt(r[f'hit@{k}'])} | {_fmt(r[f'ndcg@{k}'])} |"
            )
    lines.append("")
    lines.append(f"MRR **{_fmt(r['mrr'])}** over {r['n_evaluated']} questions with gold clauses.")
    if "multi_clause_recall@5" in r:
        lines.append(
            f" Multi-clause questions ({r['multi_clause_n']}): recall@5 "
            f"**{_fmt(r['multi_clause_recall@5'])}**."
        )
    lines.append("")

    lines.append("## Answer quality")
    lines.append("")
    a = agg["answers"]
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Exact accuracy (all key facts present) | {_fmt(a['accuracy'])} |")
    lines.append(f"| Lenient accuracy (correct or partial) | {_fmt(a['lenient_accuracy'])} |")
    lines.append(f"| Key-fact recall | {_fmt(a['key_fact_recall'])} |")
    lines.append(f"| Token F1 vs reference | {_fmt(a['token_f1'])} |")
    lines.append(f"| Correct / partial / incorrect | {a['correct']} / {a['partial']} / {a['incorrect']} |")
    lines.append("")

    lines.append("## Faithfulness")
    lines.append("")
    f = agg["faithfulness"]
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Groundedness (claims supported by retrieved context) | {_fmt(f['groundedness'])} |")
    lines.append(f"| Numeric fidelity (figures traceable to context) | {_fmt(f['numeric_fidelity'])} |")
    lines.append(f"| Hallucination rate | {_fmt(f['hallucination_rate'])} |")
    lines.append(f"| Citation precision | {_fmt(f['citation_precision'])} |")
    lines.append(f"| Citation coverage (factual sentences carrying a marker) | {_fmt(f['citation_coverage'])} |")
    lines.append(f"| Answers with no citation | {f['answers_without_citation']} |")
    lines.append("")

    lines.append("## Abstention")
    lines.append("")
    ab = agg["abstention"]
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Abstention precision | {_fmt(ab['abstention_precision'])} |")
    lines.append(f"| Abstention recall | {_fmt(ab['abstention_recall'])} |")
    lines.append(f"| Abstention F1 | {_fmt(ab['abstention_f1'])} |")
    lines.append(f"| Answer/abstain routing accuracy | {_fmt(ab['routing_accuracy'])} |")
    lines.append(f"| Correctly declined | {ab['correctly_abstained']} |")
    lines.append(f"| Answered when it should not have | {ab['missed_abstentions']} |")
    lines.append(f"| Declined when it should have answered | {ab['over_abstentions']} |")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lat = agg["latency"]["end_to_end"]
    lines.append("| Metric | ms |")
    lines.append("| --- | --- |")
    for key in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"):
        if key in lat:
            lines.append(f"| {key.replace('_ms', '').upper()} | {lat[key]} |")
    lines.append("")
    stages = agg["latency"]["stage_mean_ms"]
    if stages:
        lines.append("Mean per stage (ms): " + ", ".join(f"`{k}` {_fmt(v)}" for k, v in stages.items() if v))
        lines.append("")

    lines.append("## By question type")
    lines.append("")
    lines.append("| Kind | n | Accuracy | Recall@5 |")
    lines.append("| --- | --- | --- | --- |")
    for kind, block in agg["by_kind"].items():
        lines.append(f"| {kind} | {block['n']} | {_fmt(block['accuracy'])} | {_fmt(block['recall@5'])} |")
    lines.append("")

    if agg["judge"].get("enabled"):
        j = agg["judge"]
        lines.append("## LLM judge")
        lines.append("")
        lines.append(f"Judged {j['n']} answers. Groundedness {_fmt(j['groundedness'])}, correctness {_fmt(j['correctness'])}, verdict=correct on {j['verdict_correct']}.")
        lines.append("")

    if ablation:
        lines.append("## Ablation")
        lines.append("")
        lines.append("Same questions, same generation, different retrieval configuration.")
        lines.append("")
        lines.append("| Variant | Recall@5 | nDCG@5 | MRR | Answer accuracy | Hallucination | p95 ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, block in ablation.items():
            lines.append(
                f"| `{name}` | {_fmt(block['retrieval']['recall@5'])} | "
                f"{_fmt(block['retrieval']['ndcg@5'])} | {_fmt(block['retrieval']['mrr'])} | "
                f"{_fmt(block['answers']['accuracy'])} | "
                f"{_fmt(block['faithfulness']['hallucination_rate'])} | "
                f"{block['latency']['end_to_end'].get('p95_ms', '—')} |"
            )
        lines.append("")

    if sweep:
        lines.append("## Abstention threshold sweep (dev split)")
        lines.append("")
        lines.append(
            f"Threshold selected on dev and applied to test: **{sweep['selected_threshold']}**."
        )
        lines.append("")
        lines.append("| Threshold | Abstention P | Abstention R | Abstention F1 | Answer accuracy | Combined |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in sweep["curve"]:
            marker = " ←" if row["threshold"] == sweep["selected_threshold"] else ""
            lines.append(
                f"| {row['threshold']}{marker} | {_fmt(row['abstention_precision'])} | "
                f"{_fmt(row['abstention_recall'])} | {_fmt(row['abstention_f1'])} | "
                f"{_fmt(row['answer_accuracy'])} | {_fmt(row['overall_score'])} |"
            )
        lines.append("")

    lines.append("## How to read these numbers")
    lines.append("")
    # The non-generative path reports itself as provider "none" with model
    # "extractive", so testing the provider alone sent an extractive run down the
    # branch that claims the figures came from a model able to invent content — the
    # exact opposite of what they measure, in the one paragraph a reader relies on to
    # interpret them.
    engine = env["llm_provider"]
    if engine in {"none", "extractive"} or env.get("llm_model") == "extractive":
        lines.append(
            "- Generation ran on the **keyless extractive engine**, which composes answers "
            "from sentences copied out of retrieved clauses. It is therefore grounded by "
            "construction, and the groundedness and numeric-fidelity figures above are close "
            "to 1.0 as a property of the method rather than as an achievement. On this engine "
            "the informative metrics are retrieval quality, answer accuracy and abstention."
        )
        lines.append(
            "- Run with an LLM provider configured to make groundedness a real measurement "
            "of a model that *can* hallucinate."
        )
    else:
        lines.append(
            f"- Generation ran on `{env['llm_provider']}`/`{env['llm_model']}`, so groundedness "
            "and hallucination rate measure a model that can genuinely invent content."
        )
    if env["embedder"].get("backend") == "tfidf-svd":
        lines.append(
            "- The embedding backend is the **offline TF-IDF+SVD fallback**, used because "
            "pretrained weights could not be downloaded in this environment. Retrieval numbers "
            "with `sentence-transformers/all-MiniLM-L6-v2` are expected to be higher, "
            "particularly on paraphrased questions."
        )
    lines.append(
        "- Groundedness is a lexical entailment proxy, not an NLI model: it is strict about "
        "figures and blind to negation. The LLM judge covers the semantic cases it misses."
    )
    lines.append("")

    return "\n".join(lines)


def run_evaluation(
    settings: Settings | None = None,
    split: str = "test",
    variant: str = DEFAULT_VARIANT,
    judge_mode: str | None = None,
    ablation: bool = False,
    sweep: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    """Entry point used by the CLI and by ``make eval``."""
    cfg = settings or get_settings()
    runner = EvaluationRunner(cfg)

    qa_path = cfg.paths.eval / "golden_qa.json"
    all_questions = load_questions(qa_path, split="all")
    dataset_summary = split_summary(all_questions)

    sweep_result = None
    min_coverage = None
    if sweep:
        dev = [q for q in all_questions if q.split == "dev"]
        logger.info("sweeping abstention threshold on %d dev questions", len(dev))
        sweep_result = runner.sweep_abstention_threshold(dev, variant=variant)
        min_coverage = sweep_result["selected_threshold"]

    questions = load_questions(qa_path, split=split, limit=limit)  # type: ignore[arg-type]
    logger.info("evaluating %d questions (split=%s, variant=%s)", len(questions), split, variant)
    run = runner.run(questions, variant=variant, min_coverage=min_coverage, judge_mode=judge_mode)

    ablation_result = None
    if ablation:
        ablation_result = runner.run_ablation(questions, min_coverage=min_coverage)

    paths = write_reports(
        run,
        out_dir=cfg.paths.reports,
        dataset_summary={**dataset_summary, "evaluated_split": split, "n_evaluated": len(questions)},
        ablation=ablation_result,
        sweep=sweep_result,
    )
    return {"run": run, "ablation": ablation_result, "sweep": sweep_result, "paths": paths}
