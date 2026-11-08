"""Command line interface.

    python -m policylens.cli index
    python -m policylens.cli ask "What is the PED waiting period under Aarogya Suraksha?"
    python -m policylens.cli eval --split test --ablation --sweep
    python -m policylens.cli kg --concept maternity
    python -m policylens.cli graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import get_settings


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_index(args: argparse.Namespace) -> int:
    from .indexer import build_index

    settings = get_settings()
    index = build_index(settings, force=args.force)
    print(json.dumps(index.manifest, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent.graph import build_agent
    from .indexer import load_or_build_index

    settings = get_settings()
    index = load_or_build_index(settings)
    agent = build_agent(index, settings)
    answer = agent.answer(args.question)

    if args.json:
        print(answer.model_dump_json(indent=2, exclude={"contexts"}))
        return 0

    print()
    print(answer.answer)
    print()
    if answer.citations:
        print("Sources")
        for citation in answer.citations:
            print(f"  {citation.marker} {citation.label}")
        print()
    check = answer.self_check
    print(
        f"[{answer.status.value}] engine={check.get('engine')} "
        f"groundedness={check.get('groundedness')} "
        f"citations={check.get('citation_precision')} "
        f"latency={answer.timings_ms.get('total_ms', 0):.0f}ms"
    )
    if args.trace:
        print()
        print("Trace")
        for step in answer.trace:
            print(f"  {json.dumps(step, default=str)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import run_evaluation

    settings = get_settings()
    outcome = run_evaluation(
        settings,
        split=args.split,
        variant=args.variant,
        judge_mode=args.judge,
        ablation=args.ablation,
        sweep=args.sweep,
        limit=args.limit,
    )
    agg = outcome["run"]["aggregate"]
    print()
    print("=" * 72)
    print(f"Retrieval   recall@5 {agg['retrieval']['recall@5']}  "
          f"nDCG@5 {agg['retrieval']['ndcg@5']}  MRR {agg['retrieval']['mrr']}")
    print(f"Answers     accuracy {agg['answers']['accuracy']}  "
          f"key-fact recall {agg['answers']['key_fact_recall']}")
    print(f"Faithful    groundedness {agg['faithfulness']['groundedness']}  "
          f"hallucination {agg['faithfulness']['hallucination_rate']}  "
          f"citation precision {agg['faithfulness']['citation_precision']}")
    print(f"Abstention  F1 {agg['abstention']['abstention_f1']}  "
          f"routing accuracy {agg['abstention']['routing_accuracy']}")
    print(f"Latency     p50 {agg['latency']['end_to_end'].get('p50_ms')}ms  "
          f"p95 {agg['latency']['end_to_end'].get('p95_ms')}ms")
    print("=" * 72)
    for name, path in outcome["paths"].items():
        print(f"{name:9s} {path}")
    return 0


def cmd_kg(args: argparse.Namespace) -> int:
    from .indexer import load_or_build_index

    index = load_or_build_index(get_settings())
    if args.stats:
        print(json.dumps(index.kg.stats(), indent=2))
        return 0
    rows = index.kg.compare_across_policies(args.concept)
    if not rows:
        print(f"No knowledge-graph facts matched '{args.concept}'.")
        return 1
    for row in rows:
        detail = []
        if row.get("months") is not None:
            detail.append(f"{row['value']} {row['unit']} ({row['months']} months)")
        if row.get("amount") is not None:
            detail.append(f"{row['currency']} {row['amount']:,}")
        if row.get("percent") is not None:
            detail.append(f"{row['percent']}%")
        print(f"{row['policy']:<42s} {row['type']:<14s} {row['label']:<38s} "
              f"{', '.join(detail):<28s} {row['clause_id']}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from .agent.graph import build_agent
    from .indexer import load_or_build_index

    settings = get_settings()
    agent = build_agent(load_or_build_index(settings), settings)
    print(agent.mermaid())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("policylens.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policylens", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build the vector, lexical and graph indexes.")
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--trace", action="store_true", help="Print the agent's node trace.")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the evaluation harness.")
    p_eval.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p_eval.add_argument("--variant", default="hybrid_rerank")
    p_eval.add_argument("--judge", default=None, choices=["deterministic", "llm", "both"])
    p_eval.add_argument("--ablation", action="store_true", help="Compare retrieval variants.")
    p_eval.add_argument("--sweep", action="store_true", help="Tune the abstention threshold on dev.")
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.set_defaults(func=cmd_eval)

    p_kg = sub.add_parser("kg", help="Query the policy knowledge graph.")
    p_kg.add_argument("--concept", default="waiting")
    p_kg.add_argument("--stats", action="store_true")
    p_kg.set_defaults(func=cmd_kg)

    p_graph = sub.add_parser("graph", help="Print the agent graph as Mermaid.")
    p_graph.set_defaults(func=cmd_graph)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service.")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
