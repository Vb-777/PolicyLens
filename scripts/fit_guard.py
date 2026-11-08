#!/usr/bin/env python3
"""Fit the policy guard's thresholds on the dev split.

The guard declines a question about a provision the named policy does not contain. It
has two thresholds, and picking them by eye on the whole dataset would make the test
figures meaningless — so they are fitted here, on dev only, and the result is what the
defaults in ``config.AgentConfig`` record.

    python scripts/fit_guard.py              # fit and print the grid
    python scripts/fit_guard.py --split test # what the fitted values score on test

Reading the output: ``caught`` is unanswerable questions correctly declined, ``false``
is answerable questions wrongly declined. False abstentions cost more than misses —
declining a question the documents answer is a visible failure, while answering one
they do not is caught downstream by the coverage gate — so ties break toward fewer
false abstentions rather than toward higher F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class Sample:
    question: str
    answerable: bool
    provision_margin: float
    body_inside: float


def collect(split: str) -> list[Sample]:
    from policylens.agent.graph import build_agent
    from policylens.config import get_settings
    from policylens.eval.dataset import load_questions
    from policylens.indexer import load_or_build_index

    settings = get_settings()
    index = load_or_build_index(settings)
    agent = build_agent(index, settings)
    questions = load_questions(settings.paths.eval / "golden_qa.json", split, 0)

    samples: list[Sample] = []
    for item in questions:
        routed = agent._node_route({"question": item.question, "trace": [], "timings": {}})
        mentioned = routed.get("mentioned_policies") or []
        # The guard only applies when exactly one policy is named; a question naming
        # none or several is not the failure mode it addresses.
        if len(mentioned) != 1:
            continue
        inside, outside = agent.provision_support(item.question, mentioned[0])
        body_inside, _ = agent.policy_topical_support(item.question, mentioned[0])
        samples.append(
            Sample(
                question=item.question,
                answerable=item.answerable,
                provision_margin=outside - inside,
                body_inside=body_inside,
            )
        )
    return samples


def score(samples: list[Sample], margin: float, floor: float) -> dict:
    caught = sum(
        1
        for s in samples
        if not s.answerable and s.provision_margin >= margin and s.body_inside <= floor
    )
    false = sum(
        1
        for s in samples
        if s.answerable and s.provision_margin >= margin and s.body_inside <= floor
    )
    total_unanswerable = sum(1 for s in samples if not s.answerable)
    missed = total_unanswerable - caught
    precision = caught / (caught + false) if caught + false else 0.0
    recall = caught / total_unanswerable if total_unanswerable else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "margin": margin,
        "floor": floor,
        "caught": caught,
        "false": false,
        "missed": missed,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--json", action="store_true", help="Emit the full grid as JSON.")
    args = parser.parse_args(argv)

    if args.split == "test":
        print(
            "NOTE: thresholds are fitted on dev. Running on test reports what the "
            "fitted values score on held-out data; do not pick from this grid.\n"
        )

    samples = collect(args.split)
    unanswerable = sum(1 for s in samples if not s.answerable)
    print(
        f"split={args.split}: {len(samples)} questions naming exactly one policy "
        f"({len(samples) - unanswerable} answerable, {unanswerable} unanswerable)"
    )
    if unanswerable < 8:
        print(
            f"WARNING: only {unanswerable} unanswerable questions in this split. That is "
            "too few to fit two thresholds on with any confidence; prefer the loosest "
            "pair that avoids false abstentions over the pair with the best F1."
        )
    print()

    grid = [
        score(samples, margin / 100, floor / 100)
        for margin in range(10, 85, 5)
        for floor in range(10, 80, 5)
    ]

    # Fewest false abstentions first, then F1, then the loosest thresholds that tie.
    best = min(grid, key=lambda r: (r["false"], -r["f1"], -r["margin"]))

    print(f"{'margin':>7s} {'floor':>6s} {'caught':>7s} {'false':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    seen = set()
    for row in sorted(grid, key=lambda r: (r["false"], -r["f1"])):
        key = (row["caught"], row["false"])
        if key in seen:
            continue
        seen.add(key)
        mark = "  <-- selected" if row is best else ""
        print(
            f"{row['margin']:7.2f} {row['floor']:6.2f} {row['caught']:7d} {row['false']:6d} "
            f"{row['precision']:6.3f} {row['recall']:6.3f} {row['f1']:6.3f}{mark}"
        )
        if len(seen) >= 12:
            break

    print(
        f"\nselected: POLICYLENS_POLICY_GUARD_MARGIN={best['margin']:.2f} "
        f"POLICYLENS_POLICY_GUARD_BODY_FLOOR={best['floor']:.2f}"
    )
    print(
        f"  caught {best['caught']}, false abstentions {best['false']}, missed "
        f"{best['missed']} | P={best['precision']:.3f} R={best['recall']:.3f} "
        f"F1={best['f1']:.3f}"
    )

    fired = [
        s
        for s in samples
        if s.provision_margin >= best["margin"] and s.body_inside <= best["floor"]
    ]
    if fired:
        print("\nquestions the guard declines at these thresholds:")
        for s in fired:
            flag = "OK " if not s.answerable else "BAD"
            print(f"  [{flag}] {s.question[:88]}")

    if args.json:
        print("\n" + json.dumps(grid, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
