"""A knowledge graph over policy structure, extracted from clause text by rules.

Why a graph sits next to the vector index: some questions are structural rather than
semantic. "Which policies have a maternity waiting period, and which is shortest?" is a
comparison over typed facts scattered across six documents. Vector search answers that
badly — it returns the passages that read most like the question, not the complete set
— while a graph answers it exactly.

Extraction is rule-based rather than manifest-driven on purpose: the same rules run
over a real third-party PDF, so the graph is a property of the documents rather than of
this repository's own fixtures.

Node types
    Policy          one per document
    Clause          one per parsed clause, linked to its policy
    WaitingPeriod   normalised to months
    Exclusion       a named carve-out
    Benefit         a named cover with optional monetary or percentage limit
    Definition      a defined term
    Condition       procedural obligations (notification windows, grace periods)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from ..ingest.loaders import ParsedClause
from ..schemas import PolicyRecord

# "thirty-six (36) months" -> the parenthesised digits are the reliable signal.
_DURATION_RE = re.compile(
    r"\((\d{1,4})\)\s*(month|months|day|days|year|years|hour|hours)\b", re.IGNORECASE
)
_BARE_DURATION_RE = re.compile(
    r"\b(\d{1,4})\s*(month|months|day|days|year|years|hour|hours)\b", re.IGNORECASE
)
_RUPEE_RE = re.compile(r"Rs\.?\s*(\d[\d,]*)", re.IGNORECASE)
_USD_RE = re.compile(r"USD\s*(\d[\d,]*)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\((\d{1,3})%\)")
_BARE_PERCENT_RE = re.compile(r"\b(\d{1,3})\s*(?:%|per\s*cent|percent)\b", re.IGNORECASE)

_MONTHS_PER = {"day": 1 / 30.0, "month": 1.0, "year": 12.0, "hour": 1 / 720.0}

_WAITING_HINT = re.compile(r"waiting period|moratorium|survival period", re.IGNORECASE)
_EXCLUSION_HINT = re.compile(r"exclusion|excluded|not payable|shall not be liable", re.IGNORECASE)
_CONDITION_HINT = re.compile(
    r"grace period|free look|notif|intimat|submit|renewal|portability|revival|transfer",
    re.IGNORECASE,
)
_DEFINITION_HINT = re.compile(r"\bmeans\b", re.IGNORECASE)


def _to_int(raw: str) -> int:
    digits = raw.replace(",", "").strip()
    return int(digits) if digits.isdigit() else 0


def _normalise_months(value: int, unit: str) -> float:
    unit = unit.lower().rstrip("s")
    return round(value * _MONTHS_PER.get(unit, 1.0), 4)


def extract_durations(text: str) -> list[tuple[int, str]]:
    """Durations, preferring the parenthesised numeral used in legal drafting."""
    found = [(_to_int(v), u.lower().rstrip("s")) for v, u in _DURATION_RE.findall(text)]
    if found:
        return found
    return [(_to_int(v), u.lower().rstrip("s")) for v, u in _BARE_DURATION_RE.findall(text)]


def extract_amounts(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    out.extend(("INR", _to_int(v)) for v in _RUPEE_RE.findall(text))
    out.extend(("USD", _to_int(v)) for v in _USD_RE.findall(text))
    return out


def extract_percentages(text: str) -> list[int]:
    found = [int(v) for v in _PERCENT_RE.findall(text)]
    if found:
        return found
    return [int(v) for v in _BARE_PERCENT_RE.findall(text)]


class PolicyKnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # -- build ------------------------------------------------------------------

    def build(self, documents: list[tuple[PolicyRecord, list[ParsedClause]]]) -> None:
        self.g = nx.MultiDiGraph()
        for record, clauses in documents:
            self._add_policy(record, clauses)

    def _add_policy(self, record: PolicyRecord, clauses: list[ParsedClause]) -> None:
        pid = record.policy_id
        self.g.add_node(
            pid,
            type="Policy",
            label=record.display or record.product_name,
            insurer=record.insurer,
            product_type=record.product_type,
            uin=record.uin,
        )

        for clause in clauses:
            clause_id = f"{pid}#{clause.clause_number}"
            self.g.add_node(
                clause_id,
                type="Clause",
                label=clause.clause_title,
                policy_id=pid,
                section=clause.section_title,
                clause_number=clause.clause_number,
                text=clause.text,
            )
            self.g.add_edge(pid, clause_id, relation="HAS_CLAUSE")

            context = f"{clause.section_title} {clause.clause_title} {clause.text}"
            heading = f"{clause.section_title} {clause.clause_title}"

            if _WAITING_HINT.search(context):
                self._add_waiting_period(pid, clause_id, clause)
            if _EXCLUSION_HINT.search(heading) or _EXCLUSION_HINT.search(clause.text[:160]):
                self._add_exclusion(pid, clause_id, clause)
            if _DEFINITION_HINT.search(clause.text[:200]) and "definition" in clause.section_title.lower():
                self._add_definition(pid, clause_id, clause)
            if _CONDITION_HINT.search(heading):
                self._add_condition(pid, clause_id, clause)

            self._add_benefit(pid, clause_id, clause)

    def _node(self, node_id: str, **attrs: Any) -> None:
        if node_id in self.g:
            self.g.nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})
        else:
            self.g.add_node(node_id, **attrs)

    def _add_waiting_period(self, pid: str, clause_id: str, clause: ParsedClause) -> None:
        durations = extract_durations(clause.text)
        if not durations:
            return
        value, unit = durations[0]
        node_id = f"WP::{pid}::{clause.clause_number}"
        self._node(
            node_id,
            type="WaitingPeriod",
            label=clause.clause_title,
            policy_id=pid,
            value=value,
            unit=unit,
            months=_normalise_months(value, unit),
            clause_id=clause_id,
        )
        self.g.add_edge(pid, node_id, relation="HAS_WAITING_PERIOD")
        self.g.add_edge(node_id, clause_id, relation="STATED_IN")

    def _add_exclusion(self, pid: str, clause_id: str, clause: ParsedClause) -> None:
        node_id = f"EX::{pid}::{clause.clause_number}"
        self._node(
            node_id,
            type="Exclusion",
            label=clause.clause_title,
            policy_id=pid,
            clause_id=clause_id,
        )
        self.g.add_edge(pid, node_id, relation="EXCLUDES")
        self.g.add_edge(node_id, clause_id, relation="STATED_IN")

    def _add_definition(self, pid: str, clause_id: str, clause: ParsedClause) -> None:
        node_id = f"DEF::{pid}::{clause.clause_number}"
        self._node(
            node_id,
            type="Definition",
            label=clause.clause_title,
            policy_id=pid,
            clause_id=clause_id,
        )
        self.g.add_edge(pid, node_id, relation="DEFINES")
        self.g.add_edge(node_id, clause_id, relation="STATED_IN")

    def _add_condition(self, pid: str, clause_id: str, clause: ParsedClause) -> None:
        durations = extract_durations(clause.text)
        node_id = f"CON::{pid}::{clause.clause_number}"
        attrs: dict[str, Any] = {
            "type": "Condition",
            "label": clause.clause_title,
            "policy_id": pid,
            "clause_id": clause_id,
        }
        if durations:
            value, unit = durations[0]
            attrs.update({"value": value, "unit": unit})
        self._node(node_id, **attrs)
        self.g.add_edge(pid, node_id, relation="HAS_CONDITION")
        self.g.add_edge(node_id, clause_id, relation="STATED_IN")

    def _add_benefit(self, pid: str, clause_id: str, clause: ParsedClause) -> None:
        amounts = extract_amounts(clause.text)
        percents = extract_percentages(clause.text)
        if not amounts and not percents:
            return
        node_id = f"BEN::{pid}::{clause.clause_number}"
        attrs: dict[str, Any] = {
            "type": "Benefit",
            "label": clause.clause_title,
            "policy_id": pid,
            "clause_id": clause_id,
        }
        if amounts:
            currency, value = amounts[0]
            attrs.update({"currency": currency, "amount": value})
        if percents:
            attrs["percent"] = percents[0]
        self._node(node_id, **attrs)
        self.g.add_edge(pid, node_id, relation="PROVIDES")
        self.g.add_edge(node_id, clause_id, relation="STATED_IN")

    # -- queries ----------------------------------------------------------------

    def policies(self) -> list[dict[str, Any]]:
        return [
            {"policy_id": n, **{k: v for k, v in d.items() if k != "type"}}
            for n, d in self.g.nodes(data=True)
            if d.get("type") == "Policy"
        ]

    def nodes_of_type(self, node_type: str, policy_id: str | None = None) -> list[dict[str, Any]]:
        out = []
        for node, data in self.g.nodes(data=True):
            if data.get("type") != node_type:
                continue
            if policy_id and data.get("policy_id") != policy_id:
                continue
            out.append({"node_id": node, **data})
        return out

    def waiting_periods(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.nodes_of_type("WaitingPeriod", policy_id)
        return sorted(rows, key=lambda r: r.get("months", 0.0))

    def exclusions(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        return self.nodes_of_type("Exclusion", policy_id)

    def benefits(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        return self.nodes_of_type("Benefit", policy_id)

    def definitions(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        return self.nodes_of_type("Definition", policy_id)

    def conditions(self, policy_id: str | None = None) -> list[dict[str, Any]]:
        return self.nodes_of_type("Condition", policy_id)

    def find(self, concept: str, node_types: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        """Label substring search across typed nodes."""
        needle = concept.lower().strip()
        if not needle:
            return []
        out = []
        for node, data in self.g.nodes(data=True):
            ntype = data.get("type", "")
            if ntype in {"Policy", "Clause"}:
                continue
            if node_types and ntype not in node_types:
                continue
            if needle in str(data.get("label", "")).lower():
                out.append({"node_id": node, **data})
        return out

    def compare_across_policies(self, concept: str) -> list[dict[str, Any]]:
        """Every typed fact matching ``concept``, grouped so policies can be compared."""
        rows = self.find(concept)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row.get("policy_id", "?")].append(row)

        # One clause can be typed more than once: a grace period is both a
        # WaitingPeriod (it has a duration) and a Condition (it governs renewal), so a
        # naive listing shows the same fact twice with the same figure and reads as a
        # data-quality problem. Collapse on the fact itself and keep every type that
        # described it, which is strictly more information than either row alone.
        merged: dict[tuple, dict[str, Any]] = {}
        for policy_id, items in grouped.items():
            policy_label = self.g.nodes.get(policy_id, {}).get("label", policy_id)
            for item in items:
                # Keyed on the clause and the provision name, not on the values: the
                # types a clause is given carry different fields (an Exclusion has no
                # duration, the WaitingPeriod for the same clause does), so keying on
                # values would split one provision back into a row per type with most
                # of them empty.
                key = (policy_id, item.get("label"), item.get("clause_id"))
                existing = merged.get(key)
                if existing is not None:
                    node_type = item.get("type")
                    if node_type and node_type not in existing["types"]:
                        existing["types"].append(node_type)
                        existing["types"].sort()
                    # Fill in fields this type supplies and the first one did not.
                    for field_name in ("months", "value", "unit", "amount", "currency", "percent"):
                        if existing.get(field_name) is None and item.get(field_name) is not None:
                            existing[field_name] = item[field_name]
                    continue
                merged[key] = {
                    "policy_id": policy_id,
                    "policy": policy_label,
                    "types": [item["type"]] if item.get("type") else [],
                    "label": item.get("label"),
                    "months": item.get("months"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "amount": item.get("amount"),
                    "currency": item.get("currency"),
                    "percent": item.get("percent"),
                    "clause_id": item.get("clause_id"),
                }
        return sorted(merged.values(), key=lambda r: (r["policy_id"], str(r["label"])))

    def clause_text(self, clause_id: str) -> str:
        return str(self.g.nodes.get(clause_id, {}).get("text", ""))

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for _, data in self.g.nodes(data=True):
            counts[str(data.get("type", "Unknown"))] += 1
        relations: dict[str, int] = defaultdict(int)
        for _, _, data in self.g.edges(data=True):
            relations[str(data.get("relation", "?"))] += 1
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "node_types": dict(counts),
            "relations": dict(relations),
        }

    # -- persistence ------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="links")
        (directory / "kg.json").write_text(json.dumps(data), encoding="utf-8")
        (directory / "kg_stats.json").write_text(json.dumps(self.stats(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> PolicyKnowledgeGraph:
        path = Path(directory) / "kg.json"
        kg = cls()
        if not path.exists():
            return kg
        data = json.loads(path.read_text(encoding="utf-8"))
        kg.g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        return kg
