"""Tools the agent can call.

Each tool is a plain function plus a JSON schema. The schemas are exposed so the same
tools can be handed to a tool-calling LLM, while the graph itself can call them
directly — which is what keeps the agent's behaviour identical with and without an
API key.

Four tools, each earning its place:

``search_clauses``   hybrid retrieval over the corpus
``query_knowledge_graph``  typed structural facts, for questions that are really a
                     comparison or an enumeration rather than a passage lookup
``compare_policies`` the same fact across every policy that states it
``settlement_calculator``  the arithmetic a policy answer usually implies — a
                     proportionate room-rent deduction, a co-payment, a deductible —
                     which an LLM gets subtly wrong often enough to be worth removing
                     from its hands
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..indexer import RagIndex
from ..retrieval.hybrid import HybridRetriever
from ..schemas import ScoredChunk


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# -- settlement arithmetic ---------------------------------------------------------


def settlement_calculator(
    claim_amount: float,
    entitled_room_rent: float | None = None,
    actual_room_rent: float | None = None,
    co_payment_percent: float = 0.0,
    deductible: float = 0.0,
    sum_insured: float | None = None,
) -> dict[str, Any]:
    """Apply the standard deduction waterfall to a claim.

    Order matters and is the order policy wordings specify: proportionate deduction
    for a room upgrade first, then the deductible, then co-payment on the balance,
    then the sum insured cap. Applying co-payment before the deductible produces a
    different and wrong number.
    """
    steps: list[dict[str, Any]] = []
    amount = float(claim_amount)
    steps.append({"step": "claimed", "amount": round(amount, 2)})

    if entitled_room_rent and actual_room_rent and actual_room_rent > entitled_room_rent:
        ratio = entitled_room_rent / actual_room_rent
        amount *= ratio
        steps.append(
            {
                "step": "proportionate_deduction",
                "ratio": round(ratio, 4),
                "amount": round(amount, 2),
                "note": "room category exceeded entitlement",
            }
        )

    if deductible:
        amount = max(0.0, amount - float(deductible))
        steps.append({"step": "deductible", "deducted": float(deductible), "amount": round(amount, 2)})

    if co_payment_percent:
        co_pay = amount * (float(co_payment_percent) / 100.0)
        amount -= co_pay
        steps.append(
            {
                "step": "co_payment",
                "percent": float(co_payment_percent),
                "deducted": round(co_pay, 2),
                "amount": round(amount, 2),
            }
        )

    capped = False
    if sum_insured is not None and amount > float(sum_insured):
        amount = float(sum_insured)
        capped = True
        steps.append({"step": "sum_insured_cap", "amount": round(amount, 2)})

    return {
        "payable": round(amount, 2),
        "total_deduction": round(float(claim_amount) - amount, 2),
        "capped_at_sum_insured": capped,
        "steps": steps,
    }


# -- registry ----------------------------------------------------------------------


class ToolRegistry:
    """Binds tools to a loaded index."""

    def __init__(self, index: RagIndex, retriever: HybridRetriever) -> None:
        self.index = index
        self.retriever = retriever
        self._specs: dict[str, ToolSpec] = {}
        self._register_all()

    # -- implementations -----------------------------------------------------

    def search_clauses(
        self, query: str, top_k: int = 5, policy_id: str | None = None
    ) -> list[ScoredChunk]:
        result = self.retriever.retrieve(
            query, top_k=top_k, policy_filter=[policy_id] if policy_id else None
        )
        return result.results

    def query_knowledge_graph(
        self, concept: str, policy_id: str | None = None, fact_type: str | None = None
    ) -> list[dict[str, Any]]:
        kg = self.index.kg
        type_map = {
            "waiting_period": "WaitingPeriod",
            "exclusion": "Exclusion",
            "benefit": "Benefit",
            "definition": "Definition",
            "condition": "Condition",
        }
        if fact_type:
            node_type = type_map.get(fact_type.lower(), fact_type)
            rows = kg.nodes_of_type(node_type, policy_id)
            if concept:
                needle = concept.lower()
                rows = [r for r in rows if needle in str(r.get("label", "")).lower()]
            return rows
        rows = kg.find(concept)
        if policy_id:
            rows = [r for r in rows if r.get("policy_id") == policy_id]
        return rows

    def compare_policies(self, concept: str) -> list[dict[str, Any]]:
        return self.index.kg.compare_across_policies(concept)

    # -- registration --------------------------------------------------------

    def _register_all(self) -> None:
        self._specs["search_clauses"] = ToolSpec(
            name="search_clauses",
            description=(
                "Search policy wordings for the clauses most relevant to a question. "
                "Use for any question about what a policy says. Returns clause text "
                "with policy and clause-number provenance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "top_k": {"type": "integer", "default": 5},
                    "policy_id": {
                        "type": "string",
                        "description": "Restrict to one policy, e.g. HLT-AAROGYA.",
                    },
                },
                "required": ["query"],
            },
            func=self.search_clauses,
        )
        self._specs["query_knowledge_graph"] = ToolSpec(
            name="query_knowledge_graph",
            description=(
                "Look up structured facts extracted from the policies: waiting "
                "periods (normalised to months), exclusions, benefit limits, "
                "definitions and procedural conditions. Use when a question asks to "
                "enumerate or rank facts rather than read one clause."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "e.g. 'maternity', 'cataract'."},
                    "policy_id": {"type": "string"},
                    "fact_type": {
                        "type": "string",
                        "enum": ["waiting_period", "exclusion", "benefit", "definition", "condition"],
                    },
                },
                "required": ["concept"],
            },
            func=self.query_knowledge_graph,
        )
        self._specs["compare_policies"] = ToolSpec(
            name="compare_policies",
            description=(
                "Return the same fact across every policy that states it, for "
                "side-by-side comparison. Use for 'which policy has the shorter/"
                "better X' questions."
            ),
            parameters={
                "type": "object",
                "properties": {"concept": {"type": "string"}},
                "required": ["concept"],
            },
            func=self.compare_policies,
        )
        self._specs["settlement_calculator"] = ToolSpec(
            name="settlement_calculator",
            description=(
                "Compute the payable amount after proportionate room-rent deduction, "
                "deductible, co-payment and the sum insured cap, applied in that "
                "order. Use whenever a question asks how much would actually be paid."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "claim_amount": {"type": "number"},
                    "entitled_room_rent": {"type": "number"},
                    "actual_room_rent": {"type": "number"},
                    "co_payment_percent": {"type": "number", "default": 0},
                    "deductible": {"type": "number", "default": 0},
                    "sum_insured": {"type": "number"},
                },
                "required": ["claim_amount"],
            },
            func=settlement_calculator,
        )

    # -- access --------------------------------------------------------------

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}. Available: {sorted(self._specs)}")
        return self._specs[name]

    def call(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).func(**kwargs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        return [s.to_anthropic_schema() for s in self._specs.values()]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [s.to_openai_schema() for s in self._specs.values()]
