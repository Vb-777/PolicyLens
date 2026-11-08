"""The agent, as an explicit LangGraph state machine.

    route -> retrieve -> [kg_lookup] -> generate -> self_check -> (retry | end)

The shape is the point. A single retrieve-then-generate chain answers a direct lookup
well and fails two cases that matter in this domain:

* **Comparisons.** "Which policy has the shorter waiting period?" needs clauses from
  several documents. Plain top-k retrieval fills the window with the best-matching
  passages, which are often several clauses from the *same* policy. The ``kg_lookup``
  node fixes this structurally: it asks the knowledge graph which policies state the
  fact at all and pulls in the clause that states it for each one.
* **Weak retrieval.** When the retrieved clauses do not support the answer, the right
  move is to retrieve again more widely, not to emit it. ``self_check`` measures
  groundedness before the answer is returned and loops once if it is below the floor.

Routing is rule-based by default so that evaluation is deterministic and the agent
behaves identically with or without an API key. Set ``POLICYLENS_ROUTER=llm`` to let
the model classify instead; the graph is unchanged either way.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..generation import Generator, content_tokens
from ..grounding import citation_validity, score_groundedness
from ..indexer import RagIndex
from ..prompts.templates import INSUFFICIENT
from ..retrieval.hybrid import HybridRetriever
from ..schemas import Answer, AnswerStatus, ScoredChunk
from .tools import ToolRegistry

Route = Literal["lookup", "comparative", "structural"]

_COMPARATIVE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|better|worse|shorter|longer|lower|higher|"
    r"cheaper|difference|differs?|which (?:of|one|policy|plan)|both)\b",
    re.IGNORECASE,
)
_STRUCTURAL_RE = re.compile(
    r"\b(list|all|every|enumerate|what are the|which policies|how many)\b", re.IGNORECASE
)


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    mentioned_policies: list[str]
    top_k: int
    contexts: list[ScoredChunk]
    kg_facts: list[dict[str, Any]]
    answer: Answer
    attempts: int
    trace: list[dict[str, Any]]
    timings: dict[str, float]


class PolicyLensAgent:
    def __init__(
        self,
        index: RagIndex,
        generator: Generator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index
        self.retriever = HybridRetriever(index, self.settings.retrieval)
        self.tools = ToolRegistry(index, self.retriever)
        # The extractive engine reuses the BM25 index's IDF table to weight how much
        # each question term matters, so it needs the index that was actually built.
        self.generator = generator or Generator(
            settings=self.settings,
            idf=index.bm25.idf,
            policy_vocab=index.policy_vocabulary(),
        )
        self.router_mode = os.getenv("POLICYLENS_ROUTER", "rules").lower()
        self.graph = self._build_graph()

    # -- graph construction ------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("route", self._node_route)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("kg_lookup", self._node_kg_lookup)
        builder.add_node("generate", self._node_generate)
        builder.add_node("self_check", self._node_self_check)

        builder.set_entry_point("route")
        builder.add_edge("route", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._should_use_kg,
            {"kg_lookup": "kg_lookup", "generate": "generate"},
        )
        builder.add_edge("kg_lookup", "generate")
        builder.add_edge("generate", "self_check")
        builder.add_conditional_edges(
            "self_check",
            self._should_retry,
            {"retrieve": "retrieve", "end": END},
        )
        return builder.compile()

    # -- nodes -------------------------------------------------------------------

    def _node_route(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        question = state["question"]

        mentioned = self._mentioned_policies(question)
        if self.router_mode == "llm" and self.generator.llm.is_generative:
            route = self._route_with_llm(question)
        else:
            route = self._route_with_rules(question, mentioned)

        top_k = self.settings.retrieval.top_k
        if route in {"comparative", "structural"}:
            top_k = max(top_k, self.settings.agent.widen_top_k)

        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "route",
                "route": route,
                "mentioned_policies": mentioned,
                "top_k": top_k,
                "router_mode": self.router_mode,
            }
        )
        timings = dict(state.get("timings", {}))
        timings["route_ms"] = (time.perf_counter() - start) * 1000

        return {
            **state,
            "route": route,
            "mentioned_policies": mentioned,
            "top_k": top_k,
            "attempts": state.get("attempts", 0),
            "trace": trace,
            "timings": timings,
        }

    def _node_retrieve(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        attempts = state.get("attempts", 0)
        top_k = state.get("top_k", self.settings.retrieval.top_k)
        # A retry is only useful if it looks at material the first pass did not.
        if attempts > 0:
            top_k = max(top_k * 2, self.settings.agent.widen_top_k)

        result = self.retriever.retrieve(state["question"], top_k=top_k)

        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "retrieve",
                "attempt": attempts + 1,
                "top_k": top_k,
                "n_results": len(result.results),
                "top_clauses": [c.chunk.chunk_id for c in result.results[:5]],
                **result.debug,
            }
        )
        timings = dict(state.get("timings", {}))
        for key, value in result.timings_ms.items():
            timings[key] = timings.get(key, 0.0) + value
        timings["retrieve_node_ms"] = timings.get("retrieve_node_ms", 0.0) + (
            time.perf_counter() - start
        ) * 1000

        return {
            **state,
            "contexts": result.results,
            "attempts": attempts + 1,
            "trace": trace,
            "timings": timings,
        }

    def _node_kg_lookup(self, state: AgentState) -> AgentState:
        """Pull in clauses the vector search missed, via the knowledge graph.

        For a comparison the graph knows which policies state the fact at all, so the
        clause that states it can be fetched for each one even when its passage did
        not rank highly.
        """
        start = time.perf_counter()
        question = state["question"]
        contexts = list(state.get("contexts", []))
        present = {c.chunk.clause_id for c in contexts}

        facts: list[dict[str, Any]] = []
        for concept in self._kg_concepts(question):
            facts.extend(self.tools.call("compare_policies", concept=concept))

        # Deduplicate by clause, keep the graph's ordering.
        seen_clauses: set[str] = set()
        added: list[str] = []
        by_clause = {c.clause_id: c for c in self.index.chunks}

        budget = max(4, self.settings.agent.widen_top_k)
        for fact in facts:
            clause_id = fact.get("clause_id")
            if not clause_id or clause_id in seen_clauses:
                continue
            seen_clauses.add(clause_id)
            if clause_id in present:
                continue
            chunk = by_clause.get(clause_id)
            if chunk is None:
                continue
            contexts.append(
                ScoredChunk(chunk=chunk, score=0.0, retriever="knowledge-graph")
            )
            added.append(clause_id)
            if len(added) >= budget:
                break

        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "kg_lookup",
                "concepts": self._kg_concepts(question),
                "n_facts": len(facts),
                "clauses_added": added,
            }
        )
        timings = dict(state.get("timings", {}))
        timings["kg_ms"] = (time.perf_counter() - start) * 1000

        return {**state, "contexts": contexts, "kg_facts": facts, "trace": trace, "timings": timings}

    def _clause_tokens(self, chunk) -> set[str]:
        """Clause tokens including its headings.

        A clause states its rule in the body but names the provision in its title, so a
        question phrased in the provision's name ("no claim bonus") matches the heading
        even when the body never repeats the phrase.
        """
        return content_tokens(
            f"{chunk.meta.section_title} {chunk.meta.clause_title} {chunk.text}"
        )

    def _provision_labels(self, policy_id: str) -> list[str]:
        """The canonical names of every provision the policy contains.

        The knowledge graph types each clause it can interpret as a Benefit, Exclusion,
        Condition, WaitingPeriod or Definition, and the label is the provision's name as
        the document itself gives it ("No Claim Bonus", "Grace Period", "Room Rent
        Sub-Limit"). Taken together these form a provision *inventory* for the policy,
        and an inventory is the one thing that makes absence checkable: full-text search
        can only tell you a phrase was not found, which is equally true of a provision
        the policy lacks and a provision it paraphrases.
        """
        labels: list[str] = []
        for node_type in ("Benefit", "Exclusion", "Condition", "WaitingPeriod", "Definition"):
            labels.extend(n.get("label", "") for n in self.index.kg.nodes_of_type(node_type, policy_id))
        labels.extend(
            c.meta.clause_title for c in self.index.chunks if c.meta.policy_id == policy_id
        )
        return [label for label in labels if label]

    def provision_support(self, question: str, policy_id: str) -> tuple[float, float]:
        """How well the question matches a provision named in ``policy_id``, versus elsewhere.

        Matching against short canonical labels rather than clause bodies is what makes
        this usable: a clause body shares the corpus's common vocabulary with every
        other clause, so body coverage is high everywhere and discriminates nothing,
        while a label is a few deliberate words naming one provision.
        """
        engine = self.generator.extractive
        terms = engine.query_terms(question)
        if not terms:
            return 0.0, 0.0
        best_in = best_out = 0.0
        for policy in self.index.policies:
            best = 0.0
            for label in self._provision_labels(policy.policy_id):
                best = max(best, engine.weighted_coverage(terms, content_tokens(label)))
            if policy.policy_id == policy_id:
                best_in = best
            else:
                best_out = max(best_out, best)
        return best_in, best_out

    def policy_topical_support(self, question: str, policy_id: str) -> tuple[float, float]:
        """Best topical coverage of ``question`` inside ``policy_id`` versus outside it.

        Returns ``(best_inside, best_outside)``. Both sides are scanned exhaustively
        over the corpus rather than over the retrieved window: the whole point is to
        establish that *no* clause in the named policy addresses the question, and a
        retrieved sample cannot establish an absence. The corpus is small enough that
        an exhaustive scan costs well under a millisecond.
        """
        engine = self.generator.extractive
        terms = engine.query_terms(question)
        if not terms:
            return 0.0, 0.0
        best_in = best_out = 0.0
        for chunk in self.index.chunks:
            cov = engine.weighted_coverage(terms, self._clause_tokens(chunk))
            if chunk.meta.policy_id == policy_id:
                best_in = max(best_in, cov)
            else:
                best_out = max(best_out, cov)
        return best_in, best_out

    def _policy_guard(self, state: AgentState) -> str | None:
        """Decline a question about a provision the named policy does not contain.

        "What is the no claim bonus under the Griha Raksha householder's policy?" is
        unanswerable: the householder's policy has no such benefit, but the motor and
        health policies describe one at length, so retrieval returns confident-looking
        clauses from the wrong document.

        An earlier version of this guard asked only whether any top-ranked clause came
        *from* the named policy. That check does not work, because provenance is not
        topicality: some clause of the right policy almost always ranks well on the
        shared vocabulary of the question ("policy", "period", "claim"), and its
        presence was read as evidence the policy covers the provision. On this corpus
        it passed every case it was written to catch.

        What settles the question is whether the named policy contains any clause that
        covers what is being asked about, measured against what the rest of the corpus
        manages on the same question. A concept that scores poorly inside the named
        policy and markedly better outside it is a concept that lives elsewhere.
        """
        if not self.settings.agent.enable_policy_guard:
            return None
        mentioned = state.get("mentioned_policies") or []
        if len(mentioned) != 1:
            return None
        target = mentioned[0]
        question = state["question"]
        cfg = self.settings.agent

        # Two signals have to agree, because each alone misfires in a different way.
        #
        # The provision margin says the question names a provision that some *other*
        # policy carries far better than this one. Alone it would decline "does the term
        # plan have a maturity value?", where no provision is labelled "Maturity Value"
        # but the clause body answers the question directly.
        #
        # The body floor says no clause anywhere in the named policy covers the
        # question's rare terms. Alone it would decline any question the policy answers
        # in its own vocabulary, which is most of them: "report a claim" against a clause
        # that says "intimate".
        #
        # Requiring both means declining only when the named policy neither names the
        # provision nor discusses it, while another policy plainly does.
        prov_in, prov_out = self.provision_support(question, target)
        if (prov_out - prov_in) < cfg.policy_guard_margin:
            return None
        body_in, _ = self.policy_topical_support(question, target)
        if body_in > cfg.policy_guard_body_floor:
            return None
        record = self.index.policy_by_id(target)
        return (record.display or target) if record else target

    def _node_generate(self, state: AgentState) -> AgentState:
        blocked_policy = self._policy_guard(state)
        if blocked_policy:
            answer = Answer(
                question=state["question"],
                answer=(
                    f"{INSUFFICIENT}. No clause in {blocked_policy} addresses this; the "
                    "closest matching clauses belong to other policies."
                ),
                status=AnswerStatus.ABSTAINED,
                llm_provider=self.generator.llm.provider,
                self_check={"engine": "policy-guard", "blocked_policy": blocked_policy},
            )
            answer.contexts = state.get("contexts", [])
            trace = list(state.get("trace", []))
            trace.append({"node": "generate", "engine": "policy-guard", "status": "abstained"})
            return {**state, "answer": answer, "trace": trace}

        answer = self.generator.generate(state["question"], state.get("contexts", []))
        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "generate",
                "engine": answer.self_check.get("engine", "unknown"),
                "status": answer.status.value,
                "n_citations": len(answer.citations),
            }
        )
        timings = dict(state.get("timings", {}))
        for key, value in answer.timings_ms.items():
            timings[key] = timings.get(key, 0.0) + value
        return {**state, "answer": answer, "trace": trace, "timings": timings}

    def _node_self_check(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        answer: Answer = state["answer"]
        contexts = state.get("contexts", [])

        report = score_groundedness(answer.answer, contexts)
        citations = citation_validity(answer, contexts)

        answer.self_check.update(
            {
                **report.to_dict(),
                **citations,
                "attempts": state.get("attempts", 1),
            }
        )

        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "self_check",
                "groundedness": round(report.score, 4),
                "numeric_fidelity": round(report.numeric_fidelity, 4),
                "citation_precision": citations["citation_precision"],
            }
        )
        timings = dict(state.get("timings", {}))
        timings["self_check_ms"] = (time.perf_counter() - start) * 1000

        return {**state, "answer": answer, "trace": trace, "timings": timings}

    # -- edges -------------------------------------------------------------------

    def _should_use_kg(self, state: AgentState) -> str:
        if not self.settings.agent.enable_kg:
            return "generate"
        if state.get("attempts", 1) > 1:
            # Already expanded once; a second graph pass adds nothing.
            return "generate"
        return "kg_lookup" if state.get("route") in {"comparative", "structural"} else "generate"

    def _should_retry(self, state: AgentState) -> str:
        cfg = self.settings.agent
        if not cfg.enable_self_check:
            return "end"
        answer: Answer = state["answer"]
        if answer.status is AnswerStatus.ERROR:
            return "end"
        # An explicit abstention is a correct outcome, not a retrieval failure to fix.
        if answer.status is AnswerStatus.ABSTAINED:
            return "end"
        if state.get("attempts", 1) > cfg.max_retrieval_retries:
            return "end"
        groundedness = float(answer.self_check.get("groundedness", 1.0))
        return "retrieve" if groundedness < cfg.groundedness_floor else "end"

    # -- routing helpers ---------------------------------------------------------

    _NAME_NOISE = {
        "insurance", "policy", "plan", "package", "cover", "assurance",
        "health", "life", "term", "private", "overseas", "travel", "motor",
        "householder", "householders", "car",
    }

    def _mentioned_policies(self, question: str) -> list[str]:
        """Policies named in the question.

        Matching on any long word in the product name is too loose: "Aarogya Suraksha"
        and "Suraksha Kavach" share a word, so a question about one looked like a
        question about both and was wrongly routed as a comparison. Matching on
        distinctive *phrases* — the full name and its leading bigram — disambiguates
        them, with a single distinctive word as a last resort.
        """
        lowered = question.lower()
        hits: list[str] = []
        for record in self.index.policies:
            name = (record.display or record.product_name).lower()
            words = [w for w in re.findall(r"[a-z]{3,}", name) if w not in self._NAME_NOISE]
            phrases: list[str] = []
            if len(words) >= 2:
                phrases.append(" ".join(words[:2]))
            phrases.append(name)
            if len(words) == 1:
                phrases.append(words[0])

            if any(p in lowered for p in phrases):
                hits.append(record.policy_id)
                continue

            # Fall back to a word unique to this policy across the whole corpus.
            unique = [
                w
                for w in words
                if sum(
                    1
                    for other in self.index.policies
                    if w in (other.display or other.product_name).lower()
                )
                == 1
            ]
            if any(w in lowered for w in unique):
                hits.append(record.policy_id)
        return hits

    def _route_with_rules(self, question: str, mentioned: list[str]) -> Route:
        if _COMPARATIVE_RE.search(question) or len(mentioned) > 1:
            return "comparative"
        if _STRUCTURAL_RE.search(question):
            return "structural"
        return "lookup"

    def _route_with_llm(self, question: str) -> Route:
        system = (
            "Classify an insurance question into exactly one of: lookup, comparative, "
            "structural. lookup = answerable from one clause of one policy. "
            "comparative = needs facts from more than one policy. structural = asks to "
            "list or count facts across policies. Reply with the single word only."
        )
        try:
            response = self.generator.llm.complete(system, question)
            label = response.text.strip().lower().split()[0]
            if label in {"lookup", "comparative", "structural"}:
                return label  # type: ignore[return-value]
        except Exception:  # noqa: BLE001 - routing must never break the request
            pass
        return self._route_with_rules(question, self._mentioned_policies(question))

    def _kg_concepts(self, question: str) -> list[str]:
        """Question tokens that name something the knowledge graph actually knows."""
        tokens = [t for t in content_tokens(question) if len(t) > 3]
        known: list[str] = []
        for token in tokens:
            if self.index.kg.find(token):
                known.append(token)
        return known[:4]

    # -- public ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        final = self.graph.invoke(
            {"question": question, "attempts": 0, "trace": [], "timings": {}}
        )
        answer: Answer = final["answer"]
        answer.trace = final.get("trace", [])
        answer.timings_ms = {**final.get("timings", {}), **answer.timings_ms}
        answer.timings_ms["total_ms"] = (time.perf_counter() - start) * 1000
        answer.contexts = final.get("contexts", [])
        return answer

    def mermaid(self) -> str:
        """Graph topology as Mermaid, for the README and for debugging."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:  # noqa: BLE001
            return "graph TD; route-->retrieve-->kg_lookup-->generate-->self_check;"


def build_agent(index: RagIndex, settings: Settings | None = None) -> PolicyLensAgent:
    cfg = settings or get_settings()
    return PolicyLensAgent(
        index=index,
        generator=Generator(
            settings=cfg, idf=index.bm25.idf, policy_vocab=index.policy_vocabulary()
        ),
        settings=cfg,
    )
