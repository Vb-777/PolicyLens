"""FastAPI service.

The index, agent and models are loaded once at startup rather than per request: on
this corpus loading costs more than answering does, and doing it per request would
make the latency figures in the evaluation report unrepresentative of the service.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..agent.graph import PolicyLensAgent, build_agent
from ..config import Settings, get_settings
from ..indexer import RagIndex, load_or_build_index
from ..schemas import AnswerStatus

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("loading index...")
    index = load_or_build_index(settings)
    agent = build_agent(index, settings)
    _state.update({"settings": settings, "index": index, "agent": agent, "started": time.time()})
    logger.info(
        "ready: %d chunks, %d policies, generation=%s",
        len(index.chunks),
        len(index.policies),
        agent.generator.llm.provider,
    )
    yield
    _state.clear()


app = FastAPI(
    title="PolicyLens",
    version="0.1.0",
    description=(
        "Grounded question answering over insurance policy wordings. Every answer "
        "cites the clauses it came from, and the service reports its own "
        "groundedness so a caller can reject an answer it does not trust."
    ),
    lifespan=lifespan,
)


def get_agent() -> PolicyLensAgent:
    agent = _state.get("agent")
    if agent is None:  # pragma: no cover - only before startup completes
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return agent


def get_index() -> RagIndex:
    index = _state.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return index


# -- schemas ---------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    include_contexts: bool = False
    include_trace: bool = False


class CitationOut(BaseModel):
    marker: str
    clause_id: str
    policy_id: str
    clause_number: str
    label: str
    quote: str


class ContextOut(BaseModel):
    clause_id: str
    policy_id: str
    clause_title: str
    score: float
    retriever: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    abstained: bool
    citations: list[CitationOut]
    groundedness: float | None = None
    numeric_fidelity: float | None = None
    citation_precision: float | None = None
    engine: str
    latency_ms: float
    contexts: list[ContextOut] | None = None
    trace: list[dict[str, Any]] | None = None


class SearchHit(BaseModel):
    clause_id: str
    policy_id: str
    policy: str
    clause_number: str
    clause_title: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None
    text: str


# -- routes ----------------------------------------------------------------------


@app.get("/health", summary="Liveness and readiness.")
def health() -> dict[str, Any]:
    index: RagIndex | None = _state.get("index")
    agent: PolicyLensAgent | None = _state.get("agent")
    if index is None or agent is None:
        raise HTTPException(status_code=503, detail="starting")
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _state.get("started", time.time()), 1),
        "n_chunks": len(index.chunks),
        "n_policies": len(index.policies),
        "embedder": index.embedder.describe(),
        "llm_provider": agent.generator.llm.provider,
        "llm_model": agent.generator.llm.model,
    }


@app.get("/policies", summary="Indexed policies.")
def policies(index: RagIndex = Depends(get_index)) -> list[dict[str, Any]]:
    return [p.model_dump() for p in index.policies]


@app.post("/ask", response_model=AskResponse, summary="Ask a grounded question.")
def ask(request: AskRequest, agent: PolicyLensAgent = Depends(get_agent)) -> AskResponse:
    start = time.perf_counter()
    answer = agent.answer(request.question)
    latency = (time.perf_counter() - start) * 1000

    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        status=answer.status.value,
        abstained=answer.status is AnswerStatus.ABSTAINED,
        citations=[CitationOut(**c.model_dump()) for c in answer.citations],
        groundedness=answer.self_check.get("groundedness"),
        numeric_fidelity=answer.self_check.get("numeric_fidelity"),
        citation_precision=answer.self_check.get("citation_precision"),
        engine=str(answer.self_check.get("engine", "unknown")),
        latency_ms=round(latency, 2),
        contexts=[
            ContextOut(
                clause_id=c.chunk.clause_id,
                policy_id=c.chunk.meta.policy_id,
                clause_title=c.chunk.meta.clause_title,
                score=round(c.score, 4),
                retriever=c.retriever,
                text=c.chunk.text,
            )
            for c in answer.contexts
        ]
        if request.include_contexts
        else None,
        trace=answer.trace if request.include_trace else None,
    )


@app.get("/search", response_model=list[SearchHit], summary="Retrieval without generation.")
def search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=50),
    policy_id: str | None = None,
    agent: PolicyLensAgent = Depends(get_agent),
) -> list[SearchHit]:
    result = agent.retriever.retrieve(
        q, top_k=top_k, policy_filter=[policy_id] if policy_id else None
    )
    return [
        SearchHit(
            clause_id=c.chunk.clause_id,
            policy_id=c.chunk.meta.policy_id,
            policy=c.chunk.meta.policy_display,
            clause_number=c.chunk.meta.clause_number,
            clause_title=c.chunk.meta.clause_title,
            score=round(c.score, 4),
            dense_rank=c.dense_rank,
            lexical_rank=c.lexical_rank,
            text=c.chunk.text,
        )
        for c in result.results
    ]


@app.get("/kg/stats", summary="Knowledge graph shape.")
def kg_stats(index: RagIndex = Depends(get_index)) -> dict[str, Any]:
    return index.kg.stats()


@app.get("/kg/compare", summary="One fact across every policy that states it.")
def kg_compare(
    concept: str = Query(..., min_length=2), index: RagIndex = Depends(get_index)
) -> list[dict[str, Any]]:
    rows = index.kg.compare_across_policies(concept)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No graph facts matched '{concept}'.")
    return rows


@app.get("/metrics", summary="Latest evaluation report, if one has been run.")
def metrics(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    import json

    path = settings.paths.reports / "latest.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report yet. Run `python -m policylens.cli eval`.",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "generated_at": payload.get("generated_at"),
        "variant": payload.get("variant"),
        "dataset": payload.get("dataset"),
        "aggregate": payload.get("aggregate"),
        "environment": payload.get("environment"),
    }
