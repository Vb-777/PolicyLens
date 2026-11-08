"""Index build and load: the bridge between raw documents and retrieval.

An index is four artefacts that must stay consistent with each other — the FAISS
vectors, the chunk records, the BM25 inverted index, and (for corpus-fitted embedders)
the fitted vectoriser. They are written and loaded as a unit, and the manifest records
which embedding backend produced them so a mismatched reload fails loudly instead of
returning quietly wrong results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from .config import Settings, get_settings
from .embeddings.base import Embedder, build_embedder
from .ingest.chunker import chunk_corpus
from .ingest.loaders import load_documents
from .kg.graph import PolicyKnowledgeGraph
from .schemas import Chunk, PolicyRecord
from .store.bm25 import BM25Index
from .store.faiss_store import FaissStore

logger = logging.getLogger(__name__)

MANIFEST_FILE = "index_manifest.json"


@dataclass
class RagIndex:
    """Everything retrieval needs, loaded and consistent."""

    store: FaissStore
    bm25: BM25Index
    embedder: Embedder
    policies: list[PolicyRecord]
    kg: PolicyKnowledgeGraph
    manifest: dict

    @property
    def chunks(self) -> list[Chunk]:
        return self.store.chunks

    def policy_vocabulary(self) -> set[str]:
        """Tokens that name a policy rather than describe its content.

        Used to stop product names from being treated as content terms a clause is
        expected to contain.
        """
        from .store.bm25 import tokenize

        vocab: set[str] = set()
        for record in self.policies:
            vocab |= set(
                tokenize(
                    f"{record.display} {record.product_name} {record.insurer} {record.product_type}"
                )
            )
        return vocab

    def policy_by_id(self, policy_id: str) -> PolicyRecord | None:
        for record in self.policies:
            if record.policy_id == policy_id:
                return record
        return None


def build_index(settings: Settings | None = None, force: bool = False) -> RagIndex:
    cfg = settings or get_settings()
    corpus_dir = cfg.paths.corpus
    index_dir = cfg.paths.index

    t0 = time.perf_counter()
    documents = load_documents(corpus_dir)
    chunks, policies = chunk_corpus(documents, cfg.chunk)
    t_parse = time.perf_counter() - t0
    logger.info("parsed %d policies into %d chunks in %.2fs", len(policies), len(chunks), t_parse)

    embedder = build_embedder(cfg.embedding)
    texts = [c.for_embedding() for c in chunks]

    t1 = time.perf_counter()
    embedder.fit(texts)
    vectors = embedder.encode(texts, is_query=False)
    t_embed = time.perf_counter() - t1
    logger.info("embedded %d chunks with %s in %.2fs", len(chunks), embedder.name, t_embed)

    store = FaissStore(dim=vectors.shape[1])
    store.add(chunks, vectors)

    bm25 = BM25Index()
    # Index the heading path alongside the body. A clause states its rule in the body
    # but names it in the title — "Room Rent Sub-Limit", "AYUSH Treatment" — and
    # indexing only the body made those terms invisible to lexical search and absent
    # from the vocabulary the abstention logic consults.
    bm25.build([c.for_embedding() for c in chunks])

    kg = PolicyKnowledgeGraph()
    kg.build(documents)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)
    bm25.save(index_dir)
    embedder.save(index_dir)
    kg.save(index_dir)
    (index_dir / "policies.json").write_text(
        json.dumps([p.model_dump() for p in policies], indent=2), encoding="utf-8"
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_policies": len(policies),
        "n_chunks": len(chunks),
        "n_clauses": sum(p.n_clauses for p in policies),
        "embedder": embedder.describe(),
        "chunk_config": {
            "max_chars": cfg.chunk.max_chars,
            "overlap_chars": cfg.chunk.overlap_chars,
            "min_chars": cfg.chunk.min_chars,
        },
        "kg": kg.stats(),
        "build_seconds": {"parse": round(t_parse, 3), "embed": round(t_embed, 3)},
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return RagIndex(
        store=store, bm25=bm25, embedder=embedder, policies=policies, kg=kg, manifest=manifest
    )


def load_index(settings: Settings | None = None) -> RagIndex:
    cfg = settings or get_settings()
    index_dir = cfg.paths.index
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index in {index_dir}. Build one with `make index` or "
            "`python -m policylens.cli index`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    store = FaissStore.load(index_dir)
    bm25 = BM25Index.load(index_dir)

    # Rebuild the embedder that produced this index, not whatever the current
    # environment defaults to: a query embedded by a different model is meaningless
    # against these vectors.
    built_with = str(manifest.get("embedder", {}).get("backend", cfg.embedding.backend))
    embed_cfg = cfg.embedding
    if built_with != embed_cfg.backend and embed_cfg.backend == "auto":
        embed_cfg = type(embed_cfg)(**{**embed_cfg.__dict__, "backend": built_with})
    embedder = build_embedder(embed_cfg)
    embedder.load(index_dir)

    if embedder.name != built_with:
        raise RuntimeError(
            f"Index was built with the '{built_with}' embedding backend but the "
            f"'{embedder.name}' backend is active. Rebuild the index or set "
            f"POLICYLENS_EMBED_BACKEND={built_with}."
        )

    policies_path = index_dir / "policies.json"
    policies = [
        PolicyRecord.model_validate(p)
        for p in json.loads(policies_path.read_text(encoding="utf-8"))
    ]

    kg = PolicyKnowledgeGraph.load(index_dir)

    return RagIndex(
        store=store, bm25=bm25, embedder=embedder, policies=policies, kg=kg, manifest=manifest
    )


def load_or_build_index(settings: Settings | None = None) -> RagIndex:
    cfg = settings or get_settings()
    try:
        return load_index(cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.info("building index (%s)", exc.__class__.__name__)
        return build_index(cfg)
