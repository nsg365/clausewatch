"""Storage: Qdrant for dense vectors (+ payload filters) and a JSONL chunk store that is
the source of truth for BM25, quote verification and the contract catalog."""

from __future__ import annotations

import atexit
import json
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from clausewatch.config import get_settings
from clausewatch.retrieval.models import get_embeddings

_NS = uuid.UUID("7f1c2a52-6d0e-4a55-9c1e-2f7c1b3c9a10")


class ChunkStore:
    """Append-only JSONL of chunk records + a JSON catalog of contracts."""

    def __init__(self, data_dir: Path):
        self.path = data_dir / "chunks.jsonl"
        self.catalog_path = data_dir / "catalog.json"
        self._lock = threading.Lock()
        self._chunks: dict[str, dict] = {}
        self._catalog: dict[str, dict] = {}
        self.version = 0
        self.reload()

    def reload(self) -> None:
        self._chunks = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._chunks[rec["chunk_id"]] = rec
        self._catalog = json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else {}
        self.version += 1

    def add(self, contract: dict, chunks: list[dict]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                for c in chunks:
                    f.write(json.dumps(c) + "\n")
                    self._chunks[c["chunk_id"]] = c
            self._catalog[contract["contract_id"]] = contract
            self.catalog_path.write_text(json.dumps(self._catalog, indent=1))
            self.version += 1

    def has_contract(self, contract_id: str) -> bool:
        return contract_id in self._catalog

    def get(self, chunk_id: str) -> dict | None:
        return self._chunks.get(chunk_id)

    def all(self) -> list[dict]:
        return list(self._chunks.values())

    def contracts(self) -> dict[str, dict]:
        return dict(self._catalog)

    def contract_chunks(self, contract_id: str) -> list[dict]:
        return sorted(
            (c for c in self._chunks.values() if c["contract_id"] == contract_id),
            key=lambda c: c["char_start"],
        )


@lru_cache
def get_chunk_store() -> ChunkStore:
    return ChunkStore(get_settings().data_dir)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        client = QdrantClient(url=s.qdrant_url)
    else:
        s.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(s.qdrant_path))
    atexit.register(client.close)  # avoids a noisy __del__ during interpreter shutdown
    return client


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    s = get_settings()
    client = get_qdrant_client()
    emb = get_embeddings()
    if not client.collection_exists(s.collection):
        dim = len(emb.embed_query("dimension probe"))
        client.create_collection(
            s.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if s.qdrant_url:  # payload indexes only exist in server mode
            for field in ("metadata.contract_id", "metadata.contract_type", "metadata.clause_types"):
                client.create_payload_index(s.collection, field, models.PayloadSchemaType.KEYWORD)
    return QdrantVectorStore(client=client, collection_name=s.collection, embedding=emb)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def index_chunks(chunks: list[dict]) -> None:
    docs = [
        Document(
            page_content=f"{c['section']}\n{c['text']}",
            metadata={k: v for k, v in c.items() if k != "text"},
        )
        for c in chunks
    ]
    get_vector_store().add_documents(docs, ids=[point_id(c["chunk_id"]) for c in chunks], batch_size=64)


def build_filter(
    contract_ids: list[str] | None = None,
    clause_types: list[str] | None = None,
    contract_types: list[str] | None = None,
) -> models.Filter | None:
    must = []
    if contract_ids:
        must.append(models.FieldCondition(key="metadata.contract_id", match=models.MatchAny(any=contract_ids)))
    if clause_types:
        must.append(models.FieldCondition(key="metadata.clause_types", match=models.MatchAny(any=clause_types)))
    if contract_types:
        must.append(models.FieldCondition(key="metadata.contract_type", match=models.MatchAny(any=contract_types)))
    return models.Filter(must=must) if must else None
