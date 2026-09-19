"""Hybrid retrieval: metadata pre-filter -> dense (Qdrant) + sparse (BM25) -> RRF fusion
-> cross-encoder rerank.

Metadata filters are *soft*: if the clause-type filter leaves too few candidates (the
keyword tagger missed something), we relax it and keep only the contract filter.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from clausewatch.config import get_settings
from clausewatch.retrieval.models import get_reranker
from clausewatch.retrieval.store import build_filter, get_chunk_store, get_vector_store

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in for on by with as at be is are was were this that shall will may any all "
    "such its it from under which party parties agreement".split()
)


_SUFFIXES = ("ations", "ation", "ings", "ing", "ments", "ment", "able", "ible", "ally", "ated", "ates",
             "ate", "als", "al", "ies", "ied", "es", "ed", "s")


def _stem(tok: str) -> str:
    """Tiny suffix stripper so 'renewal'/'renew'/'renewable' and 'terminate'/'termination' match."""
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


_GENERIC = {"agreement", "agreements", "contract", "contracts", "clause", "clauses", "provision", "provisions",
            "inc", "corp", "ltd", "llc", "the", "s"}


def strip_contract_names(query: str, titles: list[str]) -> str:
    """Remove the scoped contracts' company names / document types from a query.

    Once retrieval is filtered to a contract, every chunk belongs to it, so name tokens
    add no signal and pull ranking towards title pages. Handles squashed EDGAR names:
    'Entertainment Gaming Asia' matches 'ENTERTAINMENTGAMINGASIAINC'."""
    keys, title_words = [], set()
    for t in titles:
        company, _, doctype = t.partition(" - ")
        keys.append(re.sub(r"[^a-z0-9]", "", company.lower()))
        title_words |= set(_TOKEN.findall(t.lower()))
    words = re.findall(r"\S+", query)
    norm = [re.sub(r"[^a-z0-9]", "", w.lower().replace("'s", "")) for w in words]
    drop = [False] * len(words)
    i = 0
    while i < len(words):
        best = 0
        for j in range(i + 1, min(len(words), i + 6) + 1):
            joined = "".join(norm[i:j])
            if len(joined) >= 4 and any(k.startswith(joined) or (len(joined) >= 6 and joined in k) for k in keys):
                best = j
        if best:
            for k in range(i, best):
                drop[k] = True
            i = best
        else:
            i += 1
    kept = [w for w, d, n in zip(words, drop, norm)
            if not d and n not in _GENERIC and not (n in title_words and len(n) > 3 and n not in _CLAUSE_WORDS)]
    out = " ".join(kept).strip(" ,.;:")
    return out if len(out) >= 3 else query


# Title words that are also meaningful clause vocabulary are kept.
_CLAUSE_WORDS = {"license", "licence", "services", "service", "supply", "maintenance", "support", "development",
                 "hosting", "exclusivity", "termination", "insurance", "indemnification"}


@dataclass
class Hit:
    chunk_id: str
    text: str
    meta: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None
    scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "contract_id": self.meta["contract_id"],
            "contract_title": self.meta["contract_title"],
            "section": self.meta["section"],
            "page_start": self.meta["page_start"],
            "page_end": self.meta["page_end"],
            "clause_types": self.meta.get("clause_types", []),
            "text": self.text,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf": round(self.rrf, 5),
            "rerank": None if self.rerank is None else round(self.rerank, 4),
        }


class _BM25Index:
    """BM25 over the chunk store, rebuilt lazily when the store changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version = -1
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def search(self, query: str, allowed: set[str] | None, k: int) -> list[str]:
        store = get_chunk_store()
        with self._lock:
            if self._version != store.version:
                chunks = store.all()
                self._ids = [c["chunk_id"] for c in chunks]
                corpus = [tokenize(f"{c['section']} {c['text']}") for c in chunks]
                self._bm25 = BM25Okapi(corpus) if corpus else None
                self._version = store.version
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (i for i in range(len(self._ids)) if allowed is None or self._ids[i] in allowed),
            key=lambda i: -scores[i],
        )
        return [self._ids[i] for i in ranked[:k] if scores[i] > 0]


_bm25 = _BM25Index()


def _allowed_ids(contract_ids, clause_types, contract_types) -> set[str] | None:
    if not (contract_ids or clause_types or contract_types):
        return None
    out = set()
    for c in get_chunk_store().all():
        if contract_ids and c["contract_id"] not in contract_ids:
            continue
        if contract_types and c["contract_type"] not in contract_types:
            continue
        if clause_types and not set(c.get("clause_types", [])) & set(clause_types):
            continue
        out.add(c["chunk_id"])
    return out


def hybrid_search(
    query: str,
    *,
    contract_ids: list[str] | None = None,
    clause_types: list[str] | None = None,
    contract_types: list[str] | None = None,
    top_n: int | None = None,
    rerank: bool = True,
    rerank_query: str | None = None,
    mode: str = "hybrid",  # "hybrid" | "dense" | "sparse" (for ablations)
    min_candidates: int = 4,
) -> tuple[list[Hit], dict]:
    s = get_settings()
    top_n = top_n or s.rerank_top_n
    store = get_chunk_store()
    info = {"query": query, "filters": {"contract_ids": contract_ids, "clause_types": clause_types,
                                        "contract_types": contract_types}, "relaxed": False, "mode": mode}

    if contract_ids:
        catalog = store.contracts()
        titles = [catalog[c]["title"] for c in contract_ids if c in catalog]
        query = strip_contract_names(query, titles)
        if rerank_query:
            rerank_query = strip_contract_names(rerank_query, titles)
        info["search_text"] = query

    # Contract / contract-type filters are hard. The clause-type filter is applied as extra
    # *filtered* ranked lists fused alongside the unfiltered ones: tagged chunks get boosted,
    # but a chunk the keyword tagger missed can still be retrieved.
    allowed = _allowed_ids(contract_ids, None, contract_types)
    allowed_clause = _allowed_ids(contract_ids, clause_types, contract_types) if clause_types else None
    if clause_types and len(allowed_clause or ()) < min_candidates:
        info["relaxed"] = True
        clause_types, allowed_clause = None, None

    hits: dict[str, Hit] = {}
    ranks: dict[str, list[int]] = {}

    def _hit(cid: str) -> Hit:
        if cid not in hits:
            rec = store.get(cid)
            hits[cid] = Hit(cid, rec["text"], {k: v for k, v in rec.items() if k != "text"})
            ranks[cid] = []
        return hits[cid]

    lists = [(None, allowed)] + ([(clause_types, allowed_clause)] if clause_types else [])
    for ctypes, allowed_ids in lists:
        boosted = ctypes is not None
        if mode in ("hybrid", "dense"):
            flt = build_filter(contract_ids, ctypes, contract_types)
            dense = get_vector_store().similarity_search_with_score(query, k=s.dense_k, filter=flt)
            for rank, (doc, score) in enumerate(dense, 1):
                h = _hit(doc.metadata["chunk_id"])
                ranks[h.chunk_id].append(rank)
                if not boosted:
                    h.dense_rank, h.scores["dense"] = rank, float(score)
        if mode in ("hybrid", "sparse"):
            for rank, cid in enumerate(_bm25.search(query, allowed_ids, s.sparse_k), 1):
                h = _hit(cid)
                ranks[cid].append(rank)
                if not boosted:
                    h.sparse_rank = rank

    for cid, h in hits.items():
        h.rrf = sum(1.0 / (s.rrf_k + r) for r in ranks[cid])
    fused = sorted(hits.values(), key=lambda h: -h.rrf)[: s.fused_k]
    info["n_candidates"] = len(fused)

    if rerank and fused:
        scores = get_reranker().score(rerank_query or query, [f"{h.meta['section']}\n{h.text}" for h in fused])
        for h, sc in zip(fused, scores):
            h.rerank = sc
        fused.sort(key=lambda h: -h.rerank)
    return fused[:top_n], info


def per_contract_search(query: str, contract_ids: list[str], per_contract: int = 3, **kw) -> tuple[list[Hit], dict]:
    """Retrieve a quota from each contract so comparisons see every document."""
    out, infos = [], []
    for cid in contract_ids:
        hits, info = hybrid_search(query, contract_ids=[cid], top_n=per_contract, **kw)
        out.extend(hits)
        infos.append(info)
    return out, {"per_contract": infos}
