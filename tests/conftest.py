"""Offline test fixtures: an isolated data dir with a tiny synthetic contract, and a
scripted fake LLM so the graph can be exercised without API calls."""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="clausewatch-test-")
os.environ["CLAUSEWATCH_DATA_DIR"] = _TMP
os.environ["CLAUSEWATCH_QDRANT_PATH"] = os.path.join(_TMP, "qdrant")
os.environ["CLAUSEWATCH_CATALOG_PATH"] = os.path.join(_TMP, "catalog.json")
os.environ["CLAUSEWATCH_COLLECTION"] = "test_contracts"
os.environ.setdefault("GROQ_API_KEY", "gsk_offline_test_key")  # clients are built but never called

PAGES = [
    """MASTER SUPPLY AGREEMENT
This Master Supply Agreement is entered into by and between Acme Corp ("Supplier") and Beta LLC ("Buyer").
1. Term
The initial term of this Agreement is three (3) years from the Effective Date. This Agreement shall
automatically renew for successive one (1) year terms unless either party gives notice of non-renewal
at least ninety (90) days before the end of the then-current term.
""",
    """2. Termination for Convenience
Buyer may terminate this Agreement at any time for any reason upon thirty (30) days written notice to Supplier.
Supplier may not terminate this Agreement for convenience.
3. Indemnification
Supplier shall indemnify, defend and hold harmless Buyer from any and all claims, losses and damages
arising out of the products. Buyer has no indemnification obligations under this Agreement.
4. Governing Law
This Agreement shall be governed by the laws of the State of Delaware.
""",
]


@pytest.fixture(scope="session")
def indexed_contract():
    from clausewatch.ingest.chunking import chunk_pages
    from clausewatch.retrieval.store import get_chunk_store, index_chunks
    from clausewatch.taxonomy import tag_clauses

    cid = "acme-beta-supply"
    recs = []
    for i, ch in enumerate(chunk_pages(PAGES, chunk_size=400, overlap=50)):
        recs.append({
            "chunk_id": f"{cid}::{i:04d}", "contract_id": cid, "contract_title": "Acme - Supply Agreement",
            "contract_type": "Supply", "source": "test", "section": ch.section, "page_start": ch.page_start,
            "page_end": ch.page_end, "char_start": ch.char_start,
            "clause_types": tag_clauses(ch.text, ch.section), "text": ch.text,
        })
    index_chunks(recs)
    get_chunk_store().add({"contract_id": cid, "title": "Acme - Supply Agreement", "contract_type": "Supply",
                           "source": "test", "n_pages": 2, "n_chunks": len(recs), "pdf_path": ""}, recs)
    return cid


class FakeLLM:
    """Returns scripted objects keyed by schema name; lists are consumed in order."""

    def __init__(self, script: dict):
        self.script = {k: list(v) if isinstance(v, list) else v for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, schema, system, user, **kw):
        self.calls.append((schema.__name__, user))
        item = self.script[schema.__name__]
        if isinstance(item, list):
            item = item.pop(0) if len(item) > 1 else item[0]
        return schema(**item(user)) if callable(item) else schema(**item)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(script):
        fake = FakeLLM(script)
        monkeypatch.setattr("clausewatch.llm.structured_call", fake)
        return fake

    return install
