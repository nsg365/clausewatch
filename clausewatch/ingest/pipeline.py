"""Ingest a contract PDF: chunk -> tag clause types -> (CUAD only) attach gold labels -> index."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from clausewatch.config import get_settings
from clausewatch.ingest.chunking import chunk_pages, read_pdf_pages
from clausewatch.retrieval.store import get_chunk_store, index_chunks
from clausewatch.taxonomy import CUAD_LABEL_MAP, tag_clauses
from clausewatch.textutil import norm, slugify

log = logging.getLogger(__name__)


def pretty_title(title: str) -> str:
    """'ArmstrongFlooringInc_20190107_8-K_EX-10.2_..._Intellectual Property Agreement'
    -> 'ArmstrongFlooringInc - Intellectual Property Agreement'."""
    if "_" not in title:
        return title
    company = title.split("_", 1)[0]
    tail = re.split(r"[_]|-EX-[\w.()]+-", title)[-1]
    tail = re.sub(r"^(EX-)?[\d.]+-?", "", tail).strip(" -_")
    return f"{company} - {tail.title() if tail.isupper() else tail}"


def _gold_types(chunk_text: str, labels: dict[str, list[str]]) -> list[str]:
    hay = norm(chunk_text)
    out = set()
    for cuad_cat, spans in labels.items():
        cat = CUAD_LABEL_MAP.get(cuad_cat)
        if not cat:
            continue
        for span in spans:
            probe = norm(span)[:120]
            if len(probe) >= 15 and probe in hay:
                out.add(cat)
                break
    return sorted(out)


def ingest_pdf(
    pdf_path: str | Path,
    *,
    title: str | None = None,
    contract_type: str = "Unknown",
    source: str = "upload",
    gold_labels: dict[str, list[str]] | None = None,
    force: bool = False,
) -> dict:
    s = get_settings()
    store = get_chunk_store()
    pdf_path = Path(pdf_path)
    title = title or pdf_path.stem
    contract_id = slugify(title)
    if store.has_contract(contract_id) and not force:
        log.info("skip %s (already ingested)", contract_id)
        return store.contracts()[contract_id]

    pages = read_pdf_pages(pdf_path)
    if sum(len(p) for p in pages) < 500:
        raise ValueError(f"{pdf_path.name}: no extractable text (scanned PDF? OCR is out of scope)")
    display = pretty_title(title)
    raw_chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)

    records = []
    for i, ch in enumerate(raw_chunks):
        rec = {
            "chunk_id": f"{contract_id}::{i:04d}",
            "contract_id": contract_id,
            "contract_title": display,
            "contract_type": contract_type,
            "source": source,
            "section": ch.section,
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "char_start": ch.char_start,
            "clause_types": tag_clauses(ch.text, ch.section),
            "text": ch.text,
        }
        if gold_labels is not None:
            rec["gold_clause_types"] = _gold_types(ch.text, gold_labels)
        records.append(rec)

    contract = {
        "contract_id": contract_id,
        "title": display,
        "raw_title": title,
        "contract_type": contract_type,
        "source": source,
        "n_pages": len(pages),
        "n_chunks": len(records),
        "pdf_path": str(pdf_path),
    }
    index_chunks(records)
    store.add(contract, records)
    log.info("ingested %s: %d pages, %d chunks", display, len(pages), len(records))
    return contract


def ingest_cuad_subset(n: int = 30, n_holdout: int = 3) -> list[dict]:
    from clausewatch.ingest import cuad

    manifest = cuad.raw_dir() / "subset.json"
    if manifest.exists():
        corpus, holdout = cuad.load_manifest()
    else:
        corpus, holdout = cuad.select_subset(n, n_holdout)
        cuad.save_manifest(corpus, holdout)
    out = []
    for c in corpus:
        pdf = cuad.download_pdf(c)
        out.append(
            ingest_pdf(pdf, title=c.title, contract_type=c.contract_type, source="cuad", gold_labels=c.labels)
        )
    for c in holdout:  # fetched but NOT indexed: used as "new contract" demos
        cuad.download_pdf(c)
    return out
