"""FastAPI backend used by the Streamlit UI (single process owns the vector store).

    uvicorn clausewatch.api:app --port 8000
"""

from __future__ import annotations

import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from clausewatch.agent.graph import get_graph, run, scan_risks
from clausewatch.config import get_settings
from clausewatch.ingest.pipeline import ingest_pdf
from clausewatch.retrieval.hybrid import hybrid_search
from clausewatch.retrieval.store import get_chunk_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model, reranker and BM25 index before the first request.
    hybrid_search("warm up", top_n=1)
    yield


app = FastAPI(title="ClauseWatch", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    contract_ids: list[str] = []
    options: dict = {}


class CompareRequest(BaseModel):
    clause: str
    contract_ids: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "contracts": len(get_chunk_store().contracts()), "model": get_settings().llm_model}


@app.get("/contracts")
def contracts():
    return list(get_chunk_store().contracts().values())


@app.post("/ask")
def ask(req: AskRequest):
    return run(req.question, req.contract_ids, req.options)


@app.post("/compare")
def compare(req: CompareRequest):
    if len(req.contract_ids) < 2:
        raise HTTPException(400, "need at least two contract_ids")
    q = f"Compare the {req.clause} provisions across these contracts and explain the key differences."
    return run(q, req.contract_ids, {"force_query_type": "comparison"})


@app.post("/risks/{contract_id}")
def risks(contract_id: str):
    if contract_id not in get_chunk_store().contracts():
        raise HTTPException(404, f"unknown contract {contract_id}")
    return scan_risks(contract_id)


@app.post("/upload")
def upload(file: UploadFile = File(...), contract_type: str = Form("Unknown")):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF uploads are supported")
    name = re.sub(r"[^\w.\- ]", "_", Path(file.filename).name)
    dest = get_settings().data_dir / "uploads" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return ingest_pdf(dest, title=dest.stem, contract_type=contract_type, source="upload")
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/graph")
def graph():
    return {"mermaid": get_graph().get_graph().draw_mermaid()}
