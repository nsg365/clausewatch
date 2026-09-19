"""ClauseWatch MCP server.

Run standalone (stdio, for Claude Desktop / any MCP client):
    python -m clausewatch.mcp_server
or over HTTP:
    python -m clausewatch.mcp_server --http --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from clausewatch.agent.graph import run, scan_risks
from clausewatch.retrieval.hybrid import hybrid_search
from clausewatch.retrieval.store import get_chunk_store
from clausewatch.taxonomy import CLAUSE_CATEGORIES

# stdio transport owns stdout; send logs to stderr.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s %(message)s")
for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

mcp = MCPServer(
    "clausewatch",
    instructions=(
        "Contract & compliance review over an indexed corpus of commercial contracts (CUAD). "
        "Call list_contracts first to get contract IDs. Answers are verified against source clauses "
        "and carry citations (contract, section, page); do not add facts beyond what the tools return."
    ),
)


class Citation(BaseModel):
    n: int
    contract_id: str
    contract_title: str
    section: str
    page_start: int
    page_end: int
    quote: str = Field(description="Verbatim supporting text from the contract")


class ContractInfo(BaseModel):
    contract_id: str
    title: str
    contract_type: str
    n_pages: int
    source: str


class QueryResult(BaseModel):
    answer: str = Field(description="Answer with inline [n] citation markers")
    query_type: str
    citations: list[Citation]
    verification_verdict: str = Field(description="pass | pass_pruned | fail | insufficient")
    supported_claims: int
    total_claims: int
    retries: int


class RiskFlag(BaseModel):
    rank: int
    pattern_id: str
    name: str
    severity: Literal["high", "medium", "low"]
    confidence: float
    rationale: str
    quote: str
    section: str
    page_start: int
    page_end: int


class RiskReport(BaseModel):
    contract_id: str
    contract_title: str
    flags: list[RiskFlag]
    rejected_by_verifier: int
    summary: str


class ClauseHit(BaseModel):
    contract_id: str
    contract_title: str
    section: str
    page_start: int
    page_end: int
    clause_types: list[str]
    rerank_score: float | None
    text: str


def _query_result(r: dict) -> QueryResult:
    claims = r["verification"].get("claims", [])
    return QueryResult(
        answer=r["answer"],
        query_type=r["query_type"] or "",
        citations=[Citation(**{k: c[k] for k in Citation.model_fields}) for c in r["citations"]],
        verification_verdict=r["verification"].get("verdict", "n/a"),
        supported_claims=sum(c["status"] == "supported" for c in claims),
        total_claims=len(claims),
        retries=r["verification"].get("attempt", 0),
    )


def _require(contract_ids: list[str]) -> None:
    catalog = get_chunk_store().contracts()
    unknown = [c for c in contract_ids if c not in catalog]
    if unknown:
        raise ToolError(f"Unknown contract_id(s): {unknown}. Call list_contracts for valid IDs.")


@mcp.tool()
def list_contracts() -> list[ContractInfo]:
    """List all indexed contracts with their IDs, titles and contract types."""
    return [ContractInfo(**{k: c[k] for k in ContractInfo.model_fields}) for c in get_chunk_store().contracts().values()]


@mcp.tool()
def query_contracts(
    question: Annotated[str, Field(description="Natural-language question about the contracts")],
    contract_ids: Annotated[list[str] | None, Field(description="Optional scope; empty lets the router choose")] = None,
) -> QueryResult:
    """Ask a question about one or more contracts (clause lookup or general Q&A).

    Runs the full agent: routing -> hybrid retrieval -> cross-encoder rerank -> draft ->
    claim-level verification (with re-retrieval on failure) -> cited answer.
    Leave contract_ids empty to let the router pick contracts from the question.
    """
    if contract_ids:
        _require(contract_ids)
    return _query_result(run(question, contract_ids))


@mcp.tool()
def flag_risks(
    contract_id: Annotated[str, Field(description="contract_id from list_contracts")],
    top_n: Annotated[int, Field(description="Maximum number of flags to return")] = 5,
) -> RiskReport:
    """Scan a contract for risky clauses (uncapped liability, auto-renewal, one-sided indemnity,
    broad IP assignment, restrictive non-compete, change-of-control, minimum commitments, etc.).
    Every flag quotes the source clause; flags whose quote cannot be found verbatim or that an
    independent verifier rejects are dropped."""
    _require([contract_id])
    r = scan_risks(contract_id)
    flags = [
        RiskFlag(rank=i, **{k: f[k] for k in RiskFlag.model_fields if k != "rank"})
        for i, f in enumerate(r["risk_flags"][:top_n], 1)
    ]
    return RiskReport(
        contract_id=contract_id,
        contract_title=get_chunk_store().contracts()[contract_id]["title"],
        flags=flags,
        rejected_by_verifier=len(r["rejected_flags"]),
        summary=r["answer"],
    )


@mcp.tool()
def compare_clauses(
    clause: Annotated[str, Field(description="Clause or topic, e.g. 'limitation of liability'")],
    contract_ids: Annotated[list[str], Field(description="Two or more contract_ids to compare")],
) -> QueryResult:
    """Compare how two or more contracts handle a clause/topic (e.g. 'termination for convenience',
    'limitation of liability', 'exclusivity'). Retrieval takes a quota from each contract."""
    if len(contract_ids) < 2:
        raise ToolError("compare_clauses needs at least two contract_ids")
    _require(contract_ids)
    q = f"Compare the {clause} provisions across these contracts and explain the key differences."
    return _query_result(run(q, contract_ids, options={"force_query_type": "comparison"}))


@mcp.tool(
    description="Raw hybrid search (dense + BM25 + metadata boost + cross-encoder rerank) without LLM "
    "generation. clause_types must come from: " + ", ".join(CLAUSE_CATEGORIES)
)
def search_clauses(
    query: Annotated[str, Field(description="Search text, ideally in contract language")],
    contract_ids: Annotated[list[str] | None, Field(description="Optional contract_id filter")] = None,
    clause_types: Annotated[list[str] | None, Field(description="Optional clause-category boost")] = None,
    top_n: Annotated[int, Field(description="Number of clauses to return")] = 5,
) -> list[ClauseHit]:
    if contract_ids:
        _require(contract_ids)
    bad = [c for c in clause_types or [] if c not in CLAUSE_CATEGORIES]
    if bad:
        raise ToolError(f"Unknown clause_types {bad}; valid: {list(CLAUSE_CATEGORIES)}")
    hits, _ = hybrid_search(query, contract_ids=contract_ids, clause_types=clause_types, top_n=top_n)
    return [
        ClauseHit(**{**{k: v for k, v in h.as_dict().items() if k in ClauseHit.model_fields}, "rerank_score": h.rerank})
        for h in hits
    ]


@mcp.tool()
def ingest_contract(
    pdf_path: Annotated[str, Field(description="Absolute path to a text-based PDF on the server machine")],
    title: Annotated[str | None, Field(description="Display title; defaults to the file name")] = None,
    contract_type: Annotated[str, Field(description="e.g. Supply, License, Distributor")] = "Unknown",
) -> ContractInfo:
    """Add a new contract PDF (absolute path on the server machine) to the index so it can be
    queried and risk-scanned. Returns the new contract_id."""
    from clausewatch.ingest.pipeline import ingest_pdf

    c = ingest_pdf(pdf_path, title=title, contract_type=contract_type)
    return ContractInfo(**{k: c[k] for k in ContractInfo.model_fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        mcp.run("streamable-http", port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
