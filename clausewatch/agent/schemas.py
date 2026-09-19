"""Graph state and the structured-output schemas each LLM node is constrained to."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from clausewatch.taxonomy import CLAUSE_CATEGORIES

QueryType = Literal["clause_lookup", "risk_flagging", "comparison", "general_qa"]
ClauseType = Literal[tuple(CLAUSE_CATEGORIES)]  # type: ignore[valid-type]


class AgentState(TypedDict, total=False):
    # input
    question: str
    contract_ids: list[str]  # explicit scope from the caller (UI upload, MCP arg)
    options: dict  # {"verify": bool, "retrieval_mode": "hybrid"|"dense"|"sparse", "rerank": bool}
    # router
    query_type: QueryType
    clause_types: list[str]
    search_query: str
    target_contracts: list[str]
    # retrieval
    candidates: list[dict]
    hits: list[dict]
    retrieval_info: dict
    attempt: int
    retry_query: str
    # generation / verification
    draft: dict
    verification: dict
    risk_flags: list[dict]
    # output
    answer: str
    citations: list[dict]
    trace: Annotated[list[dict], operator.add]


class RouteDecision(BaseModel):
    query_type: QueryType = Field(
        description="clause_lookup: find/explain a specific clause in one contract. "
        "risk_flagging: scan a contract for risky terms. "
        "comparison: compare a clause or term across two or more contracts. "
        "general_qa: anything else about the contracts."
    )
    clause_types: list[ClauseType] = Field(description="Clause categories the question is about (may be empty).")
    contract_ids: list[str] = Field(description="IDs from the catalog of the contracts the question refers to.")
    search_query: str = Field(description="A keyword-rich retrieval query written in contract language.")
    reason: str = Field(description="One sentence explaining the routing decision.")


class Claim(BaseModel):
    statement: str = Field(description="One atomic factual claim made in the answer.")
    source_ids: list[str] = Field(description="Source labels (e.g. S2) that support this claim.")
    supporting_quote: str = Field(description="Verbatim excerpt copied from the cited source that supports the claim.")


class DraftAnswer(BaseModel):
    answer: str = Field(description="The answer, with inline source markers like [S1].")
    claims: list[Claim]
    insufficient_context: bool = Field(description="True if the sources do not contain the answer.")
    missing_information: str = Field(description="What information is missing, if any; empty string otherwise.")


class ClaimVerdict(BaseModel):
    claim_index: int
    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    explanation: str


class VerificationResult(BaseModel):
    verdicts: list[ClaimVerdict]
    answers_question: bool = Field(description="Whether the supported claims actually answer the question.")
    retry_query: str = Field(description="If more evidence is needed, a better retrieval query; else empty string.")


class RiskAssessment(BaseModel):
    present: bool = Field(description="True only if a source clause clearly matches the risk criteria.")
    source_id: str = Field(description="Label of the source containing the risky clause, or empty string.")
    quote: str = Field(description="Verbatim excerpt (1-3 sentences) from that source, or empty string.")
    rationale: str
    severity: Literal["high", "medium", "low"]
    confidence: float = Field(description="0.0-1.0")


class FlagCheck(BaseModel):
    flag_index: int
    grounded: bool = Field(description="True if the quoted clause really supports the stated risk.")
    explanation: str


class FlagChecks(BaseModel):
    checks: list[FlagCheck]
