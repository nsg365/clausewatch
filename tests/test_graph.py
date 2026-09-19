"""End-to-end graph tests with a scripted LLM (real retrieval, embeddings and reranker)."""

import re

from clausewatch.agent.graph import build_graph, scan_risks


def _label_for(user: str, needle: str) -> str:
    """Find the [S#] label of the source whose text contains `needle`."""
    body = re.split(r"\n(?:Sources|Clauses):\n", user, maxsplit=1)[-1]
    for block in re.split(r"\n\n(?=\[S\d+\])", body):
        m = re.match(r"\s*\[(S\d+)\]", block)
        if m and needle.lower() in block.lower():
            return m.group(1)
    raise AssertionError(f"{needle!r} not in sources")


def _route(qtype, cid, clauses=("termination",)):
    return {"query_type": qtype, "clause_types": list(clauses), "contract_ids": [cid],
            "search_query": "terminate for convenience written notice", "reason": "test"}


def _run(question, cid=None):
    return build_graph().invoke({"question": question, "contract_ids": [cid] if cid else [], "options": {}, "trace": []})


def test_verified_answer_has_citations(indexed_contract, fake_llm):
    cid = indexed_contract
    quote = "Buyer may terminate this Agreement at any time for any reason upon thirty (30) days written notice"

    def draft(user):
        s = _label_for(user, "Buyer may terminate")
        return {"answer": f"Only the Buyer may terminate for convenience, on 30 days' notice [{s}].",
                "claims": [{"statement": "Buyer may terminate on 30 days notice", "source_ids": [s], "supporting_quote": quote}],
                "insufficient_context": False, "missing_information": ""}

    fake = fake_llm({
        "RouteDecision": _route("clause_lookup", cid),
        "DraftAnswer": draft,
        "VerificationResult": {"verdicts": [{"claim_index": 0, "verdict": "supported", "explanation": "ok"}],
                               "answers_question": True, "retry_query": ""},
    })
    out = _run("Can the supplier terminate for convenience?", cid)
    assert out["verification"]["verdict"] == "pass"
    assert out["answer"].endswith("[1].")
    assert out["citations"][0]["section"].startswith("2. Termination")
    assert out["citations"][0]["page_start"] == 2
    assert [c[0] for c in fake.calls] == ["RouteDecision", "DraftAnswer", "VerificationResult"]
    assert [e["node"] for e in out["trace"]] == ["router", "retrieve", "rerank", "draft", "verify", "answer"]


def test_citations_added_when_draft_omits_markers(indexed_contract, fake_llm):
    cid = indexed_contract
    quote = "This Agreement shall be governed by the laws of the State of Delaware"

    def draft(user):
        s = _label_for(user, "State of Delaware")
        return {"answer": "Delaware law governs the agreement.",  # no [S#] marker
                "claims": [{"statement": "Delaware law governs", "source_ids": [s], "supporting_quote": quote}],
                "insufficient_context": False, "missing_information": ""}

    fake_llm({
        "RouteDecision": _route("clause_lookup", cid, clauses=("governing_law",)),
        "DraftAnswer": draft,
        "VerificationResult": {"verdicts": [{"claim_index": 0, "verdict": "supported", "explanation": "ok"}],
                               "answers_question": True, "retry_query": ""},
    })
    out = _run("Which law governs?", cid)
    assert out["answer"] == "Delaware law governs the agreement. [1]"
    assert out["citations"][0]["section"].startswith("4. Governing Law")


def test_fabricated_quote_is_rejected_and_triggers_retry(indexed_contract, fake_llm):
    cid = indexed_contract

    def bad_draft(user):
        s = _label_for(user, "Buyer may terminate")
        # The LLM judge is fooled, but the quote does not exist in the source.
        return {"answer": f"Either party may terminate on 10 days notice [{s}].",
                "claims": [{"statement": "Either party may terminate on 10 days notice", "source_ids": [s],
                            "supporting_quote": "either party may terminate upon ten (10) days notice"}],
                "insufficient_context": False, "missing_information": ""}

    fake_llm({
        "RouteDecision": _route("clause_lookup", cid),
        "DraftAnswer": bad_draft,
        "VerificationResult": {"verdicts": [{"claim_index": 0, "verdict": "supported", "explanation": "looks fine"}],
                               "answers_question": True, "retry_query": "termination for convenience notice"},
    })
    out = _run("Who can terminate for convenience?", cid)
    verify_events = [e for e in out["trace"] if e["node"] == "verify"]
    assert len(verify_events) >= 3  # initial + 2 retries (each with a REJECTED line)
    assert any("REJECTED" in e["message"] for e in verify_events)
    assert [e["node"] for e in out["trace"]].count("retrieve") == 3
    assert out["verification"]["verdict"] == "fail"
    assert out["citations"] == []
    assert "could not find verifiable support" in out["answer"]


def test_risk_scan_drops_flags_with_unverifiable_quotes(indexed_contract, fake_llm):
    cid = indexed_contract

    def assess(user):
        if "One-sided indemnification" in user:
            return {"present": True, "source_id": _label_for(user, "Buyer has no indemnification"),
                    "quote": "Buyer has no indemnification obligations under this Agreement.",
                    "rationale": "Only the Supplier indemnifies.", "severity": "high", "confidence": 0.9}
        if "Auto-renewal" in user:
            return {"present": True, "source_id": "S1", "quote": "renews forever with no way out",
                    "rationale": "hallucinated", "severity": "medium", "confidence": 0.8}
        return {"present": False, "source_id": "", "quote": "", "rationale": "", "severity": "low", "confidence": 0.1}

    fake_llm({
        "RiskAssessment": assess,
        "FlagChecks": {"checks": [{"flag_index": 0, "grounded": True, "explanation": "one-sided"}]},
    })
    out = scan_risks(cid)
    assert [f["pattern_id"] for f in out["risk_flags"]] == ["one_sided_indemnification"]
    assert [f["pattern_id"] for f in out["rejected_flags"]] == ["auto_renewal"]
    assert out["citations"][0]["section"].startswith("3. Indemnification")
