"""The ClauseWatch LangGraph.

    START -> router --(risk_flagging)--> risk_scan -------------------------> verify -> answer -> END
                    \\-(lookup/compare/qa)-> retrieve -> rerank -> draft -> verify -> answer
                                              ^                               |
                                              \\---- retry (verifier feedback) -/
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from clausewatch.agent import nodes
from clausewatch.agent.schemas import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", nodes.router)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("draft", nodes.draft)
    g.add_node("verify", nodes.verify)
    g.add_node("risk_scan", nodes.risk_scan)
    g.add_node("answer", nodes.answer)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", nodes.route_after_router, ["retrieve", "risk_scan", "answer"])
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "draft")
    g.add_conditional_edges("draft", nodes.route_after_draft, ["verify", "answer"])
    g.add_edge("risk_scan", "verify")
    g.add_conditional_edges("verify", nodes.route_after_verify, ["retrieve", "answer"])
    g.add_edge("answer", END)
    return g.compile()


@lru_cache
def get_graph():
    return build_graph()


def run(question: str, contract_ids: list[str] | None = None, options: dict | None = None) -> dict:
    """Run the agent and return a JSON-serialisable result."""
    state = get_graph().invoke(
        {"question": question, "contract_ids": contract_ids or [], "options": options or {}, "trace": []},
        {"recursion_limit": 25},
    )
    return {
        "question": question,
        "query_type": state.get("query_type"),
        "target_contracts": state.get("target_contracts", []),
        "clause_types": state.get("clause_types", []),
        "search_query": state.get("search_query", ""),
        "answer": state.get("answer", ""),
        "draft": state.get("draft", {}),
        "citations": state.get("citations", []),
        "risk_flags": [f for f in state.get("risk_flags", []) if f.get("status") == "verified"],
        "rejected_flags": [f for f in state.get("risk_flags", []) if f.get("status") == "rejected"],
        "verification": state.get("verification", {}),
        "contexts": [h["text"] for h in state.get("hits", [])],
        "sources": [{k: v for k, v in h.items() if k != "text"} for h in state.get("hits", [])],
        "trace": state.get("trace", []),
    }


def scan_risks(contract_id: str) -> dict:
    """Deterministic entry point for risk flagging (skips the router)."""
    from clausewatch.agent.nodes import answer, risk_scan, verify

    state: dict = {"question": f"Flag risks in {contract_id}", "query_type": "risk_flagging",
                   "target_contracts": [contract_id], "options": {}, "trace": []}
    for node in (risk_scan, verify, answer):
        out = node(state)
        state["trace"] = state["trace"] + out.pop("trace", [])
        state.update(out)
    flags = state.get("risk_flags", [])
    return {
        "contract_id": contract_id,
        "answer": state["answer"],
        "citations": state["citations"],
        "risk_flags": [f for f in flags if f.get("status") == "verified"],
        "rejected_flags": [f for f in flags if f.get("status") == "rejected"],
        "trace": state["trace"],
    }
