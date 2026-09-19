"""ClauseWatch demo UI. Talks to the FastAPI backend (`uvicorn clausewatch.api:app`).

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("CLAUSEWATCH_API_URL", "http://localhost:8000")
TIMEOUT = 600

st.set_page_config(page_title="ClauseWatch", layout="wide")
st.title("ClauseWatch")
st.caption("Agentic RAG for contract and compliance review. Answers are verified against the cited clauses.")


@st.cache_data(ttl=30)
def contracts() -> list[dict]:
    return requests.get(f"{API}/contracts", timeout=30).json()


try:
    catalog = contracts()
except requests.RequestException:
    st.error(f"Backend not reachable at {API}. Start it with `uvicorn clausewatch.api:app --port 8000`.")
    st.stop()

by_id = {c["contract_id"]: c for c in catalog}

with st.sidebar:
    st.header("Contracts")
    up = st.file_uploader("Upload a contract (PDF)", type=["pdf"])
    if up and st.button("Index uploaded contract", type="primary"):
        with st.spinner("Chunking, tagging and embedding..."):
            r = requests.post(f"{API}/upload", files={"file": (up.name, up.getvalue(), "application/pdf")}, timeout=TIMEOUT)
        if r.ok:
            st.session_state["selected"] = [r.json()["contract_id"]]
            contracts.clear()
            st.success(f"Indexed {r.json()['title']} ({r.json()['n_chunks']} chunks)")
            st.rerun()
        else:
            st.error(r.text)
    selected = st.multiselect(
        "Scope (optional)",
        options=list(by_id),
        default=st.session_state.get("selected", []),
        format_func=lambda cid: f"{by_id[cid]['title']} [{by_id[cid]['contract_type']}]",
    )
    verify = st.toggle("Verifier node", value=True, help="Turn off to see the unverified draft (ablation).")
    st.caption(f"{len(catalog)} contracts indexed")

tab_ask, tab_risk = st.tabs(["Ask", "Risk flags"])

EXAMPLES = [
    "Can either party terminate the Papa John's endorsement agreement for convenience?",
    "What is the governing law of the Kitov Pharma manufacturing agreement?",
    "Compare the exclusivity provisions in the two distributor agreements.",
    "Which contracts cap liability, and at what amount?",
]

with tab_ask:
    ex = st.selectbox("Example questions", [""] + EXAMPLES)
    question = st.text_area("Question", value=ex, height=80)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Routing -> retrieving -> reranking -> drafting -> verifying..."):
            r = requests.post(f"{API}/ask", json={"question": question, "contract_ids": selected,
                                                   "options": {"verify": verify}}, timeout=TIMEOUT)
        if not r.ok:
            st.error(r.text)
        else:
            res = r.json()
            ver = res.get("verification", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Route", res["query_type"])
            c2.metric("Verifier verdict", ver.get("verdict", "off"))
            c3.metric("Retries", ver.get("attempt", 0))
            st.markdown(res["answer"])

            st.subheader("Cited source clauses")
            for c in res["citations"]:
                with st.expander(f"[{c['n']}] {c['ref']}"):
                    if c["quote"]:
                        st.markdown(f"> {c['quote']}")
                    src = next((s for s in res["sources"] if s["chunk_id"] == c["chunk_id"]), None)
                    idx = res["sources"].index(src) if src else None
                    if idx is not None:
                        st.text(res["contexts"][idx])

            if ver.get("claims"):
                st.subheader("Verifier: claim-level check")
                st.dataframe(pd.DataFrame(ver["claims"])[["status", "statement", "quote_verified", "explanation"]],
                             use_container_width=True, hide_index=True)
            with st.expander("Agent trace"):
                for e in res["trace"]:
                    st.markdown(f"**{e['node']}** ({e.get('ms', 0)} ms) - {e['message']}")

with tab_risk:
    target = st.selectbox("Contract to scan", options=[""] + list(by_id),
                          index=(list(by_id).index(selected[0]) + 1) if selected else 0,
                          format_func=lambda cid: by_id[cid]["title"] if cid else "Select...")
    if st.button("Scan for risks", type="primary", disabled=not target):
        with st.spinner("Scanning 10 risk patterns and verifying each flag..."):
            r = requests.post(f"{API}/risks/{target}", timeout=TIMEOUT)
        if not r.ok:
            st.error(r.text)
        else:
            res = r.json()
            sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
            if not res["risk_flags"]:
                st.info("No verified risk flags.")
            for i, f in enumerate(res["risk_flags"][:5], 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {sev_icon[f['severity']]} {f['name']}** "
                                f"({f['severity']}, confidence {f['confidence']:.2f})")
                    st.write(f["rationale"])
                    st.markdown(f"> {f['quote']}")
                    st.caption(f"{f['contract_title']} - {f['section']} - p. {f['page_start']}")
            if res["rejected_flags"]:
                with st.expander(f"{len(res['rejected_flags'])} flag(s) rejected by the verifier"):
                    for f in res["rejected_flags"]:
                        st.markdown(f"- **{f['name']}**: {f.get('verifier_note') or 'quote not found in source'}")
            with st.expander("Agent trace"):
                for e in res["trace"]:
                    st.markdown(f"**{e['node']}** ({e.get('ms', 0)} ms) - {e['message']}")
