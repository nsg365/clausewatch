"""LangGraph nodes. Each node returns a partial state update plus trace events, so every
routing / retrieval / verification decision is visible in the UI, API, MCP output and logs."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from clausewatch import llm
from clausewatch.agent import prompts
from clausewatch.agent.schemas import (
    AgentState,
    DraftAnswer,
    FlagChecks,
    RiskAssessment,
    RouteDecision,
    VerificationResult,
)
from clausewatch.config import get_settings
from clausewatch.retrieval.hybrid import hybrid_search, strip_contract_names
from clausewatch.retrieval.models import get_reranker
from clausewatch.retrieval.store import get_chunk_store
from clausewatch.risk_patterns import RISK_PATTERNS, SEVERITY_WEIGHT
from clausewatch.textutil import quote_in_text

log = logging.getLogger("clausewatch.agent")

DEFAULT_OPTIONS = {"verify": True, "retrieval_mode": "hybrid", "rerank": True}


def _opts(state: AgentState) -> dict:
    return {**DEFAULT_OPTIONS, **(state.get("options") or {})}


def event(node: str, message: str, **data) -> dict:
    log.info("[%s] %s", node, message)
    return {"node": node, "message": message, "data": data}


def traced(node: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(state: AgentState):
            t0 = time.perf_counter()
            out = fn(state)
            ms = round((time.perf_counter() - t0) * 1000)
            for e in out.get("trace", []):
                e.setdefault("ms", ms)
            return out

        return wrapper

    return deco


def label_sources(hits: list[dict]) -> list[dict]:
    return [{**h, "label": f"S{i}"} for i, h in enumerate(hits, 1)]


def render_sources(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{h['label']}] Contract: {h['contract_title']} | Section: {h['section']} | "
        f"Page {h['page_start']}{'' if h['page_end'] == h['page_start'] else '-' + str(h['page_end'])}\n{h['text']}"
        for h in hits
    )


# --------------------------------------------------------------------------- router
@traced("router")
def router(state: AgentState) -> dict:
    catalog = get_chunk_store().contracts()
    explicit = [c for c in (state.get("contract_ids") or []) if c in catalog]
    lines = "\n".join(f"{cid} | {c['title']} | {c['contract_type']}" for cid, c in catalog.items())
    scope_note = (
        f"The user is asking about these contracts: {', '.join(explicit)}\n\n" if explicit else ""
    )
    decision = llm.structured_call(
        RouteDecision,
        prompts.ROUTER_SYSTEM,
        prompts.ROUTER_USER.format(catalog=lines, scope_note=scope_note, question=state["question"]),
        llm=llm.get_chat_model(effort=get_settings().router_effort, model=get_settings().router_model),
    )
    if forced := _opts(state).get("force_query_type"):
        decision.query_type = forced
    routed = [c for c in decision.contract_ids if c in catalog]
    dropped = sorted(set(decision.contract_ids) - set(routed))
    if decision.query_type == "comparison":
        targets = list(dict.fromkeys(explicit + routed))
    else:
        targets = explicit or routed
    return {
        "query_type": decision.query_type,
        "clause_types": list(decision.clause_types),
        "search_query": decision.search_query,
        "target_contracts": targets,
        "attempt": 0,
        "trace": [
            event(
                "router",
                f"{decision.query_type} | clauses={list(decision.clause_types)} | contracts={targets}",
                reason=decision.reason,
                search_query=decision.search_query,
                unknown_contract_ids=dropped,
            )
        ],
    }


def route_after_router(state: AgentState) -> str:
    if state["query_type"] == "risk_flagging":
        return "risk_scan" if state.get("target_contracts") else "answer"
    return "retrieve"


# --------------------------------------------------------------------------- retrieval
@traced("retrieve")
def retrieve(state: AgentState) -> dict:
    opts = _opts(state)
    attempt = state.get("attempt", 0)
    qtype = state["query_type"]
    targets = state.get("target_contracts") or []
    clause_types = state.get("clause_types") if qtype in ("clause_lookup", "comparison") else None
    query = state.get("search_query") or state["question"]
    if attempt > 0:
        # Verifier asked for more evidence: use its query and relax the clause filter.
        query = state.get("retry_query") or f"{state['question']} {query}"
        clause_types = None

    common = dict(clause_types=clause_types, rerank=False, mode=opts["retrieval_mode"])
    if qtype == "comparison" and len(targets) >= 2:
        candidates, infos = [], []
        for cid in targets:
            hits, info = hybrid_search(query, contract_ids=[cid], top_n=15, **common)
            candidates += [h.as_dict() for h in hits]
            infos.append(info)
        info = {"per_contract": infos}
    else:
        hits, info = hybrid_search(query, contract_ids=targets or None, top_n=get_settings().fused_k, **common)
        candidates = [h.as_dict() for h in hits]

    relaxed = info.get("relaxed") or any(i.get("relaxed") for i in info.get("per_contract", []))
    msg = f"attempt {attempt}: {len(candidates)} candidates (dense+BM25, RRF) for {query!r}"
    if relaxed:
        msg += " [clause filter relaxed]"
    return {
        "candidates": candidates,
        "retrieval_info": info,
        "trace": [event("retrieve", msg, filters={"contracts": targets, "clause_types": clause_types})],
    }


@traced("rerank")
def rerank(state: AgentState) -> dict:
    opts = _opts(state)
    s = get_settings()
    cands = state.get("candidates", [])
    # Rerank on the router's contract-language query: the raw question often names the
    # company, which makes the cross-encoder favour title/preamble chunks.
    if state.get("attempt", 0):
        query = state.get("retry_query") or state["question"]
    else:
        query = state.get("search_query") or state["question"]
    top_n = s.rerank_top_n + (2 if state.get("attempt", 0) else 0)

    if targets := state.get("target_contracts"):
        catalog = get_chunk_store().contracts()
        query = strip_contract_names(query, [catalog[c]["title"] for c in targets if c in catalog])
    if opts["rerank"] and cands:
        scores = get_reranker().score(query, [f"{c['section']}\n{c['text']}" for c in cands])
        for c, sc in zip(cands, scores):
            c["rerank"] = round(sc, 4)
        cands = sorted(cands, key=lambda c: -c["rerank"])

    if state["query_type"] == "comparison" and len(state.get("target_contracts") or []) >= 2:
        per = max(2, top_n // len(state["target_contracts"]))
        picked = []
        for cid in state["target_contracts"]:
            picked += [c for c in cands if c["contract_id"] == cid][:per]
    else:
        picked = cands[:top_n]

    # On retries keep the evidence that already supported claims.
    kept = [h for h in state.get("hits", []) if h.get("supported")] if state.get("attempt", 0) else []
    seen, merged = set(), []
    for h in kept + picked:
        if h["chunk_id"] not in seen:
            seen.add(h["chunk_id"])
            merged.append({k: v for k, v in h.items() if k not in ("label", "supported")})
    hits = label_sources(merged[: top_n + 2])
    return {
        "hits": hits,
        "trace": [
            event(
                "rerank",
                f"cross-encoder kept {len(hits)}/{len(state.get('candidates', []))}",
                top=[
                    {"label": h["label"], "contract": h["contract_title"], "section": h["section"], "score": h.get("rerank")}
                    for h in hits
                ],
            )
        ],
    }


# --------------------------------------------------------------------------- drafting
@traced("draft")
def draft(state: AgentState) -> dict:
    feedback = ""
    ver = state.get("verification")
    if state.get("attempt", 0) and ver:
        bad = [v for v in ver.get("claims", []) if v["status"] != "supported"]
        feedback = (
            "\nA previous draft was rejected by the verifier. Unsupported claims were:\n"
            + "\n".join(f"- {v['statement']} ({v['status']}: {v['explanation']})" for v in bad)
            + "\nDo not repeat unsupported claims. Use only the sources below.\n"
        )
    result = llm.structured_call(
        DraftAnswer,
        prompts.DRAFT_SYSTEM,
        prompts.DRAFT_USER.format(question=state["question"], feedback=feedback, sources=render_sources(state["hits"])),
    )
    d = result.model_dump()
    return {
        "draft": d,
        "trace": [
            event(
                "draft",
                f"{len(d['claims'])} claims" + (" | model reports insufficient context" if d["insufficient_context"] else ""),
                answer=d["answer"],
            )
        ],
    }


def route_after_draft(state: AgentState) -> str:
    return "verify" if _opts(state)["verify"] else "answer"


# --------------------------------------------------------------------------- verification
def _verify_answer(state: AgentState) -> dict:
    s = get_settings()
    hits = {h["label"]: h for h in state["hits"]}
    d = state["draft"]
    claims = d["claims"]

    # Layer 1 (deterministic): cited sources exist and the quote appears verbatim in one of them.
    checks = []
    for c in claims:
        cited = [hits[x] for x in c["source_ids"] if x in hits]
        quote_ok = any(quote_in_text(c["supporting_quote"], h["text"]) for h in cited)
        checks.append({"cited_ok": bool(cited), "quote_ok": quote_ok})

    # Layer 2 (LLM judge, fresh context): does the cited text entail the claim?
    verdicts: dict[int, dict] = {}
    answers_question, retry_query = False, ""
    if claims:
        blocks = []
        for i, c in enumerate(claims):
            src = "\n".join(
                f"  [{x}] ({hits[x]['contract_title']}, {hits[x]['section']}): {hits[x]['text']}"
                for x in c["source_ids"]
                if x in hits
            ) or "  (no valid source cited)"
            blocks.append(f"Claim {i}: {c['statement']}\n Cited source text:\n{src}")
        vr = llm.structured_call(
            VerificationResult,
            prompts.VERIFY_SYSTEM,
            prompts.VERIFY_USER.format(question=state["question"], claims="\n\n".join(blocks)),
            llm=llm.get_chat_model(model=s.verifier_model),
        )
        verdicts = {v.claim_index: v.model_dump() for v in vr.verdicts}
        answers_question, retry_query = vr.answers_question, vr.retry_query

    results = []
    for i, (c, chk) in enumerate(zip(claims, checks)):
        v = verdicts.get(i, {"verdict": "unsupported", "explanation": "verifier returned no verdict"})
        status = v["verdict"]
        if status == "supported" and not chk["cited_ok"]:
            status, v["explanation"] = "unsupported", "cites a source that was not retrieved"
        elif status == "supported" and not chk["quote_ok"]:
            status, v["explanation"] = "unsupported", "supporting quote not found verbatim in cited source"
        results.append({"statement": c["statement"], "source_ids": c["source_ids"], "quote": c["supporting_quote"],
                        "status": status, "explanation": v["explanation"], "quote_verified": chk["quote_ok"]})

    n_sup = sum(r["status"] == "supported" for r in results)
    ratio = n_sup / len(results) if results else 0.0
    attempt = state.get("attempt", 0)
    if d["insufficient_context"] or not results:
        verdict = "insufficient"
    elif ratio == 1.0 and answers_question:
        verdict = "pass"
    elif ratio >= s.min_support_ratio and answers_question:
        verdict = "pass_pruned"
    else:
        verdict = "fail"
    can_retry = attempt < s.max_verify_retries
    decision = "retry" if verdict in ("fail", "insufficient") and can_retry else "accept"

    supported_labels = {x for r in results if r["status"] == "supported" for x in r["source_ids"]}
    new_hits = [{**h, "supported": h["label"] in supported_labels} for h in state["hits"]]
    ver = {"verdict": verdict, "decision": decision, "supported_ratio": round(ratio, 3),
           "answers_question": answers_question, "claims": results, "attempt": attempt}

    msg = f"{n_sup}/{len(results)} claims supported, verdict={verdict.upper()} -> {decision.upper()}"
    evs = [event("verify", msg, claims=results, retry_query=retry_query)]
    for r in results:
        if r["status"] != "supported":
            evs.append(event("verify", f"REJECTED claim: {r['statement'][:120]!r} ({r['status']}: {r['explanation']})"))
    out = {"verification": ver, "hits": new_hits, "trace": evs}
    if decision == "retry":
        out["attempt"] = attempt + 1
        # A different query than the first attempt; the draft's "missing information" note is prose,
        # not a search query, so fall back to the user's own question.
        out["retry_query"] = retry_query or state["question"]
    return out


def _excerpt(text: str, quote: str, width: int = 700) -> str:
    """Context window around the quote (whole chunk if short)."""
    if len(text) <= 2 * width:
        return text
    probe = re.sub(r"\s+", " ", quote).strip()[:40]
    idx = re.sub(r"\s+", " ", text).find(probe)
    flat = re.sub(r"\s+", " ", text)
    if idx < 0:
        return flat[: 2 * width]
    return flat[max(0, idx - width // 2) : idx + len(quote) + width]


def _verify_risks(state: AgentState) -> dict:
    s = get_settings()
    flags = state.get("risk_flags", [])
    evs = []
    # Layer 1: quote must exist verbatim in the cited chunk.
    grounded = []
    for f in flags:
        chunk = get_chunk_store().get(f["chunk_id"])
        f["quote_verified"] = bool(chunk) and quote_in_text(f["quote"], chunk["text"])
        if f["quote_verified"]:
            grounded.append(f)
        else:
            f["status"] = "rejected"
            evs.append(event("verify", f"REJECTED flag {f['pattern_id']}: quote not found in {f['chunk_id']}"))
    # Layer 2: independent judge re-reads each quote in context.
    # Small batches keep each request under free-tier per-request token caps.
    for start in range(0, len(grounded), 3):
        batch = grounded[start : start + 3]
        blocks = "\n\n".join(
            f"Flag {i}: {f['name']} ({f['severity']})\nRationale: {f['rationale']}\nQuote: \"{f['quote']}\"\n"
            f"Source excerpt ({f['contract_title']}, {f['section']}):\n"
            f"{_excerpt(get_chunk_store().get(f['chunk_id'])['text'], f['quote'])}"
            for i, f in enumerate(batch)
        )
        try:
            checks = llm.structured_call(FlagChecks, prompts.RISK_VERIFY_SYSTEM, blocks,
                                         llm=llm.get_chat_model(model=s.verifier_model))
            by_idx = {c.flag_index: c for c in checks.checks}
        except Exception as e:  # fail closed: an unverifiable flag is not shown
            by_idx = {}
            evs.append(event("verify", f"judge call failed ({type(e).__name__}); rejecting {len(batch)} flag(s)"))
        for i, f in enumerate(batch):
            c = by_idx.get(i)
            f["status"] = "verified" if c and c.grounded else "rejected"
            f["verifier_note"] = c.explanation if c else "no verdict from verifier"
            if f["status"] == "rejected":
                evs.append(event("verify", f"REJECTED flag {f['pattern_id']}: {f['verifier_note']}"))
    verified = [f for f in flags if f.get("status") == "verified"]
    verified.sort(key=lambda f: -SEVERITY_WEIGHT[f["severity"]] * f["confidence"])
    evs.insert(0, event("verify", f"{len(verified)}/{len(flags)} risk flags verified (quote match + judge)"))
    return {
        "risk_flags": flags,
        "verification": {"verdict": "pass", "decision": "accept", "verified": len(verified), "total": len(flags)},
        "trace": evs,
    }


@traced("verify")
def verify(state: AgentState) -> dict:
    if state["query_type"] == "risk_flagging":
        return _verify_risks(state)
    return _verify_answer(state)


def route_after_verify(state: AgentState) -> str:
    return "retrieve" if state["verification"]["decision"] == "retry" else "answer"


# --------------------------------------------------------------------------- risk scan
def _assess(contract: dict, pattern, hits: list[dict]) -> dict | None:
    if not hits:
        return None
    a = llm.structured_call(
        RiskAssessment,
        prompts.RISK_SYSTEM,
        prompts.RISK_USER.format(contract=contract["title"], name=pattern.name, criteria=pattern.criteria,
                                 sources=render_sources(hits)),
        effort=get_settings().router_effort,
    )
    if not a.present:
        return None
    src = next((h for h in hits if h["label"] == a.source_id), hits[0])
    return {
        "pattern_id": pattern.id,
        "name": pattern.name,
        "severity": a.severity,
        "confidence": max(0.0, min(1.0, a.confidence)),
        "rationale": a.rationale,
        "quote": a.quote,
        "contract_id": contract["contract_id"],
        "contract_title": contract["title"],
        "chunk_id": src["chunk_id"],
        "section": src["section"],
        "page_start": src["page_start"],
        "page_end": src["page_end"],
    }


@traced("risk_scan")
def risk_scan(state: AgentState) -> dict:
    catalog = get_chunk_store().contracts()
    jobs = []
    for cid in state["target_contracts"]:
        for p in RISK_PATTERNS:
            hits, _ = hybrid_search(p.query, contract_ids=[cid], clause_types=p.categories, top_n=3,
                                    mode=_opts(state)["retrieval_mode"])
            jobs.append((catalog[cid], p, label_sources([h.as_dict() for h in hits])))
    errors: list[str] = []

    def safe_assess(job):
        try:
            return _assess(*job)
        except Exception as e:  # one failed pattern must not sink the whole scan
            errors.append(f"{job[1].id}: {type(e).__name__}")
            return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(safe_assess, jobs))
    flags = [r for r in results if r]
    evs = [event("risk_scan", f"scanned {len(RISK_PATTERNS)} patterns x {len(state['target_contracts'])} contract(s): "
                              f"{len(flags)} candidate flags", candidates=[f["pattern_id"] for f in flags])]
    if errors:
        evs.append(event("risk_scan", f"{len(errors)} pattern check(s) failed and were skipped: {errors}"))
    return {"risk_flags": flags, "trace": evs}


# --------------------------------------------------------------------------- answer
def _page(h: dict) -> str:
    return f"p. {h['page_start']}" if h["page_start"] == h["page_end"] else f"pp. {h['page_start']}-{h['page_end']}"


def _cite(h: dict, n: int, quote: str = "") -> dict:
    quote = quote.strip().strip("\"'“”").strip()
    return {"n": n, "contract_id": h["contract_id"], "contract_title": h["contract_title"], "section": h["section"],
            "page_start": h["page_start"], "page_end": h["page_end"], "chunk_id": h["chunk_id"],
            "quote": quote, "ref": f"{h['contract_title']}, {h['section']}, {_page(h)}"}


def _answer_risks(state: AgentState) -> dict:
    s = get_settings()
    flags = [f for f in state.get("risk_flags", []) if f.get("status") == "verified"][: s.risk_top_n]
    rejected = sum(f.get("status") == "rejected" for f in state.get("risk_flags", []))
    titles = ", ".join(get_chunk_store().contracts()[c]["title"] for c in state["target_contracts"])
    if not flags:
        text = f"No verified risk flags found in {titles} across {len(RISK_PATTERNS)} patterns."
        return {"answer": text, "citations": []}
    lines = [f"**Top {len(flags)} risk flags for {titles}** "
             f"({rejected} candidate flag(s) rejected by the verifier)\n"]
    cites = []
    for n, f in enumerate(flags, 1):
        h = {**f}
        cites.append(_cite(h, n, f["quote"]))
        lines.append(f"{n}. **{f['name']}** ({f['severity'].upper()}, confidence {f['confidence']:.2f}) - "
                     f"{f['rationale']} [{n}]\n   > \"{f['quote']}\"\n   > - {cites[-1]['ref']}")
    return {"answer": "\n".join(lines), "citations": cites}


def _renumber(text: str, hits: dict, quotes: dict) -> tuple[str, list[dict]]:
    order: dict[str, int] = {}

    def sub(m):
        labels = re.findall(r"S\d+", m.group(0))
        nums = []
        for lab in labels:
            if lab in hits:
                order.setdefault(lab, len(order) + 1)
                nums.append(str(order[lab]))
        return f"[{', '.join(nums)}]" if nums else ""

    text = re.sub(r"\[(?:S\d+(?:,\s*)?)+\]", sub, text)
    cites = [_cite(hits[lab], n, quotes.get(lab, "")) for lab, n in order.items()]
    return text, cites


@traced("answer")
def answer(state: AgentState) -> dict:
    if state["query_type"] == "risk_flagging":
        if not state.get("target_contracts"):
            names = "\n".join(f"- {c['title']} (`{cid}`)" for cid, c in get_chunk_store().contracts().items())
            return {"answer": "Which contract should I scan? Available contracts:\n" + names, "citations": [],
                    "trace": [event("answer", "risk scan requested without a resolvable contract")]}
        out = _answer_risks(state)
        return {**out, "trace": [event("answer", f"{len(out['citations'])} risk flags returned")]}

    hits = {h["label"]: h for h in state.get("hits", [])}
    d = state.get("draft") or {}
    ver = state.get("verification") or {"verdict": "unverified"}
    claims = ver.get("claims") or [
        {"statement": c["statement"], "source_ids": c["source_ids"], "quote": c["supporting_quote"], "status": "unverified"}
        for c in d.get("claims", [])
    ]
    quotes = {}
    for c in claims:
        if c["status"] in ("supported", "unverified"):
            for lab in c["source_ids"]:
                # attach the quote only to the cited chunk(s) that actually contain it
                if lab in hits and quote_in_text(c["quote"], hits[lab]["text"]):
                    quotes.setdefault(lab, c["quote"])

    verdict = ver["verdict"]
    if verdict == "unverified" and d.get("insufficient_context") and not d.get("answer", "").strip():
        d = {**d, "answer": "The provided contract excerpts do not contain this information. "
                            + d.get("missing_information", "")}
    if verdict in ("pass", "unverified"):
        body = d["answer"]
        if not re.search(r"\[S\d+", body):  # model forgot inline markers: cite the claims' sources
            labels = list(dict.fromkeys(x for c in claims if c["status"] in ("supported", "unverified")
                                        for x in c["source_ids"]))
            if labels:
                body = f"{body.rstrip()} [{', '.join(labels)}]"
        text, cites = _renumber(body, hits, quotes)
    else:
        good = [c for c in claims if c["status"] == "supported"]
        if not good:
            text = ("I could not find verifiable support for this in the indexed contracts. "
                    + (f"Missing: {d.get('missing_information')}" if d.get("missing_information") else ""))
            cites = []
        else:
            body = "\n".join(f"- {c['statement']} [{', '.join(c['source_ids'])}]" for c in good)
            dropped = len(claims) - len(good)
            note = f"\n\n_{dropped} claim(s) from the draft were removed because the verifier could not ground them._" if dropped else ""
            prefix = "" if verdict == "pass_pruned" else "Only part of the answer could be verified:\n"
            text, cites = _renumber(prefix + body + note, hits, quotes)
    return {"answer": text, "citations": cites,
            "trace": [event("answer", f"verdict={verdict}, {len(cites)} citations")]}
