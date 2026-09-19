"""ClauseWatch evaluation harness.

    python eval/run_eval.py retrieval              # retrieval ablation (no LLM calls)
    python eval/run_eval.py agent --config full    # run the agent on the QA set (resumable)
    python eval/run_eval.py agent --config no_verifier
    python eval/run_eval.py score --config full    # RAGAS + deterministic metrics (resumable)
    python eval/run_eval.py risk                   # risk-flag demo on held-out CUAD contracts
    python eval/run_eval.py report                 # write eval/results/summary.md

Every per-question result is cached under eval/results/, so a run interrupted by a
rate limit continues where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import contextlib
import math
import os
import signal
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CLAUSEWATCH_WAIT_ON_DAILY_LIMIT", "1")  # eval runs unattended through free-tier quotas

from clausewatch.config import get_settings  # noqa: E402
from clausewatch.llm import acall_waiting_for_quota, call_waiting_for_quota  # noqa: E402
from clausewatch.textutil import quote_in_text  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS = EVAL_DIR / "results"
CONFIGS = {
    "full": {},
    "no_verifier": {"verify": False},
}
log = logging.getLogger("eval")


def load_qa() -> list[dict]:
    return [json.loads(l) for l in (EVAL_DIR / "qa_set.jsonl").read_text().splitlines() if l.strip()]


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(statistics.mean(xs), 3) if xs else None


# --------------------------------------------------------------------------- retrieval
def _router_queries() -> dict[str, dict]:
    """search_query / clause_types the router produced in the `full` agent run, if available."""
    out = {}
    for p in (RESULTS / "runs" / "full").glob("*.json"):
        r = json.loads(p.read_text())
        ev = next((e for e in r["trace"] if e["node"] == "router"), None)
        if ev:
            out[p.stem] = {"query": ev["data"]["search_query"], "clause_types": r.get("clause_types")}
    return out


def cmd_retrieval(args) -> None:
    from clausewatch.retrieval.hybrid import hybrid_search, per_contract_search
    from clausewatch.retrieval.store import get_chunk_store

    k = get_settings().rerank_top_n
    qa = [q for q in load_qa() if q["evidence"]]
    routed = _router_queries()
    variants = [
        ("Dense only", dict(mode="dense", rerank=False)),
        ("BM25 only", dict(mode="sparse", rerank=False)),
        ("Hybrid (RRF)", dict(mode="hybrid", rerank=False)),
        ("Hybrid + cross-encoder rerank", dict(mode="hybrid", rerank=True)),
    ]
    sources = [("question", "scoped"), ("question", "open")]
    if len(routed) == len(load_qa()):
        sources.insert(0, ("router", "scoped"))
        variants.append(("Hybrid + clause boost + rerank", dict(mode="hybrid", rerank=True, boost=True)))
    rows = []
    for source, scope in sources:
        for name, kw in variants:
            kw = dict(kw)
            boost = kw.pop("boost", False)
            if boost and source != "router":
                continue
            recalls, rr, hit1 = [], [], []
            for q in qa:
                cids = q["contracts"] if scope == "scoped" else None
                query = routed[q["id"]]["query"] if source == "router" else q["question"]
                if boost:
                    kw["clause_types"] = routed[q["id"]]["clause_types"]
                if cids and len(cids) > 1:
                    hits, _ = per_contract_search(query, cids, per_contract=k // len(cids), **kw)
                else:
                    hits, _ = hybrid_search(query, contract_ids=cids, top_n=k, **kw)
                texts = [h.text for h in hits]
                found = [any(quote_in_text(e, t) for t in texts) for e in q["evidence"]]
                recalls.append(sum(found) / len(found))
                first = next((i for i, t in enumerate(texts, 1) if any(quote_in_text(e, t) for e in q["evidence"])), None)
                rr.append(1 / first if first else 0.0)
                hit1.append(1.0 if first == 1 else 0.0)
            rows.append({"query": source, "scope": scope, "retriever": name, f"evidence_recall@{k}": _mean(recalls),
                         "mrr": _mean(rr), "hit@1": _mean(hit1)})
            print(rows[-1])

    # Clause tagger (drives the metadata filter) vs CUAD expert labels, chunk-level micro P/R.
    tp = fp = fn = 0
    for c in get_chunk_store().all():
        if "gold_clause_types" not in c:
            continue
        gold, pred = set(c["gold_clause_types"]), set(c["clause_types"])
        tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
    tagger = {"recall": round(tp / (tp + fn), 3), "precision_vs_cuad": round(tp / (tp + fp), 3)}
    _dump(RESULTS / "retrieval.json", {"k": k, "n_questions": len(qa), "rows": rows, "tagger": tagger})
    print("tagger", tagger)


# --------------------------------------------------------------------------- agent runs
def cmd_agent(args) -> None:
    from langchain_core.callbacks import get_usage_metadata_callback

    from clausewatch.agent.graph import run

    opts = CONFIGS[args.config]
    out_dir = RESULTS / "runs" / args.config
    for q in load_qa():
        if args.only and q["id"] not in args.only:
            continue
        path = out_dir / f"{q['id']}.json"
        if path.exists() and not args.force:
            continue
        t0 = time.perf_counter()
        try:
            def _run():
                with get_usage_metadata_callback() as cb:
                    return run(q["question"], [], opts), cb.usage_metadata

            r, usage = call_waiting_for_quota(_run, label=q["id"])
        except Exception as e:  # not cached, so the next invocation retries this question
            print(f"{q['id']} FAILED: {type(e).__name__}: {str(e)[:200]}")
            continue
        r["latency_s"] = round(time.perf_counter() - t0, 2)
        r["usage"] = {m: dict(u) for m, u in usage.items()}
        r["llm_calls"] = sum(1 for e in r["trace"] if e["node"] in ("router", "draft")) + sum(
            1 for e in r["trace"] if e["node"] == "verify" and "claims supported" in e["message"]
        )
        _dump(path, r)
        toks = sum(u.get("total_tokens", 0) for u in r["usage"].values())
        print(f"{q['id']} {r['query_type']:13} {r['verification'].get('verdict', 'off'):12} "
              f"{r['latency_s']:6.1f}s {toks:6d} tok  {r['answer'][:80]!r}")


# --------------------------------------------------------------------------- scoring
ALL_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
_CITE = re.compile(r"\s*\[\d+(?:,\s*\d+)*\]")


async def _score_one(q: dict, r: dict, metrics: dict, judge, wanted: tuple[str, ...]) -> dict:
    from pydantic import BaseModel, Field

    from clausewatch.retrieval.store import get_chunk_store

    response = _CITE.sub("", r["answer"]).strip()
    # Give the judge the same labelled sources the generator saw: without the contract/section
    # header it cannot tell which agreement a clause belongs to (breaks comparisons).
    contexts = [
        f"Contract: {src['contract_title']} | Section: {src['section']}\n{text}"
        for src, text in zip(r["sources"], r["contexts"])
    ] or [""]
    row = {"id": q["id"], "type": q["type"], "query_type": r["query_type"],
           "verdict": r["verification"].get("verdict", "off"), "latency_s": r["latency_s"],
           "tokens": sum(u.get("total_tokens", 0) for u in r["usage"].values()),
           "routing_ok": set(q["contracts"]) <= set(r["target_contracts"]),
           "n_citations": len(r["citations"])}

    # Citation grounding (deterministic): cited quote appears verbatim in the cited chunk.
    store = get_chunk_store()
    cites = [c for c in r["citations"] if c.get("quote")]
    row["citation_grounding"] = (
        sum(quote_in_text(c["quote"], (store.get(c["chunk_id"]) or {}).get("text", "")) for c in cites) / len(cites)
        if cites else None
    )
    row["evidence_in_context"] = (
        _mean([float(any(quote_in_text(e, t) for t in r["contexts"])) for e in q["evidence"]]) if q["evidence"] else None
    )

    if q["type"] == "unanswerable":
        class Abstain(BaseModel):
            abstained: bool = Field(description="True if the response says the information is not in the contract and does not invent an answer.")
            explanation: str

        from clausewatch.llm import structured_runnable

        runnable = structured_runnable(judge, Abstain)
        v = await acall_waiting_for_quota(lambda: runnable.ainvoke(
            f"Question: {q['question']}\nResponse: {response}\n\nDid the response correctly decline to answer?"
        ), label="judge:abstain")
        row["abstained"] = v.abstained
        return row

    async def safe(name, coro):
        if name not in wanted:
            coro.close()
            return None
        try:
            return float((await asyncio.wait_for(coro, timeout=6 * 3600)).value)
        except Exception as e:  # a metric failing on one item should not kill the run
            log.warning("%s %s failed: %s", q["id"], name, e)
            return None

    row["judges"] = {m: get_settings().judge_model for m in wanted}
    row["faithfulness"] = await safe("faithfulness", metrics["faithfulness"].ascore(
        user_input=q["question"], response=response, retrieved_contexts=contexts))
    row["answer_relevancy"] = await safe("answer_relevancy", metrics["answer_relevancy"].ascore(
        user_input=q["question"], response=response))
    row["context_precision"] = await safe("context_precision", metrics["context_precision"].ascore(
        user_input=q["question"], reference=q["reference"], retrieved_contexts=contexts))
    return row


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    """Wall-clock limit that fires even if a worker thread is sleeping (SIGALRM on the main thread)."""
    def handler(signum, frame):
        raise _Timeout(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def cmd_score(args) -> None:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHF
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    from clausewatch.llm import get_judge_model

    sys.path.insert(0, str(EVAL_DIR))
    from ragas_adapter import LangChainRagasLLM

    s = get_settings()
    judge = get_judge_model()
    rllm = LangChainRagasLLM(judge)
    emb = RagasHF(model=s.embedding_model)
    metrics = {
        "faithfulness": Faithfulness(llm=rllm),
        "answer_relevancy": AnswerRelevancy(llm=rllm, embeddings=emb, strictness=3),
        "context_precision": ContextPrecision(llm=rllm),
    }
    run_dir, score_dir = RESULTS / "runs" / args.config, RESULTS / "scores" / args.config
    for q in load_qa():
        rp, sp = run_dir / f"{q['id']}.json", score_dir / f"{q['id']}.json"
        if not rp.exists():
            continue
        if sp.exists() and not args.force:
            prev = json.loads(sp.read_text())
            metrics_done = q["type"] == "unanswerable" or all(prev.get(m) is not None for m in args.metrics)
            if metrics_done:
                continue  # re-score only items where a metric failed (e.g. rate limit)
        rllm._sem = None
        try:
            with time_limit(args.item_timeout):
                row = call_waiting_for_quota(lambda: asyncio.run(_score_one(
                    q, json.loads(rp.read_text()), metrics, judge, tuple(args.metrics))), label=q["id"])
            if sp.exists():  # keep previously computed metrics that were not requested this time
                prev = json.loads(sp.read_text())
                for m in ALL_METRICS:
                    if m not in args.metrics and prev.get(m) is not None:
                        row[m] = prev[m]
                row["judges"] = {**prev.get("judges", {}), **row.get("judges", {})}
        except (Exception, _Timeout) as e:
            print(f"{q['id']} scoring FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        row["judge_model"] = s.judge_model
        _dump(sp, row)
        print({k: v for k, v in row.items() if k in ("id", "faithfulness", "answer_relevancy", "context_precision", "abstained")})


# --------------------------------------------------------------------------- risk demo
# Risk patterns that have a matching CUAD expert label (used to sanity-check flags).
PATTERN_TO_CUAD = {
    "uncapped_liability": ["Uncapped Liability"],
    "auto_renewal": ["Renewal Term", "Notice Period To Terminate Renewal"],
    "one_sided_termination": ["Termination For Convenience"],
    "broad_ip_assignment": ["Ip Ownership Assignment"],
    "restrictive_non_compete": ["Non-Compete", "Exclusivity", "No-Solicit Of Customers"],
    "change_of_control": ["Change Of Control"],
    "minimum_commitment": ["Minimum Commitment", "Volume Restriction"],
    "liquidated_damages": ["Liquidated Damages"],
}


def cmd_risk(args) -> None:
    from clausewatch.agent.graph import scan_risks
    from clausewatch.ingest.cuad import download_pdf, load_manifest
    from clausewatch.ingest.pipeline import ingest_pdf

    _, holdout = load_manifest()
    md = ["# Risk-flagging demo on unseen contracts\n",
          "Contracts below were held out of the indexed corpus, ingested as new uploads, then scanned.\n"]
    summary = []
    for c in holdout:
        path = RESULTS / "risk" / f"{c.title[:40]}.json"
        if path.exists() and not args.force:
            r = json.loads(path.read_text())
        else:
            contract = ingest_pdf(download_pdf(c), title=c.title, contract_type=c.contract_type,
                                  source="holdout", gold_labels=c.labels)
            r = call_waiting_for_quota(lambda: scan_risks(contract["contract_id"]), label=c.title[:30])
            r["contract_title"] = contract["title"]
            _dump(path, r)
        top = r["risk_flags"][: get_settings().risk_top_n]
        consistent = checkable = 0
        for f in top:
            if f["pattern_id"] in PATTERN_TO_CUAD:
                checkable += 1
                consistent += any(c.labels.get(cat) for cat in PATTERN_TO_CUAD[f["pattern_id"]])
        summary.append({"contract": r["contract_title"], "flags": len(top), "rejected_by_verifier": len(r["rejected_flags"]),
                        "cuad_checkable": checkable, "cuad_consistent": consistent})
        md.append(f"\n## {r['contract_title']}\n\n{r['answer']}\n")
        if r["rejected_flags"]:
            md.append("\n<details><summary>Rejected by verifier</summary>\n")
            for f in r["rejected_flags"]:
                md.append(f"\n- **{f['name']}**: {f.get('verifier_note') or 'quote not found verbatim in source'}")
            md.append("\n</details>\n")
    (RESULTS / "risk_demo.md").write_text("\n".join(md))
    _dump(RESULTS / "risk_summary.json", summary)
    for s_ in summary:
        print(s_)


# --------------------------------------------------------------------------- report
def cmd_report(args) -> None:
    lines = ["# Evaluation results\n"]
    s = get_settings()
    judges: dict[str, set] = {}
    for p in (RESULTS / "scores").glob("*/*.json"):
        for metric, model in json.loads(p.read_text()).get("judges", {}).items():
            judges.setdefault(metric, set()).add(model)
    judge_txt = ", ".join(f"{m}: `{'`, `'.join(sorted(v))}`" for m, v in sorted(judges.items())) or f"`{s.judge_model}`"
    lines.append(f"Generator: `{s.llm_model}` ({s.llm_provider}) | embeddings: `{s.embedding_model}` | "
                 f"reranker: `{s.reranker_model}`\n")
    lines.append(f"RAGAS judges - {judge_txt}\n")
    summary = {}
    for cfg in CONFIGS:
        rows = [json.loads(p.read_text()) for p in sorted((RESULTS / "scores" / cfg).glob("*.json"))]
        if not rows:
            continue
        ans = [r for r in rows if r["type"] != "unanswerable"]
        una = [r for r in rows if r["type"] == "unanswerable"]
        summary[cfg] = {
            "n": len(rows),
            "faithfulness": _mean([r.get("faithfulness") for r in ans]),
            "answer_relevancy": _mean([r.get("answer_relevancy") for r in ans]),
            "context_precision": _mean([r.get("context_precision") for r in ans]),
            "citation_grounding": _mean([r.get("citation_grounding") for r in ans]),
            "evidence_in_context": _mean([r.get("evidence_in_context") for r in ans]),
            "abstention": _mean([float(r["abstained"]) for r in una]) if una else None,
            "routing": _mean([float(r["routing_ok"]) for r in rows]),
            "latency_p50_s": round(statistics.median(r["latency_s"] for r in rows), 1),
            "tokens_per_q": round(statistics.mean(r["tokens"] for r in rows)),
        }
    _dump(RESULTS / "summary.json", summary)

    def fmt(v):
        return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

    if summary:
        cols = [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer relevancy"),
                ("context_precision", "Context precision"), ("citation_grounding", "Citation grounding"),
                ("abstention", "Abstention (unanswerable)"), ("routing", "Routing acc."),
                ("latency_p50_s", "p50 latency (s)"), ("tokens_per_q", "Tokens / question")]
        lines.append("## End-to-end (RAGAS + deterministic checks)\n")
        lines.append("| Config | " + " | ".join(c[1] for c in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for cfg, m in summary.items():
            lines.append(f"| {cfg} | " + " | ".join(fmt(m[c[0]]) for c in cols) + " |")

    rp = RESULTS / "retrieval.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        k = r["k"]
        lines.append(f"\n## Retrieval ablation ({r['n_questions']} questions, no LLM)\n")
        lines.append(f"| Query | Scope | Retriever | Evidence recall@{k} | MRR | Hit@1 |")
        lines.append("|---|---|---|---|---|---|")
        for row in r["rows"]:
            lines.append(f"| {row['query']} | {row['scope']} | {row['retriever']} | {row[f'evidence_recall@{k}']:.2f} | "
                         f"{row['mrr']:.2f} | {row['hit@1']:.2f} |")
        t = r["tagger"]
        lines.append(f"\nClause tagger vs CUAD expert labels (chunk level): recall {t['recall']:.2f}, "
                     f"precision {t['precision_vs_cuad']:.2f}.\n")

    sp = RESULTS / "risk_summary.json"
    if sp.exists():
        lines.append("\n## Risk flagging on held-out contracts\n")
        lines.append("| Contract | Flags | Rejected by verifier | Consistent with CUAD labels |")
        lines.append("|---|---|---|---|")
        for x in json.loads(sp.read_text()):
            lines.append(f"| {x['contract']} | {x['flags']} | {x['rejected_by_verifier']} | "
                         f"{x['cuad_consistent']}/{x['cuad_checkable']} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("retrieval")
    for name in ("agent", "score"):
        p = sub.add_parser(name)
        p.add_argument("--config", choices=list(CONFIGS), default="full")
        p.add_argument("--only", nargs="*")
        p.add_argument("--force", action="store_true")
        if name == "score":
            p.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=list(ALL_METRICS))
            p.add_argument("--item-timeout", type=int, default=2700, help="seconds per question (includes quota waits)")
    p = sub.add_parser("risk")
    p.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
    {"retrieval": cmd_retrieval, "agent": cmd_agent, "score": cmd_score, "risk": cmd_risk,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
