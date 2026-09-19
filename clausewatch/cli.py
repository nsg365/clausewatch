"""Command line entry point.

    python -m clausewatch.cli ingest                 # download + index the CUAD subset
    python -m clausewatch.cli add path/to.pdf        # index a new contract
    python -m clausewatch.cli contracts
    python -m clausewatch.cli ask "question" [-c CONTRACT_ID ...] [--no-verify] [--json]
    python -m clausewatch.cli risk CONTRACT_ID|path/to.pdf
    python -m clausewatch.cli graph                  # print the LangGraph as mermaid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _print_trace(trace: list[dict]) -> None:
    print("\n--- trace ---", file=sys.stderr)
    for e in trace:
        print(f"[{e['node']:>9}] {e.get('ms', 0):>6} ms  {e['message']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clausewatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--holdout", type=int, default=3)
    p = sub.add_parser("add")
    p.add_argument("pdf")
    p.add_argument("--type", default="Unknown")
    sub.add_parser("contracts")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("-c", "--contract", action="append", default=[])
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("risk")
    p.add_argument("target", help="contract_id or path to a PDF (ingested first)")
    p.add_argument("--json", action="store_true")
    sub.add_parser("graph")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
    if args.cmd == "ingest":
        logging.getLogger("clausewatch").setLevel(logging.INFO)
        from clausewatch.ingest.pipeline import ingest_cuad_subset

        out = ingest_cuad_subset(args.n, args.holdout)
        print(f"indexed {len(out)} contracts, {sum(c['n_chunks'] for c in out)} chunks")
    elif args.cmd == "add":
        from clausewatch.ingest.pipeline import ingest_pdf

        print(json.dumps(ingest_pdf(args.pdf, contract_type=args.type), indent=2))
    elif args.cmd == "contracts":
        from clausewatch.retrieval.store import get_chunk_store

        for cid, c in get_chunk_store().contracts().items():
            print(f"{cid:58} {c['contract_type']:22} {c['title']}")
    elif args.cmd == "ask":
        from clausewatch.agent.graph import run

        r = run(args.question, args.contract, {"verify": not args.no_verify})
        if args.json:
            print(json.dumps(r, indent=2))
            return
        _print_trace(r["trace"])
        print("\n" + r["answer"] + "\n")
        for c in r["citations"]:
            print(f"[{c['n']}] {c['ref']}")
            if c["quote"]:
                print(f'    "{c["quote"].strip(chr(34))}"')
    elif args.cmd == "risk":
        from clausewatch.agent.graph import scan_risks

        target = args.target
        if target.lower().endswith(".pdf") and Path(target).exists():
            from clausewatch.ingest.pipeline import ingest_pdf

            target = ingest_pdf(target)["contract_id"]
        r = scan_risks(target)
        if args.json:
            print(json.dumps(r, indent=2))
            return
        _print_trace(r["trace"])
        print("\n" + r["answer"])
    elif args.cmd == "graph":
        from clausewatch.agent.graph import get_graph

        print(get_graph().get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
