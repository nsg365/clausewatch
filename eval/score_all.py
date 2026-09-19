"""Score the QA set one question per subprocess, with a hard timeout per question.

RAGAS + a rate-limited free-tier judge can wedge a process (a blocking sleep on the event
loop thread survives in-process timeouts), so each question runs in its own process and is
killed and retried if it overruns.

    python eval/score_all.py --config full --metrics faithfulness context_precision --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def load_ids() -> list[str]:
    return [json.loads(l)["id"] for l in (ROOT / "eval" / "qa_set.jsonl").read_text().splitlines() if l.strip()]


def complete(path: Path, metrics: list[str]) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    if row.get("type") == "unanswerable":
        return row.get("abstained") is not None
    return all(row.get(m) is not None for m in metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    ap.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=list(ALL_METRICS))
    ap.add_argument("--force", action="store_true", help="re-score even if the metrics are already present")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per question before the process is killed")
    ap.add_argument("--attempts", type=int, default=4)
    args = ap.parse_args()

    scores = ROOT / "eval" / "results" / "scores" / args.config
    todo = []
    for qid in load_ids():
        path = scores / f"{qid}.json"
        if args.force or not complete(path, args.metrics):
            todo.append(qid)
    print(f"[score_all] {args.config}: {len(todo)} question(s) to score: {todo}", flush=True)

    for qid in todo:
        path = scores / f"{qid}.json"
        before = path.read_text() if path.exists() else None
        for attempt in range(1, args.attempts + 1):
            cmd = [sys.executable, "eval/run_eval.py", "score", "--config", args.config,
                   "--only", qid, "--metrics", *args.metrics]
            # After the first pass the row exists but is incomplete, so --force is needed to redo it.
            if args.force or before is not None:
                cmd.append("--force")
            t0 = time.time()
            try:
                subprocess.run(cmd, cwd=ROOT, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"[score_all] {qid}: attempt {attempt} timed out after {args.timeout}s", flush=True)
                continue
            if complete(path, args.metrics):
                print(f"[score_all] {qid}: done in {time.time() - t0:.0f}s "
                      f"-> { {k: v for k, v in json.loads(path.read_text()).items() if k in ALL_METRICS or k == 'abstained'} }",
                      flush=True)
                break
            print(f"[score_all] {qid}: attempt {attempt} incomplete, retrying", flush=True)
        else:
            print(f"[score_all] {qid}: giving up after {args.attempts} attempts", flush=True)
    print("[score_all] finished", flush=True)


if __name__ == "__main__":
    main()
