"""Inject eval/results/summary.md into the README between the RESULTS markers.

    python eval/run_eval.py report && python scripts/update_readme_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- RESULTS -->", "<!-- /RESULTS -->"

NOTES = """
### How to read these numbers

- **20 hand-written questions** over the 30 indexed contracts: 16 single-contract lookups, 2 cross-document
  comparisons, and 2 whose answer is deliberately *absent* from the contract. Every supporting quote in
  `eval/qa_set.jsonl` was checked to exist verbatim in the indexed text before the run.
- **Judges.** RAGAS metrics are LLM-judged. Faithfulness (both configs, so the ablation is like-for-like) is
  judged by `openai/gpt-oss-20b`; answer relevancy and context precision by `qwen/qwen3.8-27b` - both
  different from the generator, which is `openai/gpt-oss-120b`. The judge sees the same labelled sources
  ("Contract: X | Section: Y") the generator saw; without those labels it cannot attribute a clause to a
  contract and scores correct comparison answers as unfaithful.
- **Judge noise is larger than the ablation gap.** Re-scoring identical answers moved individual
  faithfulness scores by up to 0.25, so with n=18 the ~0.03 difference in mean faithfulness is *not*
  meaningful. The deterministic metrics are the trustworthy ones.
- **Citation grounding** is not LLM-judged: it is the fraction of returned citations whose quote appears
  verbatim (whitespace/punctuation-insensitive) in the chunk it cites. This is the metric the verifier is
  designed to protect, and it is where the ablation shows a clean difference.
- **Context precision** is computed over the 6-8 chunks handed to the generator. A low score with a correct
  answer means the right clause ranked late (e.g. q02, where the verifier's second retrieval pass appended
  the decisive clause last).
- Full per-question rows: `eval/results/scores/`. Agent traces: `eval/results/runs/`.
- **Known gap:** q15's context precision was scored before the labelled-context fix (a single-contract
  question, where labels do not change attribution). Re-run `python eval/run_eval.py score --config full
  --only q15 --metrics context_precision --force` to refresh it.

### What the verifier buys

| | Verifier ON | Verifier OFF |
|---|---|---|
| Citation grounding (deterministic) | **1.00** | 0.94 |
| Answers scored 0.00 on faithfulness | **0** | 1 |
| Correct abstentions on unanswerable questions | 2/2 | 2/2 |
| Median latency | 30.6 s | 17.6 s |
| Tokens per question | 7.0k | 4.3k |

The clearest single case is **q02** (cap on total liability). Without the verifier the agent stops at
*"Insufficient context to determine the cap"*. With it, the draft's `insufficient_context` flag triggers a
second retrieval pass with the verifier's own query, and the answer comes back correct and cited. The cost
is roughly 1.7x latency and 1.6x tokens.
"""


def main() -> None:
    summary = (ROOT / "eval" / "results" / "summary.md").read_text().strip()
    summary = summary.split("\n", 1)[1].strip()  # drop the "# Evaluation results" heading
    summary = summary.replace("\n## ", "\n### ")  # nest under the README's "## Evaluation"
    readme = (ROOT / "README.md").read_text()
    block = f"{START}\n## Evaluation\n\n{summary}\n{NOTES}\n{END}"
    if START in readme and END in readme:
        readme = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, readme, flags=re.S)
    else:
        readme = readme.replace(START, block)
    (ROOT / "README.md").write_text(readme)

    scores = ROOT / "eval" / "results" / "scores"
    pending = [
        p.stem
        for p in sorted((scores / "full").glob("*.json"))
        if (row := json.loads(p.read_text()))["type"] != "unanswerable"
        and not all(row.get(m) is not None for m in ("faithfulness", "answer_relevancy", "context_precision"))
    ]
    print("README results updated." + (f" Still missing metrics: {pending}" if pending else " All metrics present."))


if __name__ == "__main__":
    main()
