#!/usr/bin/env bash
# Full evaluation pipeline. Every step is cached/resumable: re-run this script after a
# rate-limit interruption and it continues where it stopped.
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python

steps=(
  "agent --config full"
  "risk"
  "agent --config no_verifier"
  "retrieval"
  "report"
)
for step in "${steps[@]}"; do
  echo "### $step"
  # shellcheck disable=SC2086
  $PY eval/run_eval.py $step 2>&1 | grep -v -i -E "warn|Loading weights|it/s\]"
done
# One process per question, with a hard timeout (see eval/score_all.py).
# Faithfulness (the verifier ablation) runs on a second model family with its own daily quota,
# so the two metric groups draw from separate free-tier buckets.
FAITH_JUDGE=${CLAUSEWATCH_FAITHFULNESS_JUDGE:-openai/gpt-oss-20b}
echo "### score_all faithfulness, both configs (judge: $FAITH_JUDGE)"
CLAUSEWATCH_JUDGE_MODEL=$FAITH_JUDGE CLAUSEWATCH_JUDGE_MAX_TOKENS=2500 \
  $PY eval/score_all.py --config full --metrics faithfulness --timeout 10800
CLAUSEWATCH_JUDGE_MODEL=$FAITH_JUDGE CLAUSEWATCH_JUDGE_MAX_TOKENS=2500 \
  $PY eval/score_all.py --config no_verifier --metrics faithfulness --timeout 10800
echo "### score_all full: context precision (judge: default)"
$PY eval/score_all.py --config full --metrics context_precision ${FORCE_CP:+--force} --timeout 10800
echo "### report"
$PY eval/run_eval.py report
echo "### ALL DONE"
