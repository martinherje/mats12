#!/usr/bin/env bash
# Follow-up for the lp_4b run: the by-hand recompute, the three side checks (markedness, geometry, borderline
# scores), then the table and the figure. The just-ask baseline of record ran on 10 Sep 20:45 (data/raw/ask_lp_4b_*.jsonl,
# data/processed/ask_lp_4b_*.json) and is not re-run here: ask_model.py would move the raw file of record to .bak
# and answer again. Run from the repo root on the Colab runtime that holds data/processed/acts_lp_4b*.npz:
#   bash scripts/colab_followup.sh
set -e
RUN=${1:-lp_4b}
for r in $RUN ${RUN}_prompted; do
  echo; echo "=== recompute: $r"; python scripts/recompute_headline.py --run $r
  echo; echo "=== markedness: $r"; python scripts/markedness.py --run $r | grep -E '^\[|^Reading'
  echo; echo "=== geometry: $r"; python scripts/geometry.py --run $r | head -6
  echo; echo "=== borderline scores: $r"; python scripts/borderline_scores.py --run $r | head -24
done
echo; echo "=== table and figure"; python scripts/results_table.py --run $RUN; python scripts/figures.py --run $RUN   # writes journal/results.md and figures/fig1_$RUN.png
echo ALLDONE
