#!/usr/bin/env bash
# The follow-up cell for the lp_4b run: the just-ask baseline (cell 9), the by-hand recompute (cell 10),
# the three checks (markedness, geometry, borderline scores), then the table and figures. Run from the repo root
# on the Colab runtime that holds data/processed/acts_lp_4b*.npz:   bash scripts/colab_followup.sh
set -e
RUN=${1:-lp_4b}
echo "=== just-ask baseline (cell 9)"
python scripts/ask_model.py --run $RUN --label legal 2>&1 | grep -v 'it/s]' | tail -4
python scripts/ask_model.py --run $RUN --label harmful 2>&1 | grep -v 'it/s]' | tail -4
for r in $RUN ${RUN}_prompted; do
  echo; echo "=== recompute (cell 10): $r"; python scripts/recompute_headline.py --run $r
  echo; echo "=== markedness: $r"; python scripts/markedness.py --run $r | grep -E '^\[|VERDICT'
  echo; echo "=== geometry: $r"; python scripts/geometry.py --run $r | head -6
  echo; echo "=== borderline scores: $r"; python scripts/borderline_scores.py --run $r | head -12
done
echo; echo "=== table and figures"; python scripts/results_table.py --run $RUN | tail -6; python scripts/figures.py --run $RUN --title Qwen3.5-4B
echo ALLDONE
