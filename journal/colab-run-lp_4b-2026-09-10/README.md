# Colab run lp_4b, 10 Sep 2026 evening — raw cell outputs, saved verbatim

Qwen3.5-4B on a Colab T4, the full 371-row hand-checked dataset (250 design rows, 6 excluded; 60 simple anchors; 61 negations).
One file per notebook cell, copied from the notebook Martin downloaded after the run (`legality_probe_colab_with_outputs.ipynb`
in this folder is that notebook, outputs included). Nothing edited except stripping terminal colour codes.

Known problems in this run, to be fixed by re-running the cell after `git pull`:
- **09 just-ask:** did not run — the pilot's `ask_lp_4b_*.jsonl` files from 9 Sep were still on Drive and the old script refused to
  overwrite. The "just asking the model" lines in 08b therefore come from the 9 Sep pilot on the *old* dataset, and the fair-baseline
  line is missing. The script now backs the old file up; cell 9 must be re-run.
- **10 recompute:** ran the old cell, which let the 121 check rows into the test set (0.683 vs the script's 0.700, and "mistakes"
  that include plain acts). The corrected cell is `notebooks/cell_10_replacement.py`.
- **12 push:** rejected as non-fast-forward (the Mac pushed in the meantime). Journal and figures are committed from the Mac instead.
- **08 table:** the markdown exporter crashed on a stale pilot file; the printed table is complete and correct.

The numbers in 07 and 08 are the results of record for the four conditional-generalisation tests in both conditions.
