# Colab run lp_4b, 10 Sep 2026 evening — raw cell outputs, saved verbatim

Qwen3.5-4B on a Colab T4, the full 371-row hand-checked dataset (250 design rows, 6 excluded; 60 simple anchors; 61 negations).
One file per notebook cell, copied from the notebook Martin downloaded after the run (`legality_probe_colab_with_outputs.ipynb`
in this folder is that notebook, outputs included). Nothing edited except stripping terminal colour codes.

Known problems in this run:
- **09 just-ask:** did not run in the main session; the pilot's `ask_lp_4b_*.jsonl` files from 9 Sep were still on Drive and the old script refused to
  overwrite. The script now backs the old file up; cell 9 ran at 20:45, see 13-followup.
- **08b:** the lines "81% right overall, 62% on the off-diagonal" and "91% / 82%" are the 9 Sep pilot's just-ask numbers on the old dataset. Never quote them.
- **10 recompute:** ran the old cell, which let the 121 check rows into the test set (0.683 vs the script's 0.700, and "mistakes"
  that include plain acts). Its mistake list is invalid. The corrected recompute is `notebooks/verify_by_hand.py`.
- **11 steering:** ran once on the prompted condition; cut from the plan on 10 Sep and not reported anywhere.
- **13 followup** (run from the Mac, same files): holds the just-ask baseline of record (20:45 CEST) and an agent recompute of the headline
  numbers, the mistakes and the fair baseline (0.81 vs 0.74). The recompute does not count until Martin repeats it and logs it in `verification-log.md`.
- **12 push:** rejected as non-fast-forward (the Mac pushed in the meantime). Journal and figures are committed from the Mac instead.
- **08 table:** the markdown exporter crashed on a stale pilot file; the printed table is complete and correct.

The numbers in 07 and 08 are the results of record for the four conditional-generalisation tests in both conditions.
