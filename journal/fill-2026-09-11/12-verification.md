## Verification and reproducibility, paste-ready (Martin's voice; edit the bracketed parts to match what is actually done at submission)

### Verification

Every number in this write-up was produced by scripts Claude wrote, so before quoting one I re-ran it on my own machine from the files of record, following a written plan with expected outputs (journal/verification-plan.md). Claude had run the same commands earlier the same day (journal/verification-dryrun-2026-09-11-claude.md); I used that output as the thing to compare against, not as the check. A number counts as verified only once I have run it and written the log row in my own words (journal/verification-log.md).

Done and matching at the time of writing, from data/scenarios.csv alone: the dataset counts (371 rows read, 250 design rows, 60 plain acts, 61 negations; 6 excluded; borderline on legality / harm / both 66 / 46 / 23; quadrants 62 / 65 / 62 / 61) and the six random rows drawn with a fixed seed after the dataset was frozen (s286, s163, s123, s210, s206, s215, listed under the summary). I would be astonished if these were wrong; I made every one of those marks myself, and git would show any edit to the CSV after the draw.

[Paste this paragraph once step 1c and step 2 have matched on the PC; otherwise use the next one.] Also re-run by me: the topic split falls out of seed 0 on the sorted topic list and is identical in all nine result files, and the table numbers were refit from the saved activations by a second code path (notebooks/verify_by_hand.py: sklearn refit at the saved layer and C, rows counted, AUROC by the rank formula written out in numpy, no call into probe_eval.py). Sentence only: 21 of 30 right, AUROC 0.742, 12 of 30 called illegal, interval 0.53 to 0.83; plain acts 33 of 60 called legal, negations 61 of 61; cos(d_illegal, d_harm) +0.09 at layer 25, inside a fresh 200-swap band of −0.28 to +0.30; the illegality direction within the harmless rows 0.676 as is and 0.676 with the harm direction projected out; a fixed-layer null of 200 fresh shuffles beaten 200 of 200. Sentence + question: 15 of 30, AUROC 0.742, 0 of 30 called illegal, interval 0.50 to 0.50; 58 of 60 and 58 of 61; cos +0.70 at layer 15, outside a band of −0.54 to +0.51; 0.849 to 0.738; null beaten 77 of 200. Every value equals the table of record. Two differences from the script's numbers are expected and are not errors: the sentence + question reverse-legality AUROC comes out 0.911 against the script's 0.907 (one rank pair; a different lbfgs build on my machine), and bootstrap interval edges move by one row because the resampling seed differs.

[Use this paragraph instead if the doc is submitted before step 2 has run.] Not yet re-run by me at submission: the topic-split check and the refit of the table numbers from the activations. Claude's own dry run of those commands matched the table exactly (sentence only 21 of 30, 0.742, 12 of 30 called illegal; sentence + question 15 of 30, 0.742, 0 of 30 called illegal), but that is the agent checking its own work, so every table number stands as agent-produced until I have run notebooks/verify_by_hand.py myself. I would be very surprised if they were wrong, since seed 0 fixes the split and the refit path is independent of the evaluation script, but surprised is not checked.

[Paste once step 4 has run.] The fair baseline I recomputed from the raw answers file (data/raw/ask_lp_4b_legal.jsonl) with the same rank formula: on the 30 held-out harmless-stratum rows the probe is tested on, the model's own Yes−No logit sorts illegal from legal at 0.81 against the probe's 0.74; on the 30 harmful-stratum rows of the reverse test, 1.00; the model answers "No" to all 60 plain acts and says "not illegal" to 10 of the 15 illegal-harmless test rows. I would be surprised if this were wrong and I am not surprised by the direction; the pre-registration named it as a likely outcome.

What I did not verify, and the honest sentence for each:

- That batched extraction equals an unbatched forward pass on the 4B. Claude checked this on a Qwen2.5-0.5B stand-in on 8 Sep (max relative difference 4.5e-4, fp16 rounding); nobody has repeated it on the 4B. Low surprise if wrong, but it would be my error to own.
- The activations of record were stored in half precision (the fp32 fix landed forty minutes after the extraction ran). Values are finite, max |value| 104 in the sentence-only run and 36 in the sentence + question run, so nothing overflowed; a fresh extraction stores fp32 and will move numbers in the third decimal, and I have not done it.
- The 100-shuffle selection-inclusive nulls were not rerun (about 9 minutes per design on the T4, 80 for all eight; the JSON keeps only mean, p95, max and the count). The substitute is checking that each file's null maximum sits on the correct side of its test accuracy, plus the cheaper fixed-layer 200-shuffle null in verify_by_hand.py, which gives the same verdict for every design.
- The recompute cell that ran on Colab on 10 Sep was an old version that leaked the 121 check rows into the test set and printed 0.683 against 0.700. That number and its mistake list are invalid and are not used anywhere; the corrected path is verify_by_hand.py.
- The no-cue rerun (0.74, 100 of 100), the markedness, geometry and borderline side checks were not recomputed by hand and are quoted only with that label, or not at all. Steering was cut and is not reported.
- The labels are mine and cannot be machine-checked: one annotator, US law as of 2025. The model itself calls about two thirds of my illegal-but-harmless rows legal, and I did not get a second lawyer to arbitrate.
- The hours table is reconstructed from my estimates and commit timestamps, not from a timer; about 8 counted hours before the write-up (journal/hours.md).

### Reproducing it

The code, the dataset (data/scenarios.csv, column guide in data/SCENARIOS_COLUMNS.md), the table of record (journal/results.md, written by scripts/results_table.py), the raw cell outputs of the run including the notebook with outputs (journal/colab-run-lp_4b-2026-09-10/) and the figure (figures/fig1_lp_4b.png) are in the repository github.com/martinherje/mats12. The result JSONs, the raw answer files and the activations are gitignored for size; they are in a Google Drive folder, mats12_runs/data/processed and data/raw (probeeval_lp_4b*.json, ask_lp_4b_*.json, acts_lp_4b.npz and acts_lp_4b_prompted.npz, 58 MB each).

A full rerun is notebooks/legality_probe_colab.ipynb, cells 0 to 12, on a free Colab T4, loading Qwen/Qwen3.5-4B from the Hugging Face weights in fp16 (the T4 has no bf16); extraction batch size 16, max length 256; the eight probe evaluations take about 64 minutes. Seed 0 fixes the split, the bootstrap and the shuffles, so probe_eval.py reproduces the JSON bit for bit from the same activations; that shows determinism, not independence. A fresh extraction stores fp32, so expect third-decimal drift against the numbers here.

Without a GPU, every number in the tables can be recomputed from the saved activations in seconds: `uv run python notebooks/verify_by_hand.py --run lp_4b --acts-dir <folder with the npz files>`, the same with `--run lp_4b_prompted`, and `--tag L_nh2h`, `H_i2l`, `H_l2i` for the other designs; `scripts/recompute_headline.py` is the shorter headline-only check; `scripts/results_table.py --run lp_4b` regenerates the table.

## Links that belong in the doc or the form, with status

| Link | URL | Where found | Public? |
|---|---|---|---|
| Google Doc (the deliverable) | https://docs.google.com/document/d/1m1WZNDAqC2N7Rf0eBecJhofWlo0cI_3DTu_rtOxhn10/edit | run sheet line 6; vault CHANGELOG 11 Sep 16:14 | Not yet. Sharing "anyone with the link" was left to Martin (CHANGELOG 11 Sep 16:14). Must be set before the form checkbox is ticked. |
| Repo | https://github.com/martinherje/mats12 | README; vault CHANGELOG 8 Sep 19:52 | Private (confirmed `gh repo view`: visibility PRIVATE). Form's optional code link only if made public, or say "private, access on request". |
| Colab working copy (trimmed 10 Sep, the one the run used) | https://colab.research.google.com/drive/1LuHofS0IRvu9eKmjOWx9I2jBrVPEk-E_ | run sheet line 18 | In Martin's Drive; private unless shared. Cell 3 clones the private repo with a Colab secret, so a reader can read it but not run it without repo access. |
| Colab 9 Sep copy | https://colab.research.google.com/drive/1kApytH2NJIXPBXDcDpgFjHwkVvjnQQU1 | vault CHANGELOG 9 Sep 16:35 | Stale ("the 9 Sep copy is stale", run sheet line 18). Do not link. |
| Colab GitHub-sourced link | https://colab.research.google.com/github/martinherje/mats12/blob/main/notebooks/mats12_colab.ipynb | journal/archive-value-leakage.md line 7 | Points at the abandoned value-leakage notebook and needs repo access. Do not link. (The legality equivalent would be the same path ending `legality_probe_colab.ipynb`; not found in any file, only works if the repo goes public.) |
| Drive folder with JSONs and activations | none in any file; the folder is My Drive/mats12_runs (mounted at ~/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/) | verification-plan.md line 75 | No share link exists. If the doc says "on request", nothing to do; if it is to be linked, Martin creates the share link in Drive himself. |
| Airtable form | https://airtable.com/appnMboxg76F1QIDc/pagqu7wWWrUCZkNVI/form | journal/form-answers.md line 3 | The submission target, not a link for the doc. |

## Errors and traps found while checking the sources for this section

1. Form Q4, Martin's draft: "Model: Qwen 3.5 4b (through GitHub API calls)". Correct: "Model: Qwen3.5-4B, loaded from the Hugging Face weights on a free Colab T4 in fp16 (no API)". Sources: data/processed/acts_lp_4b.json manifest (`dtype: torch.float16`, `device: cuda`); journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt (Tesla T4, 15360 MiB); scripts/common.py `pick_dtype`; writeup-facts.md "Model and hardware".

2. writeup-facts.md line 39 says "Independent hand refit (journal/verification-log.md row 2026-09-11 ...): headline 21 of 30, AUROC 0.742". No Martin-authored row exists; those rows sit under the heading "Drafted by Claude 10 Sep, each row to be re-run and reworded by Martin before it counts" (verification-log.md line 50). The repo at 8123557 has a clean working tree and no journal/verify_20260911.txt. Do not carry that line's framing into the doc.

3. Skeleton limitation 12 (writeup-skeleton.md line 88) and form-answers Q6 still say the corrected recompute "has not yet been run by Martin". True as of the repo; rewrite to whatever is true at submission, using the two alternative paragraphs above.

4. verification-plan.md step 4a tells him to confirm the fair baseline with scripts/report.py; that script was deleted in cf40386. The same line is the last line of journal/results.md, printed by `uv run python scripts/results_table.py --run lp_4b`.

5. "9 of 15" versus "10 of 15" illegal-harmless test rows answered "not illegal": the plan says 9 (line 78), the drafted log row says 10 (line 65). Recomputed tonight from data/raw/ask_lp_4b_legal.jsonl with the plan's own step 4a one-liner: 10 of 15 said No; 6 of 30 rows got Yes (5 illegal, 1 legal); answer accuracy 0.633; logit AUROC 0.809. Team-pass summary section 4 also gives 10. Expect 10 when he runs it; it is his run that settles it.

6. Martin's draft: "relabelling 4 (on harmfulness/legality) from Claude's original labels". Step 1a prints `relabelled 6` from the CSV column (dry run line 284; plan line 110), and writeup-facts.md "Who changed what" lists 4 labels changed by Martin (s087, s126, s218, s057) against a column count of 6. A reader who runs step 1a sees 6. Either quote the column's 6 or add one clause saying the column also counts label fixes made at merge time; do not leave 4 and 6 to meet unexplained. Standing convention (4) not re-litigated; the mismatch is flagged only.

7. The prompted cosine leaves the band at layer 14, not 13 (verification-plan.md finding 6; probeeval_lp_4b_prompted_L_h2nh.json cosine_curve, layer 13 = +0.26 inside, layer 14 = +0.61 outside). The vault run sheet's 20:30 decision row still says "from layer 13" and "0.85 → 0.64"; the 21:35 row corrects both. Do not copy from the 20:30 row.

## Source table for every number in the paste text

| Number | File |
|---|---|
| 371 / 250 / 60 / 61; 6 excluded; 66 / 46 / 23; 62 / 65 / 62 / 61 | verification-dryrun-2026-09-11-claude.md lines 283 to 286; verification-plan.md step 1a expected line 110 |
| s286, s163, s123, s210, s206, s215 | dryrun lines 292 to 297; plan line 119 |
| 21 of 30, 0.742, 12 of 30 called illegal, 0.53 to 0.83 | dryrun lines 311 to 314; journal/results.md row 1; probeeval_lp_4b_L_h2nh.json (test_cross_acc 0.700, test_cross_auroc 0.7422) |
| 33 of 60, 61 of 61 | dryrun lines 349 to 350 |
| cos +0.091 at layer 25, band −0.28 to +0.30 | dryrun lines 352 to 353 (JSON key cos_dlegal_dharm is −0.0909, sign flipped) |
| 0.676 to 0.676; 200 of 200 | dryrun lines 354 to 355, 360 |
| 15 of 30, 0.742, 0 of 30 called illegal, 0.50 to 0.50 | dryrun lines 373 to 376 |
| 58 of 60, 58 of 61 | dryrun lines 411 to 412 |
| cos +0.695 at layer 15, band −0.54 to +0.51 | dryrun lines 414 to 415 |
| 0.849 to 0.738; 77 of 200 | dryrun lines 416 to 417, 422 |
| 0.911 vs 0.907; bootstrap edges by one row | verification-plan.md finding 3; team-pass-summary section 4 |
| 0.81 vs 0.74; 1.00; 60 plain acts all No; 10 of 15 | journal/results.md line 43; recomputed tonight from data/raw/ask_lp_4b_legal.jsonl (plan step 4a one-liner); ask_lp_4b_legal.json git c61feea, n 244 |
| 4.5e-4 on the 0.5B, 8 Sep | verification-log.md line 46 |
| fp16 storage; max 104 / 36; fix landed after the run | dryrun lines 308, 370; team-pass-summary section 4 (extraction at d920864 18:48, fp32 fix 6cd9f80 19:42) |
| 9 min per design, 80 for eight | verification-plan.md "What cannot be verified" |
| 0.683 vs 0.700, old cell 10 | verification-log.md line 48; journal/colab-run-lp_4b-2026-09-10/README.md |
| no-cue 0.74, 100 of 100 | journal/results.md; data/processed/probeeval_lp_4b_L_h2nh_nocue.json |
| about two thirds of illegal-harmless rows called legal | journal/results.md line 40 (illegal-harmless 34% right) |
| about 8 h counted | journal/hours.md |
| batch 16, max length 256, 64 min, T4 fp16 | writeup-facts.md "Model and hardware"; colab-run 06-extraction.txt, 01-gpu-check.txt |
| 58 MB npz files, Drive timestamps | Drive listing: acts_lp_4b.npz 58,021,836 B (10 Sep 18:48), acts_lp_4b_prompted.npz 57,897,696 B (18:50); ask JSONs 20:45 and 20:47 |
| JSONs gitignored | .gitignore lines 4 and 5 (`data/raw`, `data/processed`); `git check-ignore` confirms |
| repo private | `gh repo view martinherje/mats12`: visibility PRIVATE |