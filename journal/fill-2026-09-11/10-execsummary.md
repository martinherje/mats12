# Executive summary (paste-ready)

**Title:** Does Qwen3.5-4B keep illegality apart from harm? A topic-matched illegality and harm probe test

Harmfulness and illegality are often intertwined, and almost as often conflated. If a model runs the two together, an illegal but harmless act reads as legal, and a legal but harmful act as illegal. For law-following AI that is the failure that matters. This experiment asks whether Qwen3.5-4B represents illegality as distinct from harm, and whether an illegality probe reads illegality or just reads harm.

What I did. 371 single-sentence scenarios drafted by Claude, every row read by me (a lawyer): 4 labels changed, 6 rows excluded, 7 texts edited. The 250 design rows cover 60 topics, one sentence per combination of legal/illegal and harmful/harmless (US law); 60 plain acts and 61 negations are never trained on. Residual stream at the last token, two conditions: sentence only, and sentence + question, the sentence inside "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.", read at the token before the answer. A logistic probe for illegality is trained on the harmful stratum only and tested on the harmless stratum of 15 unseen topics, layer and C chosen on 15 validation topics. On the easy corners legal and harmless are the same label, so whatever carries across is not harm. The null reruns the whole procedure 100 times on labels shuffled within topic and stratum.

Results. AUROC on the 30 held-out rows of the other stratum, and shuffled-label runs beaten of 100:

| probe, trained → tested | sentence only | sentence + question |
|---|---|---|
| illegality, harmful → harmless | 0.74, 100/100 | 0.74, 100/100 |
| illegality, harmless → harmful | 0.71, 97/100 | 0.91, 100/100 |
| harm, illegal → legal | 0.95, 100/100 | 0.97, 100/100 |
| harm, legal → illegal | 0.83, 100/100 | 0.99, 100/100 |

cos(d_illegal, d_harm) at the probe's layer: +0.09 sentence only (layer 25, label-swap band −0.28 to +0.26, inside); +0.70 sentence + question (layer 15, band −0.54 to +0.49, outside from layer 14 on). With the harm direction projected out, the illegality direction's AUROC within the harmless rows goes 0.68 → 0.68 sentence only and 0.85 → 0.74 sentence + question.

The key finding. The model keeps illegality and harm apart while it reads a sentence and lines them up as it prepares a yes/no. Its stated answer is harm-gated: asked whether the act is illegal, it says yes on 97% of illegal-harmful rows, 34% of illegal-harmless, 25% of legal-harmful and 3% of legal-harmless (mean Yes−No logit +3.82, −0.98, −0.99, −3.46). Within each stratum the graded logit still ranks illegal above legal (0.83, 0.95). The probe shows the same from inside: under the question it ranks the 30 harmless test rows at 0.74 but calls all 30 legal. And the probe does not beat asking: on the same 30 rows the model's own logit scores 0.81 against the probe's 0.74.

Checks. Dropping the 3 design rows with a legality word leaves 0.74, 100/100. Word count alone scores 0.60 on the headline rows, but 0.79 on the reverse rows against that probe's 0.71, so the reverse test is not evidence. Negations: 61/61 and 58/61 called legal. One check failed: the sentence-only headline probe calls only 33 of 60 plain acts legal. The cut-off fails off-distribution; the plain acts are also short (7.7 words against 11 to 14), untested.

Limitations. One model, one split seed, 30 test rows per test (accuracy interval 53% to 83%), activations stored in fp16. One annotator; LLM-written sentences rewritten twice (43 rows after a cue-word audit, 60 after a length confound); harmful stratum not length-matched. Correlational only. Under the question the label-swap band is itself about ±0.8 from layer 17, so the sign carries more than "outside the band".

Next. A neutral-question control (any yes/no question) to tell a legality-specific alignment from a collapse onto the answer axis. Then steering along the illegality direction with harm projected out, against norm-matched random directions.

**Word count:** 594 words of prose (`wc -w` excluding the table); 656 including the 41 words in the table cells. The skeleton's rule counts the table outside the 600 (`journal/writeup-skeleton.md`, "about 600 words of prose plus the two tables"). If you want the total under 600 with the table, cut in this order: "(mean Yes−No logit +3.82, −0.98, −0.99, −3.46)" (8), "For law-following AI that is the failure that matters." (9), "Negations: 61/61 and 58/61 called legal." (6), ", activations stored in fp16" (4), "7 texts edited" and "(a lawyer)" (5), "The null reruns the whole procedure 100 times on labels shuffled within topic and stratum." (15), "(US law)" (2), ", untested" (1). That is 50 words; the rest goes in the body.

# Sources for every number (all re-read tonight from the files of record, not from the facts card alone)

- 371 / 250 / 60 / 61 / 60 topics; 6 excluded: `data/scenarios.csv` (recounted: main 250, simple 60, negated 61; exclude=1 on 6 rows). 4 labels changed (s087, s126, s218, s057), 7 texts edited (s077, s057, s113, s256, s297, s339, s340): `journal/writeup-facts.md` §02 "Who changed what" (the CSV `relabelled` column reads 6; the write-up says 4 by you and the body should note the column counts 6). Drafted by Claude: git `bf79a51`.
- The question string: `scripts/common.py` `QUESTIONS["legal"]`, `JURISDICTION`. Token position: `journal/writeup-facts.md` §03 "The two conditions"; dry run step 5a.
- 30/15/15 topic split, seed 0, selection on validation AUROC, 100 within-cell shuffles: `scripts/probe_eval.py` (`select_and_test`, `shuffle_within_cells`, `--n-perm 100`).
- Table AUROCs and beaten counts: `data/processed/probeeval_lp_4b_{L_h2nh,L_nh2h,H_i2l,H_l2i}.json` and `..._prompted_...json`, keys `test_cross_auroc`, `perm_null.shuffles_beaten_auroc` (0.7422/100, 0.7111/97, 0.9467/100, 0.8267/100; 0.7422/100, 0.9067/100, 0.9689/100, 0.9911/100); same in `journal/results.md`.
- Cosines and bands: `probeeval_lp_4b_L_h2nh.json` `factorial.cos_dlegal_dharm` −0.0909 → +0.09, layer 25, `cosine_curve` band at 25 = −0.28..+0.26; `probeeval_lp_4b_prompted_L_h2nh.json` −0.6953 → +0.70, layer 15, band −0.54..+0.49; curve outside the band at layers 14–32 (layer 13 is +0.26 against a top of +0.31; layer 14 is +0.61 against +0.49). `journal/results.md` "Directions".
- Projection 0.68 → 0.68 and 0.85 → 0.74: same two files, `factorial.dlegal_auroc_within_harmless` and `dlegal_minus_harm_top1_auroc_within_harmless` (0.6756 → 0.6756; 0.8489 → 0.7378).
- Harm-gated answer, 97/34/25/3 % and mean logits +3.82/−0.98/−0.99/−3.46; within-stratum AUROC 0.83/0.95: recomputed tonight from `data/raw/ask_lp_4b_legal.jsonl` (244 main rows, exclude=0) with the command at the end of `journal/writeup-skeleton.md`: 96.8/34.4/25.0/3.3 %, 0.828/0.951. Per-quadrant accuracies 97/34/75/97 % are `data/processed/ask_lp_4b_legal.json` `per_quadrant` (the 34 % and 75 % are the same facts as "said illegal 34 %" and "said illegal 25 %"). These are still (agent) until your verification-log row exists.
- All 30 harmless test rows called legal under the question, accuracy 50 %: `probeeval_lp_4b_prompted_L_h2nh.json` `test_cross_acc` 0.5, `test_cross_acc_ci95` [0.5, 0.5], `shuffles_beaten_acc` 36; dry run "predicted illegal: 0 of 30".
- Fair baseline 0.81 vs 0.74: `journal/results.md` last line (`results_table.py` via `common.just_ask_auroc`); recomputed tonight from the raw jsonl on the 15 test topics, harmless stratum, n=30: 0.809 (harmful stratum: 1.000). (agent) until logged.
- No-cue rerun 0.74, 100/100, 241 rows: `data/processed/probeeval_lp_4b_L_h2nh_nocue.json`.
- Word count alone 0.60 / 0.79 / probe 0.71: `length_only_test_auroc` in the L_h2nh and L_nh2h files; `journal/results.md` table note.
- Negations 61/61 and 58/61, plain acts 33/60: `extra_sets.negated.frac_predicted_legal_1` 1.0 and 0.9508 (×61), `extra_sets.simple.frac_predicted_legal_1` 0.55 (×60) in the two L_h2nh files; `journal/results.md` "Checks".
- Accuracy interval 53–83 %: `probeeval_lp_4b_L_h2nh.json` `test_cross_acc_ci95` [0.533, 0.834].
- fp16 storage: dry run header "stored as float16, finite: True, max |value| 104"; `journal/team-pass-summary-2026-09-10.md` §4; fix commit `6cd9f80` after extraction at `d920864`.
- 43 rows rewritten after the cue-word audit: git `d35014c`. 60 legal-harmless rows replaced by length-matched twins (7.7 vs 13.1 words; word count 0.96 → 0.60): git `e2f171c`; `journal/writeup-facts.md` §02 "Review history". Harmful stratum 11.2 vs 13.9 words, plain acts 7.7 words, design rows 11.2–13.9: `journal/writeup-facts.md` §02 "Mean words per quadrant".
- Band about ±0.8 from layer 17 under the question: `probeeval_lp_4b_prompted_L_h2nh.json` `cosine_curve` (upper edge +0.72 to +0.92 over layers 17–32, lower −0.93 to −0.74); `journal/results.md` "Directions" bullets.
- Neutral-question control and steering cut on 10 Sep; steering scaffold on branch `steering-scaffold`: `README.md` "The design"; `journal/highlights.md` "Things to not forget".
- Colab T4, Hugging Face weights, fp16: `journal/writeup-facts.md` §03 "Model and hardware" (`01-gpu-check.txt`, `04-model-smoke.txt`, manifest `acts_lp_4b.json`).

# Errors in your draft and form, with the corrected sentence

1. Form Q4: "Model: Qwen 3.5 4b (through GitHub API calls)". Corrected: "Model: Qwen/Qwen3.5-4B, Hugging Face weights loaded on a free Colab T4 in fp16 (transformers, one forward pass per sentence); no API." Source: `journal/writeup-facts.md` §03 "Model and hardware"; `scripts/common.py` `load_model`; `journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt`.

2. Form Q4: "(2) the sentence followed by "is this legal? Yes or No", read at the token before the answer". Corrected: "(2) the sentence inside the question "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.", read at the token before the answer." Source: `scripts/common.py` `QUESTIONS["legal"]`. Note the question asks about illegality, not legality, and the probe is illegality-positive throughout.

3. Draft: "I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization." Corrected: "The harmless stratum was length-matched on 10 Sep after word count alone separated the original legal-harmless rows at AUROC 0.96: the 60 legal-harmless rows were replaced by length-matched twins of each topic's illegal-harmless row, and word count on the headline test rows fell to 0.60. The harmful stratum was not matched (11.2 vs 13.9 words); word count alone reaches 0.79 on the reverse test rows. Word count is reported as a scorer on every test set, not as a behavioural test." Source: git `e2f171c`; `scripts/probe_eval.py` `length_only_test_auroc`; `journal/results.md` table; `journal/writeup-facts.md` §02.

4. Draft: "Of the 371 hand-checked scenarios, I ended up ... flagging 43 as borderline only on legality, flagging 23 as borderline only on harmfulness and flagging 23 as borderline on both". The 43/23/23 split holds for the 250 design rows; over all 371 the harm-only count is 25 (48 harm flags minus 23 both) because 2 flags sit on twins/negations. Corrected: "Of the 250 design rows, 43 are flagged borderline on legality only, 23 on harm only and 23 on both (89 rows)." Source: `journal/writeup-facts.md` §02 "Counts"; `data/scenarios.csv`.

5. Draft: "371 LLM-generated and LLM-labelled (Claude 5.1 Fable) single sentence scenarios" and form: "(Claude Fable 5.1)". Use one name in both places: Claude Fable 5.1. Also the sentences were rewritten twice after review (43 cue-word rows, 60 twins), so "LLM-generated" should read "drafted by Claude on 9 Sep and rewritten twice on review findings, every row then read by me". Source: git `d35014c`, `e2f171c`; `README.md` "The dataset, counted once".

6. Draft: "S112: "You vote in a local election." (legal_harmless)" is listed as a design example. It is a plain act (`set=simple`), never trained on. Say so, or swap in a design-row example such as s003 legal_harmful "You honk and gesture furiously at a nervous elderly driver until she pulls over in tears." Source: `data/scenarios.csv` (s112 set=simple); `journal/writeup-facts.md` §02 "Example sentences".

7. Draft: "Qwen 3.5 4b" throughout. The model id is Qwen/Qwen3.5-4B; write "Qwen3.5-4B". Source: `data/processed/acts_lp_4b.json` manifest; `README.md`.

8. Form Q3 hypothesis "Illegality will be similarly represented to harmfulness (lie in a similar direction)" is what your pre-registration predicted (cos 0.25, "higher than random"). The result is +0.09 inside the band sentence only and +0.70 sentence + question, so Q3 should state the hypothesis and then say it held only under the question. Source: `journal/highlights.md` "What I expect the hat test to show"; `journal/results.md` "Directions".

# Still (agent) until your verification-log row exists

The harm-gated table (97/34/25/3, logits, 0.83/0.95) and the fair baseline 0.81 are recomputed by me tonight from `data/raw/ask_lp_4b_legal.jsonl` and match the skeleton. Neither is yet in `journal/verification-log.md` in your hand. Run the command at the end of `journal/writeup-skeleton.md` and step 4 of `journal/verification-plan.md`, log the rows, and the summary above stands as written; if you cannot, drop the logit parenthesis and the 0.81 sentence and the rest still holds.