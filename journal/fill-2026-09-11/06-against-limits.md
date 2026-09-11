## Form Q5. "What is the strongest evidence you found against these hypotheses?"

Paste text (about 330 words):

Two of the hypotheses I wrote down before the run went the wrong way. I expected illegality and harm to lie in a similar direction; while the model only reads the sentence they do not (cosine +0.09 at the probe's layer, inside the label-swap band at 32 of 33 layers, and an illegality direction built there predicts harm at 0.50). They line up only once the question is asked (+0.70). I also guessed the 60 plain legal acts would be called legal 58 times; the sentence-only headline probe called 27 of them illegal.

Against the claim I do make, that the model keeps illegality apart from harm while reading, the strongest evidence is that the harmless stratum is not flat in harm. The harm probe, trained on illegal rows and never shown a legality label, sorts illegal from legal among the same 30 harmless test rows at 0.61 (0.74 under the question), so part of the headline 0.74 can be graded harm that the binary label does not capture, and projecting out the between-stratum harm contrast cannot remove a within-stratum gradient. Second, the 27 of 60 plain acts called illegal: whatever the cut-off reads off distribution, it is not legality, and the plain acts are short (7.7 words against 11 to 14), so a length effect is possible and untested. Third, the illegality direction is weakest where it matters, 0.68 within the harmless rows against 0.87 within the harmful ones. Fourth, "inside the band" is a null result about alignment, not a demonstration of independence in 2560 dimensions; the sentence-only cosine is small but positive at every layer from 17 on (+0.09 to +0.19).

Against the alignment under the question: from layer 17 the band is itself about ±0.8, so any within-cell contrast aligns with harm there, and the probe-layer figure (+0.70 against −0.54 to +0.49) is the only place the statement has a margin. The neutral-question control that would tell a legality-specific alignment from a collapse onto the answer axis was cut, so I cannot say which it is. The cut-off fitted under the question calls all 30 harmless test rows legal (accuracy 50%; beats 36 of 100 shuffles on accuracy while beating 100 on AUROC).

Against the probe being useful: on the same 30 rows the model's own Yes−No logit sorts illegal from legal at 0.81 against the probe's 0.74, and at 1.00 against 0.71 and 0.91 on the reverse rows. The probe does not beat asking. The sentence-only reverse test (0.71) is not evidence at all, because word count alone reaches 0.79 on those rows.

Sources for Q5, in order of appearance:
- Hypotheses and the 58 of 60 guess: `journal/highlights.md` ("cos 0.25, above random"; "58 of 60").
- cos +0.09, 32 of 33 layers, +0.70, band −0.54 to +0.49, layers 17–32 +0.09 to +0.19 and prompted band about ±0.8: `journal/results.md` "Directions"; `probeeval_lp_4b_L_h2nh.json` and `probeeval_lp_4b_prompted_L_h2nh.json` `factorial.cos_dlegal_dharm` (sign flipped) and `cosine_curve`.
- Illegality direction predicting harm 0.50: `probeeval_lp_4b_L_h2nh.json` `factorial.dlegal_auroc_predicting_harm_test` 0.5022, printed as 1 − stored.
- 27 of 60: `probeeval_lp_4b_L_h2nh.json` `extra_sets.simple.frac_predicted_legal_1` 0.55; `journal/results.md` "Checks".
- 7.7 vs 11 to 14 words: `data/scenarios.csv`, means 7.68 (set=simple) and 11.18 to 13.87 (kept design rows by quadrant); `journal/writeup-facts.md` "Mean words per quadrant".
- 0.61 / 0.74 harm probe within the harmless stratum: agent-computed, not in any JSON. Confirmed tonight by refit at the saved layer and C: 0.613 sentence only (layer 14, C 0.01), 0.738 sentence + question (layer 15, C 10); factorial d_harm gives 0.636 / 0.800. Script: `/private/tmp/claude-501/-Users-martinherje-Documents-PhDAI/71a2bd58-5af7-4580-9111-9da6c7bbabbd/scratchpad/harm_within_harmless.py`, run with `cd ~/mats12 && uv run python <path>`. Run it yourself and log the row, or drop that sentence from the form.
- 0.68 within harmless vs 0.87 within harmful: `probeeval_lp_4b_L_h2nh.json` `factorial.dlegal_auroc_within_harmless` 0.6756, `..._within_harmful` 0.8711.
- Neutral-question control cut: `README.md` "The design"; `journal/writeup-facts.md` §A.
- Accuracy 50%, 36/100 on accuracy, 100/100 on AUROC: `probeeval_lp_4b_prompted_L_h2nh.json` `test_cross_acc`, `perm_null.shuffles_beaten_acc/auroc`; `journal/results.md`.
- 0.81 vs 0.74; 1.00 vs 0.71 and 0.91: `journal/results.md` last line, computed by `scripts/results_table.py` via `common.just_ask_auroc` from `data/raw/ask_lp_4b_legal.jsonl`. Still (agent) until your step 4a is in `verification-log.md`.
- Word count 0.79 vs probe 0.71: `probeeval_lp_4b_L_nh2h.json` `length_only_test_auroc` 0.7933, `test_cross_auroc` 0.7111.

## Form Q6. "What are the biggest limitations to your results? Could you have addressed them?"

Paste text (about 560 words):

Thirty test sentences per test. Every accuracy interval is 23 to 30 points wide (headline 53% to 83%), no AUROC interval was computed, and all eight tests share the same 15 test topics under one split seed, so they are one draw of topics, not eight checks. Rotating the test topics four ways would have cost about four hours of free T4 time, which I had overnight; I fixed one split in advance and never re-rolled it, and this is the first thing I would do with another day.

One model, one run, and the activations of record are stored in half precision. The fix that stores fp32 landed 54 minutes after the extraction of record. The stored values are finite (largest 104 sentence only, 36 under the question), so a re-extraction would move numbers in the third decimal at most, and I have not done it. Re-extraction plus the eight evaluations is about an hour and a half on Colab; it was doable, and I spent that time on verifying the numbers by hand instead.

The p-value the script writes is computed on accuracy, not AUROC, while the pre-registered statistic is the AUROC beat count. I report beat counts and no p-value. One line to fix; not fixed.

The data. Claude drafted all 371 sentences and their labels. I read every row, changed 4 labels, excluded 6 rows and edited 7 texts, so the labels are one annotator's, and that annotator is a Norwegian lawyer labelling US law. 66 of the 250 design rows are flagged borderline on legality, 46 on harm, 23 on both, and the borderline rows stayed in training and test (the script has a flag to drop them; the rerun is about eight minutes per test and I did not run it). The model itself calls only 34% of the illegal-harmless rows illegal, and no second lawyer arbitrated. Writing the sentences myself would have taken three to four hours I did not have; a US-trained second reader was not available inside the budget. LLM-written sentences can carry surface regularities a probe reads; the two I found, rule-naming words and sentence length, I fixed by rewriting.

Length is handled by twins, not fully. The 60 legal-harmless rows were replaced by length-matched twins after word count alone separated the harmless stratum at 0.96 (0.60 now on the headline rows). The harmful stratum, which is where the headline probe is trained, was not matched: illegal-harmful rows average 11.2 words, legal-harmful 13.9, and word count alone separates the training rows at 0.68. Sixty more twins would have been about an hour of drafting, half an hour of checking and a re-extraction; the confound was found on the afternoon of 10 Sep and the run went ahead without them.

No neutral-question control. Under any yes/no question the within-cell contrasts may collapse onto the answer axis; the condition that separates that from a legality-specific alignment is one more extraction and one evaluation, about half an hour of Colab. It was cut under the one-test rule on 10 Sep, and it is the cut I regret.

Correlational only. A steering scaffold was built and run once on the sentence + question condition, not analysed and not reported (it sits on a branch). A causal test with norm-matched random directions and the answers actually read is about three hours, and after two dataset rewrites that time was not there.

The calibration failure under the question, every harmless test row called legal while the ranking holds at 0.74, is described, not explained. Refitting the intercept on validation rows would have been cheap and would have hidden it; I left it and report the failed accuracy null. "Alignment" is likewise a description of two mean-difference directions and a widening band, not a mechanism; nothing here says which components do it.

The fair baseline, 0.81 against 0.74, is 182 against 167 of 225 rank pairs on 30 rows, with no interval. A paired topic bootstrap is ten lines I did not write. The direction repeats on the reverse rows (1.00 against 0.71 and 0.91), which is why I state it as "does not beat asking" and not more.

Hours are reconstructed from commit times and my own estimates; no timer ran.

Sources for Q6, in order of appearance:
- 30 rows, 53% to 83%, shared test topics, seed 0: `probeeval_lp_4b_L_h2nh.json` `test_cross_acc_ci95` [0.533, 0.834], `args.seed` 0, `split.test_topics` identical in all eight files; `journal/results.md` table note. Four-fold estimate: eight evaluations took 64 min on the T4 (`journal/writeup-facts.md` "Model and hardware", notebook cell 13 text), so four rotations ≈ 4.3 h.
- fp16: `acts_lp_4b.json` `dtype: torch.float16`, `git d920864`, timestamp 2026-09-10T16:48 UTC (18:48 CEST); fix commit `6cd9f80` 2026-09-10 19:42 CEST; max |value| 104 / 36 from `journal/team-pass-summary-2026-09-10.md` §4.
- Accuracy p-value: `scripts/probe_eval.py` line 179 (`p_value_vs_null` from `null_acc`); pre-registered count is the AUROC one per `journal/results.md` table note and the probe_eval docstring.
- 371 drafted by Claude; 4 labels, 6 excluded, 7 texts: `data/scenarios.csv` `source` column; `journal/writeup-facts.md` "Who changed what" (your own tally; the CSV `relabelled` column counts 6 because s077 and s079 were text edits/flags, say 4).
- 66 / 46 / 23 borderline: `data/scenarios.csv` (set=main), recounted tonight; `--drop-borderline` flag in `scripts/probe_eval.py`, `args.drop_borderline` false in every JSON of record.
- 34% of illegal-harmless rows called illegal: `ask_lp_4b_legal.json` `per_quadrant.illegal_harmless` 0.3443.
- 0.96 → 0.60: `journal/writeup-facts.md` "Review history" (third review) and `journal/results.md` word-count column; twins commit `e2f171c`.
- 11.2 vs 13.9 words: `data/scenarios.csv` kept design rows, means 11.18 / 13.87. Word count separating the harmful training rows at 0.68: computed tonight from `scenarios.csv` and `split.train_topics` of `probeeval_lp_4b_L_h2nh.json` (0.683, n=61); one-liner: `uv run python -c "import json,pandas as pd;from sklearn.metrics import roc_auc_score;df=pd.read_csv('data/scenarios.csv');tr=set(json.load(open('<D>/probeeval_lp_4b_L_h2nh.json'))['split']['train_topics']);h=df[(df.set=='main')&(df.exclude==0)&(df.harmful==1)&df.topic.isin(tr)];print(roc_auc_score(h.legal,h.text.str.split().str.len()))"`.
- Neutral-question control cut 10 Sep: `README.md` "The design"; CHANGELOG 10 Sep 15:11 via `journal/writeup-facts.md` §A.
- Steering: branch `steering-scaffold` (commit 7940169, `git branch -a`); `README.md` "The design"; `journal/writeup-skeleton.md` limitation 10.
- Calibration failure: `probeeval_lp_4b_prompted_L_h2nh.json` `test_cross_acc` 0.5, interval [0.5, 0.5], `shuffles_beaten_acc` 36; `journal/writeup-facts.md` §4.
- Rank pairs: 0.7422 × 225 = 167, 0.8089 × 225 = 182 (`journal/writeup-facts.md` §7 fair baseline 0.8089; `probeeval_lp_4b_L_h2nh.json` `test_cross_auroc` 0.7422).
- Hours reconstructed: `journal/hours.md` header.

## Errors in your draft that touch these two answers

1. Form Q4, your sentence: "Model: Qwen 3.5 4b (through GitHub API calls)". Corrected: "Model: Qwen3.5-4B, weights from Hugging Face, run locally on a free Colab T4 in float16 (no API)." Source: `acts_lp_4b.json` (`model: Qwen/Qwen3.5-4B`, `device: cuda`, `dtype: torch.float16`); `scripts/common.py` `load_model` uses `AutoModelForCausalLM.from_pretrained`; `journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt` (Tesla T4).

2. Form Q4, your sentence: "(2) the sentence followed by 'is this legal? Yes or No'". Corrected: "(2) the sentence inside the question 'Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.', read at the token before the answer." Source: `scripts/common.py` `QUESTIONS["legal"]`, `JURISDICTION`.

3. Write-up draft, your sentence: "I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization." Corrected: "I adjusted for length in the harmless stratum only: the 60 legal-harmless rows were replaced by length-matched twins of each topic's illegal-harmless row after word count alone separated them at AUROC 0.96 (0.60 after the swap). The harmful stratum was left as written (11.2 against 13.9 words). Word count alone is reported as its own AUROC line on every test's rows, which is a baseline, not a behavioural test." Sources: commit `e2f171c`; `journal/results.md` word-count column; `scripts/probe_eval.py` line 162 (`length_only_test_auroc`).

4. Write-up draft, "Claude 5.1 Fable": the model is named Claude Fable 5.1 elsewhere in your form (Q4); use one spelling.

5. Your borderline counts (43 legality only, 23 harm only, 23 both) are the exclusive split and agree with the CSV's inclusive 66 / 46 / 23; keep whichever you use consistently in the doc and the form. Source: `data/scenarios.csv`, recounted tonight.