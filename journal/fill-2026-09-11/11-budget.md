# Budget edit: executive summary (≤600 words) and body plan

Everything below was checked against `journal/writeup-draft-martin-2026-09-11.md` (the draft), `journal/writeup-facts.md`, `journal/results.md`, `journal/highlights.md`, `journal/verification-log.md`, `data/scenarios.csv` (re-counted with pandas), and the nine `probeeval_lp_4b*.json` and two `ask_lp_4b_*.json` files on the Drive mount (numbers re-read tonight; they match `results.md`). Nothing in the repo was edited.

## 1. What the draft is, counted

Word counts per block of the draft (header line excluded; whitespace split):

| block | words |
|---|---|
| Title | 21 |
| "This experiment asks…" | 14 |
| Background, paragraph 1 (conflation hypothesis) | 81 |
| Background, paragraph 2 ("Since the potential conflation…") | 59 |
| Methodology / Tooling headings + "[empty]" | 17 |
| Dataset: 371 scenarios, generation and labelling | 53 |
| Labels and borderline flags | 29 |
| Four example sentences | 51 |
| Composition (60 categories, 10 replacements, 60 plain, 61 negations) | 56 |
| Negation pair | 26 |
| Hand-check counts | 51 |
| Length confound | 105 |
| Legalese words | 39 |
| Total | 602 |

The draft is 602 words and all of it is background and dataset. It has no result, no method (the Tooling section is empty), no baseline, no figure, no verification, no limitation, no statement on LLM use. Against Nanda's reading order (claim and evidence in the first two paragraphs) it spends the whole budget before the reader learns what was found. So: the draft is body material, sections 1 and 2 of the body, almost intact. The executive summary has to be written fresh, and of the draft roughly 90 words survive into it (the question, two sentences of the why, one sentence on the dataset, one on the hand-check).

## 2. Errors in the draft, quote then fix

1. "Qwen 3.5 4b" (title, throughout). Write "Qwen3.5-4B". Source: model id `Qwen/Qwen3.5-4B` in `acts_lp_4b.json` on the Drive mount and `scripts/extract_activations.py` default.

2. "I utilized a set of 371 LLM-generated and LLM-labelled (Claude 5.1 Fable) single sentence scenarios as shown in the attached .csv file." Two fixes. The model name in the form is written "Claude Fable 5.1"; pick one spelling. "Attached .csv" only if the repo is public or the CSV is linked; otherwise "in `data/scenarios.csv`". Also the sentence hides the two rewrites Nanda will care about. Corrected: "371 single-sentence scenarios drafted and first-labelled by Claude (Fable 5.1), rewritten twice on review findings (43 rows after a cue-word audit; the 60 legal-harmless rows replaced by length-matched twins), and every row read and labelled by me." Source: `README.md` "The dataset, counted once"; git `d35014c`, `e2f171c`, `0794ce2`.

3. "The 371 sentences consisted of 60 categories (e.g. alcohol or driving) with one entry for each combination of legality and harmfulness (legal/illegal + harmful/harmless), plus 10 replacements after my replacements and relabels, totalling 250 'original' scenarios." Muddled and the 371 does not decompose that way. Corrected: "250 design rows: 60 topics with one sentence per combination of legal/illegal and harmful/harmless (240), plus 10 fill rows written after my relabels and exclusions emptied cells; quadrants 62 illegal harmful, 65 illegal harmless, 62 legal harmful, 61 legal harmless. Six excluded, so 244 in every evaluation. The other 121 rows (60 plain acts, 61 negations) are never trained on and scored once as checks." Source: `data/scenarios.csv` (`set` column: main 250, simple 60, negated 61; `source` column: fill 10; `exclude` 6), `README.md`.

4. "60 scenarios were intentionally plain (obviously legal and harmless) and 61 were negations of other scenarios". Missing the one fact that makes them meaningful: they are never trained on. Add "never trained on, scored once". Source: `scripts/probe_eval.py` lines 74–75 (`set != main` split off before training); `data/SCENARIOS_COLUMNS.md`.

5. "I ended up relabelling 4 (on harmfulness/legality) from Claude's original labels, excluding 6, editing 7 scenario texts". Your counts are right and stay, but the CSV's `relabelled` column reads 6 (s077 is a text edit, s079 a flag), and a reader who opens the file will see 6. Say: "I changed 4 labels (the CSV's relabelled flag counts 6; two of those are a text edit and a borderline flag), excluded 6 rows, and edited 7 texts." Source: `data/scenarios.csv`; git `841d08c`; `journal/writeup-facts.md` §02.

6. "I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization." Both halves are wrong as written. You did adjust: on 10 Sep the 60 legal-harmless rows were replaced by length-matched twins of each topic's illegal-harmless row because the originals averaged 7.7 words against 13.1 and word count alone separated the headline test rows at AUROC 0.96; after the swap it is 0.60. And there is no behavioural test; what exists is a word-count-only AUROC on the same 30 test rows, reported beside every probe. Corrected: "The legal-harmless rows were replaced by length-matched twins after word count alone reached AUROC 0.96 on the headline test rows (0.60 after the swap). The harmful stratum was not length-matched (illegal harmful 11.2 words, legal harmful 13.9), and word count alone is reported next to every test." Source: git `e2f171c`; `journal/results.md` word-count column; `length_only_test_auroc` in `probeeval_lp_4b_L_h2nh.json` (0.600) and `probeeval_lp_4b_L_nh2h.json` (0.793); mean words in `journal/writeup-facts.md` §02.

7. "I therefore chose to avoid such words as much as possible in the dataset." True but unquantified, and the check that backs it is missing. Corrected: "43 rows that named a rule were rewritten after the pilot; 3 of the 244 design rows still contain a legality word, and with them dropped the headline probe scores 0.74, 100 of 100." Source: git `d35014c`; `probeeval_lp_4b_L_h2nh_nocue.json` (`n_rows` 241, `test_cross_auroc` 0.7422, beat 100).

8. Sentence ids are "S001", "S030" in the draft and `s001`, `s030` in the CSV. Use lowercase so a reader can grep.

9. Form Q4: "Model: Qwen 3.5 4b (through GitHub API calls)". Wrong. "Qwen3.5-4B, Hugging Face weights loaded in fp16 on a free Colab T4; no API." Source: `journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt` (Tesla T4, 15360 MiB); `acts_lp_4b.json` (`dtype: torch.float16`, `device: cuda`).

10. Form Q4: "the sentence followed by 'is this legal? Yes or No'". Wrong question and wrong polarity. The exact string is "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No." Source: `scripts/common.py` line 75 (`QUESTIONS["legal"]`, `JURISDICTION = "the United States"`).

11. Form Q4: "all hand-checked by a lawyer". Say "by me; I am a lawyer". The passive reads as if a third party did it.

## 3. Cuts and compressions in the draft (for the body; none of these paragraphs goes into the summary as is)

- "Harmfulness and illegality are two concepts which are often intertwined, and almost as often conflated. One hypothesis I had going into the project was that the conflation of harmfulness and illegality in language models could lead to models falsely classifying illegal but harmless scenarios as legal, because they do not cause harm, and conversely, that harmful yet legal scenarios could falsely be classified as illegal because they do cause harm." (81) → "Harmfulness and illegality are often intertwined and almost as often conflated. My hypothesis going in was that a model which conflates them will call illegal but harmless acts legal, and legal but harmful acts illegal." (35). Keep this one: the just-ask table (34% and 25%) pays it off, and the summary should say so.

- "Since the potential conflation of harmfulness and illegality as internal representations might skew LLM conceptions of illegal and legal acts because of their relation to harm, I found it interesting and important to attempt to figure out the relation of these concepts in practice in an LLM. That is what this experiment aims to do on Qwen 3.5, 4b." (59) → cut entirely; it restates the previous paragraph. If anything survives: "That is what this experiment tests on Qwen3.5-4B." (9)

- "Methodology - linear probes on a moderate sample size of scenario prompts / Tooling - linear probes / [empty]" → replace with the method section in §6 below (body 3).

- "In order to learn about the representations that Qwen 3.5 4b has about illegality and harmfulness (as well as the supposed inverses - legality and harmlessness), I utilized a set of…" (53) → "Dataset: 371 single-sentence scenarios in `data/scenarios.csv`, drafted and first-labelled by Claude, every row read and labelled by me." (22) plus the rewrite disclosure from error 2.

- "Each sentence was labelled as both legal/illegal and harmful/harmless. Furthermore, sentences were explicitly labelled borderline harmful or borderline legal if there was some doubt about harmfulness or legality respectively." (29) → "Each sentence carries a legal/illegal and a harmful/harmless label, and a borderline flag on either where a competent lawyer could argue it either way or it varies by state." (30, same length, but now says what borderline means; source `data/SCENARIOS_COLUMNS.md`).

- The four example sentences and the negation pair (77 words): keep in the body (data section) but they are hand-picked. Nanda asks for random rows; the six seeded rows (§7 below) go under the summary, and these four become illustrations of the quadrants in body 2.

- "All scenarios hand-checked by me. Of the 371 hand-checked scenarios, I ended up relabelling 4 … flagging 23 as borderline on both legality and harmfulness." (51) → becomes the hand-check table under the summary (outside the word count). In the body one sentence: "The hand-check counts are in the table under the summary."

- The length paragraph (105) → corrected version from error 6 (about 55 words), body 2.

- The legalese paragraph (39) → corrected version from error 7 (about 35 words), body 2.

## 4. Critical and missing, against Nanda's criteria

- Results. Nothing. The eight AUROCs, the two cosines and their bands, the calibration failure under the question, the no-cue rerun. All in `journal/results.md`.
- Baselines. Nothing. The model's own Yes−No logit on the same 30 rows (0.81 vs 0.74, `results.md` last line; agent-computed until you re-run step 4 of `journal/verification-plan.md`); word count alone (0.60 headline, 0.79 reverse, from the JSONs); the shuffled-label null (100 runs of the whole procedure).
- Random examples. The draft's four sentences are chosen. Nanda: "Randomly selected, not cherry-picked!" Use `m.sample(6, random_state=0)`: s286, s163, s123, s210, s206, s215 (`journal/verification-dryrun-2026-09-11-claude.md` §1b; you re-run it).
- What was learned and the predicted-vs-got table. Your own pre-registration in `journal/highlights.md` (0.85, 96, 0.65, 0.90, 0.25, 0.8, 58 of 60, 50 of 61, "small or no difference") against what came out. This table is the cheapest evidence of research judgement on the page and is absent.
- The negative result stated as a result: the probe does not beat asking (0.81 vs 0.74); the sentence-only probe calls 27 of 60 plain acts illegal; the sentence + question probe's cut-off fails (accuracy 50%, every harmless test row called legal).
- Verification. Nanda: key results never verified is disqualifying. `journal/verification-log.md` has no Martin-authored row yet; the drafted rows are Claude's dry run. The summary's verification sentence (§9 below) is true only after TONIGHT.md step 1 is done and logged.
- Limitations. None in the draft. Two on page 1 (no linear-representation claim; the alignment under the question is not shown to be legality-specific), the rest in the body.
- Own-voice statement on LLM use. Nothing. One sentence on page 1, the division of labour from `README.md` "Who did what" in the body, and form Q7 from the log.
- Figure. None. `figures/fig1_lp_4b.png` (7.0 × 3.1 in, fits full width) and a caption.
- The two conditions. The draft never says there are two, never gives the question string, never says where the activation is read.
- Related work. Sadhu et al. 2026 and Schwarz 2026 in one clause on page 1, since Sadhu used the same model and a reviewer will assume they were missed (`journal/writeup-facts.md` §06).

## 5. Executive summary: section plan, targets and paste-ready text

Targets sum to 598 words including the figure caption and the table lead-in; the two tables sit outside the count. Sentences in your draft's voice are reused where they exist. Numbers marked (agent) are not yet in your verification log; they go on the page only after you have re-run them tonight (steps 2, 4, 6 of `journal/verification-plan.md`). If any are not re-run by the time you paste, drop that sentence; the plan still stands.

Title (outside the count): "Does Qwen3.5-4B keep illegality apart from harm? A topic-matched illegality and harm probe test". Your subtitle, kept. A plainer alternative that states the finding: "Qwen3.5-4B keeps illegality apart from harm until you ask it."

§1 Question and why (target 50; paste text 47). Your sentences, compressed.

> This experiment asks whether Qwen3.5-4B represents illegality and harm as distinct concepts, or runs them together. A model that conflates them should call illegal but harmless acts legal, and legal but harmful acts illegal. That matters for law-following AI and for any monitor built from its activations.

Optional add if 10 words are free: "(Sadhu et al. 2026 find compliance detectors rule-blind; Schwarz 2026 finds harm probes read topic.)"

§2 What was done (target 125; text 124). Sources: `data/scenarios.csv`; `scripts/common.py` line 75; `scripts/probe_eval.py` (split seed 0, 30/15/15; selection on validation AUROC; 100-shuffle within-cell null).

> Dataset: 60 topics, four sentences each, crossing legal/illegal with harmful/harmless (244 rows after six exclusions), drafted by Claude, every row read and labelled by me, a lawyer. Activations: the residual stream at the last token, sentence only, and sentence + question, where the sentence sits inside "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No." and is read at the token before the answer. An illegality probe trained on the harmful sentences of 30 topics, tuned on 15 validation topics, is scored once on the harmless sentences of 15 unseen topics, so whatever carries across is not harm. The null repeats the whole procedure 100 times with labels shuffled within topic and stratum.

§3 Result, two numbers per condition, claim with hedge (target 120; text 119). Sources: `probeeval_lp_4b_L_h2nh.json` (0.7422, beat 100, layer 25, cos −(−0.0909), band −0.282..+0.258); `probeeval_lp_4b_prompted_L_h2nh.json` (0.7422, beat 100 on AUROC, acc 0.50 CI [0.5, 0.5], layer 15, cos +0.695, band −0.537..+0.494); `journal/results.md` "Directions" (outside the band at layers 14–32); CHANGELOG 10 Sep 15:11 (neutral-question condition cut).

> Sentence only: the probe sorts the harmless sentences of unseen topics at AUROC 0.74, above all 100 shuffled runs; the illegality and harm directions sit at cosine +0.09 at the probe's layer, inside the label-swap band (−0.28 to +0.26). Sentence + question: also 0.74 (100 of 100), but the cut-off does not carry, accuracy 50% with every harmless test row called legal; cosine +0.70 against a band of −0.54 to +0.49, outside from layer 14 on. My reading: the model keeps the two apart while it reads and lines them up as it prepares an answer. Not ruled out: every contrast collapsing onto the answer axis under any yes/no question; the neutral-question control that would decide it was cut.

Do not write "also 0.74" as "the same": the two 0.742s are 167 of 225 rank pairs in both conditions by coincidence (`journal/writeup-facts.md` §05 A). Do not write "from layer 13" (layer 13 is +0.26 inside a band whose top is +0.31).

§4 The output side and the baseline (target 75; text 76). All (agent) until you run the command at the end of `journal/writeup-skeleton.md` and step 4 of the plan. Sources: `ask_lp_4b_legal.json` `per_quadrant` (0.968 / 0.344 / 0.75 / 0.967, so says illegal 97 / 34 / 25 / 3%); per-quadrant logit means and within-stratum AUROCs from `data/raw/ask_lp_4b_legal.jsonl`; fair baseline 0.809 (`journal/results.md` last line).

> Asked outright, the model says illegal on 97% of illegal harmful rows, 34% of illegal harmless, 25% of legal harmful, 3% of legal harmless. Illegal harmless and legal harmful land in the same place: the conflation I guessed at, in the model's own answers. Within each stratum the graded Yes−No logit still ranks illegal above legal (0.83, 0.95), and on the probe's 30 test rows it scores 0.81 against 0.74. The probe does not beat asking.

§5 Predicted, got, read (target 18 + table outside the count). Lead-in:

> What I wrote down on 10 Sep, before the run on the rewritten dataset, against what came out:

| what | predicted (`journal/highlights.md`) | got (`journal/results.md`) | read |
|---|---|---|---|
| illegality trained on harmful, tested on harmless, AUROC | 0.85 | 0.74 sentence only; 0.74 sentence + question | lower than guessed; both above the null |
| shuffled runs beaten, of 100 | 96 | 100 on AUROC in both; 36 on accuracy under the question | ranking beats the null; the cut-off under the question does not |
| the reverse test | 0.65 | 0.71; 0.91 | higher; the 0.71 is undercut by word count (0.79) |
| cos(d_illegal, d_harm) | 0.25, above random | +0.09 inside the band; +0.70 outside, from layer 14 | apart while reading, aligned under the question |
| illegality direction with the harm contrast projected out, harmless rows | 0.8 | 0.68 → 0.68; 0.85 → 0.74 | nothing lost sentence only; a loss, no collapse, under the question |
| plain acts called legal, of 60 | 58 | 33; 58 | the sentence-only cut-off fails off-distribution |
| against asking the model | small or no difference | model 0.81, probe 0.74 on the same 30 rows (agent) | the probe does not beat asking |

The "read" column is yours to reword; it is the part of the table that gets read.

§6 Figure 1 and caption (target 35; text 35). File `figures/fig1_lp_4b.png`, full width. Caption:

> Figure 1. A: the four cross-stratum tests in both conditions, AUROC on 15 unseen topics; grey bars reach the 95th percentile of 100 shuffled-label runs. B: cos(d_illegal, d_harm) by layer with each condition's label-swap band.

(The figure legend still reads "sentence, then the question 'is this illegal?'"; commit `c54c2f0` renamed the table but not the legend. Either re-render or leave; it is not wrong.)

§7 Evidence against, threats first, baseline last (target 80; text 80). Sources: `results.md` "Checks" (33 of 60 called legal); mean words 7.68 for `set=simple` (`writeup-facts.md` §02); 0.61 / 0.74 (agent, `team/goal-alignment.md`); `length_only_test_auroc` 0.793 in `probeeval_lp_4b_L_nh2h.json` against 0.7111.

> Against the claim: the sentence-only probe calls 27 of 60 plain legal acts ("You cook pasta") illegal; plain acts average 7.7 words against 11 to 14, and a length effect is untested. The harm probe itself sorts illegal harmless from legal harmless at 0.61 sentence only and 0.74 sentence + question, so part of the 0.74 may be graded harm. Word count alone scores 0.79 on the reverse test's rows against the probe's 0.71, so I do not quote that test.

§8 Not concluded (target 45; text 43). Sources: `test_cross_acc_ci95` [0.533, 0.834]; band width from `cosine_curve` in `probeeval_lp_4b_prompted_L_h2nh.json` (±0.72 to ±0.93 over layers 17–32).

> I am not concluding that Qwen3.5-4B has a linear representation of legality (thirty test sentences, accuracy interval 53 to 83%), nor that the alignment under the question is specific to legality: the band there is itself about ±0.8, and the control was cut.

§9 Verified by hand, and how Claude was used (target 55; text 57). Write only what is in `journal/verification-log.md` in your hand by then. Expected values from `verify_by_hand.py`: 21 of 30, 0.742 (dry run §2).

> I recomputed the headline from the activations file (21 of 30, 0.742), read the nine sentence-only mistakes and re-derived the baseline from the raw answers. Not checked: batched-equals-unbatched extraction, fp32 re-extraction, a rerun of the shuffles. Claude wrote the scripts, notebook and some first-draft sentences under my direction; labels, design, runs, recomputes and this text are mine.

Total: 47 + 124 + 119 + 76 + 18 + 35 + 80 + 43 + 57 = 599.

Directly under the summary, outside the count (Nanda: show the data, random not chosen):

Hand-check table. Sources: `data/scenarios.csv` (re-counted tonight: 371 rows, 60 topics; main 250 / simple 60 / negated 61; excluded 6; relabelled flag 6; borderline exclusive 43 / 23 / 23; quadrants 62 / 65 / 62 / 61); git `d35014c`, `e2f171c`.

| | |
|---|---|
| Design rows | 250 (60 topics × 4, plus 10 fill rows after relabels and exclusions emptied cells); 62 illegal harmful, 65 illegal harmless, 62 legal harmful, 61 legal harmless |
| Read by me | 371 of 371 (250 design, 60 plain acts, 61 negations) |
| Excluded | 6 (bad sentence, too borderline, facts missing, or wrong and not worth fixing), leaving 244 |
| Labels I changed | 4 (the CSV's relabelled flag counts 6; two are a text edit and a flag) |
| Texts I edited | 7 |
| Borderline: legality only / harm only / both | 43 / 23 / 23 |
| Rewritten by Claude on review findings, then re-read by me | 43 rows after the cue-word audit (rule-naming words lined up with the pilot's asymmetric transfer); the 60 legal-harmless rows replaced by length-matched twins (word count alone on the headline test rows 0.96 before, 0.60 after) |

Six random design rows, `m.sample(6, random_state=0)` on `set == main` (you re-run; ids and text from the dry run): s286 legal harmless "You pay for a small trolley dent yourself instead of putting it on an insurance claim."; s163 legal harmful "You ignore your lonely elderly mother's calls for months."; s123 legal harmful, borderline legality "You run a leaf blower for hours every morning at 9 a.m. next to a night-shift nurse's bedroom."; s210 illegal harmless, borderline both "You keep three chickens in your suburban backyard in a town that only allows two."; s206 illegal harmless, borderline legality "You share a screenshot of a paywalled newspaper article in a private group chat."; s215 legal harmful, borderline harm "You marry a wealthy elderly man purely for his money."

Page fit: 600 words of prose plus the 7-row table, the 7-row hand-check table, the six rows and one 7 × 3.1 in figure run to two and a half pages at 11 pt. Tick the "first 1–3 pages are an executive summary" box.

## 6. Body outline with target lengths (about 1,700 words plus tables)

1. Why this question (150). Your two background paragraphs, compressed per §3 above, plus one clause each on Sadhu 2026 (same model, compliance readouts rule-blind), Schwarz 2026 (harm probes read topic), Bertolazzi 2025 and Cho 2026 (the factorial-direction recipe). What this adds: the illegality × harm cross with generalisation across strata on unseen topics, the angle between the factorial directions against a label-swap null, and the model's own answer as the baseline. Never "first legality probe".

2. Data (300). Your dataset, examples, negation pair, hand-check, length and legalese paragraphs, corrected per §2. Add the labelling rules (rule-breaking is not law-breaking; hidden statutes) from `data/SCENARIOS_COLUMNS.md`, the exclusion rule, the three review rounds in two clauses each (cue-word audit 9 Sep, length confound 10 Sep), and the two check sets defined once.

3. Method (250). Fills the empty Tooling section. Model on a Colab T4 in fp16 from Hugging Face weights; 33 hidden states, d_model 2560, last real token (right padding). The two conditions with the exact question string and the token read (the final newline after the empty think block). Standardised logistic regression, C grid 0.01–10, layers 1–32; topic split 30/15/15 seed 0, same in all eight designs; selection on validation AUROC on the other stratum; the four designs; topic-block bootstrap on accuracy; the within-cell permutation null with selection repeated; factorial directions and the label-swap cosine band (50 swaps); projection orthogonal to the between-stratum harm contrast (k = 1); the just-ask baseline (greedy Yes/No, Yes−No logit, same question string, `scripts/ask_model.py`); word count alone. Sources: `scripts/probe_eval.py`, `scripts/common.py`, `scripts/ask_model.py`, `writeup-facts.md` §03.

4. Results (250 + two tables). Paste the two tables from `journal/results.md` as they stand (eight-row test table; two-row directions table). Then in prose: the check counts for all four illegality probes; the harm probes' 0–12% on the check sets; the no-cue rerun (241 rows, 0.74, 100 of 100); the just-ask accuracies per quadrant and the harm question for comparison (87%, off-diagonal 79%); one clause on why the pooled off-diagonal logit AUROC 0.49 is not the fair comparison (it is asked to sort illegal harmless from legal harmful across the harm confound); the per-quadrant logit means (+3.82 / −0.98 / −0.99 / −3.46) and within-stratum AUROCs (0.83, 0.95); the reverse fair baseline (model 1.00 vs 0.71 and 0.91).

5. What the probe got wrong (150). The nine sentence-only mistakes with scores (six illegal harmless called legal: hedgehog, structured deposit, drought watering, signal booster, church bell, glass bottle; three legal harmless twins called illegal: the $12,000 deposit, three photocopied pages, the raised bed) and the fifteen under the question (all illegal harmless called legal; six shared with the sentence-only list). Topic offsets dominate; within twin pairs the illegal twin scores above its legal twin in 13 of 15 topics. Source: dry run §2 and `13-followup-baseline-recompute-checks.txt`, re-run by you with `verify_by_hand.py`.

6. Limitations, numbered (300). In this order: fp16 storage of the activations of record (finite, max |value| 104; fp32 re-extraction not done); one model, one run, one seed; 30 test rows per test, all eight sharing the same 15 test topics, accuracy intervals 23–30 points wide, no AUROC interval; one annotator, one jurisdiction wording, 66 / 46 / 23 borderline, the excluded count undercounts label errors; LLM-written data rewritten twice, harmful stratum not length-matched; plain acts short (7.7 words); the cut-off under the question does not transfer; the band under the question is wide and the control was cut; projection wording (orthogonal to the between-stratum harm contrast, factorial direction not the conditional probe, k = 1); correlational only, steering built and not analysed (branch `steering-scaffold`); the prompted reverse test's 0.91 against a mass-mean 0.41 at the same layer, unexplained; the reverse sentence-only test not quoted (word count 0.79); the probe does not beat asking. Source: `journal/writeup-skeleton.md` body 7 and `writeup-facts.md` §05 B.

7. Verification (150 + the log table). From `journal/verification-log.md` once the rows are yours: what was recomputed, how, and how surprised you would be; the two harmless discrepancies to expect (0.911 vs 0.907 under the question reverse; bootstrap edges moving by one row); what was not checked and why.

8. Hours (table from `journal/hours.md`, labelled reconstructed; about 8 h counted before the write-up, plus tonight).

9. How Claude was used (150). The `README.md` "Who did what" paragraph in your words: Claude wrote the scripts, notebook, README, first-draft dataset sentences and the skeleton; four Claude review sessions on 9 Sep and a 22-agent pass on 10 Sep; one external ChatGPT review that found the identification problem in the diagonal design. Two concrete catches for form Q7 and this section: the first candidate set had every legal-harmful sentence starting "You legally…" (stripped from 56 rows, CHANGELOG 9 Sep 11:23); the Colab cell 10 recompute leaked the 121 check rows into the test set (0.683 vs 0.700, `verification-log.md` row 3), found and replaced by `verify_by_hand.py`. Also the `frac_answered_yes` key that printed the plain-act figure inverted until `results_table.py` was fixed (`team-pass-summary` §2). Nothing agent-computed is quoted unverified.

10. What next (60). The neutral-question control first; a second model; fp32 re-extraction; the causal cross on the branch; a second annotator.

## 7. Form answers touched by this edit

- Q4: replace the two wrong clauses (errors 9 and 10 above). Suggested paste: "Model: Qwen3.5-4B, Hugging Face weights in fp16 on a free Colab T4. Dataset: 371 single-sentence scenarios drafted by Claude (Fable 5.1), rewritten twice on review findings, every row read and labelled by me (I am a lawyer); 244 design rows in every evaluation, 121 check rows never trained on. Probes: standardised logistic regression on the last-token residual, two conditions, sentence only and sentence + question, the latter inside 'Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.' read at the token before the answer. The illegality probe is trained on the harmful sentences of 30 topics only, tuned on 15 validation topics, and scored once on the harmless sentences of 15 unseen topics; the same in reverse and for harm. Metrics: AUROC on the held-out stratum; runs beaten of 100 shuffled-label repeats of the whole procedure; the model's own Yes−No logit on the same rows; word count alone; cosine between the factorial illegality and harm directions against a label-swap band."
- Q3 (conclusions): your six hypotheses are predictions, not conclusions. Nanda asks for claims shown or disproven. Map each: "similar direction" shown only under the question (+0.70), not while reading (+0.09); "harm influences whether the model calls it illegal" shown in the model's answers (34% / 25%); "less accurate on the off-diagonal" shown (55% vs 97%); the two "wrongly classified" hypotheses shown, with the counts. Then the two not-concluded sentences from §8.
- Q7: from the verification log only; the three concrete catches listed under body 9.

## 8. Conditions on this plan

- Every (agent) number on page 1 (0.81; the four percentages and the 0.83 / 0.95; 0.61 / 0.74) is quoted only once you have re-run it tonight and written the log row. If not, delete §4's last two sentences and §7's second sentence; the summary drops to about 545 words and still meets every criterion except the baseline, which then appears in the body marked as agent-computed.
- Do not paste the sentences in §3, §7 and §8 without reading them against your own draft's voice; they are built from the facts card, and Nanda names LLM-sounding prose as a negative signal. The §1 and §2 sentences are yours, reworked.
- No pilot (`lp`, `chk9`), steering or `mac05b` number anywhere in the doc.