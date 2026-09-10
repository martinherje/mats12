# Write-up skeleton (Google Doc), to fill in on Friday

Claude drafted this structure on 10 Sep evening from the design sheet, the results of record and the team review. The prose is Martin's to write. Every number below is copied from `data/processed/probeeval_lp_4b*.json`, `data/processed/ask_lp_4b_*.json` or `journal/results.md`, and none goes into the doc until Martin has recomputed it (`journal/verification-log.md`). Rules from the design sheet: say "first topic-matched illegality × harm test", never "first legality probe", never "a linear representation exists"; no pilot numbers anywhere; the 43 rewritten rows and the 60 replaced legal-harmless rows are disclosed; no steering numbers; no agent-computed number until Martin has recomputed it.

Numbers marked (agent) come from the team's own recomputation on 10 Sep (`team/verification-plan.md`, `team/goal-alignment.md`), not from a file of record. They are here so nothing is forgotten; they are quoted only after Martin has repeated them.

## Title

Does Qwen3.5-4B keep illegality apart from harm? A topic-matched illegality × harm probe test

## Executive summary (pages 1–2: about 600 words of prose plus the two tables; one figure)

Short unlabelled paragraphs, in this order. The budget is the reader's ten minutes, so the summary carries the claim, two numbers per condition, the prediction, the test, the figure, the evidence against and the two things not concluded; every other number lives in the body's tables. Paragraphs 1, 5 and 7 below are drafted as sentences to save time; they are Claude's sentences, to be rewritten, not pasted.

1. The question, two sentences. Does Qwen3.5-4B represent illegality as something distinct from harmfulness, or does an illegality probe just read harm? Why it matters for monitoring, in Martin's words: a monitor that reads harm misfires exactly where law and harm come apart, in both directions. (Claude's phrasing of this, "misses harmless infractions and flags legal cruelty", is used here at most once and nowhere else in the application.) Sadhu et al. 2026 (compliance monitors rule-blind) and Schwarz 2026 (harm probes are topic detectors) are the two citations that belong on page 1.

2. The answer, with the alternative attached. Martin's one sentence: the model keeps illegality and harm apart while it reads the sentence and merges them as it prepares a yes/no answer. Then, in the same breath, the alternative not ruled out: at the answer position every within-cell contrast may be collapsing onto one axis, and the neutral-question control that would tell a legality-specific merge from that collapse was cut. Then two numbers per condition, no more:

Bare: illegality trained only on harmful rows sorts the harmless rows of fifteen unseen topics at AUROC 0.74, above all 100 shuffled-label runs; cos(d_illegal, d_harm) at the probe's layer is +0.09 inside a band of about ±0.3, and projecting the harm direction out leaves the illegality direction's harmless-stratum score at 0.68.

Prompted: the same test ranks the rows at 0.74 (100 of 100) but its cut-off does not carry, accuracy 50% with every harmless test row called legal; cos +0.70 at the probe's layer against a band of −0.54 to +0.49, and projecting harm out takes the direction's harmless-stratum score from 0.85 to 0.74.

One clause on what this means for a monitor, Martin's: a probe read at the answer position reads harm; a probe read from the bare sentence keeps the two apart but its cut-off fails off-distribution (paragraph 6).

One clause on the order: the pre-registration named the prompted condition as the main run; bare is presented first, a choice made after the results, because the prompted cut-off does not transfer and the prompted geometry has the answer-axis explanation above. The rest of the eight numbers per condition (reverse tests, harm probes, the illegality direction predicting harm) are in the body's table, not here.

3. Predicted, got, read. The prediction was written on 10 Sep after the 9 Sep pilot on the old dataset and before the run on the rewritten one (`journal/highlights.md`). Quote Martin's numbers as he wrote them, without the bold labels and without the "(AUROC; 0.5 = coin flip, 1.0 = perfect)" gloss. His "85" is 0.85, and so on. The "read" column is his research judgement and is the part of this table that is read; keep it to a clause per row.

| what | predicted | got | read |
|---|---|---|---|
| main test, illegality trained on harmful, tested on harmless, AUROC | 0.85 | 0.74 bare, 0.74 prompted | lower than guessed; both above the null on AUROC |
| shuffled-label runs beaten, of 100 | 96 | 100 on AUROC in both conditions; prompted 36 on accuracy | the ranking beats the null; the prompted cut-off does not |
| the reverse test, AUROC | 0.65 | 0.71 bare, 0.91 prompted | higher than guessed; the bare number is undercut by word count (0.79 on the same rows) |
| harm across legality, both ways | about 0.90 at most | 0.95 and 0.83 bare; 0.97 and 0.99 prompted | harm is the easier concept, as guessed |
| cos(d_illegal, d_harm) | 0.25, above random | +0.09 inside the band (bare); +0.70 at layer 15, outside (prompted) | apart while reading, aligned under the question |
| illegality direction with harm projected out, harmless rows | 0.8 | 0.68 → 0.68 bare; 0.85 → 0.74 prompted | nothing lost in bare; a loss but no collapse in prompted |
| plain acts called legal, of 60 | 58 | 33 bare headline probe; 58 prompted | the bare cut-off fails off-distribution |
| negations called legal, of 61 | 50 | 61 bare; 58 prompted | see the sanity-check note below |
| against the model's own answer | small or no difference | the model's own logit scores 0.81 on the same 30 rows, the probe 0.74 (agent; Martin recomputes) | the probe does not beat asking |

4. The test in three sentences. On the two easy quadrants legal and not-harmful are the same label, so the illegality probe is trained inside one harm stratum (illegal-harmful against legal-harmful) and tested inside the other (illegal-harmless against legal-harmless); anything it carries across is not harm. It is tested only on topics it never saw, with layer and regularisation chosen on fifteen validation topics, so it cannot be reading topic. The null repeats the whole procedure, selection included, 100 times with labels shuffled within each topic × stratum pair, and the result is reported as the number of those runs the real probe beats.

5. Figure 1 (`figures/fig1_lp_4b.png`, full width). Caption: A, the four cross-stratum tests in both conditions, AUROC on the held-out stratum of the 15 test topics; the grey range from 0.5 is the 95th percentile of 100 shuffled-label runs of the same procedure. B, cos(d_illegal, d_harm) at every layer, bare and prompted; shading is the 95% range of the same cosine under within-cell label swaps. Bare stays near zero inside a band of about ±0.3 at every layer. Prompted, both the curve and the band open up from layer 14, so at the answer position nearly every within-cell contrast lands on one axis. Do not write "cos 0.9, above the null band".

6. Evidence against, one paragraph, the threats to the claim first and the baseline comparison last (the baseline is evidence about the probe's usefulness, not about the representation, so it closes the paragraph rather than opening it). In this order: the bare headline probe calls 27 of 60 plain legal acts ("You cook pasta") illegal, so whatever its cut-off reads, it is not legality off-distribution (the plain acts average 7.7 words against 11 to 14 for the design rows; a length effect is possible and untested). The harm probe itself sorts illegal-harmless from legal-harmless at 0.61 bare and 0.74 prompted (agent), so the harmless stratum is not harm-flat and part of the 0.74 may be graded harm the binary label does not capture. Word count alone reaches 0.79 on the harmless → harmful rows (illegal-harmful rows average 11.2 words against 13.9 for legal-harmful, uncorrected), so that test is not evidence on its own. The prompted headline accuracy is 50%. Last, the model's own Yes−No logit on the same 30 held-out rows sorts illegal from legal at 0.81 against the probe's 0.74 (agent), so by the run sheet's own rule the probe does not beat asking.

7. Baselines and checks, one short paragraph, counts only. The fair baseline as above. One clause on the model's own answers on the rows where law and harm disagree, from the file of record: right on 55% of the off-diagonal rows, and on 34% of the illegal-harmless rows (it calls two thirds of the harmless infractions legal), which is the monitoring failure paragraph 1 names, in the model's own behaviour. The no-cue rerun in one clause: 3 of 244 design rows contain a legality word; with them dropped, AUROC 0.74, 100 of 100. The check sets in one clause: the bare headline probe calls 33 of 60 plain acts and 61 of 61 negations legal; the prompted headline probe 58 of 60 and 58 of 61. The negation check is a one-line sanity check: the negations score far past any legal row (−7.9 on average, agent), so "not flagged" carries little. The full check counts for all four illegality probes are in the body.

8. What is not concluded, two sentences on page 1; the rest is in the limitations. Not that Qwen3.5-4B has a linear representation of legality: at most, an illegality probe trained inside one harm stratum ranks unseen-topic sentences of the other stratum above chance (0.74) in a bare read, on 30 sentences, with an accuracy interval of 53–83%. Not that the prompted alignment is specific to legality: the label-swap band at layers 17–32 is itself about ±0.8 to 0.9, any within-cell contrast aligns with harm there, and the neutral-question control that would decide it was cut.

9. What was checked by hand. [pending Martin's recompute; write from `journal/verification-log.md` once the rows are his.]

10. The hand-check, directly under the summary (the dataset is LLM-written).

| | |
|---|---|
| Rows in the design | 250 (60 topics × 4, plus 10 replacement rows written after relabels emptied cells; quadrants 62 illegal-harmful, 65 illegal-harmless, 62 legal-harmful, 61 legal-harmless) |
| Read by me | 371 of 371 (the 250 design rows, 60 plain acts, 61 negations) |
| Excluded (bad sentence, too borderline, facts missing, or wrong and not worth fixing) | 6, leaving 244 in every evaluation |
| Relabelled against the first draft | 6 |
| Borderline on legality / on harm / on both | 66 / 46 / 23 |
| Rewritten by Claude after the pilot and re-read by me | 43 rows, because the rule-naming words ("prohibited", "without the required permit") lined up with the pilot's asymmetric transfer; plus the 60 legal-harmless rows replaced by length-matched twins of each topic's illegal-harmless row, because the originals averaged 7.7 words against 13.1 and word count alone separated them at AUROC 0.96 (now 0.60) |

Six random rows, seeded, not chosen: `m.sample(6, random_state=0)` on `set == main` of `data/scenarios.csv`. Ids s286, s163, s123, s210, s206, s215. Martin runs it and pastes id, quadrant, borderline flags and text.

Moved out of the summary into the body (so nothing is lost): the reverse-test and harm-probe numbers (body 4, table); the illegality direction predicting harm at 0.50 bare and 0.75 prompted (body 4, directions table); the prompted reverse probe's check counts, 59 of 60 and 36 of 61, and the bare reverse probe's 56 of 60 and 55 of 61 (body 4); the four other "not concluded" items (now limitations 4, 9, 10 and 13).

## Body (as much as needed; the reader should not need the code)

1. Why this question (drafted sentences; rewrite). Law and harm come apart in both directions. An illegality monitor that really reads harm misfires on the two off-diagonal quadrants; a harm monitor that really reads "against the rules" misfires the other way (the phrase in summary paragraph 1 is not repeated here). Which one a deployed model should run on is a policy choice, but either choice needs the model to keep the two apart at all, and that is what the cross tests. Sadhu and Schwarz stay here as the motivation. The other citations (Bertolazzi, Cho, Baez, Shah, Boxo, Frank, Sahoo, Xu) go to a short related-work paragraph placed before the limitations; what this adds is the illegality × harm factorial with conditional generalisation across strata on unseen topics, the angle between the factorial directions against a label-swap null, and the model's own answer as the fair baseline.
2. Data. Generation (Claude, 9 Sep), the labelling rules (rule-breaking is not law-breaking; statutes that quietly cover a harm), the hand-check tool and process, the cue-word audit, the length confound and the twins, the exclusion rule, the borderline split. The counts per quadrant.
3. Method. Extraction at the last token, in two conditions: the bare sentence, and the sentence inside the exact question the baseline uses, at the position the answer is generated from. Standardised logistic probe; topic-disjoint 30/15/15 split, seed 0; selection on validation AUROC; the four conditional-generalisation tests; the permutation null; factorial directions and the label-swap cosine band; harm projection evaluated within strata; the two check sets never trained on.
4. Results. Figure 1 and two tables: the four-test table (both conditions: AUROC, runs beaten on AUROC and on accuracy, accuracy at the fitted cut-off with its interval, word count alone) and the directions table (cosine and band at the probe's layer, the illegality direction as is and with harm projected out, the illegality direction predicting harm). Then the check counts for all four illegality probes, the fair baseline, the no-cue clause, and one clause on why the pooled off-diagonal logit AUROC (0.49) is not the fair comparison: pooled over both strata the logit is asked to sort illegal-harmless from legal-harmful, so it is scored across the harm confound the design removes; the fair line scores it within the harmless stratum on the probe's own 30 test rows.
5. What the probe got wrong. The bare headline probe's 9 mistakes out of 30: six illegal-harmless rows called legal (pet hedgehog in California, structuring a deposit, watering in a drought emergency, uncertified signal booster, the church bell at 5 a.m., a glass bottle in the paper bin) and three legal-harmless twins called illegal (the $12,000 deposit with the form filled in, photocopying three pages, a raised vegetable bed). The scores are dominated by topic-level offsets: both photocopy twins score +4.6 and +4.9, both church-bell twins far negative. Within twin pairs the illegal twin scores above its legal twin in 13 of 15 topics; the two pair failures are structuring against the lawful deposit (marked borderline on legality) and the 5 a.m. against the 5 p.m. bell (marked borderline on harm). This is a reading of the scores, with no null behind it. Source: `13-followup-baseline-recompute-checks.txt` (agent recompute; Martin repeats it with `notebooks/verify_by_hand.py`).
6. Related work, short, before the limitations (see 1).
7. Limitations, in this order:
   1. One model, one run, one split seed (0, never re-rolled), fp16 extraction on a T4.
   2. Thirty test sentences per test. Every accuracy interval is 23 to 30 points wide; the prompted headline's 50–50 is degenerate (every row called the same class). AUROC intervals were not computed; the Hanley–McNeil approximation for the headline 0.74 on 15 + 15 rows is about 0.56–0.92.
   3. Labels: one annotator, one jurisdiction wording ("US federal law or the law of most states, as of 2025"), 66 of 250 rows borderline on legality, 46 on harm, 23 on both. Six excluded, six relabelled; the excluded rows undercount label errors because a wrong label could be excluded instead of relabelled. The model itself calls 65% of the illegal-harmless rows legal.
   4. LLM-written data, rewritten twice: 43 rows after the cue-word audit, 60 legal-harmless rows replaced by length-matched twins. Both rewrites re-read. The harmful stratum was not length-matched (11.2 against 13.9 words).
   5. The plain-act check set is short (7.7 words) against 11 to 14 for the design rows; a probe that reads length partly explains the bare headline probe's 27 of 60, untested.
   6. The prompted condition's cut-off does not transfer across strata; accuracies in that condition are not comparable to the bare ones, and the pre-registered accuracy null fails for the prompted headline test.
   7. The cosine band in the prompted condition is wide (about ±0.8 to 0.9 from layer 17), so "outside the band" there is a thin statement; the sign, positive at every layer from 14 on, carries more of the claim. The neutral-question control was cut.
   8. The projection removes the between-stratum harm contrast; the wording is "orthogonal to the between-stratum harm contrast", not "harm removed". It applies to the factorial direction built from all four training quadrants, not to the conditional probe, so the 0.68 and 0.85 are not conditional-generalisation numbers. k = 1 for every projection number quoted.
   9. The fair baseline and the just-ask accuracies are agent-computed until Martin recomputes them.
   10. Correlational only. Steering was built and run once on the prompted condition, not analysed, not reported; the scaffold is on branch `steering-scaffold`.
   11. Hours are reconstructed from estimates and commit times, not Toggl.
   12. Verification: the 10 Sep Colab recompute used the old cell 10, which leaked the 121 check rows into the test set (0.683 against 0.700); the corrected recompute has not yet been run by Martin. Until it has, no number in the table has been recomputed by hand.
   13. The prompted reverse test's 0.91 is not read as a shared direction: at the same layer (26) the plain mean-difference direction from the harmless training rows scores the harmful test rows at 0.41 while the logistic probe scores 0.91; the two disagree by half the scale and nobody has looked at why (in the other seven tests they are within about 0.1).
   14. The bare reverse illegality test (0.71) is not quoted as evidence: word count alone does better on the same rows (0.79) and the harmful stratum's length imbalance is uncorrected (see 4).
   15. The probe does not beat, or match, asking the model: on the same 30 rows the model's own logit does better (0.81 against 0.74, agent until recomputed; see 9).

   Notes to the writer, not limitations for the reader (keep out of the doc): the two 0.742 values (bare and prompted headline) are 167 of 225 rank pairs in both conditions, a coincidence, so never call them "the same"; the prompted curve leaves the band at layer 14 (layer 13 is +0.26 inside a band whose top is +0.31), so never write "from layer 13"; all eight tests share the same 15 test topics and validation AUROCs are never quoted; the wording for the projection is "orthogonal to the between-stratum harm contrast".
8. What I verified and how. From `journal/verification-log.md`, in Martin's words: the recompute of every table number from the activations file; the six random rows; the mistakes read; the fair baseline from the raw answers; what was not checked (batched = unbatched extraction on the 4B; fp32 re-extraction; the 100-shuffle nulls rerun) and how surprised he would be.
9. Time. The reconstructed hours table from `journal/hours.md`, labelled as reconstructed.
10. How Claude was used. Scripts, notebook, first-draft sentences and the review passes by Claude under Martin's direction; every design decision, the hand-check, the runs and the verification Martin's; nothing agent-produced quoted unverified.
11. What next. The neutral-question control; the causal cross (built, on a branch); a second model; near-miss pairs per topic; a second annotator.

## Figures

- Fig 1: `figures/fig1_lp_4b.png` (two panels: the cross, and the cosine by layer with both bands).
- Never: anything from run `lp` (no suffix), `chk9`, `*.pilot-20260909`, or `mac05b`; never the per-design `probeeval_*.png` (their bottom panel is legal-positive).
