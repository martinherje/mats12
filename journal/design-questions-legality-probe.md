# Design sheet — legality probe (clean version, 9 Sep 2026)

This is the single statement of the design. The step-by-step is the vault run sheet (`plans/applications/MATS 12 - Run Sheet (legality probe).md`); the execution is `notebooks/legality_probe_colab.ipynb`; the hand-check columns are explained in `data/SCENARIOS_COLUMNS.md`; the *why* is the vault spec (`MATS 12 - Project Spec (Legality Probe)`). Earlier versions of this sheet are archived in `design-questions-legality-probe-archive.md`.

Answer the numbered decisions below in a sentence each, copy the answers into `highlights.md` under "Initial hypothesis" with the date, start Toggl. The clock starts when that is done.

## The question
Does Qwen3.5-4B represent *illegality* as a concept distinct from *harmfulness*, or does an illegality probe just read harm?

## The design (fixed unless you change it here)
- **Data.** `data/scenarios.csv`: 240 sentences, 60 topics × 4 quadrants (illegal-harmful, illegal-harmless, legal-harmful, legal-harmless), one per quadrant per topic, second person, US law (federal or most states, as of 2025). Written by Claude on 9 Sep; every row starts `hand_checked=0`.
- **Unit of analysis.** The topic (four sentences). All splits are by topic.
- **Test.** *Conditional generalisation.* Train the legality probe on harmful rows only (legal-harmful vs illegal-harmful), test it on harmless rows only (legal-harmless vs illegal-harmless) from topics it never saw. Then the reverse. Then the symmetric pair for harm. Training on the two "easy" corners is **not** used, because there legal = not-harmful on every row and the probe cannot tell them apart.
- **Selection.** Topics split 30 train / 15 validation / 15 test. Layer and regularisation (C ∈ {0.01, 0.1, 1, 10}) chosen on cross-stratum AUROC over validation topics only. Test topics scored once. Interval: topic-block bootstrap. Null: the whole procedure, selection included, repeated 100 times on labels shuffled within each topic × stratum cell; reported as "beat N of 100".
- **Directions.** Factorial contrasts on training topics: d_legal = ½[(LH − IH) + (LH̄ − IH̄)], d_harm likewise. Report cos(d_legal, d_harm) at every layer with a label-swap null band; score d_legal within each harm stratum on held-out topics, and again with the top 1–3 harm components projected out; also the plain half-contrast (train-stratum direction on other-stratum test rows).
- **Three extraction conditions.** (A) bare sentence, last token; (B) the sentence inside "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No." at the token before the answer; (C) the sentence inside a neutral Yes/No question ("Is the following sentence written in the second person?"), same position, as the control for B. The four designs run on A and B; C supplies the cosine curve.
- **Controls.** Permutation null (above). Rerun with every row containing legality vocabulary dropped. Just-ask baseline for legal and for harmful, with invalid answers scored separately, and the model's Yes−No logit scored as a classifier on the same held-out rows as the probe (the fair baseline). Rows marked `exclude=1` dropped and reported.
- **Stretch.** Steering with the prompted-run d_legal and its harm-removed version at their mid layer while the model answers the legality question: Yes−No logit shift, strengths at 0.05/0.1/0.2 × the activation norm, five norm-matched random directions.

## Decisions that are yours (answer each)
1. **Jurisdiction wording** in the write-up: "US federal law or the law of most US states, as of 2025" — keep, or narrow to one state?
2. **Hand-check scope.** All 240, or the two off-diagonal quadrants first and the rest if time allows? (Recommended: all, off-diagonal first.)
3. **Exclusion rule.** *Answered 10 Sep:* `exclude=1` is the broad bin — a bad sentence, a case too borderline to keep, a label that depends on facts not given, or a wrong label not worth fixing. `borderline=1` keeps a row a competent lawyer could argue either way. Excluded rows are dropped from the probe, the just-ask baseline and steering alike, and are counted (with reasons where noted) in the write-up; the relabelled count therefore undercounts label errors, and the write-up says so.
4. **Which condition is the headline**, bare (A) or prompted (B)? (Recommended: B is the headline, A is the "unprompted representation" secondary result.)
5. **Split seed and sizes.** Keep 30/15/15 with seed 0, or change? (Do not re-roll the seed after seeing results.)
6. **The prediction, with numbers.** "I predict the legality probe trained on harmful rows reaches ≥ ___ on harmless rows from unseen topics (and the reverse ≥ ___), beating at least ___ of 100 within-cell shuffles; that |cos(d_legal, d_harm)| ≤ ___ in the bare condition at the chosen layer; and that the legality direction with the top-3 harm components removed still sorts the harmless-stratum held-out rows at AUROC ≥ ___." Write the predicted-in-advance negative too: cross-stratum accuracy at the null and d_legal collapsing once harm is removed.
7. **Stop rule.** If hour 12 arrives with only the hand-check done: run cells 6–9, skip 10's paragraph and 11, write. Confirm or change.

## Second review, 9 Sep evening (four reviewers: originality, interestingness, statistics, Nanda fit) — what changed
- **Dataset:** 43 rows that named a rule ("prohibited", "without the required permit", "within the rules") rewritten so legality is inferable from common knowledge; the cue rate is now zero in every quadrant. Reason: the cue rate was 28/60 in illegal-harmless and near zero elsewhere, which lined up exactly with the pilot's asymmetric transfer. Weapons topic rebalanced.
- **Null:** labels shuffled within each topic × stratum cell (keeps the design's balance), 100 shuffles, reported as "beat N of 100"; with 20 shuffles the smallest p was 1/21 and all three "successes" sat on that floor.
- **Selection:** layer and C chosen on validation AUROC, not accuracy on 30 rows (which produced ties and odd layers).
- **Cosine:** reported as a curve over all layers with a within-cell label-swap null band, for three conditions (bare, prompted, neutral question), not one number at each probe's chosen layer.
- **Harm removal:** evaluated within each harm stratum on held-out topics (on a balanced set removing harm cannot hurt by construction), with the top 1–3 harm components projected out, not one vector.
- **Fair baseline:** the model's own Yes−No logit scored as a classifier on the same held-out rows as the probe.
- **Neutral-question control:** if the legality–harm cosine also collapses under "is this sentence in the second person?", the collapse is the answer axis, not legality.
- **Demoted:** asymmetric transfer, explained by the cue confound and the model's own 50% on the illegal-harmless quadrant; a footnote at most.
- **Citations added:** Bertolazzi, Pezzelle & Bernardi 2025 (arXiv 2510.06700), the same recipe on logical validity × plausibility; the defence is the concept pair and the topic-matched cross, not the method.
- **Write-up:** counts (read / relabelled / excluded / borderline) and six random rows directly under the executive summary; no pilot numbers anywhere; "first topic-matched test", never "first legality probe" and never "legality is linearly represented".
- **Steering:** prompted-run directions at their mid layer, strengths ≤ 0.2 of the activation norm (the pilot broke the model above 0.5 and used the final layer).

## Third review, 10 Sep afternoon (flaw review, literature scan, logistics) — what changed
- **Length confound in the harmless stratum:** legal-harmless rows averaged 7.7 words against 13.1 for illegal-harmless; a length-only scorer reaches AUROC 0.96 on the headline test set. Fix: the 60 legal-harmless rows are replaced by length-matched legal twins of each topic's illegal-harmless row (Martin's decision pending at the time of writing); a length-only AUROC line goes in the results table.
- **Prompt string:** the prompted extraction was receiving a literal backslash-n from the notebook shell line; the questions are defined once in `common.py` and extraction uses `--question legal`.
- **Storage and split:** activations stored fp32; the topic split is drawn from the full topic list before any exclusion.
- **Minimal rule (Martin):** one test, one null, one baseline, one figure; steering, the neutral-question condition, nested CV, bag-of-words and split-half extras are cut. The causal-steering scaffold lives on branch `steering-scaffold`, unmerged.
- **Wording:** the projected-out direction is "orthogonal to the between-stratum harm contrast", not "harm removed"; the 43 rows rewritten after the pilot are disclosed as such; the hand-check counts and six random rows sit under the executive summary.
- **10 Sep evening, after the run:** the reported core is one test (illegality, harmful → harmless) run in both conditions, one null (the AUROC beat-count, as pre-registered), one baseline (the model's own logit on the same 30 rows), one two-panel figure; the harm projection is reported at k = 1 only; the half-contrast and the no-cue rerun are one clause each; steering ran once on the prompted condition and is not reported. The top-1..3 projection, the three-condition cosine and the 240-row count above are superseded by this line.

## Boring explanations, named in advance
- The probe reads legality *words* → no-cue rerun.
- The probe reads topic → topic-disjoint splits and topic-matched quadruples.
- The probe reads harm → conditional generalisation and the harm-removed direction.
- Layer selection inflated the result → selection on validation topics only, permutation null with the same selection.
- The model refuses or rambles on the just-ask baseline → invalid answers scored separately.
- Labels are wrong → the hand-check counts (read, relabelled, excluded) are reported, and borderline rows are analysed separately.

## Prior work to cite in the first paragraph
Bertolazzi, Pezzelle & Bernardi 2025 (arXiv 2510.06700): logical validity and plausibility are linearly represented and strongly aligned, plausibility vectors bias validity judgements, debiasing vectors disentangle them — the closest method precedent. Sadhu et al. 2026 (arXiv 2608.16852): compliance readouts are rule-blind; a crossed rule × scenario benchmark with group-disjoint evaluation; the model list must be taken from the full text before quoting sizes. Schwarz 2026 (2607.13075): harm probes fail surface-matched benign controls. Baez et al. 2026 (2607.07003): dissociation by cross-subtype transfer and direction cosine. Shah et al. 2025 (2507.21141): harm subconcept directions form a strikingly low-rank subspace with a dominant direction. Boxo et al. 2025 (2509.21344): probes lean on textual cues. Cho, Li & Leshinskaya 2026 (arXiv 2602.19101, *Value Entanglement*): the closest sibling — moral, grammatical and economic value probed by mean-difference vectors on factorial stimuli in Qwen3-family models, moral value contaminates the neighbouring judgements, ablating the moral vector repairs them; no cross-stratum generalisation, no cosine null, no legality. Methodological pre-emptions to cite in one sentence each: Frank 2026 (2603.18280: probe accuracy is non-diagnostic, held-out-category generalisation is the test; ablating an entangled direction in Qwen3-8B causes confabulation), Sahoo et al. 2026 (2606.02907: perfect probes on Qwen3-14B read task format; residualise format), Xu, Rusnak & Kaplan 2026 (2603.23659: asymmetric cross-framework transfer of ethics probes, which partly read template surface features — the reason asymmetric transfer stays a footnote here). Sadhu's model list is confirmed from the full text and includes Qwen3.5-4B; quote its Table 8 numbers for that checkpoint when comparing. What this project adds relative to all of them: the legality × harm factorial with conditional generalisation across strata on unseen topics, the angle between factorial directions against a label-swap null, and the model's own answer as the fair baseline. Say "first topic-matched legality × harm test", never "first legality probe".

## Review history
- 13 Aug: first sheet (diagonal-train / off-diagonal-test).
- 8 Sep: scan found Sadhu and Schwarz; project briefly switched to value leakage, then back.
- 9 Sep: external review (ChatGPT) showed the diagonal training confounds legality with not-harm; design replaced with conditional generalisation, topic-disjoint selection, permutation null, factorial directions, two conditions, logit-based steering. Fictional-stipulated-rule alternative rejected (it is Sadhu's design).
