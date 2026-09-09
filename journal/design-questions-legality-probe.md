# Design sheet — legality probe (clean version, 9 Sep 2026)

This is the single statement of the design. The step-by-step is the vault run sheet (`plans/applications/MATS 12 - Run Sheet (legality probe).md`); the execution is `notebooks/legality_probe_colab.ipynb`; the hand-check columns are explained in `data/SCENARIOS_COLUMNS.md`; the *why* is the vault spec (`MATS 12 - Project Spec (Legality Probe)`). Earlier versions of this sheet are archived in `design-questions-legality-probe-archive.md`.

Answer the numbered decisions below in a sentence each, copy the answers into `highlights.md` under "Initial hypothesis" with the date, start Toggl. That is the clock's starting gun.

## The question
Does Qwen3.5-4B represent *illegality* as a concept distinct from *harmfulness*, or is a legality probe just a harm probe wearing a hat?

## The design (fixed unless you change it here)
- **Data.** `data/scenarios.csv`: 240 sentences, 60 topics × 4 quadrants (illegal-harmful, illegal-harmless, legal-harmful, legal-harmless), one per quadrant per topic, second person, US law (federal or most states, as of 2025). Written by Claude on 9 Sep; every row starts `hand_checked=0`.
- **Unit of analysis.** The topic (four sentences). All splits are by topic.
- **Test.** *Conditional generalisation.* Train the legality probe on harmful rows only (legal-harmful vs illegal-harmful), test it on harmless rows only (legal-harmless vs illegal-harmless) from topics it never saw. Then the reverse. Then the symmetric pair for harm. Training on the two "easy" corners is **not** used, because there legal = not-harmful on every row and the probe cannot tell them apart.
- **Selection.** Topics split 30 train / 15 validation / 15 test. Layer and regularisation (C ∈ {0.01, 0.1, 1, 10}) chosen on cross-stratum accuracy over validation topics only. Test topics scored once. Interval: topic-block bootstrap. Null: the whole procedure, selection included, repeated 20 times on shuffled labels.
- **Directions.** Factorial contrasts on training topics: d_legal = ½[(LH − IH) + (LH̄ − IH̄)], d_harm likewise. Report cos(d_legal, d_harm); score each on held-out topics; also d_legal with its harm component removed.
- **Two extraction conditions.** (A) bare sentence, last token; (B) the sentence inside "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No." at the token before the answer. Every design runs in both.
- **Controls.** Permutation null (above). Rerun with every row containing legality vocabulary dropped. Just-ask baseline for legal and for harmful, with invalid answers scored separately. Rows marked `exclude=1` dropped and reported.
- **Stretch.** Steering with d_legal and its harm-removed version while the model answers the legality question: Yes−No logit shift, strengths at 0.25/0.5/1/2 × the activation norm, five norm-matched random directions.

## Decisions that are yours (answer each)
1. **Jurisdiction wording** in the write-up: "US federal law or the law of most US states, as of 2025" — keep, or narrow to one state?
2. **Hand-check scope.** All 240, or the two off-diagonal quadrants first and the rest if time allows? (Recommended: all, off-diagonal first.)
3. **Exclusion rule.** When does a row get `exclude=1` rather than `borderline=1`? (Recommended: exclude when the label depends on facts the sentence does not give; borderline when a competent lawyer could argue either way on the facts given.)
4. **Which condition is the headline**, bare (A) or prompted (B)? (Recommended: B is the headline, A is the "unprompted representation" secondary result.)
5. **Split seed and sizes.** Keep 30/15/15 with seed 0, or change? (Do not re-roll the seed after seeing results.)
6. **The prediction, with numbers.** "I predict the legality probe trained on harmful rows reaches ≥ ___ on harmless rows from unseen topics (and the reverse ≥ ___), above the permutation null's 95th percentile; that |cos(d_legal, d_harm)| ≤ ___; and that the harm-removed legality direction predicts legality on held-out topics at AUROC ≥ ___." Write the predicted-in-advance negative too: cross-stratum accuracy at the null and d_legal collapsing once harm is removed.
7. **Stop rule.** If hour 12 arrives with only the hand-check done: run cells 6–9, skip 10's paragraph and 11, write. Confirm or change.

## Boring explanations, named in advance
- The probe reads legality *words* → no-cue rerun.
- The probe reads topic → topic-disjoint splits and topic-matched quadruples.
- The probe reads harm → conditional generalisation and the harm-removed direction.
- Layer selection inflated the result → selection on validation topics only, permutation null with the same selection.
- The model refuses or rambles on the just-ask baseline → invalid answers scored separately.
- Labels are wrong → the hand-check counts (read, relabelled, excluded) are reported, and borderline rows are analysed separately.

## Prior work to cite in the first paragraph
Sadhu et al. 2026 (arXiv 2608.16852): compliance readouts are rule-blind; a crossed rule × scenario benchmark with group-disjoint evaluation; the model list must be taken from the full text before quoting sizes. Schwarz 2026 (2607.13075): harm probes fail surface-matched benign controls. Baez et al. 2026 (2607.07003): dissociation by cross-subtype transfer and direction cosine. Shah et al. 2025 (2507.21141): harm subconcept directions form a strikingly low-rank subspace with a dominant direction. Boxo et al. 2025 (2509.21344): probes lean on textual cues. What this project adds relative to all of them: the legality × harm factorial with conditional generalisation across strata, the angle between factorial directions, and an intervention.

## Review history
- 13 Aug: first sheet (diagonal-train / off-diagonal-test).
- 8 Sep: scan found Sadhu and Schwarz; project briefly switched to value leakage, then back.
- 9 Sep: external review (ChatGPT) showed the diagonal training confounds legality with not-harm; design replaced with conditional generalisation, topic-disjoint selection, permutation null, factorial directions, two conditions, logit-based steering. Fictional-stipulated-rule alternative rejected (it is Sadhu's design).
