# Write-up skeleton (Google Doc) — fill in Friday; every number comes from data/processed/*.json and is recomputed by hand first

Claude drafted this structure on 10 Sep evening from the design sheet and Nanda's doc. The prose is yours; the brackets are the numbers to fill. Rules from the design sheet: say "first topic-matched legality × harm test", never "first legality probe", never "a linear representation exists"; no pilot numbers anywhere; the 43 rewritten rows and the 60 replaced legal-harmless rows are disclosed.

## Title
*Is a legality probe a harm probe wearing a hat? A topic-matched legality × harm test on Qwen3.5-4B*

## Executive summary (pages 1–2, ≤ 600 words, bullets fine, one figure)

- **Question.** Does Qwen3.5-4B carry a representation of *illegality* that is separate from *harmfulness*? Compliance monitors built from activation probes are rule-blind (Sadhu et al. 2026) and harm probes are largely topic detectors (Schwarz 2026); a monitor that reads harm misfires exactly where law and harm come apart.
- **Design.** 60 topics × 4 matched sentences (illegal-harmful, illegal-harmless, legal-harmful, legal-harmless; US law; second person), every row read by a lawyer. The legality probe is trained inside one harm stratum and tested on the other, on held-out topics (conditional generalisation), because on the easy corners legal = not-harmful. Layer and C chosen on validation topics only; test scored once; null = the whole procedure on labels shuffled within topic × stratum, 100 times.
- **Headline result.** [prompted condition] Legality, harmful → harmless: AUROC [ ] (accuracy [ ]%, CI [ ]–[ ]), beat [N]/100 shuffles. Reverse: AUROC [ ], beat [N]/100. Harm across legality strata: [ ] and [ ]. → *[one sentence: distinct / not distinct / one-way]*.
- **The hat test.** cos(d_legal, d_harm) at layer [ ] = [ ] (label-swap band [ ]..[ ]); the legality direction with the top harm component projected out still sorts the harmless-stratum held-out rows at AUROC [ ] (0.5 = nothing left). Figure 1: cosine by layer, bare vs prompted, with the null band.
- **Baselines and checks.** The model's own Yes/No logit on the same held-out rows: AUROC [ ] vs the probe's [ ]. Word count alone: [ ]. Every sentence with legality vocabulary removed: [ ]. 60 plain legal acts never trained on: the probe calls [ ]/60 legal. 61 negated illegal acts ("You do not …"), never trained on: the probe calls [ ]/61 legal.
- **What I'd conclude, and what I wouldn't.** [two sentences]
- **The dataset is the project.** [see table below] — this line goes here so it is on page 1.

### The hand-check, directly under the summary (required: LLM-written data)
| | |
|---|---|
| Rows in the design | 250 (60 topics × 4, plus 10 replacement rows for cells emptied by exclusions) |
| Read by me | 250 of 250 |
| Excluded (bad sentence, too borderline, facts missing, or wrong and not worth fixing) | 6 |
| Relabelled against the first draft | [6] |
| Borderline on legality / on harm / both | [64 / 46 / 23] → recount after the twins pass |
| Rewritten by Claude after the pilot and re-read by me | 43 (rule-naming words removed) + 60 (legal-harmless rows replaced by length-matched twins: the originals averaged 7.7 words vs 13.1 for illegal-harmless, and word count alone separated them at AUROC 0.96) |

Six random rows (seeded, not chosen): [run `python -c` with seed 0 over data/scenarios.csv set=main and paste id · text · labels · flags].

## Body (as much as needed; the reader should not need the code)

1. **Why this question** — Martin's framing (10 Sep, his words to write): law and harm come apart in both directions. A legality monitor that really reads harm misses harmless infractions and flags legal cruelty; a harm monitor that really reads "against the rules" moralises the law and flags jaywalking as a wrong. Which one a deployed model should run on is a policy choice, but either choice needs the model to keep the two apart at all, and that is what the cross tests. Then Sadhu, Schwarz, Bertolazzi, Cho (value entanglement), Baez, Shah, Boxo in one paragraph; what this adds (the factorial, cross-stratum generalisation on unseen topics, the cosine with a null, the model's own answer as the fair baseline).
2. **Data** — generation (Claude, 9 Sep), the labelling rules (rule-breaking ≠ law-breaking; hidden statutes), the hand-check tool and process, the cue-word audit, the length confound and the twins, the exclusion rule, the borderline split. Table of counts per quadrant.
3. **Method** — extraction (last token; bare, and inside the exact question the baseline uses), standardised logistic probe, topic-disjoint 30/15/15, selection on validation AUROC, the four conditional-generalisation tests, the permutation null, factorial directions and the label-swap cosine band, harm projection evaluated within strata, the extra check sets.
4. **Results** — the four-test table (both conditions), Figure 1 cosine by layer, Figure 2 validation curve for the headline, the check-set lines, the fair baseline, the no-cue rerun, the length-only line.
5. **What the probe got wrong** — the held-out mistakes, read and characterised (cell 10).
6. **Evidence against the hypothesis** — whatever the null, the fair baseline, the harm-projection and the check sets say against the headline; write it straight.
7. **Limitations** — one model; 30-sentence test set per design (CI ±15 points); US law as of 2025 as the ground truth, one annotator (me); LLM-written data, twice rewritten; correlational (steering designed, not run); "orthogonal to the between-stratum harm contrast", not "harm removed".
8. **What I verified and how** — from `verification-log.md`: the by-hand recompute of the headline accuracy from the activations file; the six random rows; the mistakes read; what was not checked and how surprised I'd be.
9. **Time** — reconstructed hours table from `hours.md`, labelled as reconstructed (no Toggl).
10. **How Claude was used** — scripts, notebook, first-draft sentences and the review passes by Claude under my direction; every design decision, the hand-check, the runs and the verification mine; nothing agent-produced quoted unverified.
11. **What next** — the causal cross (built, on a branch), a second model, near-miss pairs per topic, a second annotator.

## Figures
- Fig 1: `figures/cosine_by_layer_<run>.png` (cell 8b)
- Fig 2: `figures/probeeval_<run>_prompted_L_h2nh.png` (validation curve, chosen layer, test point, shuffled ceiling)
- Never: the seven `pilot_*` PNGs.
