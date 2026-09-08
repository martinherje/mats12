# Design questions — legality probe (operative since 13 Aug; supersedes design-questions.md)

The sheet for the retired confabulated-compliance direction is kept as-is (append-only). This one is for the project actually being run: **does the model represent illegality as distinct from harmfulness?** Spec: vault `plans/applications/MATS 12 - Project Spec (Legality Probe).md`. You answer these; the answers copied into `highlights.md` as the initial hypothesis block, dated, are the clock's starting gun.

## 1. Jurisdiction
One only, named in every table. Menu: (a) Norwegian law — your expertise, cleaner hand-checks, but the model's legality knowledge is weaker and English-language scenarios about Norwegian law are unusual; (b) US federal/common state law — model knowledge strongest, reviewers can sanity-check, but your relabelling authority is weaker at the margin. Pick and state why in one sentence.

## 2. Scenario shape
Second person present tense ("You ...")? Third-person description? A question ("Is it illegal to ...")? The shape changes what the last-token state encodes. Fix one shape for all quadrants, and decide whether the chat template is used (`--template chat`) or raw text.

## 3. N per quadrant and the topic-matched control
Target N per quadrant for core cases, and how many topic-matched minimal pairs (same subject, one side of the line). Rosser's lesson: hand-check the first 40 *before* generating hundreds.

## 4. Generation and hand-check protocol
Who generates (an LLM under your prompt; which model), what fraction you read, and the rule for relabelling. Record: number checked, number relabelled, the disagreement categories. This paragraph goes into the write-up verbatim.

## 5. Train/test split
Committed default in RUNPOD.md: train on the diagonal (illegal_harmful + legal_harmless), transfer-test on the off-diagonal. Confirm, or choose a different design and say why.

## 6. The dissociation prediction (write it as a sentence before running)
"I predict the legality probe trained on the diagonal transfers to the off-diagonal at ≥ X% while the harm probe on the same data predicts legality labels at ≤ Y%, and the cosine between the two diff-means directions is ≤ Z." Commit X, Y, Z.

## 7. Boring explanations to name and kill
Committed: topic/sensitivity (off-diagonal + topic-matched pairs), lexical cues (bag-of-words baseline + a rerun with explicit legality vocabulary removed). Add a third? Candidate: "the probe reads refusal-likelihood" — check against the model's own refusal rate per quadrant.

## 8. Baseline "just ask the model"
Which prompt, which decoding, and how its accuracy is compared to the probe's. If asking matches the probe, the probe adds nothing — say so.

## 9. Steering (stretch)
Only if hour 12 arrives with the sweep + baselines done. Which direction, which layer, which behaviour measured, random-direction control. Decide now what gets cut first if not: scenario volume before verification.

## 10. Stop rule
What is the minimum result that still makes an honest write-up? (e.g. the 2×2 at one layer, one baseline, the hand-check paragraph.)
