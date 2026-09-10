# Highlights doc (running)

Running doc: hypotheses, key graphs and dead ends, appended in the order they happened, with dates. The write-up is built from it.

## Hypotheses

Plain-language version. Fill in the blanks in your own words; once cell 6 has run on the 4B model, this block is frozen and gets quoted in the write-up as the prediction made in advance.

### What I expect to find — written down before the real run (10 Sep 2026)

**The question in one line.** Does the model know the difference between "against the law" and "hurts someone", or does it only know the second and use it for both?

**Which run counts as the main one.** The probe reads the model's state right before it answers "Is this illegal? Yes or No" (the *prompted* condition). The other run, where the model just reads the sentence with no question (*bare*), is reported second. — *Keep this or swap it: ______*

**What I expect the main test to show.** Train the legality probe only on harmful sentences, test it only on harmless sentences from topics it has never seen. If the model really tracks legality, the probe should still sort legal from illegal there.
- My guess for how well it sorts them (AUROC; 0.5 = coin flip, 1.0 = perfect): 85 (Claude's suggestion: about 0.75)
- How many of the 100 shuffled-label runs it should beat: 96 (suggestion: 95 or more)
- The same test the other way round (train on harmless, test on harmful): 65 (because harmlessness is more complex than just not(harmful)) (claude suggestion: about 0.70)
- The harm probe across legality (train on illegal, test on legal, and back): 90 ish maximum (claude suggestion: 0.80 or more both ways; harm is the easier concept)

**What I expect the "hat test" to show.** Cosine similarity between the linear representations encoded by the probes on harm and illegality will be higher than random.
- The angle between them (cosine; 0 = unrelated, ±1 = same line): 0.25 (claude's suggestion: no more than 0.4, and inside the shuffled band)
- If I remove the harm part from the legality arrow, does the rest still sort legal from illegal among harmless sentences? My guess (AUROC): Hopefully 0.8 (suggestion: about 0.70)

**Against the model's own answer.** The probe has to beat, or at least match, just asking the model "is this illegal?" on the same sentences. Likely small or no difference between the probe and the model, if any difference, only marginally increased accuracy of separation. 

**The checks.** The 60 plain legal acts ("You cook pasta") should all be called legal; my guess for how many: 58 of 60. The 61 "You do not …" sentences should all be called legal; my guess is that slightly fewer will be labelled correctly, because there is more uncertainty because adding "do not" complicates the structure of the sentence: 50 of 61.

**The result I would also count as a real finding.** If the legality probe does no better than the shuffled labels, or the legality arrow collapses once harm is removed, then on this model a "compliance monitor" built from activations is a harm detector. That would be in accordance with the literature, and totally publishable too, but not as interesting as if we could distinguish illegality and harm.

**Fixed choices.** 30 training topics, 15 validation, 15 test; split seed 0, never re-rolled. Dataset: 250 design rows (6 excluded), 60 simple anchors and 61 negations as checks, all read by me.

## Key results (graph + one paragraph each)

## Dead ends / pivots

## Things to not forget in the limitations section

2026-09-10, after the run (facts, from `data/processed/probeeval_lp_4b*.json`, `ask_lp_4b_legal.json` and the team's 10 Sep recomputation; the ones marked (agent) are not yet recomputed by Martin):

- In the prompted condition the label-swap band is itself about ±0.8 to 0.9 from layer 17 on, so "outside the band" there is a thin statement; the sign, positive at every layer from 14, carries more. The neutral-question control, cut on 10 Sep, is what would separate a legality-specific merge from a collapse onto the answer axis.
- The model's own Yes−No logit on the 30 headline test rows scores 0.81; the probe scores 0.74 (agent).
- The bare headline probe calls 27 of 60 plain acts illegal.
- The harmful stratum is length-imbalanced: illegal-harmful rows average 11.2 words, legal-harmful 13.9; word count alone reaches 0.79 on the reverse test's rows, against the probe's 0.71. Not corrected.
- The harm probe sorts illegal-harmless from legal-harmless at 0.61 bare and 0.74 prompted (agent), so the harmless stratum is not harm-flat.
- At prompted layer 26 the plain mean-difference direction scores the reverse test's rows at 0.41 where the logistic probe scores 0.91. Unexplained.
- The labels are one annotator's; the model disagrees with the illegal-harmless label on 65% of those rows (it calls them legal).
