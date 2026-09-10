# Results — run `lp_4b` (copied from data/processed/*.json; recompute by hand before quoting)

| condition | probe: trained → tested | AUROC | beaten by AUROC (null p95) | accuracy at fitted cut-off (95% CI) | beaten by accuracy (null p95) | word count alone |
|---|---|---|---|---|---|---|
| bare | illegality, harmful → harmless | 0.74 | 100/100 (0.62) | 70% (53%–83%) | 100/100 (60%) | 0.60 |
| bare | illegality, harmless → harmful | 0.71 | 97/100 (0.69) | 67% (57%–80%) | 93/100 (67%) | 0.79 |
| bare | harm, illegal → legal | 0.95 | 100/100 (0.68) | 77% (63%–90%) | 98/100 (67%) | 0.57 |
| bare | harm, legal → illegal | 0.83 | 100/100 (0.63) | 80% (67%–90%) | 100/100 (63%) | 0.33 |
| prompted | illegality, harmful → harmless | 0.74 | 100/100 (0.59) | 50% (50%–50%) | 36/100 (57%) | 0.60 |
| prompted | illegality, harmless → harmful | 0.91 | 100/100 (0.68) | 70% (57%–83%) | 98/100 (63%) | 0.79 |
| prompted | harm, illegal → legal | 0.97 | 100/100 (0.69) | 63% (53%–77%) | 87/100 (67%) | 0.57 |
| prompted | harm, legal → illegal | 0.99 | 100/100 (0.65) | 87% (73%–97%) | 100/100 (63%) | 0.33 |

The arrow reads trained inside → tested on; 15 test topics × 2 rows of the tested stratum = 30 test sentences per design; all eight designs share the same test topics (seed 0). The beaten count is how many of the 100 shuffled-label runs of the whole procedure the probe scored above; the pre-registered count is the AUROC one.
The word-count column is the AUROC of sentence length for the label the JSON is written in (legal, or harmful): above 0.5 means longer sentences read as legal (or harmful), below 0.5 the reverse. Word count alone reaches the probe's own AUROC on the bare illegality, harmless → harmful rows (0.79 against 0.71), so that row is not evidence on its own.
In the prompted condition the illegality, harmful → harmless probe put all 30 harmless test rows on the same side of its cut-off (accuracy 50%, interval 50%–50%); its ranking of the same rows is 0.74. The cut-off fitted among harmful rows does not carry to harmless rows.

## Directions

| condition | layer | cos(d_illegal, d_harm) | label-swap band at that layer | illegality direction within harmless: as is → top harm component projected out | illegality direction predicting harm |
|---|---|---|---|---|---|
| bare | 25 | +0.09 | -0.28 to +0.26 | 0.68 → 0.68 | 0.50 |
| prompted | 15 | +0.70 | -0.54 to +0.49 | 0.85 → 0.74 | 0.75 |

A cosine of +1 means the illegality and harm directions coincide; the band is the 95% range of the same cosine with the legality labels swapped within each topic × harm cell (the JSON stores cos(d_legal, d_harm); the sign is flipped here). The last column is the AUROC of the illegality direction for harmful vs harmless on the test rows, printed as 1 − the stored d_legal value. Projection is k = 1 (the harm direction itself).
- bare: the curve is outside the band at 1 of 33 layers (layer 9); over layers 17–32 it runs +0.09 to +0.19 and the band's upper edge runs +0.19 to +0.35.
- prompted: the curve is outside the band at 19 of 33 layers (layers 14–32); over layers 17–32 it runs +0.85 to +0.95 and the band's upper edge runs +0.72 to +0.92.

## Checks

- bare, illegality, harmful → harmless: plain acts (60, never trained on): 33 of 60 called legal; negated (61, never trained on): 61 of 61 called legal.
- bare, illegality, harmless → harmful: plain acts (60, never trained on): 56 of 60 called legal; negated (61, never trained on): 55 of 61 called legal.
- prompted, illegality, harmful → harmless: plain acts (60, never trained on): 58 of 60 called legal; negated (61, never trained on): 58 of 61 called legal.
- prompted, illegality, harmless → harmful: plain acts (60, never trained on): 59 of 60 called legal; negated (61, never trained on): 36 of 61 called legal.
- the harm probes call 0% to 12% of the check rows harmful (plain acts and negations, both conditions).
- bare: 3 of 244 design rows contain a legality word; with them dropped (241 rows) the harmful → harmless illegality probe scores AUROC 0.74, beat 100/100 on AUROC.

## Just asking the model

- 'is this illegal?': right on 76% of the design rows; by quadrant illegal-harmful 97%, illegal-harmless 34%, legal-harmful 75%, legal-harmless 97%; 55% on the off-diagonal rows; 0% unparsed. Yes−No logit AUROC pooled over both strata, all off-diagonal rows: 0.49 (not the fair comparison; see below). The model calls 100% of the plain acts legal and 87% of the negations legal.
- 'is this harmful?': right on 87% of the design rows; by quadrant illegal-harmful 98%, illegal-harmless 67%, legal-harmful 90%, legal-harmless 92%; 79% on the off-diagonal rows; 0% unparsed. Yes−No logit AUROC pooled over both strata, all off-diagonal rows: 0.84 (not the fair comparison; see below). The model calls 0% of the plain acts harmful and 11% of the negations harmful.

Fair baseline: on the same 30 held-out harmless-stratum sentences the harmful → harmless illegality probe is tested on, the model's own Yes−No logit sorts illegal from legal at AUROC 0.81; the probe scores 0.74 (bare) and 0.74 (prompted). On the 30 harmful-stratum rows of the reverse test: model 1.00, probes 0.71 (bare) and 0.91 (prompted).
