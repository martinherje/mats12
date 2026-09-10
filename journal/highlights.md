# Highlights doc (running)

Nanda-process running doc: hypotheses as they form, key graphs as they land, dead ends as they die. Append-only, timestamped. This becomes the spine of the write-up.

## Hypotheses

### Pre-registration, 10 Sep 2026 written before running notebook on the full dataset.
- **Headline condition:** prompted (B). Bare (A) is the secondary "unprompted representation" result. (Design-sheet decision 4.)
- **Prediction (decision 6):** the legality probe trained on harmful rows reaches AUROC ≥ 0.75 on harmless rows from unseen topics, and the reverse ≥ 0.70, each beating ≥ 95 of 100 within-cell shuffles. The harm probe does the same across legality strata (≥ 0.80 both ways). |cos(d_legal, d_harm)| at the chosen layer in the bare condition ≤ 0.4 and inside or near the label-swap band; the legality direction with the top harm component projected out still sorts the harmless-stratum held-out rows at AUROC ≥ 0.70. The probe beats the model's own Yes−No logit on the same rows by ≥ 0.05 AUROC, or ties it.
- **Predicted-in-advance negative:** cross-stratum AUROC within the null's 95th percentile, or the harm-removed legality direction at ≤ 0.6 within the harmless stratum → legality is not separably represented on this model; a compliance monitor built this way is a harm detector. That is a headline too.
- **Split:** 30/15/15 topics, seed 0, not re-rolled. Dataset: 250 rows, 6 excluded, 60 legal-harmless rows replaced by length-matched twins on 10 Sep before any run on the checked data.

## Key results (graph + one paragraph each)

## Dead ends / pivots

## Things to not forget in the limitations section
