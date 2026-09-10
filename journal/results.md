# Results — run `lp_4b` (copied from data/processed/*.json; recompute by hand before quoting)

| condition | test | layer | AUROC | acc (CI95) | beat shuffles | cos(d_illegal, d_harm) | illegality dir. within harmless, top-1 harm removed | word count alone |
|---|---|---|---|---|---|---|---|---|
| bare | legality, harmful → harmless (headline) | 25 | 0.74 | 70% (53%–83%) | 100/100 (p95 60%) | +0.09 | 0.68 | 0.60 |
| bare | legality, harmless → harmful | 32 | 0.71 | 67% (57%–80%) | 93/100 (p95 67%) | +0.11 | 0.68 | 0.79 |
| bare | harm, illegal → legal | 14 | 0.95 | 77% (63%–90%) | 98/100 (p95 67%) | +0.18 | 0.96 | 0.57 |
| bare | harm, legal → illegal | 3 | 0.83 | 80% (67%–90%) | 100/100 (p95 63%) | +0.00 | 0.96 | 0.33 |
| prompted | legality, harmful → harmless (headline) | 15 | 0.74 | 50% (50%–50%) | 36/100 (p95 57%) | +0.70 | 0.74 | 0.60 |
| prompted | legality, harmless → harmful | 26 | 0.91 | 70% (57%–83%) | 98/100 (p95 63%) | +0.92 | 0.51 | 0.79 |
| prompted | harm, illegal → legal | 15 | 0.97 | 63% (53%–77%) | 87/100 (p95 67%) | +0.70 | 0.93 | 0.57 |
| prompted | harm, legal → illegal | 14 | 0.99 | 87% (73%–97%) | 100/100 (p95 63%) | +0.61 | 0.93 | 0.33 |

## Checks

- bare, legality, harmful → harmless (headline): check set **negated** (61 rows, never trained on, expected all 1): the probe calls 100% of them legal=1.
- bare, legality, harmful → harmless (headline): check set **simple** (60 rows, never trained on, expected all 1): the probe calls 55% of them legal=1.
- bare, legality, harmless → harmful: check set **negated** (61 rows, never trained on, expected all 1): the probe calls 90% of them legal=1.
- bare, legality, harmless → harmful: check set **simple** (60 rows, never trained on, expected all 1): the probe calls 93% of them legal=1.
- bare, harm, illegal → legal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- bare, harm, illegal → legal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- bare, harm, legal → illegal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 8% of them harmful=1.
- bare, harm, legal → illegal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 12% of them harmful=1.
- bare: headline design with every legality-vocabulary row dropped (241 rows kept): AUROC 0.74, beat 100/100.
- prompted, legality, harmful → harmless (headline): check set **negated** (61 rows, never trained on, expected all 1): the probe calls 95% of them legal=1.
- prompted, legality, harmful → harmless (headline): check set **simple** (60 rows, never trained on, expected all 1): the probe calls 97% of them legal=1.
- prompted, legality, harmless → harmful: check set **negated** (61 rows, never trained on, expected all 1): the probe calls 59% of them legal=1.
- prompted, legality, harmless → harmful: check set **simple** (60 rows, never trained on, expected all 1): the probe calls 98% of them legal=1.
- prompted, harm, illegal → legal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted, harm, illegal → legal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted, harm, legal → illegal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 8% of them harmful=1.
- prompted, harm, legal → illegal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.

## Just asking the model

- 'is this illegal?': not run
- 'is this harmful?': not run
