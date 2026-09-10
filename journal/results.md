# Results — run `mac05b` (copied from data/processed/*.json; recompute by hand before quoting)

| condition | test | layer | AUROC | acc (CI95) | beat shuffles | cos(dL,dH) | dL within harmless, top-1 harm removed | word count alone |
|---|---|---|---|---|---|---|---|---|
| bare | legality, harmful → harmless (headline) | 23 | 0.66 | 63% (50%–77%) | 98/100 (p95 60%) | -0.15 | 0.62 | 0.60 |
| bare | legality, harmless → harmful | 9 | 0.75 | 70% (53%–83%) | 99/100 (p95 63%) | -0.09 | 0.61 | 0.79 |
| bare | harm, illegal → legal | 11 | 0.95 | 83% (70%–93%) | 100/100 (p95 67%) | -0.15 | 1.00 | 0.57 |
| bare | harm, legal → illegal | 13 | 0.99 | 93% (83%–100%) | 100/100 (p95 67%) | -0.13 | 1.00 | 0.33 |
| prompted | legality, harmful → harmless (headline) | 10 | 0.57 | 57% (43%–70%) | 79/100 (p95 60%) | -0.21 | 0.61 | 0.60 |
| prompted | legality, harmless → harmful | 13 | 0.69 | 60% (40%–80%) | 81/100 (p95 67%) | -0.24 | 0.58 | 0.79 |
| prompted | harm, illegal → legal | 21 | 0.89 | 73% (60%–87%) | 97/100 (p95 67%) | -0.40 | 0.90 | 0.57 |
| prompted | harm, legal → illegal | 20 | 0.95 | 87% (70%–100%) | 100/100 (p95 63%) | -0.45 | 0.91 | 0.33 |

## Checks

- bare, legality, harmful → harmless (headline): check set **negated** (61 rows, never trained on, expected all 1): the probe calls 97% of them legal=1.
- bare, legality, harmful → harmless (headline): check set **simple** (60 rows, never trained on, expected all 1): the probe calls 72% of them legal=1.
- bare, legality, harmless → harmful: check set **negated** (61 rows, never trained on, expected all 1): the probe calls 82% of them legal=1.
- bare, legality, harmless → harmful: check set **simple** (60 rows, never trained on, expected all 1): the probe calls 25% of them legal=1.
- bare, harm, illegal → legal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- bare, harm, illegal → legal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 5% of them harmful=1.
- bare, harm, legal → illegal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 49% of them harmful=1.
- bare, harm, legal → illegal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 7% of them harmful=1.
- bare: headline design with every legality-vocabulary row dropped (241 rows kept): AUROC 0.55, beat 44/100.
- prompted, legality, harmful → harmless (headline): check set **negated** (61 rows, never trained on, expected all 1): the probe calls 84% of them legal=1.
- prompted, legality, harmful → harmless (headline): check set **simple** (60 rows, never trained on, expected all 1): the probe calls 20% of them legal=1.
- prompted, legality, harmless → harmful: check set **negated** (61 rows, never trained on, expected all 1): the probe calls 66% of them legal=1.
- prompted, legality, harmless → harmful: check set **simple** (60 rows, never trained on, expected all 1): the probe calls 13% of them legal=1.
- prompted, harm, illegal → legal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted, harm, illegal → legal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted, harm, legal → illegal: check set **negated** (61 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted, harm, legal → illegal: check set **simple** (60 rows, never trained on, expected all 0): the probe calls 0% of them harmful=1.
- prompted: headline design with every legality-vocabulary row dropped (241 rows kept): AUROC 0.59, beat 96/100.

## Just asking the model

- 'is this illegal?': 52% right overall, 49% on the off-diagonal quadrants, 0% invalid; Yes−No logit AUROC off-diagonal 0.51. Check set negated: answered yes 92%. Check set simple: answered yes 83%.
- 'is this harmful?': 50% right overall, 50% on the off-diagonal quadrants, 0% invalid; Yes−No logit AUROC off-diagonal 0.54. Check set negated: answered yes 3%. Check set simple: answered yes 35%.
