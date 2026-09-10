"""recompute_headline.py — the by-hand check of the headline number, as a script (same code as notebook cell 10).

Refits the illegality probe exactly as probe_eval.py did (same layer, same C, same training rows: the harmful
stratum of the training topics), scores the harmless-stratum sentences of the held-out test topics, and prints
accuracy and AUROC next to what probe_eval.py reported. Then lists the sentences it got wrong, with the probe's
score (+ = called illegal) and both borderline flags. The check sets and excluded rows are left out, as in the
evaluation. Independent code path: sklearn is called directly here, no probe_eval.py import.

  python scripts/recompute_headline.py --run lp_4b
  python scripts/recompute_headline.py --run lp_4b_prompted
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh"); a = p.parse_args()
r = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text()); L, C = r["layer"], r["C"]
z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")}); df["legal"] = df.legal.astype(int); df["harmful"] = df.harmful.astype(int)
keep = (df.exclude.astype(int) == 0) & (df.get("set", pd.Series(["main"] * len(df))).astype(str) == "main")
df, acts = df[keep].reset_index(drop=True), acts[keep.to_numpy()]
train = df.topic.isin(r["split"]["train_topics"]) & (df.harmful == 1)
test = df.topic.isin(r["split"]["test_topics"]) & (df.harmful == 0)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(acts[train.to_numpy(), L], df.legal[train])
pred = clf.predict(acts[test.to_numpy(), L]); truth = df.legal[test].to_numpy(); score = -clf.decision_function(acts[test.to_numpy(), L])
print(f"run {a.run} · layer {L} · C {C} · train n={int(train.sum())} · test n={int(test.sum())}")
print(f"accuracy  recomputed {(pred == truth).mean():.3f}   probe_eval.py {r['test_cross_acc']:.3f}")
print(f"AUROC     recomputed {roc_auc_score(1 - truth, score):.3f}   probe_eval.py {r['test_cross_auroc']:.3f}")
wrong = df[test][pred != truth].assign(score=score[pred != truth])
print(f"\n{len(wrong)} of {int(test.sum())} held-out harmless-stratum sentences wrong (score + = called illegal):")
for _, w in wrong.iterrows(): print(f"  {w.score:+.2f} [{w.quadrant:17s} bl_legal={w.borderline_legal} bl_harm={w.borderline_harm}] {w.text}")
