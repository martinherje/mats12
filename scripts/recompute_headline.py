"""recompute_headline.py — the by-hand check of one design's headline number, as a script.

Reads probeeval_<run>_<tag>.json, refits the probe exactly as probe_eval.py did (same target, same training
stratum, same layer, same C, same training topics), scores the other stratum's rows from the held-out test
topics, and prints accuracy and AUROC next to what probe_eval.py reported. Then lists the sentences it got wrong,
with the probe's score (+ = called illegal, or called harmful for a harm probe) and both borderline flags. The
check sets and excluded rows are left out, as in the evaluation. Independent code path: sklearn is called
directly here rather than through probe_eval.py. What the probe predicts and which stratum it was trained inside are taken
from the JSON's saved arguments, so --tag L_nh2h refits that design and not the headline strata.

The 4B activations are not in data/processed on the Mac; they are on the Drive mount. Pass --acts-dir:
  python scripts/recompute_headline.py --run lp_4b --acts-dir "$HOME/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed"
  python scripts/recompute_headline.py --run lp_4b_prompted --acts-dir "..."
On Colab, where the file is in data/processed:
  python scripts/recompute_headline.py --run lp_4b
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh")
p.add_argument("--acts-dir", default=None, help="folder holding acts_<run>.npz if not data/processed (e.g. the Drive mount)")
a = p.parse_args()
r = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text()); L, C = r["layer"], r["C"]
target = r["args"]["target"]; other = "harmful" if target == "legal" else "legal"
stratum = r["args"]["train_stratum"]; sval = {"harmful": 1, "harmless": 0, "illegal": 0, "legal": 1}[stratum]
pos_name = "illegal" if target == "legal" else "harmful"
acts_path = (Path(a.acts_dir) if a.acts_dir else ROOT / "data/processed") / f"acts_{a.run}.npz"
z = np.load(acts_path); acts = z["acts"].astype(np.float32); print("reading", acts_path)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")}); df["legal"] = df.legal.astype(int); df["harmful"] = df.harmful.astype(int)
keep = (df.exclude.astype(int) == 0) & (df.get("set", pd.Series(["main"] * len(df))).astype(str) == "main")
df, acts = df[keep].reset_index(drop=True), acts[keep.to_numpy()]
train = df.topic.isin(r["split"]["train_topics"]) & (df[other] == sval)
test = df.topic.isin(r["split"]["test_topics"]) & (df[other] != sval)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(acts[train.to_numpy(), L], df[target][train])
pred = clf.predict(acts[test.to_numpy(), L]); truth = df[target][test].to_numpy()
score = clf.decision_function(acts[test.to_numpy(), L]) * (-1 if target == "legal" else 1)   # + = called illegal / called harmful
pos = 1 - truth if target == "legal" else truth
print(f"run {a.run} · {target} probe trained inside {stratum} rows · layer {L} · C {C} · train n={int(train.sum())} · test n={int(test.sum())}")
print(f"accuracy  recomputed {(pred == truth).mean():.3f}   probe_eval.py {r['test_cross_acc']:.3f}")
print(f"AUROC     recomputed {roc_auc_score(pos, score):.3f}   probe_eval.py {r['test_cross_auroc']:.3f}")
wrong = df[test][pred != truth].assign(score=score[pred != truth])
print(f"\n{len(wrong)} of {int(test.sum())} held-out other-stratum sentences wrong (score + = called {pos_name}):")
for _, w in wrong.iterrows(): print(f"  {w.score:+.2f} [{w.quadrant:17s} bl_legal={w.borderline_legal} bl_harm={w.borderline_harm}] {w.text}")
