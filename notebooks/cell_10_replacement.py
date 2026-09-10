import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
r = json.load(open(f"data/processed/probeeval_{RUN}_L_h2nh.json")); L, C = r["layer"], r["C"]
z = np.load(f"data/processed/acts_{RUN}.npz"); acts = z["acts"].astype(np.float32); cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
df = pd.DataFrame({k: v for k, v in cols.items()}); df["legal"] = df.legal.astype(int); df["harmful"] = df.harmful.astype(int)
# the design rows only: excluded rows out, and the simple / negated check sets out (they are never trained or tested on)
keep = (df.exclude.astype(int) == 0) & (df["set"].astype(str) == "main"); df, acts = df[keep].reset_index(drop=True), acts[keep.to_numpy()]
train = df.topic.isin(r["split"]["train_topics"]) & (df.harmful == 1)          # illegality probe trained inside the harmful stratum
test  = df.topic.isin(r["split"]["test_topics"])  & (df.harmful == 0)          # tested on the harmless stratum, unseen topics
clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(acts[train.to_numpy(), L], df.legal[train])
pred = clf.predict(acts[test.to_numpy(), L]); truth = df.legal[test].to_numpy()
score = -clf.decision_function(acts[test.to_numpy(), L])                      # + = called illegal
print(f"recomputed cross-stratum test accuracy at layer {L}, C={C}: {(pred == truth).mean():.3f}   (script reported {r['test_cross_acc']:.3f}; n = {test.sum()})")
from sklearn.metrics import roc_auc_score
print(f"recomputed AUROC: {roc_auc_score(1 - truth, score):.3f}   (script reported {r['test_cross_auroc']:.3f})")
wrong = df[test][pred != truth].assign(score=score[pred != truth])
print(f"\n{len(wrong)} mistakes on the {int(test.sum())} held-out harmless-stratum sentences (score + = called illegal; read them, and note which were borderline):")
for _, w in wrong.iterrows(): print(f"  {w.score:+.2f} [{w.quadrant:17s} bl_legal={w.borderline_legal} bl_harm={w.borderline_harm}] {w.text}")
