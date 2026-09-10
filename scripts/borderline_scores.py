"""borderline_scores.py — does the probe's continuous score respect the nuance the binary labels flatten?

IN PLAIN LANGUAGE
The labels are 0/1, but the probe's output is a continuous score (positive = called illegal, negative = called
legal; the further from zero, the more confident). If the model represents nuance, the sentences Martin
flagged as borderline on legality should get scores closer to zero than the clear ones. This script refits the
headline probe exactly as probe_eval.py did (same layer, same C, same training rows), scores every design
sentence, and compares |score| for clear vs borderline rows, separately for the training and the held-out
topics. It also lists the ten sentences closest to the line, so you can read whether they are the ones a
lawyer would hesitate over. Output: a table on stdout, data/processed/borderline_scores_<run>.json, and
figures/borderline_scores_<run>.png (score histograms, clear vs borderline).

  uv run python scripts/borderline_scores.py --run lp_4b            # after cell 7
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh"); p.add_argument("--target", default="legal", choices=["legal", "harmful"]); a = p.parse_args()
z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")})
for c in ("legal", "harmful", "exclude", "borderline_legal", "borderline_harm"): df[c] = df[c].astype(int)
ev = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text()); L, C = ev["layer"], ev["C"]
other = "harmful" if a.target == "legal" else "legal"; stratum = ev["args"]["train_stratum"]
sval = {"harmful": ("harmful", 1), "harmless": ("harmful", 0), "illegal": ("legal", 0), "legal": ("legal", 1)}[stratum]
main = (df.get("set", "main").astype(str) == "main") & (df["exclude"] == 0)
is_train = main & df["topic"].isin(ev["split"]["train_topics"]) & (df[sval[0]] == sval[1])
X = acts[:, L]; y = df[a.target].to_numpy()
clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[is_train], y[is_train])
score = clf.decision_function(X)                      # continuous: sign = side, magnitude = confidence
if a.target == "legal": score = -score                 # illegality convention: positive = called illegal
flag = df["borderline_legal"] if a.target == "legal" else df["borderline_harm"]
held = main & df["topic"].isin(ev["split"]["val_topics"] + ev["split"]["test_topics"])
rows = []
for name, m in (("training topics", is_train), ("held-out topics (val+test), same stratum", held & (df[sval[0]] == sval[1])), ("held-out topics, other stratum", held & (df[sval[0]] != sval[1]))):
    for lab, f in (("clear", flag == 0), ("borderline", flag == 1)):
        mm = m & f
        if mm.sum(): rows.append({"rows": name, "flag": lab, "n": int(mm.sum()), "median |score|": float(np.median(np.abs(score[mm]))), "mean |score|": float(np.abs(score[mm]).mean()), "frac within 0.5 of the line": float((np.abs(score[mm]) < 0.5).mean())})
tab = pd.DataFrame(rows); print(f"run={a.run} · {a.target} probe from {a.tag} (layer {L}, C={C}); |score| = distance from the decision line\n"); print(tab.round(3).to_string(index=False))
near = df[main].assign(score=score[main]).assign(absscore=lambda d: d["score"].abs()).sort_values("absscore").head(10)
print("\nThe ten design sentences closest to the line (score ≈ 0 = the probe cannot decide; + = called illegal):")
for _, r in near.iterrows(): print(f"  {r['score']:+.2f}  {r['quadrant']:17s} bl_legal={r['borderline_legal']} bl_harm={r['borderline_harm']}  {r['text'][:90]}")
out = {"layer": L, "C": C, "table": rows, "closest_to_line": near[["id", "quadrant", "borderline_legal", "borderline_harm", "score", "text"]].to_dict("records")}
(ROOT / "data/processed" / f"borderline_scores_{a.run}.json").write_text(json.dumps(out, indent=1))
fig, ax = plt.subplots(figsize=(7, 3.6)); m = held
ax.hist(score[m & (flag == 0)], bins=25, alpha=0.6, label=f"clear (n={int((m & (flag == 0)).sum())})"); ax.hist(score[m & (flag == 1)], bins=25, alpha=0.6, label=f"borderline (n={int((m & (flag == 1)).sum())})")
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(f"probe score (+ = called {'illegal' if a.target == 'legal' else 'harmful'}); held-out topics"); ax.set_ylabel("sentences"); ax.legend(); ax.set_title(f"Do borderline sentences sit closer to the line? {a.run}, layer {L}", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "figures" / f"borderline_scores_{a.run}.png", dpi=150); print(f"\nwrote data/processed/borderline_scores_{a.run}.json, figures/borderline_scores_{a.run}.png")
