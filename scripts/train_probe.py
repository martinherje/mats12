"""Layer-sweep linear probes on saved activations. Generic infrastructure; the *design* is the CLI.

Nothing here decides the experiment: which label, which quadrants train vs test, which method,
and which controls run are all arguments. Defaults are deliberately the plainest choice.

  --label       column to predict (e.g. legal or harmful; must be 0/1 in the CSV)
  --train       quadrants used for training (comma list; default: all)
  --test        quadrants held out for the transfer test (comma list; default: none → CV only)
  --method      logreg (L2 logistic regression, standardized) | diffmeans (difference-of-means direction)
  --control     also fit on shuffled labels (Hewitt & Liang-style control) to show chance level
  --bow         also fit a bag-of-words logistic baseline on the raw text (lexical-confound check)
  --contrast    a second label column; reports per-layer cosine between the diff-means directions
                of --label and --contrast (Zhao et al. style dissociation number)

Outputs data/processed/probe_<run>_<label>.csv (per-layer table) and figures/probe_<run>_<label>.png.

Example:
  uv run python scripts/train_probe.py --run v1_last_raw --label legal \
      --train illegal_harmful,legal_harmless --test illegal_harmless,legal_harmful \
      --contrast harmful --control --bow
"""
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import ROOT, QUADRANTS, manifest, write_json

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--label", required=True)
p.add_argument("--train", default="all")
p.add_argument("--test", default="")
p.add_argument("--method", default="logreg", choices=["logreg", "diffmeans"])
p.add_argument("--C", type=float, default=1.0)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--control", action="store_true")
p.add_argument("--bow", action="store_true")
p.add_argument("--contrast", default="")
p.add_argument("--eval-label", default="", help="score the transfer test against THIS label instead of --label (e.g. train a harm probe, test whether it predicts legality)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = np.random.default_rng(a.seed)

z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz")
acts = z["acts"].astype(np.float32)                      # [N, L+1, d]
cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
y = cols[a.label].astype(int)
quad = cols.get("quadrant")
N, L1, d = acts.shape
print(f"run={a.run} · N={N} · layers+1={L1} · d={d} · label={a.label} · positives={int(y.sum())}")


def subset(names: str):
    if names in ("", "all") or quad is None:
        return np.ones(N, bool) if names in ("", "all") else np.zeros(N, bool)
    want = set(names.split(","))
    bad = want - set(QUADRANTS); assert not bad, f"unknown quadrants {bad}"
    return np.isin(quad, list(want))


tr = subset(a.train); te = subset(a.test) if a.test else np.zeros(N, bool)
if a.test:
    tr &= ~te
print(f"train n={int(tr.sum())} (pos {int(y[tr].sum())}) · test n={int(te.sum())} (pos {int(y[te].sum())})")
assert y[tr].min() != y[tr].max(), "training set has a single class — check --train quadrants against --label"
if a.contrast:
    yc_all = cols[a.contrast].astype(int); corr = float(np.corrcoef(y, yc_all)[0, 1])
    print(f"label/contrast correlation over all rows: {corr:+.2f} (should be ~0 for a balanced 2x2; else the cosine is confounded)")


y_eval = cols[a.eval_label].astype(int) if a.eval_label else y


def fit_eval(X, ytr, Xte=None, yte=None):
    if a.method == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
        cv = cross_val_score(clf, X, ytr, cv=StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)).mean()
        clf.fit(X, ytr)
        score = lambda Z: clf.decision_function(Z)
    else:  # difference of means, thresholded at the projected midpoint
        w = X[ytr == 1].mean(0) - X[ytr == 0].mean(0); w /= np.linalg.norm(w) + 1e-8
        proj = X @ w; thr = (proj[ytr == 1].mean() + proj[ytr == 0].mean()) / 2
        cv = ((proj > thr).astype(int) == ytr).mean()  # in-sample; diff-means barely overfits but report as such
        score = lambda Z: Z @ w - thr
    res = {"train_acc_cv": cv}
    if Xte is not None and len(yte) and yte.min() != yte.max():
        s = score(Xte)
        res["test_acc"] = float(((s > 0).astype(int) == yte).mean())
        res["test_auroc"] = float(roc_auc_score(yte, s))
    return res


rows = []
for l in range(L1):
    X = acts[:, l]
    r = {"layer": l, **fit_eval(X[tr], y[tr], X[te] if a.test else None, y_eval[te] if a.test else None)}
    if a.control:
        r["control_acc_cv"] = fit_eval(X[tr], rng.permutation(y[tr]))["train_acc_cv"]
    if a.contrast:
        # Diff-means directions over ALL rows (train+test): on the diagonal alone the two labels are perfect
        # complements and the cosine is trivially -1. Only meaningful when the 2x2 is balanced.
        yc = cols[a.contrast].astype(int)
        w1 = X[y == 1].mean(0) - X[y == 0].mean(0)
        w2 = X[yc == 1].mean(0) - X[yc == 0].mean(0)
        r[f"cos_{a.label}_vs_{a.contrast}"] = float(w1 @ w2 / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8))
    rows.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

tab = pd.DataFrame(rows)
if a.bow:
    from sklearn.feature_extraction.text import CountVectorizer
    text = cols.get("text")
    if text is None:  # older npz without text: fall back to the scenarios path recorded in the manifest
        import json
        man = json.loads((ROOT / "data/processed" / f"acts_{a.run}.json").read_text())
        df = pd.read_csv(ROOT / man["scenarios"]); text = df.set_index("id").loc[cols["id"], "text"].to_numpy()
    vec = CountVectorizer(min_df=2, ngram_range=(1, 2)); Xb = vec.fit_transform(text.astype(str))
    r = {}
    clf = LogisticRegression(C=a.C, max_iter=2000)
    r["bow_train_acc_cv"] = cross_val_score(clf, Xb[tr], y[tr], cv=StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)).mean()
    if a.test and y_eval[te].min() != y_eval[te].max():
        clf.fit(Xb[tr], y[tr]); s = clf.decision_function(Xb[te])
        r["bow_test_acc"] = float(((s > 0).astype(int) == y_eval[te]).mean()); r["bow_test_auroc"] = float(roc_auc_score(y_eval[te], s))
    print("bag-of-words baseline:", {k: round(v, 3) for k, v in r.items()})
    for k, v in r.items():
        tab[k] = v

stem = f"probe_{a.run}_{a.label}" + (f"_eval-{a.eval_label}" if a.eval_label else "")
outc = ROOT / "data/processed" / f"{stem}.csv"; tab.to_csv(outc, index=False)
best = tab.iloc[tab["test_acc"].idxmax()] if "test_acc" in tab else tab.iloc[tab["train_acc_cv"].idxmax()]
write_json(outc.with_suffix(".json"), manifest(args=vars(a), n_train=int(tr.sum()), n_test=int(te.sum()),
           best_layer=int(best["layer"]), best={k: (None if pd.isna(v) else float(v)) for k, v in best.items()}))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tab["layer"], tab["train_acc_cv"], marker="o", label=f"train CV acc ({a.train})")
if "test_acc" in tab: ax.plot(tab["layer"], tab["test_acc"], marker="s", label=f"transfer acc ({a.test})")
if "control_acc_cv" in tab: ax.plot(tab["layer"], tab["control_acc_cv"], ls="--", color="gray", label="shuffled-label control")
if a.bow and "bow_test_acc" in tab: ax.axhline(tab["bow_test_acc"].iloc[0], ls=":", color="k", label="bag-of-words transfer acc")
ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("accuracy"); ax.set_ylim(0.3, 1.02); ax.legend(fontsize=8)
ax.set_title(f"{a.method} probe trained on '{a.label}'" + (f", scored against '{a.eval_label}'" if a.eval_label else "") + f" — run {a.run}")
outf = ROOT / "figures" / f"{stem}.png"; fig.tight_layout(); fig.savefig(outf, dpi=150)
(ROOT / "figures" / f"{stem}.txt").write_text(f"Layer sweep of a {a.method} probe predicting '{a.label}' from pooled residual-stream activations, run {a.run}. "
    f"Train quadrants: {a.train}; transfer test quadrants: {a.test or 'none'}; N_train={int(tr.sum())}, N_test={int(te.sum())}. "
    f"{'Shuffled-label control shown. ' if a.control else ''}{'Bag-of-words baseline shown. ' if a.bow else ''}\n")
print(f"wrote {outc.relative_to(ROOT)} and {outf.relative_to(ROOT)}")
print("HAND-CHECK: recompute one layer's test accuracy from the saved activations by hand (10 lines of numpy) before quoting any number.")
