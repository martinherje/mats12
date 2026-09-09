"""Conditional-generalisation probe evaluation with topic-disjoint splits. Generic infrastructure.

IN PLAIN LANGUAGE
The question is whether legality can be read off the model's state *independently of harm*. Training a probe
where legal = not-harmful (the two "easy" corners) cannot answer that: the probe learns one separator that is
both. So this script trains the legality probe INSIDE one harm stratum (legal-harmful vs illegal-harmful) and
tests it on the OTHER stratum (legal-harmless vs illegal-harmless), on topics the probe never saw. If it still
works, legality is decodable invariantly across harm. The reverse direction and the symmetric harm experiment
are run the same way.

Statistics done properly: topics (four sentences each) are the unit. Topics are split into train / validation /
test. Layer and regularisation are chosen on validation topics ONLY; the test topics are scored once. The
confidence interval is a topic-block bootstrap. A permutation null repeats the whole procedure (including the
selection) on shuffled labels, so the selection cannot inflate the headline. Optionally drops rows containing
explicit legality vocabulary, and rows flagged exclude=1 by the hand-check.

Also estimates factorial directions on training topics — d_legal = ½[(LH − IH) + (LH̄ − IH̄)], d_harm likewise —
reports the cosine between them, and scores each on held-out topics. That angle is meaningful because each
contrast is taken within a stratum of the other factor.

Outputs data/processed/probeeval_<run>_<tag>.json (all numbers), figures/probeeval_<run>_<tag>.png (validation
curve per layer with the chosen layer marked, plus the test point).

  uv run python scripts/probe_eval.py --run lp_4b --target legal --train-stratum harmful --tag L_h2nh
  uv run python scripts/probe_eval.py --run lp_4b --target legal --train-stratum harmless --tag L_nh2h
  uv run python scripts/probe_eval.py --run lp_4b --target harmful --train-stratum illegal --tag H_i2l
  uv run python scripts/probe_eval.py --run lp_4b --target harmful --train-stratum legal --tag H_l2i
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, manifest, write_json

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--target", required=True, choices=["legal", "harmful"], help="what the probe predicts")
p.add_argument("--train-stratum", required=True, help="value of the OTHER factor to train inside: harmful|harmless (target legal) or illegal|legal (target harmful)")
p.add_argument("--tag", required=True)
p.add_argument("--val-topics", type=int, default=15); p.add_argument("--test-topics", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--C-grid", default="0.01,0.1,1,10")
p.add_argument("--layers", default="all", help="'all' or comma list of hidden-state indices to consider")
p.add_argument("--drop-cue-rows", action="store_true", help="exclude rows whose text contains explicit legality vocabulary")
p.add_argument("--n-perm", type=int, default=20, help="permutation-null repeats (label shuffle + full selection procedure)")
p.add_argument("--n-boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(a.seed)

z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
df = pd.DataFrame({k: v for k, v in cols.items()})
df["legal"] = df["legal"].astype(int); df["harmful"] = df["harmful"].astype(int)
keep = np.ones(len(df), bool)
if "exclude" in df:
    keep &= df["exclude"].astype(int).to_numpy() == 0
CUE = re.compile(r"\b(illegal|illegally|unlawful|lawful|legal|legally|prohibit\w*|banned|ban|in violation of|violat\w*|required|require\w*|permit\w*|licen[cs]e\w*|law|rule|rules|ordinance|restriction\w*|forbid\w*)\b", re.I)
if a.drop_cue_rows:
    keep &= ~df["text"].astype(str).str.contains(CUE).to_numpy()
df = df[keep].reset_index(drop=True); acts = acts[keep]
other = "harmful" if a.target == "legal" else "legal"
stratum_val = {"harmful": ("harmful", 1), "harmless": ("harmful", 0), "illegal": ("legal", 0), "legal": ("legal", 1)}[a.train_stratum]
assert stratum_val[0] == other, f"--train-stratum must be a value of {other}"
in_train_stratum = df[other].to_numpy() == stratum_val[1]
y = df[a.target].to_numpy()

# topic-disjoint split
topics = np.array(sorted(df["topic"].unique())); rng.shuffle(topics)
test_t, val_t, train_t = set(topics[:a.test_topics]), set(topics[a.test_topics:a.test_topics + a.val_topics]), set(topics[a.test_topics + a.val_topics:])
t = df["topic"].to_numpy()
is_train, is_val, is_test = np.isin(t, list(train_t)), np.isin(t, list(val_t)), np.isin(t, list(test_t))
tr = is_train & in_train_stratum                     # train inside the stratum, on train topics
va_in, va_out = is_val & in_train_stratum, is_val & ~in_train_stratum
te_in, te_out = is_test & in_train_stratum, is_test & ~in_train_stratum
L1 = acts.shape[1]
layers = list(range(1, L1)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
Cs = [float(c) for c in a.C_grid.split(",")]
print(f"run={a.run} · target={a.target} · train inside {other}={stratum_val[1]} · rows kept {len(df)} · topics train/val/test {len(train_t)}/{len(val_t)}/{len(test_t)}")
print(f"train n={tr.sum()} · val in-stratum n={va_in.sum()} cross-stratum n={va_out.sum()} · test in-stratum n={te_in.sum()} cross-stratum n={te_out.sum()}")


def fit(X, yy, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, yy)


def acc(clf, X, yy):
    return float((clf.predict(X) == yy).mean()) if len(yy) and yy.min() != yy.max() else float("nan")


def auroc(clf, X, yy):
    return float(roc_auc_score(yy, clf.decision_function(X))) if len(yy) and yy.min() != yy.max() else float("nan")


def select_and_test(yv):
    """Full procedure on labels yv: choose (layer, C) by CROSS-stratum validation accuracy, then score test once."""
    best = (-1, None, None); curve = {}
    for l in layers:
        for C in Cs:
            clf = fit(acts[tr, l], yv[tr], C)
            v = acc(clf, acts[va_out, l], yv[va_out])
            curve.setdefault(l, {})[C] = v
            if v > best[0]:
                best = (v, l, C)
    _, l, C = best
    clf = fit(acts[tr, l], yv[tr], C)
    return {"layer": l, "C": C, "val_cross_acc": best[0],
            "test_cross_acc": acc(clf, acts[te_out, l], yv[te_out]), "test_cross_auroc": auroc(clf, acts[te_out, l], yv[te_out]),
            "test_in_acc": acc(clf, acts[te_in, l], yv[te_in])}, curve, clf


res, curve, clf = select_and_test(y)
# topic-block bootstrap on the cross-stratum test set
test_topics_arr = t[te_out]; yte = y[te_out]; pred = clf.predict(acts[te_out, res["layer"]])
tt = np.array(sorted(set(test_topics_arr))); boots = []
for _ in range(a.n_boot):
    pick = rng.choice(tt, len(tt), replace=True); idx = np.concatenate([np.where(test_topics_arr == k)[0] for k in pick])
    boots.append((pred[idx] == yte[idx]).mean())
res["test_cross_acc_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
# permutation null with identical selection
null = []
for i in range(a.n_perm):
    yp = y.copy(); yp[is_train | is_val | is_test] = rng.permutation(yp[is_train | is_val | is_test])
    null.append(select_and_test(yp)[0]["test_cross_acc"])
res["perm_null_test_cross_acc"] = {"mean": float(np.nanmean(null)), "p95": float(np.nanquantile(null, 0.95)), "n": a.n_perm}
res["p_value_vs_null"] = float((np.sum(np.array(null) >= res["test_cross_acc"]) + 1) / (len(null) + 1))

# factorial directions on training topics, evaluated on test topics
def mean_state(mask, l): return acts[mask, l].mean(0)
def factorial_dir(target_col, l, mask):
    oth = "harmful" if target_col == "legal" else "legal"; d = np.zeros(acts.shape[2], np.float32)
    for s in (0, 1):
        m1 = mask & (df[target_col].to_numpy() == 1) & (df[oth].to_numpy() == s); m0 = mask & (df[target_col].to_numpy() == 0) & (df[oth].to_numpy() == s)
        if m1.any() and m0.any(): d += 0.5 * (mean_state(m1, l) - mean_state(m0, l))
    return d / (np.linalg.norm(d) + 1e-8)
l = res["layer"]
dL, dH = factorial_dir("legal", l, is_train), factorial_dir("harmful", l, is_train)
proj = lambda d, mask, col: float(roc_auc_score(df[col].to_numpy()[mask], acts[mask, l] @ d)) if mask.any() else float("nan")
res["factorial"] = {"layer": l, "cos_dlegal_dharm": float(dL @ dH),
                    "dlegal_auroc_on_test_topics": proj(dL, is_test, "legal"), "dharm_auroc_on_test_topics": proj(dH, is_test, "harmful"),
                    "dlegal_auroc_predicting_harm_on_test": proj(dL, is_test, "harmful"), "dharm_auroc_predicting_legal_on_test": proj(dH, is_test, "legal")}
dLr = dL - (dL @ dH) * dH; dLr /= (np.linalg.norm(dLr) + 1e-8)
res["factorial"]["dlegal_residualised_auroc_on_test_topics"] = proj(dLr, is_test, "legal")
np.savez_compressed(ROOT / "data/processed" / f"factorial_dirs_{a.run}_{a.tag}.npz", d_legal=dL, d_harm=dH, d_legal_resid=dLr, layer=l)

out = ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json"
write_json(out, manifest(args=vars(a), split={"train_topics": sorted(train_t), "val_topics": sorted(val_t), "test_topics": sorted(test_t)}, n_rows=len(df), **res))
print("\n=== headline (selected on validation topics, scored once on test topics, cross-stratum) ===")
print(f"layer {res['layer']} · C {res['C']} · val cross-stratum acc {res['val_cross_acc']:.3f}")
print(f"TEST cross-stratum acc {res['test_cross_acc']:.3f}  CI95 {res['test_cross_acc_ci95'][0]:.2f}–{res['test_cross_acc_ci95'][1]:.2f}  auroc {res['test_cross_auroc']:.3f}  (in-stratum test acc {res['test_in_acc']:.3f})")
print(f"permutation null (same selection): mean {res['perm_null_test_cross_acc']['mean']:.3f}, p95 {res['perm_null_test_cross_acc']['p95']:.3f} · p = {res['p_value_vs_null']:.3f}")
print(f"factorial directions @ layer {l}: cos(d_legal, d_harm) = {res['factorial']['cos_dlegal_dharm']:+.3f} · d_legal→legal AUROC {res['factorial']['dlegal_auroc_on_test_topics']:.3f} · d_harm→harm {res['factorial']['dharm_auroc_on_test_topics']:.3f} · d_legal→harm {res['factorial']['dlegal_auroc_predicting_harm_on_test']:.3f} · residualised d_legal→legal {res['factorial']['dlegal_residualised_auroc_on_test_topics']:.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for C in Cs:
    ax.plot(layers, [curve[l_][C] for l_ in layers], marker=".", label=f"validation, C={C}")
ax.axvline(res["layer"], ls="--", color="gray"); ax.scatter([res["layer"]], [res["test_cross_acc"]], color="k", zorder=5, label="test (once)")
ax.axhline(res["perm_null_test_cross_acc"]["p95"], ls=":", color="r", label="permutation null p95")
ax.set_xlabel("layer"); ax.set_ylabel("cross-stratum accuracy"); ax.set_ylim(0.3, 1.02); ax.legend(fontsize=7)
ax.set_title(f"{a.target} probe trained inside {other}={stratum_val[1]}, tested on the other stratum, held-out topics")
fig.tight_layout(); fig.savefig(ROOT / "figures" / f"probeeval_{a.run}_{a.tag}.png", dpi=150)
print("wrote", out.relative_to(ROOT))
print("HAND-CHECK: recompute the test accuracy at the chosen layer from acts_<run>.npz using the split saved in the JSON.")
