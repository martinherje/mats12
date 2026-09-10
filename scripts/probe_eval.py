"""Conditional-generalisation probe evaluation with topic-disjoint splits. Generic infrastructure.

IN PLAIN LANGUAGE
The question is whether legality can be read off the model's state *independently of harm*. Training a probe
where legal = not-harmful (the two "easy" corners) cannot answer that: the probe learns one separator that is
both. So this script trains the legality probe INSIDE one harm stratum (legal-harmful vs illegal-harmful) and
tests it on the OTHER stratum (legal-harmless vs illegal-harmless), on topics the probe never saw. If it still
works, legality is decodable invariantly across harm. The reverse direction and the symmetric harm experiment
are run the same way.

Statistics (after the 9 Sep review): topics (four sentences each) are the unit. Topics are split into train /
validation / test. Layer and regularisation are chosen on validation AUROC ONLY; the test topics are scored once.
The confidence interval is a topic-block bootstrap. A permutation null repeats the whole procedure (selection
included) on labels shuffled WITHIN each topic x stratum cell, so the null keeps the design's balance. Report
"beat N of M shuffles", not a p-value: with M shuffles the smallest p is 1/(M+1).

Directions: factorial contrasts on training topics — d_legal = ½[(LH − IH) + (LH̄ − IH̄)], d_harm likewise —
with (a) the cosine between them AT EVERY LAYER plus a within-cell label-swap null band, (b) each direction's
AUROC on held-out topics, evaluated WITHIN each stratum of the other factor (a balanced all-quadrant AUROC
cannot be hurt by removing harm, so it proves nothing), (c) the legality direction with the top-k harm
components projected out (k = 1..3; Shah et al. say the harm subspace is low-rank, not rank-1), and (d) the
plain half-contrast test: the direction LH − IH from training topics scored on LH̄ vs IH̄ test rows.

Two checks after the headline (10 Sep): rows with set=simple (the original short legal-harmless anchors, "You
cook pasta") and set=negated (illegal-harmless rows with "do not", legal by construction, Marks & Tegmark-style)
are NEVER trained on; the chosen probe scores them once and the JSON reports the fraction it calls legal
(expected: all). A length-only AUROC on the same test rows is reported too (0.5 = word count carries nothing).

Outputs data/processed/probeeval_<run>_<tag>.json (all numbers), figures/probeeval_<run>_<tag>.png.

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
p.add_argument("--drop-borderline", action="store_true", help="exclude rows flagged borderline=1")
p.add_argument("--n-perm", type=int, default=100, help="permutation-null repeats (within-cell label shuffle + full selection procedure)")
p.add_argument("--n-cos-null", type=int, default=50, help="label-swap repeats for the cosine null band")
p.add_argument("--n-boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(a.seed)

z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
assert np.isfinite(acts).all(), "activations contain inf/nan (fp16 overflow?) — re-extract with the fp32 build of extract_activations.py"
cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
df = pd.DataFrame({k: v for k, v in cols.items()})
df["legal"] = df["legal"].astype(int); df["harmful"] = df["harmful"].astype(int)
assert "exclude" in df.columns, "activations were extracted from a CSV without the exclude column — re-run extract_activations.py on the current scenarios.csv"
if "set" in df.columns:   # simple / negated rows are never trained on; they are scored once after selection (see extra_sets)
    extra_mask = df["set"].astype(str).to_numpy() != "main"; extra_df = df[extra_mask].reset_index(drop=True); extra_acts = acts[extra_mask]
    df = df[~extra_mask].reset_index(drop=True); acts = acts[~extra_mask]
else:
    extra_df, extra_acts = df.iloc[0:0], acts[:0]
all_topics = np.array(sorted(df["topic"].unique()))   # split drawn from the full topic list, so exclusions never shift it
keep = df["exclude"].astype(int).to_numpy() == 0
if a.drop_borderline:
    bcol = f"borderline_{'legal' if a.target == 'legal' else 'harm'}"   # axis-specific flag (tag.py); falls back to the old single flag
    keep &= df[bcol if bcol in df.columns else "borderline"].astype(int).to_numpy() == 0
CUE = re.compile(r"\b(?:illegal|illegally|unlawful|lawful|legal|legally|prohibit\w*|banned|ban|in violation of|violat\w*|required|require\w*|permit\w*|licen[cs]e\w*|law|rule|rules|ordinance|restriction\w*|forbid\w*|prosecut\w*|breach\w*|criminalis\w*|criminaliz\w*|disorderly conduct|fair use)\b", re.I)
if a.drop_cue_rows:
    keep &= ~df["text"].astype(str).str.contains(CUE, regex=True).to_numpy()
df = df[keep].reset_index(drop=True); acts = acts[keep]
other = "harmful" if a.target == "legal" else "legal"
stratum_val = {"harmful": ("harmful", 1), "harmless": ("harmful", 0), "illegal": ("legal", 0), "legal": ("legal", 1)}[a.train_stratum]
assert stratum_val[0] == other, f"--train-stratum must be a value of {other}"
in_train_stratum = df[other].to_numpy() == stratum_val[1]
y = df[a.target].to_numpy(); y_other = df[other].to_numpy()

# topic-disjoint split
topics = all_topics.copy(); rng.shuffle(topics)
test_t, val_t, train_t = set(topics[:a.test_topics]), set(topics[a.test_topics:a.test_topics + a.val_topics]), set(topics[a.test_topics + a.val_topics:])
t = df["topic"].to_numpy()
is_train, is_val, is_test = np.isin(t, list(train_t)), np.isin(t, list(val_t)), np.isin(t, list(test_t))
tr = is_train & in_train_stratum
va_out = is_val & ~in_train_stratum
te_in, te_out = is_test & in_train_stratum, is_test & ~in_train_stratum
L1 = acts.shape[1]
layers = list(range(1, L1)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
Cs = [float(c) for c in a.C_grid.split(",")]
print(f"run={a.run} · target={a.target} · train inside {other}={stratum_val[1]} · rows kept {len(df)} · topics train/val/test {len(train_t)}/{len(val_t)}/{len(test_t)}")
print(f"train n={tr.sum()} · validation cross-stratum n={va_out.sum()} · test in-stratum n={te_in.sum()} cross-stratum n={te_out.sum()}")


def fit(X, yy, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, yy)


def acc(clf, X, yy):
    return float((clf.predict(X) == yy).mean()) if len(yy) and yy.min() != yy.max() else float("nan")


def auroc_clf(clf, X, yy):
    return float(roc_auc_score(yy, clf.decision_function(X))) if len(yy) and yy.min() != yy.max() else float("nan")


def auroc_proj(d, mask, yy):
    return float(roc_auc_score(yy[mask], acts_l[mask] @ d)) if mask.any() and yy[mask].min() != yy[mask].max() else float("nan")


def select_and_test(yv):
    """Choose (layer, C) by cross-stratum validation AUROC, then score the test rows once."""
    best = (-1.0, None, None); curve = {}
    for l in layers:
        for C in Cs:
            clf = fit(acts[tr, l], yv[tr], C)
            v = auroc_clf(clf, acts[va_out, l], yv[va_out])
            curve.setdefault(l, {})[C] = v
            if v > best[0]:
                best = (v, l, C)
    _, l, C = best
    clf = fit(acts[tr, l], yv[tr], C)
    return {"layer": l, "C": C, "val_cross_auroc": best[0],
            "test_cross_acc": acc(clf, acts[te_out, l], yv[te_out]), "test_cross_auroc": auroc_clf(clf, acts[te_out, l], yv[te_out]),
            "test_in_acc": acc(clf, acts[te_in, l], yv[te_in])}, curve, clf


def shuffle_within_cells(yv):
    """Permute labels within each topic x (other-factor) cell, keeping the design balanced."""
    yp = yv.copy()
    for tp in np.unique(t):
        for s in (0, 1):
            idx = np.where((t == tp) & (y_other == s))[0]
            if len(idx) > 1:
                yp[idx] = rng.permutation(yp[idx])
    return yp


res, curve, clf = select_and_test(y)
# extra sets (simple anchors, negated rows): never trained on; fraction the probe calls label=1 (legal / harmful) at the chosen layer
res["extra_sets"] = {}
if len(extra_df):
    for name in sorted(set(extra_df["set"].astype(str))):
        m = extra_df["set"].astype(str).to_numpy() == name
        pr = clf.predict(extra_acts[m, res["layer"]])
        res["extra_sets"][name] = {"n": int(m.sum()), f"frac_predicted_{a.target}_1": float(pr.mean()), "expected": "all 1" if a.target == "legal" else "all 0"}
# length-only baseline on the same cross-stratum test rows (0.5 = word count carries nothing about the label)
wl = df["text"].astype(str).str.split().str.len().to_numpy()
res["length_only_test_auroc"] = float(roc_auc_score(y[te_out], wl[te_out])) if y[te_out].min() != y[te_out].max() else float("nan")
# topic-block bootstrap on the cross-stratum test rows
test_topics_arr = t[te_out]; yte = y[te_out]; pred = clf.predict(acts[te_out, res["layer"]])
tt = np.array(sorted(set(test_topics_arr))); boots = []
for _ in range(a.n_boot):
    pick = rng.choice(tt, len(tt), replace=True); idx = np.concatenate([np.where(test_topics_arr == k)[0] for k in pick])
    boots.append((pred[idx] == yte[idx]).mean())
res["test_cross_acc_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
# permutation null with identical selection, labels shuffled within cells
null_acc, null_auroc = [], []
for i in range(a.n_perm):
    r0 = select_and_test(shuffle_within_cells(y))[0]
    null_acc.append(r0["test_cross_acc"]); null_auroc.append(r0["test_cross_auroc"])
null_acc = np.array(null_acc); null_auroc = np.array(null_auroc)
res["perm_null"] = {"n": a.n_perm, "acc_mean": float(np.nanmean(null_acc)), "acc_p95": float(np.nanquantile(null_acc, 0.95)), "acc_max": float(np.nanmax(null_acc)),
                    "auroc_mean": float(np.nanmean(null_auroc)), "auroc_p95": float(np.nanquantile(null_auroc, 0.95)),
                    "shuffles_beaten_acc": int((null_acc < res["test_cross_acc"]).sum()), "shuffles_beaten_auroc": int((null_auroc < res["test_cross_auroc"]).sum())}
res["p_value_vs_null"] = float((np.sum(null_acc >= res["test_cross_acc"]) + 1) / (len(null_acc) + 1))

# ---------- factorial directions ----------
def factorial_dir(target_col, l, mask):
    oth = "harmful" if target_col == "legal" else "legal"; d = np.zeros(acts.shape[2], np.float32)
    for s in (0, 1):
        m1 = mask & (df[target_col].to_numpy() == 1) & (df[oth].to_numpy() == s); m0 = mask & (df[target_col].to_numpy() == 0) & (df[oth].to_numpy() == s)
        if m1.any() and m0.any(): d += 0.5 * (acts_l[m1].mean(0) - acts_l[m0].mean(0))
    return d / (np.linalg.norm(d) + 1e-8)


def cos(u, v): return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))


# (a) cosine curve over layers with a within-cell label-swap null band, on training topics
cos_curve, cos_null_lo, cos_null_hi = [], [], []
yl, yh = df["legal"].to_numpy(), df["harmful"].to_numpy()
for l in range(L1):
    acts_l = acts[:, l]
    dL, dH = factorial_dir("legal", l, is_train), factorial_dir("harmful", l, is_train)
    cos_curve.append(cos(dL, dH))
    nulls = []
    for _ in range(a.n_cos_null):
        # swap legal labels within topic x harm cells, recompute d_legal, cosine with the real d_harm
        yp = yl.copy()
        for tp in np.unique(t):
            for s in (0, 1):
                idx = np.where((t == tp) & (yh == s) & is_train)[0]
                if len(idx) > 1: yp[idx] = rng.permutation(yp[idx])
        d = np.zeros(acts.shape[2], np.float32)
        for s in (0, 1):
            m1 = is_train & (yp == 1) & (yh == s); m0 = is_train & (yp == 0) & (yh == s)
            if m1.any() and m0.any(): d += 0.5 * (acts_l[m1].mean(0) - acts_l[m0].mean(0))
        nulls.append(cos(d, dH))
    cos_null_lo.append(float(np.quantile(nulls, 0.025))); cos_null_hi.append(float(np.quantile(nulls, 0.975)))
res["cosine_curve"] = {"layers": list(range(L1)), "cos_dlegal_dharm": cos_curve, "null_lo": cos_null_lo, "null_hi": cos_null_hi}

# (b)-(d) at the selected layer
l = res["layer"]; acts_l = acts[:, l]
dL, dH = factorial_dir("legal", l, is_train), factorial_dir("harmful", l, is_train)
harmful_mask, harmless_mask = is_test & (yh == 1), is_test & (yh == 0)
legal_mask, illegal_mask = is_test & (yl == 1), is_test & (yl == 0)
fa = {"layer": l, "cos_dlegal_dharm": cos(dL, dH),
      "dlegal_auroc_test_all": auroc_proj(dL, is_test, yl), "dharm_auroc_test_all": auroc_proj(dH, is_test, yh),
      "dlegal_auroc_within_harmful": auroc_proj(dL, harmful_mask, yl), "dlegal_auroc_within_harmless": auroc_proj(dL, harmless_mask, yl),
      "dharm_auroc_within_legal": auroc_proj(dH, legal_mask, yh), "dharm_auroc_within_illegal": auroc_proj(dH, illegal_mask, yh),
      "dlegal_auroc_predicting_harm_test": auroc_proj(dL, is_test, yh)}
# (c) project out top-k harm components: harm contrast vectors per training topic (within legality strata), PCA
hv = []
for tp in train_t:
    for s in (0, 1):
        m1 = (t == tp) & (yh == 1) & (yl == s); m0 = (t == tp) & (yh == 0) & (yl == s)
        if m1.any() and m0.any(): hv.append(acts_l[m1].mean(0) - acts_l[m0].mean(0))
hv = np.array(hv); hv = hv - hv.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(hv, full_matrices=False)
for k in (1, 2, 3):
    B = np.vstack([dH] + [Vt[i] for i in range(min(k - 1, len(Vt)))]) if k > 1 else dH[None]
    Q, _ = np.linalg.qr(B.T); dLr = dL - Q @ (Q.T @ dL); dLr /= (np.linalg.norm(dLr) + 1e-8)
    fa[f"dlegal_minus_harm_top{k}_auroc_within_harmful"] = auroc_proj(dLr, harmful_mask, yl)
    fa[f"dlegal_minus_harm_top{k}_auroc_within_harmless"] = auroc_proj(dLr, harmless_mask, yl)
    fa[f"dlegal_minus_harm_top{k}_auroc_test_all"] = auroc_proj(dLr, is_test, yl)
    if k == 1: dLr1 = dLr
# (d) half-contrast (mass-mean) conditional test: direction from the training stratum only, scored on the other stratum's test rows
tr_s = is_train & in_train_stratum
mm = acts_l[tr_s & (y == 1)].mean(0) - acts_l[tr_s & (y == 0)].mean(0); mm /= (np.linalg.norm(mm) + 1e-8)
fa["massmean_half_contrast_auroc_cross_stratum_test"] = auroc_proj(mm, te_out, y)
fa["massmean_half_contrast_auroc_in_stratum_test"] = auroc_proj(mm, te_in, y)
res["factorial"] = fa
np.savez_compressed(ROOT / "data/processed" / f"factorial_dirs_{a.run}_{a.tag}.npz", d_legal=dL, d_harm=dH, d_legal_resid=dLr1, layer=l)

out = ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json"
write_json(out, manifest(args=vars(a), split={"train_topics": sorted(train_t), "val_topics": sorted(val_t), "test_topics": sorted(test_t)}, n_rows=len(df), **res))
pn = res["perm_null"]
print("\n=== headline (layer and C chosen on validation AUROC, test topics scored once, cross-stratum) ===")
print(f"layer {res['layer']} · C {res['C']} · validation cross-stratum AUROC {res['val_cross_auroc']:.3f}")
print(f"TEST cross-stratum acc {res['test_cross_acc']:.3f}  CI95 {res['test_cross_acc_ci95'][0]:.2f}–{res['test_cross_acc_ci95'][1]:.2f}  AUROC {res['test_cross_auroc']:.3f}  (in-stratum test acc {res['test_in_acc']:.3f})")
print(f"null (labels shuffled within topic×stratum, same selection, {pn['n']} shuffles): acc mean {pn['acc_mean']:.3f}, p95 {pn['acc_p95']:.3f}, max {pn['acc_max']:.3f} · beat {pn['shuffles_beaten_acc']}/{pn['n']} on accuracy, {pn['shuffles_beaten_auroc']}/{pn['n']} on AUROC")
print(f"cos(d_legal, d_harm) at layer {l}: {fa['cos_dlegal_dharm']:+.3f} (null band {cos_null_lo[l]:+.2f}..{cos_null_hi[l]:+.2f}); curve over layers saved")
print(f"d_legal AUROC on test topics — within harmful {fa['dlegal_auroc_within_harmful']:.3f}, within harmless {fa['dlegal_auroc_within_harmless']:.3f}; d_legal predicting harm {fa['dlegal_auroc_predicting_harm_test']:.3f}")
print("d_legal with harm removed (top-1/2/3 harm components) — within harmful: " + ", ".join(f"{fa[f'dlegal_minus_harm_top{k}_auroc_within_harmful']:.3f}" for k in (1, 2, 3)) +
      " · within harmless: " + ", ".join(f"{fa[f'dlegal_minus_harm_top{k}_auroc_within_harmless']:.3f}" for k in (1, 2, 3)))
print(f"mass-mean half-contrast (train-stratum direction on other-stratum test rows): AUROC {fa['massmean_half_contrast_auroc_cross_stratum_test']:.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
for C in Cs:
    ax.plot(layers, [curve[l_][C] for l_ in layers], marker=".", label=f"validation AUROC, C={C}")
ax.axvline(res["layer"], ls="--", color="gray"); ax.scatter([res["layer"]], [res["test_cross_auroc"]], color="k", zorder=5, label="test AUROC (once)")
ax.axhline(pn["auroc_p95"], ls=":", color="r", label=f"null p95 ({pn['n']} shuffles)")
ax.set_ylabel("cross-stratum AUROC"); ax.set_ylim(0.3, 1.02); ax.legend(fontsize=7)
ax.set_title(f"{a.target} probe trained inside {other}={stratum_val[1]}, tested on the other stratum, held-out topics")
ax2.plot(range(L1), cos_curve, color="k", label="cos(d_legal, d_harm), training topics")
ax2.fill_between(range(L1), cos_null_lo, cos_null_hi, color="gray", alpha=0.3, label="within-cell label-swap null (95%)")
ax2.axhline(0, color="gray", lw=0.5); ax2.set_xlabel("layer"); ax2.set_ylabel("cosine"); ax2.set_ylim(-1, 1); ax2.legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "figures" / f"probeeval_{a.run}_{a.tag}.png", dpi=150)
print("wrote", out.relative_to(ROOT))
print("HAND-CHECK: recompute the test accuracy at the chosen layer from acts_<run>.npz using the split saved in the JSON.")
