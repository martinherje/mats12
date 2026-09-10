"""verify_by_hand.py — recompute every number of record for one design from the activations file, without probe_eval.py.

What it does, for one run and one design tag (default: the headline, L_h2nh):
  1. rebuilds the training and test rows from the topic split saved in the JSON (design rows only, excluded rows out,
     check sets out), refits the probe with sklearn at the saved layer and C, and prints accuracy as a count
     ("21 of 30") and AUROC twice: sklearn's, and the Mann-Whitney rank formula written out in numpy;
  2. lists every test sentence with its score (+ = called illegal / harmful), so the mistakes can be read;
  3. scores the two check sets with the same refit probe and prints the counts ("33 of 60 called legal");
  1b. the topic-block bootstrap interval on that accuracy (2000 resamples of the test topics);
  4. recomputes the factorial directions in plain numpy (mean differences on training topics, averaged over the other
     factor), the cosine at the chosen layer with the illegality-positive sign, a fresh label-swap band (200 swaps,
     seed 1), the illegality direction's AUROC within the other stratum before and after projecting out the harm
     direction (Gram-Schmidt, no QR), and the illegality direction's AUROC for predicting harm;
  5. a fixed-layer permutation null: labels shuffled within topic x stratum, 200 times, seed 1, refit at the same
     layer and C, no selection (cheaper than the script's selection-inclusive null; prints how many it beat);
  6. word count alone on the same test rows, and the probe score's rank correlation with word count.
Every line prints the script's number next to the recomputed one; every number comes from the file.
This is the single by-hand check path: it supersedes notebooks/cell_10_replacement.py (deleted) and covers what
scripts/recompute_headline.py covers plus the directions, the check sets, a fixed-layer null and word count.

  uv run python notebooks/verify_by_hand.py --run lp_4b                     # headline, bare
  uv run python notebooks/verify_by_hand.py --run lp_4b_prompted            # headline, prompted
  uv run python notebooks/verify_by_hand.py --run lp_4b --tag L_nh2h        # the reverse; H_i2l, H_l2i for harm
  uv run python notebooks/verify_by_hand.py --run lp_4b --n-null 0          # skip the null (seconds instead of a minute)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh")
p.add_argument("--n-null", type=int, default=200); p.add_argument("--n-swap", type=int, default=200); p.add_argument("--seed", type=int, default=1)
p.add_argument("--acts-dir", default=None, help="folder holding acts_<run>.npz if not data/processed (e.g. the Drive mount)")
a = p.parse_args(); rng = np.random.default_rng(a.seed)

r = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text())
L, C = r["layer"], r["C"]; target = r["args"]["target"]; other = "harmful" if target == "legal" else "legal"
stratum_val = {"harmful": 1, "harmless": 0, "illegal": 0, "legal": 1}[r["args"]["train_stratum"]]
fa = r["factorial"]; pn = r["perm_null"]

acts_path = (Path(a.acts_dir) if a.acts_dir else ROOT / "data/processed") / f"acts_{a.run}.npz"
z = np.load(acts_path); acts_all = z["acts"]; print("reading", acts_path)
print(f"activations: shape {acts_all.shape}, stored as {acts_all.dtype}, finite: {bool(np.isfinite(acts_all.astype(np.float32)).all())}, max |value| {float(np.abs(acts_all.astype(np.float32)).max()):.0f}")
X_all = acts_all[:, L].astype(np.float32)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")})
df["legal"] = df.legal.astype(int); df["harmful"] = df.harmful.astype(int); df["exclude"] = df.exclude.astype(int)
main = (df["set"].astype(str) == "main") & (df.exclude == 0)
d = df[main].reset_index(drop=True); X = X_all[main.to_numpy()]
y = d[target].to_numpy(); yo = d[other].to_numpy(); t = d.topic.to_numpy()
is_train = d.topic.isin(r["split"]["train_topics"]).to_numpy(); is_test = d.topic.isin(r["split"]["test_topics"]).to_numpy()
tr = is_train & (yo == stratum_val); te = is_test & (yo != stratum_val)
sign = -1.0 if target == "legal" else 1.0          # + = called illegal (target legal) or called harmful (target harmful)
pos_name = "illegal" if target == "legal" else "harmful"


def auroc_rank(labels, scores):
    """Mann-Whitney: the share of (positive, negative) pairs the positive scores higher, ties half. No sklearn."""
    labels = np.asarray(labels); scores = np.asarray(scores, float); P = scores[labels == 1]; N = scores[labels == 0]
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))


def fit(Xtr, ytr):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, ytr)


# 1. the headline number
print(f"\n== {a.run} · {a.tag}: {target} probe trained inside {other}={stratum_val} on {is_train.sum() // 4} training topics (n={tr.sum()}), tested on the other stratum of {is_test.sum() // 4} held-out topics (n={te.sum()}) · layer {L}, C {C}")
clf = fit(X[tr], y[tr]); pred = clf.predict(X[te]); score = sign * clf.decision_function(X[te])
right = int((pred == y[te]).sum())
print(f"accuracy: {right} of {te.sum()} right = {right / te.sum():.3f}      script {r['test_cross_acc']:.3f}")
print(f"AUROC   : sklearn {roc_auc_score(1 - y[te] if target == 'legal' else y[te], score):.3f} · rank formula {auroc_rank((1 - y[te]) if target == 'legal' else y[te], score):.3f}      script {r['test_cross_auroc']:.3f}")
print(f"predicted {pos_name}: {int((sign * clf.decision_function(X[te]) > 0).sum())} of {te.sum()} test rows (all one class would explain a 50% accuracy with a 0.50–0.50 interval)")
tt = np.array(sorted(set(t[te]))); hits = (pred == y[te]); tte = t[te]; boots = []
for _ in range(2000):
    pick = rng.choice(tt, len(tt), replace=True); idx = np.concatenate([np.where(tte == k)[0] for k in pick]); boots.append(hits[idx].mean())
print(f"topic-block bootstrap 95% interval on accuracy (2000 resamples of the {len(tt)} test topics, seed {a.seed}): {np.quantile(boots, 0.025):.2f}–{np.quantile(boots, 0.975):.2f}      script {r['test_cross_acc_ci95'][0]:.2f}–{r['test_cross_acc_ci95'][1]:.2f}")

# 2. the test rows, read
print(f"\ntest rows (score + = called {pos_name}; * = wrong):")
for i in np.where(te)[0]:
    s = sign * clf.decision_function(X[i:i + 1])[0]; wrong = "*" if clf.predict(X[i:i + 1])[0] != y[i] else " "
    print(f"  {wrong} {s:+5.2f} [{d.quadrant[i]:17s} bl_legal={d.borderline_legal[i]} bl_harm={d.borderline_harm[i]}] {d.text[i]}")

# 3. the check sets, counted
print("\ncheck sets (never trained on), same refit probe:")
for name in ("simple", "negated"):
    m = (df["set"].astype(str) == name).to_numpy()
    if not m.any(): continue
    pr = clf.predict(X_all[m]); ex = r["extra_sets"].get(name, {}); k = [k for k in ex if k.startswith("frac_predicted_")]
    print(f"  {name}: {int((pr == 1).sum())} of {m.sum()} called {target}=1 ({'legal' if target == 'legal' else 'harmful'}) = {pr.mean():.3f}, expected {'all' if target == 'legal' else 'none'}      script frac_predicted_{target}_1 = {ex[k[0]]:.3f}" if k else f"  {name}: {int((pr == 1).sum())} of {m.sum()} called {target}=1")

# 4. factorial directions in plain numpy
def mean_diff_dir(target_labels, other_labels, mask):
    dvec = np.zeros(X.shape[1], np.float32)
    for s in (0, 1):
        m1 = mask & (target_labels == 1) & (other_labels == s); m0 = mask & (target_labels == 0) & (other_labels == s)
        dvec += 0.5 * (X[m1].mean(0) - X[m0].mean(0))
    return dvec / np.linalg.norm(dvec)


yl, yh = d.legal.to_numpy(), d.harmful.to_numpy()
d_illegal = -mean_diff_dir(yl, yh, is_train); d_harm = mean_diff_dir(yh, yl, is_train)
cos = float(d_illegal @ d_harm)
swaps = []
for _ in range(a.n_swap):
    yp = yl.copy()
    for tp in np.unique(t):
        for s in (0, 1):
            idx = np.where((t == tp) & (yh == s) & is_train)[0]
            if len(idx) > 1: yp[idx] = rng.permutation(yp[idx])
    swaps.append(float(-mean_diff_dir(yp, yh, is_train) @ d_harm))
lo, hi = np.quantile(swaps, [0.025, 0.975])
cc = r["cosine_curve"]
print(f"\ncos(d_illegal, d_harm) at layer {L}: {cos:+.3f}      script {-fa['cos_dlegal_dharm']:+.3f}   (illegality-positive)")
print(f"label-swap band, {a.n_swap} swaps, seed {a.seed}: {-hi:+.2f}..{-lo:+.2f}      script (50 swaps) {-cc['null_hi'][L]:+.2f}..{-cc['null_lo'][L]:+.2f}   → {'outside' if (cos < -hi or cos > -lo) else 'inside'} the fresh band")
harmless_te, harmful_te = is_test & (yh == 0), is_test & (yh == 1)
d_ill_resid = d_illegal - (d_illegal @ d_harm) * d_harm; d_ill_resid /= np.linalg.norm(d_ill_resid)
print(f"illegality direction, AUROC for illegal within held-out harmless rows: {auroc_rank(1 - yl[harmless_te], X[harmless_te] @ d_illegal):.3f}      script {fa['dlegal_auroc_within_harmless']:.3f}")
print(f"  same with the harm direction projected out (Gram-Schmidt):          {auroc_rank(1 - yl[harmless_te], X[harmless_te] @ d_ill_resid):.3f}      script top-1 {fa['dlegal_minus_harm_top1_auroc_within_harmless']:.3f}")
print(f"  within held-out harmful rows, as is / harm out:                    {auroc_rank(1 - yl[harmful_te], X[harmful_te] @ d_illegal):.3f} / {auroc_rank(1 - yl[harmful_te], X[harmful_te] @ d_ill_resid):.3f}      script {fa['dlegal_auroc_within_harmful']:.3f} / {fa['dlegal_minus_harm_top1_auroc_within_harmful']:.3f}")
print(f"illegality direction predicting harm on held-out rows: {auroc_rank(yh[is_test], X[is_test] @ d_illegal):.3f}      script {1 - fa['dlegal_auroc_predicting_harm_test']:.3f}")
print(f"harm direction predicting harm within held-out legal / illegal rows: {auroc_rank(yh[is_test & (yl == 1)], X[is_test & (yl == 1)] @ d_harm):.3f} / {auroc_rank(yh[is_test & (yl == 0)], X[is_test & (yl == 0)] @ d_harm):.3f}      script {fa['dharm_auroc_within_legal']:.3f} / {fa['dharm_auroc_within_illegal']:.3f}")

# 5. fixed-layer permutation null
if a.n_null:
    null = []
    for _ in range(a.n_null):
        yp = y.copy()
        for tp in np.unique(t):
            for s in (0, 1):
                idx = np.where((t == tp) & (yo == s))[0]
                if len(idx) > 1: yp[idx] = rng.permutation(yp[idx])
        null.append(float((fit(X[tr], yp[tr]).predict(X[te]) == yp[te]).mean()))
    null = np.array(null)
    print(f"\nfixed-layer null (layer {L}, C {C}, labels shuffled within topic×stratum, {a.n_null} shuffles, seed {a.seed}): mean {null.mean():.3f}, p95 {np.quantile(null, 0.95):.3f}, max {null.max():.3f} · the real accuracy beat {int((null < right / te.sum()).sum())} of {a.n_null}")
    print(f"  script's selection-inclusive null (100 shuffles, layer and C re-chosen each time): mean {pn['acc_mean']:.3f}, p95 {pn['acc_p95']:.3f}, max {pn['acc_max']:.3f} · beat {pn['shuffles_beaten_acc']}/{pn['n']}")

# 6. word count
wc = d.text.astype(str).str.split().str.len().to_numpy()
from scipy.stats import spearmanr
print(f"\nword count alone on the same test rows, AUROC for {target}=1: {auroc_rank(y[te], wc[te]):.3f}      script {r['length_only_test_auroc']:.3f}")
print(f"probe score vs word count on the test rows: Spearman r = {spearmanr(score, wc[te])[0]:+.2f}; a value near 0 means the score does not track length")
