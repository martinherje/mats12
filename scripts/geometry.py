"""geometry.py — every sentence as a point in the plane spanned by the harm direction (x) and the illegality
direction (y), at the probe's chosen layer.

IN PLAIN LANGUAGE
Two arrows are computed from the training topics only: d_harm (harmful minus harmless, averaged over both
legality strata) and d_illegal (illegal minus legal, averaged over both harm strata). Every sentence's state is
projected onto both. If the model keeps the two concepts apart, the four quadrants sit in four corners: harm
separates left from right, legality separates bottom from top, and the corners form a rectangle. If the legality
direction is really the harm direction, the points fall on one diagonal line and the second axis adds nothing. Held-out topics are
drawn solid, training topics faint; the plain acts (set=simple) and the negations (set=negated) are drawn as
hollow markers so you can see where sentences the probe never trained on land. The angle between the arrows
is printed in the title. Note the axes are the two directions themselves, so the picture is 2-D by
construction; what is informative is where the four quadrants and the check sets fall, not the axes.

  uv run python scripts/geometry.py --run lp_4b                 # bare
  uv run python scripts/geometry.py --run lp_4b_prompted        # inside the question
Optional: --layer N to override the chosen layer; --tag to pick which evaluation's split/layer to use.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh"); p.add_argument("--layer", type=int); p.add_argument("--title", default=""); a = p.parse_args()
z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")}); yl = df["legal"].astype(int).to_numpy(); yh = df["harmful"].astype(int).to_numpy()
ev = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text()); split = ev["split"]; layer = a.layer or ev["layer"]
sset = df.get("set", pd.Series(["main"] * len(df))).astype(str).to_numpy(); excl = df.get("exclude", pd.Series([0] * len(df))).astype(int).to_numpy()
main = (sset == "main") & (excl == 0); topic = df["topic"].astype(str).to_numpy()
is_train = main & np.isin(topic, split["train_topics"]); is_test = main & np.isin(topic, split["test_topics"]); is_val = main & np.isin(topic, split["val_topics"])
X = acts[:, layer]

def fdir(target, other):
    d = np.zeros(X.shape[1], np.float32)
    for s in (0, 1):
        m1 = is_train & (target == 1) & (other == s); m0 = is_train & (target == 0) & (other == s)
        if m1.any() and m0.any(): d += 0.5 * (X[m1].mean(0) - X[m0].mean(0))
    return d / (np.linalg.norm(d) + 1e-8)

dH, dL = fdir(yh, yl), -fdir(yl, yh); cos = float(dH @ dL)   # dL = illegal minus legal (the illegality direction)
# centre on the training mean, project, and express in units of the training rows' spread along each axis
mu = X[is_train].mean(0); px, py = (X - mu) @ dH, (X - mu) @ dL
sx, sy = px[is_train].std() + 1e-8, py[is_train].std() + 1e-8; px, py = px / sx, py / sy
QUADS = [("illegal_harmful", 0, 1, "C3"), ("illegal_harmless", 0, 0, "C1"), ("legal_harmful", 1, 1, "C0"), ("legal_harmless", 1, 0, "C2")]
fig, ax = plt.subplots(figsize=(7.2, 6.4))
for name, lq, hq, col in QUADS:
    m = (yl == lq) & (yh == hq)
    ax.scatter(px[m & (is_train | is_val)], py[m & (is_train | is_val)], s=14, color=col, alpha=0.18)
    ax.scatter(px[m & is_test], py[m & is_test], s=44, color=col, alpha=0.95, edgecolor="k", lw=0.4, label=f"{name.replace('_', ' + ')} (held-out topics)")
    mx, my = px[m & is_test].mean(), py[m & is_test].mean(); ax.plot(mx, my, marker="X", ms=14, color=col, mec="k", mew=1.2)
for name, mk, lab in (("simple", "o", "plain legal acts (never trained on)"), ("negated", "^", "'You do not …' negations (never trained on)")):
    m = sset == name
    if m.any(): ax.scatter(px[m], py[m], s=34, facecolor="none", edgecolor="k", marker=mk, lw=0.8, label=lab)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("← harmless        projection on the harm direction        harmful →"); ax.set_ylabel("← legal        projection on the illegality direction        illegal →")
sup = f" — {a.title}" if a.title else ""
ax.set_title(f"Harm and illegality as two axes, layer {layer}{sup}\ncos(d_illegal, d_harm) = {cos:+.2f}; X = held-out quadrant mean; faint = training/validation topics", fontsize=9)
ax.legend(fontsize=7, loc="best"); fig.tight_layout()
out = ROOT / "figures" / f"geometry_{a.run}_L{layer}.png"; fig.savefig(out, dpi=150)
tm = {name: (float(px[(yl == lq) & (yh == hq) & is_test].mean()), float(py[(yl == lq) & (yh == hq) & is_test].mean())) for name, lq, hq, _ in QUADS}
print(f"layer {layer} · cos(d_illegal, d_harm) {cos:+.2f} · held-out quadrant means (harm axis, illegality axis):"); [print(f"  {k:18s} {v[0]:+.2f}, {v[1]:+.2f}") for k, v in tm.items()]
print(f"wrote {out.relative_to(ROOT)}")
