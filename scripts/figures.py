"""figures.py — the one write-up figure, figures/fig1_<run>.png, from data/processed/*.json only.

IN PLAIN LANGUAGE
Two panels. A: the four probes (illegality trained inside the harmful rows and tested on the harmless rows,
the reverse, and the same two for harm) in both conditions, as AUROC on the other stratum on topics the probe
never saw; the grey range from 0.5 is the 95th percentile of 100 runs of the same procedure with labels shuffled
within topic and stratum. B: cos(d_illegal, d_harm) at every layer, bare and prompted, each with the 95% band the
same cosine takes when the legality labels are swapped within each cell. A curve inside the band is no more aligned than
chance; +1 means the two directions coincide. Nothing is computed here beyond reading numbers out of the JSON files;
the cosine is stored legal-positive and is negated (d_illegal = −d_legal), so the band's ends swap too.

  uv run python scripts/figures.py --run lp_4b
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, DESIGNS

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--title", default="", help="accepted for compatibility; the caption names the model"); a = p.parse_args()
conds = [(a.run, "bare sentence", "#000000", "o"), (f"{a.run}_prompted", "inside 'Is this illegal? Yes or No'", "#0072B2", "s")]
J = lambda cond, tag: (lambda f: json.loads(f.read_text()) if f.exists() else None)(ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json")
ticks = [label.replace(", ", "\n").replace(" → ", " →\n") for _, _, _, label in DESIGNS]
(ROOT / "figures").mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

# A. the cross: AUROC on the other stratum, held-out topics; grey = shuffled-label 95th percentile, drawn from 0.5
x = np.arange(len(DESIGNS)); w = 0.34
for j, (cond, name, col, mk) in enumerate(conds):
    for i, (tag, _, _, _) in enumerate(DESIGNS):
        r = J(cond, tag)
        if not r: continue
        pn = r["perm_null"]; xx = x[i] + (j - 0.5) * w
        axA.bar(xx, pn["auroc_p95"] - 0.5, bottom=0.5, width=w * 0.85, color="#d9d9d9", edgecolor="none",
                label=f"shuffled labels, 95th pct of {pn['n']}" if (i, j) == (0, 0) else None)
        axA.plot(xx, r["test_cross_auroc"], mk, color=col, ms=6.5, label=name if i == 0 else None, zorder=5)
axA.axhline(0.5, color="gray", lw=0.6); axA.set_ylim(0.45, 1.02); axA.set_xlim(-0.6, len(DESIGNS) - 0.4)
axA.set_xticks(x); axA.set_xticklabels(ticks, fontsize=7)
axA.set_ylabel("AUROC, other stratum, unseen topics")
axA.set_title("A. Trained in one harm (or legality) stratum,\ntested in the other on unseen topics", loc="left")
for s in ("top", "right"): axA.spines[s].set_visible(False)

# B. cosine by layer, illegality-positive (d_illegal = -d_legal, so the stored curve and band are negated and the band's ends swap)
n_layers = 32
for cond, name, col, mk in conds:
    r = J(cond, "L_h2nh")
    if not r: continue
    c = r["cosine_curve"]; L = c["layers"]; n_layers = L[-1]
    cos = [-v for v in c["cos_dlegal_dharm"]]; lo = [-v for v in c["null_hi"]]; hi = [-v for v in c["null_lo"]]
    axB.fill_between(L, lo, hi, color=col, alpha=0.10, lw=0)
    axB.plot(L, cos, color=col, lw=1.8, label=name)
axB.axhline(0, color="gray", lw=0.6); axB.set_ylim(-1, 1); axB.set_xlim(0, n_layers); axB.set_xticks([l for l in (0, 8, 16, 24, 32, 40, 48) if l <= n_layers])
axB.set_xlabel("layer"); axB.set_ylabel("cos(d_illegal, d_harm)")
axB.set_title("B. cos(d_illegal, d_harm) by layer\n(+1 = the two directions coincide)", loc="left")
axB.text(n_layers - 0.5, -0.62, "shaded: 95% band from\nlabel-swapped directions", ha="right", va="top", fontsize=6.5, color="dimgray")
for s in ("top", "right"): axB.spines[s].set_visible(False)

h, l = axA.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=1.2, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=1.2, rect=[0, 0.07, 1, 1])
out = ROOT / "figures" / f"fig1_{a.run}.png"; fig.savefig(out, dpi=220); plt.close(fig)
print("wrote", out.relative_to(ROOT))
