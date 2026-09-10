"""figures.py — the write-up figures, from data/processed/*.json only.

IN PLAIN LANGUAGE
Three pictures, each answering one question:
  1. cosine_by_layer_<run>.png   Do the illegality and harm directions share a line? cos(d_illegal, d_harm) at every
     layer, bare and prompted, with the label-swap null band (inside the band = no more aligned than chance).
  2. cross_<run>.png             The 2x2 cross. For each of the four tests and both conditions: the cross-stratum
     test AUROC (dot with its bootstrap interval on accuracy shown as text), the null's 95th percentile (grey bar),
     and "beat N/100" printed on the bar. Word count alone is the hollow marker.
  3. checks_<run>.png            The checks: fraction of the simple anchors and the negated rows the probe calls
     legal (expected 100%), the harm-projected legality AUROC within the harmless stratum for top-1/2/3 removed,
     and the fair baseline (model's Yes/No logit vs probe) on the same rows.
Nothing is computed here beyond reading numbers out of the JSON files.

  uv run python scripts/figures.py --run lp_4b
"""
import argparse, json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DESIGNS = [("L_h2nh", "legality\nharmful → harmless"), ("L_nh2h", "legality\nharmless → harmful"), ("H_i2l", "harm\nillegal → legal"), ("H_l2i", "harm\nlegal → illegal")]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--title", default=""); a = p.parse_args()
conds = [(a.run, "bare sentence", "k"), (f"{a.run}_prompted", "inside 'is this illegal?'", "C3")]
J = lambda cond, tag: json.loads((ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json").read_text()) if (ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json").exists() else None
(ROOT / "figures").mkdir(exist_ok=True); sup = f" — {a.title}" if a.title else ""

# 1. cosine by layer
fig, ax = plt.subplots(figsize=(8, 4))
for cond, name, col in conds:
    r = J(cond, "L_h2nh")
    if not r: continue
    c = r["cosine_curve"]; ax.plot(c["layers"], [-v for v in c["cos_dlegal_dharm"]], color=col, lw=2, label=name)   # d_illegal = −d_legal, so the sign flips
    ax.fill_between(c["layers"], [-v for v in c["null_hi"]], [-v for v in c["null_lo"]], color=col, alpha=0.12, label=f"label-swap null 95% ({name.split()[0]})")
    ax.axvline(r["layer"], color=col, ls=":", lw=1)
ax.axhline(0, color="gray", lw=0.6); ax.set_ylim(-1, 1); ax.set_xlabel("layer"); ax.set_ylabel("cos(d_illegal, d_harm)   (+1 = same line, same way)")
ax.set_title(f"Do illegality and harm share a direction?{sup}\n(dotted = the layer each probe chose on validation topics)"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(ROOT / "figures" / f"cosine_by_layer_{a.run}.png", dpi=150); plt.close(fig)

# 2. the cross
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(DESIGNS)); w = 0.36
for j, (cond, name, col) in enumerate(conds):
    for i, (tag, lab) in enumerate(DESIGNS):
        r = J(cond, tag)
        if not r: continue
        pn = r["perm_null"]; xx = x[i] + (j - 0.5) * w
        ax.bar(xx, pn["auroc_p95"], width=w * 0.9, color="lightgray", edgecolor="gray", label="null AUROC 95th pct (100 shuffles)" if (i, j) == (0, 0) else None)
        ax.plot(xx, r["test_cross_auroc"], "o", color=col, ms=9, label=f"test AUROC, {name}" if i == 0 else None)
        ax.plot(xx, r.get("length_only_test_auroc", np.nan), "D", mfc="none", mec=col, ms=6, label="word count alone" if (i, j) == (0, 0) else None)
        ax.text(xx, 0.05, f"beat\n{pn['shuffles_beaten_acc']}/{pn['n']}", ha="center", va="bottom", fontsize=7, color="dimgray")
ax.axhline(0.5, color="gray", lw=0.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels([l for _, l in DESIGNS], fontsize=9); ax.set_ylim(0, 1.08); ax.set_ylabel("cross-stratum test AUROC (held-out topics)")
ax.set_title(f"The 2×2 cross: train inside one stratum, test on the other, on topics never seen{sup}", fontsize=10); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
fig.tight_layout(); fig.savefig(ROOT / "figures" / f"cross_{a.run}.png", dpi=150); plt.close(fig)

# 3. checks
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j, (cond, name, col) in enumerate(conds):
    r = J(cond, "L_h2nh")
    if not r: continue
    ex = r.get("extra_sets") or {}; names = sorted(ex); vals = [ex[n][[k for k in ex[n] if k.startswith("frac_predicted_")][0]] for n in names]
    axes[0].bar(np.arange(len(names)) + (j - 0.5) * 0.36, vals, width=0.33, color=col, alpha=0.8, label=name)
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels([f"{n}\n(n={ex[n]['n']})" for n in names])
    fa = r["factorial"]; ks = ["dlegal_auroc_within_harmless"] + [f"dlegal_minus_harm_top{k}_auroc_within_harmless" for k in (1, 2, 3)]
    axes[1].plot(range(4), [fa.get(k, np.nan) for k in ks], "o-", color=col, label=name)
    axes[2].bar(j, r["test_cross_auroc"], width=0.6, color=col, alpha=0.8, label=f"probe, {name}")
axes[0].axhline(1, color="gray", ls="--", lw=0.8); axes[0].set_ylim(0, 1.05); axes[0].set_title("Check sets: share not flagged illegal\n(never trained on; expected 100%)", fontsize=9); axes[0].legend(fontsize=7)
axes[1].axhline(0.5, color="gray", ls="--", lw=0.8); axes[1].set_ylim(0.3, 1.0); axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["as is", "top-1\nharm out", "top-2", "top-3"]); axes[1].set_title("Illegality direction, harmless stratum\n(held-out; AUROC after projecting harm out)", fontsize=9); axes[1].legend(fontsize=7)
# fair baseline from the ask file
try:
    ask = [json.loads(l) for l in (ROOT / "data/raw" / f"ask_{a.run}_legal.jsonl").open()]
    from sklearn.metrics import roc_auc_score
    for j, (cond, name, col) in enumerate(conds[:1]):
        r = J(cond, "L_h2nh") or J(conds[1][0], "L_h2nh")
        if not r: continue
        rows = [q for q in ask if q["topic"] in r["split"]["test_topics"] and int(q["harmful"]) == 0 and str(q.get("set", "main")) == "main" and int(q.get("exclude", 0)) == 0 and q.get("yes_minus_no_logit") is not None]
        au = roc_auc_score([int(q["legal"]) for q in rows], [-q["yes_minus_no_logit"] for q in rows])
        axes[2].bar(2.5, au, width=0.6, color="gray", alpha=0.8, label="model's own Yes/No logit, same rows")
except Exception: axes[2].text(0.5, 0.92, "just-ask baseline not run yet", fontsize=8, ha="center", transform=axes[2].transAxes)
axes[2].axhline(0.5, color="gray", ls="--", lw=0.8); axes[2].set_ylim(0, 1.05); axes[2].set_xticks([0.5, 2.5]); axes[2].set_xticklabels(["probe (bare, prompted)", "just ask"]); axes[2].set_title("Fair baseline, headline test rows\n(AUROC, illegal vs legal, harmless stratum)", fontsize=9); axes[2].legend(fontsize=7)
fig.suptitle(f"Checks{sup}", fontsize=10); fig.tight_layout(); fig.savefig(ROOT / "figures" / f"checks_{a.run}.png", dpi=150); plt.close(fig)
print("wrote", [f"figures/{n}_{a.run}.png" for n in ("cosine_by_layer", "cross", "checks")])
