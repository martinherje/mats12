"""markedness.py — is the model's concept "illegal", "legal", or both? (Martin's question, 10 Sep evening)

IN PLAIN LANGUAGE
A two-class probe cannot tell "the model represents illegality, and legal is just its absence" from the
reverse. This script uses the 60 plain legal acts (set=simple, "You cook pasta") as a neutral baseline and
asks which quadrant is displaced from it. For each layer it projects every sentence's state onto the
illegality direction (illegal minus legal, the negative of the factorial d_legal, from training topics; held-out
topics only are reported) and measures each quadrant's mean distance from the plain-act mean, in units of the plain acts'
own spread. The marked concept is the one whose sentences move away from the baseline:
  illegal rows far, legal rows near  → the concept is illegality (legal = default)
  legal rows far, illegal rows near  → the concept is legality
  both far, opposite signs           → both are represented
  both far, the same sign            → the plain acts are not a neutral baseline along this direction (register
                                       confound: short mundane sentences); the ordering is graded, not a marked pole
  both near                          → the direction is not about either (or too weak to say)
Also reported: the same for the harm direction with harmless plain acts as baseline. Output:
data/processed/markedness_<run>.json and figures/markedness_<run>.png (distance by layer, one line per quadrant).

  uv run python scripts/markedness.py --run lp_4b            # after cell 6 and cell 7
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tag", default="L_h2nh"); a = p.parse_args()
z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz"); acts = z["acts"].astype(np.float32)
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")}); df["legal"] = df["legal"].astype(int); df["harmful"] = df["harmful"].astype(int)
split = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text())["split"]
sset = df.get("set", pd.Series(["main"] * len(df))).astype(str).to_numpy(); excl = df.get("exclude", pd.Series([0] * len(df))).astype(int).to_numpy()
main = (sset == "main") & (excl == 0); simple = sset == "simple"
is_train = main & df["topic"].isin(split["train_topics"]).to_numpy(); is_test = main & df["topic"].isin(split["test_topics"]).to_numpy()
yl, yh = df["legal"].to_numpy(), df["harmful"].to_numpy()
QUADS = {"illegal_harmful": (0, 1), "illegal_harmless": (0, 0), "legal_harmful": (1, 1), "legal_harmless": (1, 0)}

def factorial_dir(X, target, other):   # ½[(T1 − T0 | other=0) + (T1 − T0 | other=1)] on training topics
    d = np.zeros(X.shape[1], np.float32)
    for s in (0, 1):
        m1 = is_train & (target == 1) & (other == s); m0 = is_train & (target == 0) & (other == s)
        if m1.any() and m0.any(): d += 0.5 * (X[m1].mean(0) - X[m0].mean(0))
    return d / (np.linalg.norm(d) + 1e-8)

L = acts.shape[1]; out = {"illegality_direction": {q: [] for q in QUADS}, "harm_direction": {q: [] for q in QUADS}, "layers": list(range(L))}
for l in range(L):
    X = acts[:, l]
    for key, target, other in (("illegality_direction", yl, yh), ("harm_direction", yh, yl)):
        d = factorial_dir(X, target, other)
        if key == "illegality_direction": d = -d   # report along the ILLEGALITY direction: positive = displaced toward illegal
        proj = X @ d
        base_mu, base_sd = proj[simple].mean(), proj[simple].std() + 1e-8   # the plain acts: the neutral baseline
        for q, (lq, hq) in QUADS.items():
            m = is_test & (yl == lq) & (yh == hq)
            out[key][q].append(float((proj[m].mean() - base_mu) / base_sd) if m.any() else float("nan"))
# summary at the probe's chosen layer and averaged over the last third of layers
chosen = json.loads((ROOT / "data/processed" / f"probeeval_{a.run}_{a.tag}.json").read_text())["layer"]
summ = {}
for key in ("illegality_direction", "harm_direction"):
    at = {q: out[key][q][chosen] for q in QUADS}; late = {q: float(np.nanmean(out[key][q][2 * L // 3:])) for q in QUADS}
    summ[key] = {"layer_used": chosen, "at_chosen_layer": at, "mean_over_last_third_of_layers": late}
out["summary"] = summ
def verdict(at):
    ill = np.mean([at["illegal_harmful"], at["illegal_harmless"]]); leg = np.mean([at["legal_harmful"], at["legal_harmless"]])   # signs kept
    if abs(ill) > 1 and abs(leg) > 1 and np.sign(ill) == np.sign(leg):
        return ("every quadrant is displaced the same way from the plain acts, so the plain acts are not a neutral baseline along this direction "
                "(register confound: short mundane sentences); the ordering is graded, not a marked pole")
    if abs(ill) > 1 and abs(leg) < 0.5: return "illegal rows displaced, legal rows near the baseline: illegality is the marked side"
    if abs(leg) > 1 and abs(ill) < 0.5: return "legal rows displaced, illegal rows near the baseline: legality is the marked side"
    if abs(ill) > 1 and abs(leg) > 1: return "illegal and legal rows displaced in opposite directions: both sides represented"
    return "neither side is clearly displaced along this direction (too weak to say)"
out["verdict_illegality_direction"] = verdict(summ["illegality_direction"]["at_chosen_layer"])
(ROOT / "data/processed").mkdir(exist_ok=True); (ROOT / "data/processed" / f"markedness_{a.run}.json").write_text(json.dumps(out, indent=1))
print(f"run={a.run} · baseline = {int(simple.sum())} plain legal acts · held-out topics only · distances in plain-act standard deviations\n")
for key in ("illegality_direction", "harm_direction"):
    print(f"[{key}] at layer {chosen}:  " + "  ".join(f"{q} {summ[key]['at_chosen_layer'][q]:+.2f}" for q in QUADS))
print(f"\nReading at layer {chosen} (illegality direction): {out['verdict_illegality_direction']}")
print("Reading: a quadrant near 0 sits where the plain acts sit; |value| > 1 means it has moved more than one plain-act spread away.")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, key, title in zip(axes, ("illegality_direction", "harm_direction"), ("along the illegality direction", "along the harm direction")):
    for q, col in zip(QUADS, ("C3", "C1", "C0", "C2")): ax.plot(out["layers"], out[key][q], color=col, lw=2, label=q.replace("_", "-"))
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(chosen, color="gray", ls=":", lw=1); ax.set_xlabel("layer"); ax.set_title(f"Distance from the plain-act baseline, {title}", fontsize=9)
axes[0].set_ylabel("distance (plain-act standard deviations)"); axes[0].legend(fontsize=7)
fig.suptitle("Distance from the plain legal acts, by quadrant (held-out topics)", fontsize=10); fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True); fig.savefig(ROOT / "figures" / f"markedness_{a.run}.png", dpi=150); print(f"wrote data/processed/markedness_{a.run}.json, figures/markedness_{a.run}.png")
