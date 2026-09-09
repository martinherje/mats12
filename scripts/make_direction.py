"""Build per-layer difference-of-means directions from an activation .npz. Generic infrastructure.

  uv run python scripts/make_direction.py --run gonogo_pod --label good_side --filter "bet==1" --out direction_v1_good_side
  uv run python scripts/make_direction.py --run gonogo_pod --label bet --out direction_v1_bet      # topic control (bet vs no bet)

--label   binary column in the activation npz (from the scenarios CSV): direction = mean(label==1) - mean(label==0)
--filter  pandas query over the label columns to select rows first (e.g. drop baseline rows for the good_side direction)
Writes data/processed/<out>.npz with dirs [layers+1, d] (unit vectors) plus a sidecar .json manifest.
Also prints, per layer, the separation (cosine between each row and the direction, mean by class) as a sanity number.

IN PLAIN LANGUAGE
What it does: turns the saved snapshots into one vector per layer — the average state under label=1 minus
the average under label=0. With label good_side that is "what changes inside the model when the good side
flips from below the threshold to above it". With label bet it is "what changes when a bet is mentioned at
all". No training, no fitting; one subtraction.
What comes out: data/processed/<out>.npz holding unit vectors, one per layer, ready for the ablation.
Caveat printed in the table: the two prompt versions differ in a few words, so the direction separates
them trivially. That is not the result; the result is what happens when the direction is removed.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, manifest, write_json
from steer import save_directions

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--label", required=True); p.add_argument("--filter", default="")
p.add_argument("--out", required=True); p.add_argument("--think", default="", help="optionally restrict to think==on|off rows")
a = p.parse_args()

z = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz")
acts = z["acts"].astype(np.float32); cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
df = pd.DataFrame({k: v for k, v in cols.items() if k != "text"})
mask = np.ones(len(df), bool)
if a.filter:
    mask &= df.eval(a.filter).to_numpy()
if a.think:
    mask &= (df["think"].astype(str) == a.think).to_numpy()
y = df.loc[mask, a.label].astype(int).to_numpy(); X = acts[mask]
assert set(np.unique(y)) == {0, 1}, f"label {a.label} must be binary after filtering; got {np.unique(y)}"
print(f"rows used {mask.sum()}/{len(df)} · positives {int(y.sum())} · layers+1 {X.shape[1]} · d {X.shape[2]}")
out = ROOT / "data/processed" / f"{a.out}.npz"
save_directions(out, X, y, a.label, {"run": a.run, "filter": a.filter, "think": a.think})
dirs = np.load(out)["dirs"]
print("layer  mean cos(pos)  mean cos(neg)   (separation sanity; trivially large where the prompts differ lexically)")
for l in range(X.shape[1]):
    Xl = X[:, l]; c = (Xl @ dirs[l]) / (np.linalg.norm(Xl, axis=1) + 1e-8)
    print(f"{l:5d}  {c[y==1].mean():+.3f}       {c[y==0].mean():+.3f}")
write_json(out.with_suffix(".json"), manifest(run=a.run, label=a.label, filter=a.filter, think=a.think, n=int(mask.sum()), shape=list(dirs.shape)))
print("wrote", out.relative_to(ROOT))
