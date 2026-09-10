# %% [markdown]
# # Look at the data: what a probe sees, and how the two-axes picture is made
#
# Open this file in VS Code. Each `# %%` block is a cell: hover it and click Run Cell
# (or put the cursor in it and press Shift+Enter). Outputs appear in the Interactive window
# on the right. Everything is read from files in this repo; nothing is fabricated in between.
#
# Files used (the Qwen3.5-4B run of record, lp_4b):
#   acts_lp_4b.npz on the Drive mount (ACTS_DIR)      the model's internal states + the labels
#   data/processed/probeeval_lp_4b_L_h2nh.json        the headline evaluation's output
#   scripts/geometry.py                               the script that draws the two-axes figure
# For the 0.5B stand-in made on this Mac, set RUN = "mac05b"; its acts file is in data/processed/.

# %%
import json
from pathlib import Path
import numpy as np, pandas as pd
RUN = "lp_4b"           # "mac05b" for the 0.5B stand-in
LAYER = None            # None = the layer the headline probe chose; or a number, e.g. 20
ACTS_DIR = Path.home() / "Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed"   # where acts_lp_4b.npz lives
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
acts_path = (ACTS_DIR if (ACTS_DIR / f"acts_{RUN}.npz").exists() else ROOT / "data/processed") / f"acts_{RUN}.npz"
z = np.load(acts_path); print("reading", acts_path)
acts = z["acts"]
print("array of model states:", acts.shape, "= (sentences, layers, numbers per state), stored as", acts.dtype)
print("largest magnitude anywhere:", float(np.abs(acts.astype(np.float32)).max()), "(half-precision overflows at 65504)")

# %% [markdown]
# ## 1. One sentence, one layer = one row of numbers
# The probe sees this row of numbers, not the words.

# %%
i = 0                                   # sentence index; try 1, 2, 3 ...
ev = json.loads((ROOT / "data/processed" / f"probeeval_{RUN}_L_h2nh.json").read_text())
layer = LAYER or ev["layer"]
print(z["col_id"][i], "|", z["col_text"][i])
print("legal =", z["col_legal"][i], " harmful =", z["col_harmful"][i], " quadrant =", z["col_quadrant"][i], " set =", z["col_set"][i])
print(f"its state at layer {layer}, first 12 of {acts.shape[2]} numbers:")
print(np.array2string(acts[i, layer, :12], precision=3))

# %% [markdown]
# ## 2. The table the probe is trained on
# Labels on the left, the numbers on the right (only the first 6 shown). One row per sentence.

# %%
df = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith("col_")})
table = df[["id", "quadrant", "set", "legal", "harmful"]].copy()
for j in range(6):
    table[f"x{j}"] = np.round(acts[:, layer, j].astype(np.float32), 3)
table["text"] = df["text"].str.slice(0, 55)
print(table.head(12).to_string(index=False))
print("...", len(table), "rows in total;", int((df["set"] == "main").sum()), "in the design,", int((df["set"] == "simple").sum()), "plain acts,", int((df["set"] == "negated").sum()), "negations")

# %% [markdown]
# ## 3. What the evaluation wrote out
# These keys are what every number in the report is copied from.

# %%
print("layer chosen on validation topics:", ev["layer"], "  regularisation C:", ev["C"])
print("test (held-out topics, other stratum): accuracy", round(ev["test_cross_acc"], 3), " range", [round(v, 2) for v in ev["test_cross_acc_ci95"]], " AUROC", round(ev["test_cross_auroc"], 3))
print("shuffled-label null:", ev["perm_null"])
print("word count alone on the same rows, AUROC:", round(ev["length_only_test_auroc"], 3))
print("check sets:", ev["extra_sets"])
print("cos(d_illegal, d_harm) at the chosen layer:", round(-ev["factorial"]["cos_dlegal_dharm"], 3), "(the JSON stores the legal-positive value; the sign is flipped here)")
print("test topics:", ev["split"]["test_topics"])

# %% [markdown]
# ## 4. The two arrows, step by step (this is what scripts/geometry.py does)
# Harm arrow = mean(harmful) − mean(harmless), computed inside the illegal rows and inside the legal rows, then averaged, so legality cancels.
# Illegality arrow = mean(illegal) − mean(legal), inside harmful and inside harmless, averaged, so harm cancels.
# Training topics only, so the held-out topics never touch the arrows.

# %%
yl = df["legal"].astype(int).to_numpy(); yh = df["harmful"].astype(int).to_numpy()
main = (df["set"] == "main").to_numpy() & (df["exclude"].astype(int).to_numpy() == 0)
is_train = main & df["topic"].isin(ev["split"]["train_topics"]).to_numpy()
is_test = main & df["topic"].isin(ev["split"]["test_topics"]).to_numpy()
X = acts[:, layer].astype(np.float32)

def arrow(target, other):
    d = np.zeros(X.shape[1], np.float32)
    for s in (0, 1):                                   # inside each level of the other concept
        m1 = is_train & (target == 1) & (other == s)
        m0 = is_train & (target == 0) & (other == s)
        d += 0.5 * (X[m1].mean(0) - X[m0].mean(0))
    return d / np.linalg.norm(d)                       # length 1

d_harm = arrow(yh, yl); d_illegal = -arrow(yl, yh)      # arrow(yl, yh) points toward legal; the illegality arrow is its negative
print("harm arrow, first 6 of", len(d_harm), ":", np.round(d_harm[:6], 4))
print("illegality arrow, first 6:", np.round(d_illegal[:6], 4))
print("angle between them, as a cosine:", round(float(d_harm @ d_illegal), 3))

# %% [markdown]
# ## 5. Every sentence as a point: x = along the harm arrow, y = along the illegality arrow
# A dot product with a unit arrow is "how far along that arrow the point sits".

# %%
mu = X[is_train].mean(0)                               # centre on the training cloud
px = (X - mu) @ d_harm; py = (X - mu) @ d_illegal
px = px / px[is_train].std(); py = py / py[is_train].std()   # same units on both axes
pts = pd.DataFrame({"id": df["id"], "quadrant": df["quadrant"], "set": df["set"], "x_harm": np.round(px, 2), "y_illegal": np.round(py, 2)})
print(pts[is_test].head(8).to_string(index=False))
print("\nheld-out quadrant means (x_harm, y_illegal):")
print(pts[is_test].groupby("quadrant")[["x_harm", "y_illegal"]].mean().round(2))

# %% [markdown]
# ## 6. The picture
# If the model keeps the two apart, the four quadrants sit in four corners; if not, they fall on one diagonal.

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 6))
for q, col in (("illegal_harmful", "C3"), ("illegal_harmless", "C1"), ("legal_harmful", "C0"), ("legal_harmless", "C2")):
    m = (df["quadrant"] == q).to_numpy()
    ax.scatter(px[m & is_train], py[m & is_train], s=12, color=col, alpha=0.15)
    ax.scatter(px[m & is_test], py[m & is_test], s=40, color=col, edgecolor="k", lw=0.4, label=q + " (held-out)")
    ax.plot(px[m & is_test].mean(), py[m & is_test].mean(), "X", ms=13, color=col, mec="k")
for name, mk in (("simple", "o"), ("negated", "^")):
    m = (df["set"] == name).to_numpy()
    ax.scatter(px[m], py[m], s=30, facecolor="none", edgecolor="k", marker=mk, label=name + " (never trained on)")
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("← harmless   along the harm arrow   harmful →"); ax.set_ylabel("← legal   along the illegality arrow   illegal →")
ax.set_title(f"{RUN}, layer {layer}, cos(d_illegal, d_harm) = {float(d_harm @ d_illegal):+.2f}"); ax.legend(fontsize=7)
out = ROOT / "figures" / f"look_{RUN}_L{layer}.png"; fig.savefig(out, dpi=150); print("saved", out)
plt.show(block=False)          # shows inline in the VS Code interactive window; does not block when run as a script

# %% [markdown]
# The same figure, produced by the repo script (so you can check they agree):
# `uv run python scripts/geometry.py --run lp_4b` → `figures/geometry_lp_4b_L25.png`.
