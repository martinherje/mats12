"""Sanity-check a scenarios CSV before spending GPU time on it.

Reports quadrant counts, label/quadrant consistency, jurisdiction uniformity, hand-check and relabel
fractions, duplicates, and length spread. Exits non-zero on hard errors.
  uv run python scripts/validate_scenarios.py data/scenarios.csv
"""
import sys
import pandas as pd
from common import ROOT, QUADRANTS

path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/scenarios.csv")
df = pd.read_csv(path)
need = ["id", "text", "jurisdiction", "legal", "harmful", "quadrant", "topic", "borderline", "hand_checked", "relabelled"]
optional_flags = ["exclude", "borderline_legal", "borderline_harm"]
if "set" in df.columns:
    print("sets:", dict(df["set"].value_counts()), "— only set=main enters the probe design; simple and negated rows are scored once as checks")
    df_all = df; df = df[df["set"] == "main"].reset_index(drop=True)
missing = [c for c in need if c not in df.columns]
errs = []
if missing:
    errs.append(f"missing columns: {missing}")
else:
    if len(df) == 0:
        print(f"{path.name}: 0 rows (template only)"); sys.exit(0)
    if not df["id"].is_unique: errs.append("duplicate ids")
    dup = df["text"].duplicated().sum()
    if dup: errs.append(f"{dup} duplicate texts")
    for c in ["legal", "harmful", "borderline", "hand_checked", "relabelled"] + [c for c in optional_flags if c in df.columns]:
        if not set(df[c].dropna().unique()) <= {0, 1}: errs.append(f"column {c} must be 0/1")
    exp = df.apply(lambda r: f"{'legal' if r.legal == 1 else 'illegal'}_{'harmful' if r.harmful == 1 else 'harmless'}", axis=1)
    bad = (exp != df["quadrant"]).sum()
    if bad: errs.append(f"{bad} rows where quadrant disagrees with legal/harmful columns")
    if df["jurisdiction"].nunique() > 1: errs.append(f"mixed jurisdictions: {df['jurisdiction'].unique().tolist()} — pick one")
    print(f"{path.name}: {len(df)} rows · jurisdiction={df['jurisdiction'].unique().tolist()}")
    if "borderline_legal" in df.columns: print(f"borderline_legal: {int(df.borderline_legal.sum())}   borderline_harm: {int(df.borderline_harm.sum())}")
    print("per quadrant (core / borderline):")
    for q in QUADRANTS:
        sub = df[df.quadrant == q]
        print(f"  {q:18s} {len(sub):4d}   core={int((sub.borderline == 0).sum()):4d}  borderline={int((sub.borderline == 1).sum()):4d}")
    print(f"hand_checked: {df.hand_checked.mean():.0%} ({int(df.hand_checked.sum())}/{len(df)}) · relabelled among checked: "
          f"{(df[df.hand_checked == 1].relabelled.mean() if df.hand_checked.sum() else float('nan')):.0%}")
    if "exclude" in df.columns:
        print(f"excluded rows: {int(df.exclude.sum())} (dropped from the probe evaluation, reported in the write-up)")
        per_topic = df[df.exclude == 0].groupby("topic").size(); print(f"topics with all 4 rows still usable: {int((per_topic == 4).sum())}/{df.topic.nunique()}")
    print(f"topics: {df.topic.nunique()} · text length words: min {df.text.str.split().str.len().min()}, "
          f"median {int(df.text.str.split().str.len().median())}, max {df.text.str.split().str.len().max()}")
    offdiag = df.quadrant.isin(["illegal_harmless", "legal_harmful"]).sum()
    if offdiag < 40: print(f"WARNING: only {offdiag} off-diagonal items; the cross-stratum test is underpowered")
if errs:
    print("ERRORS:"); [print("  -", e) for e in errs]; sys.exit(1)
print("OK")
