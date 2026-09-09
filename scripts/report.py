"""Plain-language report of the legality-probe results. Generic infrastructure.

IN PLAIN LANGUAGE
Reads the result files cells 7-9 wrote and says, for each design, what was trained on what, how it did on the
held-out sentences, what shuffled labels would have scored, and a one-word verdict. Also lists the figures.
  uv run python scripts/report.py --run lp_4b
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); a = p.parse_args()
DESIGNS = [("L_h2nh", "legality", "harmful sentences", "harmless sentences"), ("L_nh2h", "legality", "harmless sentences", "harmful sentences"),
           ("H_i2l", "harm", "illegal sentences", "legal sentences"), ("H_l2i", "harm", "legal sentences", "illegal sentences")]
COND = [(a.run, "bare sentence"), (f"{a.run}_prompted", "inside the 'is this illegal?' question")]

def verdict(r):
    pn = r.get("perm_null") or {"acc_p95": r["perm_null_test_cross_acc"]["p95"]}
    acc, p95, p = r["test_cross_acc"], pn["acc_p95"], r["p_value_vs_null"]
    if acc >= p95 + 0.1 and p <= 0.05: return "CLEAR: well above what shuffled labels achieve"
    if acc > p95: return "LIKELY: above the shuffled-label ceiling, but not by much"
    return "NO: not distinguishable from shuffled labels"

for cond, cond_name in COND:
    print(f"\n=== Condition: {cond_name} ===")
    for tag, what, train_on, test_on in DESIGNS:
        f = ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json"
        if not f.exists(): print(f"  [{tag}] not run yet"); continue
        r = json.loads(f.read_text()); n_te = r["split"] and len(r["split"]["test_topics"])
        lo, hi = r["test_cross_acc_ci95"]
        print(f"\n  A {what} probe trained on {train_on} (layer {r['layer']} chosen on validation topics), tested on {test_on} from {n_te} topics it never saw:")
        pn = r.get("perm_null") or {"acc_p95": r["perm_null_test_cross_acc"]["p95"], "n": r["perm_null_test_cross_acc"]["n"], "shuffles_beaten_acc": None}
        beat = f"; beat {pn['shuffles_beaten_acc']}/{pn['n']} shuffles" if pn.get("shuffles_beaten_acc") is not None else ""
        print(f"    got {r['test_cross_acc']:.0%} right (likely range {lo:.0%}–{hi:.0%}), AUROC {r.get('test_cross_auroc', float('nan')):.2f}; shuffled labels would get up to {pn['acc_p95']:.0%}{beat}.  → {verdict(r)}")
        fa = r["factorial"]
        if tag == "L_h2nh":
            print(f"    The legality and harm directions sit at cosine {fa['cos_dlegal_dharm']:+.2f} (0 = unrelated, ±1 = the same line).")
            if "dlegal_minus_harm_top1_auroc_within_harmless" in fa:
                print(f"    Within the harmless sentences of held-out topics, the legality direction sorts legal from illegal at AUROC {fa['dlegal_auroc_within_harmless']:.2f}; with the top 1/2/3 harm components removed: "
                      + "/".join(f"{fa[f'dlegal_minus_harm_top{k}_auroc_within_harmless']:.2f}" for k in (1, 2, 3)) + " (0.5 = coin flip).")
            else:
                print(f"    After removing the harm component, the legality direction still sorts held-out sentences at AUROC {fa['dlegal_residualised_auroc_on_test_topics']:.2f} (0.5 = coin flip).")
    nc = ROOT / "data/processed" / f"probeeval_{cond}_L_h2nh_nocue.json"
    if nc.exists():
        r = json.loads(nc.read_text()); pn = r.get("perm_null") or {"acc_p95": r["perm_null_test_cross_acc"]["p95"]}; print(f"\n  Same main design with every sentence containing an explicit legality word removed: {r['test_cross_acc']:.0%} (shuffled ceiling {pn['acc_p95']:.0%}). If this is close to the number above, the probe is not just reading the words.")
# fair baseline: the model's own Yes-No logit as a classifier on the SAME held-out rows the main design was tested on
try:
    import numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score
    for cond, cond_name in COND:
        f = ROOT / "data/processed" / f"probeeval_{cond}_L_h2nh.json"; g = ROOT / "data/raw" / f"ask_{a.run}_legal.jsonl"
        if not (f.exists() and g.exists()): continue
        r = json.loads(f.read_text()); ask = pd.DataFrame([json.loads(l) for l in g.open()])
        if "yes_minus_no_logit" not in ask: print("\n(just-ask logits not recorded in this run; re-run cell 9 for the fair baseline)"); break
        m = ask.topic.isin(r["split"]["test_topics"]) & (ask.harmful == 0)   # the harmless-stratum rows from the test topics = the main design's test set
        au = roc_auc_score(ask.legal[m], -ask.yes_minus_no_logit[m])
        print(f"\nFAIR BASELINE ({cond_name}): on the same {int(m.sum())} held-out harmless-stratum sentences, the model's own Yes/No judgement sorts legal from illegal at AUROC {au:.2f}; the probe scored {r['test_cross_auroc']:.2f}.")
except Exception as e:
    print("fair-baseline block skipped:", type(e).__name__, e)
for lab in ("legal", "harmful"):
    f = ROOT / "data/processed" / f"ask_{a.run}_{lab}.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"\nJust asking the model 'is this {'illegal' if lab=='legal' else 'harmful'}?': {r['accuracy']:.0%} right overall, {r['offdiagonal_accuracy']:.0%} on the off-diagonal sentences, {r['refused_or_unparsed_frac']:.0%} invalid answers.")
figs = sorted((ROOT / "figures").glob(f"probeeval_{a.run}*.png"))
print("\nFigures (validation curve per layer, chosen layer, test point, shuffled-label ceiling):"); [print("  ", f.relative_to(ROOT)) for f in figs]
