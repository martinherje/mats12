"""results_table.py — the results tables for the write-up, as markdown, from data/processed/*.json.

IN PLAIN LANGUAGE
Reads every probeeval_<run>[_prompted]_<tag>.json and ask_<run>_<label>.json that exists and writes
journal/results.md: one row per design and condition (test AUROC, accuracy with its interval, beat N/100,
the cosine, the harm-projected legality AUROC within the harmless stratum), then the check lines (word count
alone, the simple and negated sets, the no-cue rerun) and the just-ask baseline. Nothing here is computed;
it only copies numbers already in the JSON files, so you can check every cell against its file.

  uv run python scripts/results_table.py --run lp_4b
"""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DESIGNS = [("legal", "harmful", "L_h2nh", "legality, harmful → harmless (headline)"), ("legal", "harmless", "L_nh2h", "legality, harmless → harmful"),
           ("harmful", "illegal", "H_i2l", "harm, illegal → legal"), ("harmful", "legal", "H_l2i", "harm, legal → illegal")]
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--out", default="journal/results.md"); a = p.parse_args()
out = [f"# Results — run `{a.run}` (copied from data/processed/*.json; recompute by hand before quoting)\n"]
out.append("| condition | test | layer | AUROC | acc (CI95) | beat shuffles | cos(dL,dH) | dL within harmless, top-1 harm removed | word count alone |")
out.append("|---|---|---|---|---|---|---|---|---|")
checks = []
for cond, cname in ((a.run, "bare"), (f"{a.run}_prompted", "prompted")):
    for target, stratum, tag, label in DESIGNS:
        f = ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json"
        if not f.exists(): out.append(f"| {cname} | {label} | – | not run | | | | | |"); continue
        r = json.loads(f.read_text()); pn = r["perm_null"]; fa = r["factorial"]; lo, hi = r["test_cross_acc_ci95"]
        within = fa.get("dlegal_minus_harm_top1_auroc_within_harmless") if target == "legal" else fa.get("dharm_auroc_within_legal")
        out.append(f"| {cname} | {label} | {r['layer']} | {r['test_cross_auroc']:.2f} | {r['test_cross_acc']:.0%} ({lo:.0%}–{hi:.0%}) | {pn['shuffles_beaten_acc']}/{pn['n']} (p95 {pn['acc_p95']:.0%}) | {fa['cos_dlegal_dharm']:+.2f} | {within if within is None else f'{within:.2f}'} | {r.get('length_only_test_auroc', float('nan')):.2f} |")
        for name, ex in (r.get("extra_sets") or {}).items():
            k = [k for k in ex if k.startswith("frac_predicted_")][0]
            checks.append(f"- {cname}, {label}: check set **{name}** ({ex['n']} rows, never trained on, expected {ex['expected']}): the probe calls {ex[k]:.0%} of them {k.split('_')[2]}=1.")
    nc = ROOT / "data/processed" / f"probeeval_{cond}_L_h2nh_nocue.json"
    if nc.exists():
        r = json.loads(nc.read_text())
        if "perm_null" not in r: checks.append(f"- {cname}: no-cue rerun file is from an older run format (stale?) — rerun it"); continue
        checks.append(f"- {cname}: headline design with every legality-vocabulary row dropped ({r['n_rows']} rows kept): AUROC {r['test_cross_auroc']:.2f}, beat {r['perm_null']['shuffles_beaten_acc']}/{r['perm_null']['n']}.")
out.append("\n## Checks\n"); out.extend(checks or ["- (none run yet)"])
out.append("\n## Just asking the model\n")
for lab, q in (("legal", "is this illegal?"), ("harmful", "is this harmful?")):
    f = ROOT / "data/processed" / f"ask_{a.run}_{lab}.json"
    if not f.exists(): out.append(f"- '{q}': not run"); continue
    r = json.loads(f.read_text())
    out.append(f"- '{q}': {r['accuracy']:.0%} right overall, {r['offdiagonal_accuracy']:.0%} on the off-diagonal quadrants, {r['refused_or_unparsed_frac']:.0%} invalid; Yes−No logit AUROC off-diagonal {r.get('logit_auroc_offdiagonal', float('nan')):.2f}."
               + "".join(f" Check set {n}: answered yes {e['frac_answered_yes']:.0%}." for n, e in (r.get('extra_sets') or {}).items() if e.get('frac_answered_yes') is not None))
Path(ROOT / a.out).write_text("\n".join(out) + "\n", encoding="utf-8"); print("\n".join(out)); print(f"\nwrote {a.out}")
