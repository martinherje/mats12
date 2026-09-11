"""results_table.py — journal/results.md, the tables of record, copied from data/processed/*.json.

IN PLAIN LANGUAGE
Reads every probeeval_<run>[_prompted]_<tag>.json and ask_<run>_<label>.json that exists and writes
journal/results.md: the main table (one row per probe and condition: AUROC on the other stratum, how many of the
100 shuffled-label runs it beat on AUROC, accuracy at the fitted cut-off with its interval, the same count on
accuracy, and word count alone), a two-row directions table (cosine at the chosen layer, the label-swap band
there, the illegality direction within the harmless rows before and after the harm direction is projected out,
and the illegality direction predicting harm), the check-set counts, the no-cue rerun and the just-ask baseline.
Every number is copied from a JSON file except one: the fair-baseline line, which scores the model's own Yes−No
logit on the same 30 held-out rows the illegality probe is tested on (common.just_ask_auroc).
Signs: the JSON files are legal-positive (d_legal, cos_dlegal_dharm, frac_predicted_legal_1). Everything printed
here is illegality-positive: d_illegal = −d_legal, so the stored cosine is printed as its negative and
"d_legal predicting harm" as 1 − the stored value.

  uv run python scripts/results_table.py --run lp_4b
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, DESIGNS, just_ask_auroc

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--out", default="journal/results.md"); a = p.parse_args()
CONDS = ((a.run, "sentence only"), (f"{a.run}_prompted", "sentence + question"))
J = lambda cond, tag: (lambda f: json.loads(f.read_text()) if f.exists() else None)(ROOT / "data/processed" / f"probeeval_{cond}_{tag}.json")
cnt = lambda frac, n: int(round(frac * n))


def ranges(layers):
    """[9] -> 'layer 9'; [14, 15, ..., 32] -> 'layers 14–32'; gaps become commas."""
    runs, start, prev = [], layers[0], layers[0]
    for l in layers[1:] + [None]:
        if l is not None and l == prev + 1: prev = l; continue
        runs.append(f"{start}" if start == prev else f"{start}–{prev}")
        if l is not None: start = prev = l
    return ("layer " if len(layers) == 1 else "layers ") + ", ".join(runs)

out = [f"# Results — run `{a.run}` (copied from data/processed/*.json; recompute by hand before quoting)\n",
       "| condition | probe: trained → tested | AUROC | beaten by AUROC (null p95) | accuracy at fitted cut-off (95% CI) | beaten by accuracy (null p95) | word count alone |",
       "|---|---|---|---|---|---|---|"]
one_sided, checks, harm_fracs, nocue, wc_beats = [], [], [], [], []
n_test_rows = None; n_perm = 100
for cond, cname in CONDS:
    for tag, target, stratum, label in DESIGNS:
        r = J(cond, tag)
        if r is None: out.append(f"| {cname} | {label} | not run | | | | |"); continue
        pn = r["perm_null"]; lo, hi = r["test_cross_acc_ci95"]; n_test_rows = 2 * len(r["split"]["test_topics"]); n_perm = pn["n"]
        out.append(f"| {cname} | {label} | {r['test_cross_auroc']:.2f} | {pn['shuffles_beaten_auroc']}/{pn['n']} ({pn['auroc_p95']:.2f}) | "
                   f"{r['test_cross_acc']:.0%} ({lo:.0%}–{hi:.0%}) | {pn['shuffles_beaten_acc']}/{pn['n']} ({pn['acc_p95']:.0%}) | {r.get('length_only_test_auroc', float('nan')):.2f} |")
        if r.get("length_only_test_auroc", 0) >= r["test_cross_auroc"]:
            wc_beats.append(f"the {cname} {label} rows ({r['length_only_test_auroc']:.2f} against {r['test_cross_auroc']:.2f})")
        if lo == hi == r["test_cross_acc"]:
            tested = "harmless" if stratum == "harmful" else "harmful" if stratum == "harmless" else "legal" if stratum == "illegal" else "illegal"
            one_sided.append(f"In the {cname} condition the {label} probe put all {n_test_rows} {tested} test rows on the same side of its cut-off "
                             f"(accuracy {r['test_cross_acc']:.0%}, interval {lo:.0%}–{hi:.0%}); its ranking of the same rows is {r['test_cross_auroc']:.2f}. "
                             f"The cut-off fitted among {stratum} rows does not carry to {tested} rows.")
        ex = r.get("extra_sets") or {}
        if target == "legal" and ex:
            parts = []
            for name, words in (("simple", "plain acts"), ("negated", "negated")):
                if name in ex:
                    e = ex[name]; k = f"frac_predicted_{target}_1"
                    parts.append(f"{words} ({e['n']}, never trained on): {cnt(e[k], e['n'])} of {e['n']} called legal")
            checks.append(f"- {cname}, {label}: " + "; ".join(parts) + ".")
        elif ex:
            harm_fracs += [e[f"frac_predicted_{target}_1"] for e in ex.values()]
    r = J(cond, "L_h2nh_nocue"); r0 = J(cond, "L_h2nh")
    if r is not None and r0 is not None and "perm_null" in r:
        nocue.append(f"- {cname}: {r0['n_rows'] - r['n_rows']} of {r0['n_rows']} design rows contain a legality word; with them dropped ({r['n_rows']} rows) "
                     f"the harmful → harmless illegality probe scores AUROC {r['test_cross_auroc']:.2f}, beat {r['perm_null']['shuffles_beaten_auroc']}/{r['perm_null']['n']} on AUROC.")
out.append("")
out.append(f"The arrow reads trained inside → tested on; {n_test_rows // 2 if n_test_rows else 15} test topics × 2 rows of the tested stratum = {n_test_rows or 30} test sentences per design; "
           f"all eight designs share the same test topics (seed 0). The beaten count is how many of the {n_perm} shuffled-label runs of the whole procedure the probe scored above; "
           "the pre-registered count is the AUROC one.")
out.append("The word-count column is the AUROC of sentence length for the label the JSON is written in (legal, or harmful): above 0.5 means longer sentences read as legal "
           "(or harmful), below 0.5 the reverse." + (f" Word count alone reaches the probe's own AUROC on {'; '.join(wc_beats)}, so " +
           ("that row is not evidence on its own." if len(wc_beats) == 1 else "those rows are not evidence on their own.") if wc_beats else ""))
out.extend(one_sided)

# directions, from the headline design's JSON in each condition
out.append("\n## Directions\n")
out.append("| condition | layer | cos(d_illegal, d_harm) | label-swap band at that layer | illegality direction within harmless: as is → top harm component projected out | illegality direction predicting harm |")
out.append("|---|---|---|---|---|---|")
band_lines = []
for cond, cname in CONDS:
    r = J(cond, "L_h2nh")
    if r is None: out.append(f"| {cname} | not run | | | | |"); continue
    fa = r["factorial"]; c = r["cosine_curve"]; i = c["layers"].index(r["layer"])
    out.append(f"| {cname} | {r['layer']} | {-fa['cos_dlegal_dharm']:+.2f} | {-c['null_hi'][i]:+.2f} to {-c['null_lo'][i]:+.2f} | "
               f"{fa['dlegal_auroc_within_harmless']:.2f} → {fa.get('dlegal_minus_harm_top1_auroc_within_harmless', float('nan')):.2f} | "
               f"{1 - fa['dlegal_auroc_predicting_harm_test']:.2f} |")
    cos = [-v for v in c["cos_dlegal_dharm"]]; top = [-v for v in c["null_lo"]]; bot = [-v for v in c["null_hi"]]
    outside = [l for l, v, t, b in zip(c["layers"], cos, top, bot) if v > t or v < b]
    late = [l for l in c["layers"] if l > len(c["layers"]) // 2]
    band_lines.append(f"- {cname}: the curve is outside the band at {len(outside)} of {len(c['layers'])} layers"
                      + (f" ({ranges(outside)})" if outside else "")
                      + f"; over layers {late[0]}–{late[-1]} it runs {min(cos[l] for l in late):+.2f} to {max(cos[l] for l in late):+.2f} "
                      f"and the band's upper edge runs {min(top[l] for l in late):+.2f} to {max(top[l] for l in late):+.2f}.")
out.append("")
out.append("A cosine of +1 means the illegality and harm directions coincide; the band is the 95% range of the same cosine with the legality labels swapped within each topic × harm cell "
           "(the JSON stores cos(d_legal, d_harm); the sign is flipped here). The last column is the AUROC of the illegality direction for harmful vs harmless on the test rows, "
           "printed as 1 − the stored d_legal value. Projection is k = 1 (the harm direction itself).")
out.extend(band_lines)

out.append("\n## Checks\n"); out.extend(checks or ["- (none run yet)"])
if harm_fracs:
    out.append(f"- the harm probes call {min(harm_fracs):.0%} to {max(harm_fracs):.0%} of the check rows harmful (plain acts and negations, both conditions).")
out.extend(nocue)

# just asking the model
out.append("\n## Just asking the model\n")
QUAD = (("illegal_harmful", "illegal-harmful"), ("illegal_harmless", "illegal-harmless"), ("legal_harmful", "legal-harmful"), ("legal_harmless", "legal-harmless"))
for lab, q, word in (("legal", "is this illegal?", "legal"), ("harmful", "is this harmful?", "harmful")):
    f = ROOT / "data/processed" / f"ask_{a.run}_{lab}.json"
    if not f.exists(): out.append(f"- '{q}': not run"); continue
    r = json.loads(f.read_text()); pq = r["per_quadrant"]
    line = (f"- '{q}': right on {r['accuracy']:.0%} of the design rows; by quadrant " + ", ".join(f"{name} {pq[k]:.0%}" for k, name in QUAD)
            + f"; {r['offdiagonal_accuracy']:.0%} on the off-diagonal rows; {r['refused_or_unparsed_frac']:.0%} unparsed. "
            f"Yes−No logit AUROC pooled over both strata, all off-diagonal rows: {r.get('logit_auroc_offdiagonal', float('nan')):.2f} (not the fair comparison; see below).")
    ex = r.get("extra_sets") or {}
    parts = []
    for name, words in (("simple", "plain acts"), ("negated", "negations")):
        e = ex.get(name) or {}
        v = e.get("frac_answered_yes") if e.get("frac_answered_yes") is not None else e.get(f"frac_predicted_{lab}_1")   # both keys = mean(pred) = fraction called label=1
        if v is not None: parts.append(f"{v:.0%} of the {words} {word}")
    if parts: line += " The model calls " + " and ".join(parts) + "."
    out.append(line)

# the fair baseline: the model's own logit on the exact rows the illegality probes are tested on (same split in both conditions, seed 0)
r0 = J(a.run, "L_h2nh") or J(f"{a.run}_prompted", "L_h2nh")
if r0 is not None:
    tt = r0["split"]["test_topics"]
    au0, n0 = just_ask_auroc(a.run, tt, harmful=0); au1, n1 = just_ask_auroc(a.run, tt, harmful=1)
    if n0:
        pr = lambda tag: " and ".join(f"{J(c, tag)['test_cross_auroc']:.2f} ({n})" for c, n in CONDS if J(c, tag) is not None)
        out.append(f"\nFair baseline: on the same {n0} held-out harmless-stratum sentences the harmful → harmless illegality probe is tested on, the model's own "
                   f"Yes−No logit sorts illegal from legal at AUROC {au0:.2f}; the probe scores {pr('L_h2nh')}. "
                   f"On the {n1} harmful-stratum rows of the reverse test: model {au1:.2f}, probes {pr('L_nh2h')}.")
    else:
        out.append("\nFair baseline: not computed (data/raw/ask_<run>_legal.jsonl missing or without logits).")
Path(ROOT / a.out).write_text("\n".join(out) + "\n", encoding="utf-8"); print("\n".join(out)); print(f"\nwrote {a.out}")
