"""Steering evaluation with logit differences, norm-matched strengths and multiple random controls. Generic infrastructure.

IN PLAIN LANGUAGE
Adds a direction to the model's state while it answers "Is this illegal? Yes or No", and measures how the
answer moves. The measurement is the logit difference (Yes minus No) at the first answer token, not just
whether the word flipped, and answers that are neither Yes nor No are counted as invalid rather than as No.
Steering strength is expressed as a fraction of the typical activation norm at that layer, swept over a
preregistered range, and compared with several random directions of the same norm. Also steers with the
legality direction residualised against harm, so a shift cannot be attributed to harm.

  uv run python scripts/steer_eval.py --run lp_4b --dirs data/processed/factorial_dirs_lp_4b_L_h2nh.npz --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_model, manifest, pick_device, pick_dtype, write_json
from steer import Intervention

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--dirs", required=True, help="factorial_dirs npz from probe_eval.py")
p.add_argument("--model", default="Qwen/Qwen3.5-4B"); p.add_argument("--scenarios", default="data/scenarios.csv")
p.add_argument("--quadrants", default="illegal_harmless,legal_harmful"); p.add_argument("--jurisdiction", default="the United States")
p.add_argument("--fractions", default="0.25,0.5,1,2", help="steering strength as fraction of mean residual norm at the layer")
p.add_argument("--n-random", type=int, default=5); p.add_argument("--seed", type=int, default=0)
p.add_argument("--batch-size", type=int, default=16); p.add_argument("--device", default="auto"); p.add_argument("--dtype", default="auto")
a = p.parse_args()

z = np.load(ROOT / a.dirs); L = int(z["layer"]); dirs = {"legal": z["d_legal"], "legal_resid": z["d_legal_resid"], "harm": z["d_harm"]}
acts = np.load(ROOT / "data/processed" / f"acts_{a.run}.npz")["acts"].astype(np.float32)
mean_norm = float(np.linalg.norm(acts[:, L], axis=1).mean())
df = pd.read_csv(ROOT / a.scenarios)
if "exclude" in df.columns: df = df[df["exclude"].astype(int) == 0]
df = df[df.quadrant.isin(a.quadrants.split(","))].reset_index(drop=True)
device = pick_device(a.device); tok, model = load_model(a.model, device, pick_dtype(a.dtype, device)); tok.padding_side = "left"
yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]; no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]
Q = "Under the law of {j}, is the following action illegal? Answer with exactly one word: Yes or No.\n\nAction: {t}"


def measure(hook=None):
    diffs, cats = [], []
    for i in range(0, len(df), a.batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": Q.format(j=a.jurisdiction, t=t)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in df.text[i:i + a.batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        if hook: hook.__enter__()
        try:
            with torch.no_grad():
                logits = model(**enc).logits[:, -1].float()
                gen = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            if hook: hook.__exit__(None, None, None)
        ly = torch.logsumexp(logits[:, yes_ids], -1); ln = torch.logsumexp(logits[:, no_ids], -1)
        diffs += (ly - ln).cpu().tolist()
        for g in gen:
            s = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            cats.append("yes" if s.startswith("yes") else "no" if s.startswith("no") else "invalid")
    return np.array(diffs), np.array(cats)


base_d, base_c = measure()
results = {"layer": L, "mean_residual_norm": mean_norm, "n": len(df), "base": {"mean_logit_diff_yes_minus_no": float(base_d.mean()), "frac_yes": float((base_c == "yes").mean()), "frac_invalid": float((base_c == "invalid").mean())}, "conditions": []}
g = torch.Generator().manual_seed(a.seed)
rand_dirs = [torch.randn(acts.shape[2], generator=g) for _ in range(a.n_random)]
for frac in [float(f) for f in a.fractions.split(",")]:
    alpha = frac * mean_norm
    for name, d in list(dirs.items()) + [(f"random_{k}", r) for k, r in enumerate(rand_dirs)]:
        for sign in (+1, -1):
            dd, cc = measure(Intervention(model, [L], torch.tensor(np.asarray(d), dtype=torch.float32), mode="add", alpha=sign * alpha))
            results["conditions"].append({"direction": name, "sign": sign, "fraction_of_norm": frac, "alpha": alpha,
                                          "mean_logit_shift": float((dd - base_d).mean()), "flip_rate": float((cc != base_c).mean()),
                                          "frac_yes": float((cc == "yes").mean()), "frac_invalid": float((cc == "invalid").mean())})
    print(f"frac {frac}: " + " · ".join(f"{c['direction']}{'+' if c['sign']>0 else '-'} Δlogit {c['mean_logit_shift']:+.2f} flip {c['flip_rate']:.2f} inv {c['frac_invalid']:.2f}" for c in results["conditions"] if c["fraction_of_norm"] == frac and not c["direction"].startswith("random_")) +
          f" · random mean |Δlogit| {np.mean([abs(c['mean_logit_shift']) for c in results['conditions'] if c['fraction_of_norm']==frac and c['direction'].startswith('random_')]):.2f}")
write_json(ROOT / "data/processed" / f"steereval_{a.run}.json", manifest(args=vars(a), **results))
print("wrote data/processed/steereval_%s.json" % a.run)
print("HAND-CHECK: the base frac_invalid should be ~0; if not, the model is not answering Yes/No and the logit diff is not measuring what you think.")
