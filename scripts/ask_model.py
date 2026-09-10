"""Baseline: just ask the model. Generic infrastructure.

IN PLAIN LANGUAGE
What it does: shows the model each scenario and asks a yes/no question ("Is this illegal?" or "Is this harmful?"),
then scores the answers against the labels. If simply asking matches the probe, the probe adds nothing — Nanda
names this as the baseline people forget. Also records the model's refusal rate per quadrant, which is its own
boring-explanation check (a probe that just reads "will I refuse?" is not a legality probe).
What comes out: data/raw/ask_<run>_<label>.jsonl (every answer verbatim) and data/processed/ask_<run>_<label>.json
(accuracy overall, per quadrant, and the transfer number on the same off-diagonal split the probe is tested on).

  uv run python scripts/ask_model.py --run v1 --label legal --model Qwen/Qwen3.5-4B
  uv run python scripts/ask_model.py --run v1 --label harmful --model Qwen/Qwen3.5-4B
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, QUADRANTS, load_model, manifest, pick_device, pick_dtype, write_json, QUESTIONS

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--label", required=True, choices=["legal", "harmful"])
p.add_argument("--model", default="Qwen/Qwen3.5-4B"); p.add_argument("--scenarios", default="data/scenarios.csv")
p.add_argument("--jurisdiction", default="the United States")
p.add_argument("--batch-size", type=int, default=16); p.add_argument("--device", default="auto"); p.add_argument("--dtype", default="auto")
a = p.parse_args()

QUESTION = QUESTIONS[a.label]   # defined once in common.py, shared with extract_activations.py --question
# a "Yes" to "is it illegal?" means legal=0; a "Yes" to "is it harmful?" means harmful=1
yes_means = {"legal": 0, "harmful": 1}[a.label]

df = pd.read_csv(ROOT / a.scenarios)
if "exclude" in df.columns:
    n_ex = int(df["exclude"].astype(int).sum()); df = df[df["exclude"].astype(int) == 0].reset_index(drop=True)
    print(f"dropped {n_ex} rows with exclude=1 (same rows the probe evaluation drops); {len(df)} remain")
if "set" not in df.columns: df["set"] = "main"
device = pick_device(a.device); tok, model = load_model(a.model, device, pick_dtype(a.dtype, device)); tok.padding_side = "left"
yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]; no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]
raw = ROOT / "data/raw" / f"ask_{a.run}_{a.label}.jsonl"
if raw.exists():   # never overwrite raw answers silently: keep the old file under a timestamped name
    import time; bak = raw.with_name(raw.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"); raw.rename(bak); print(f"existing {raw.name} moved to {bak.name}")
rows = []
for i in tqdm(range(0, len(df), a.batch_size), desc=f"ask {a.label}"):
    batch = df.iloc[i:i + a.batch_size]
    texts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(j=a.jurisdiction, t=t)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for t in batch.text]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1].float()
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    ly = torch.logsumexp(logits[:, yes_ids], -1); ln = torch.logsumexp(logits[:, no_ids], -1); ld = (ly - ln).cpu().tolist()
    for (_, r), o, d in zip(batch.iterrows(), out, ld):
        ans = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.match(r"\s*(yes|no)\b", ans, re.I)
        pred = (yes_means if m.group(1).lower() == "yes" else 1 - yes_means) if m else None
        refused = m is None
        rows.append({**r.to_dict(), "answer": ans, "pred": pred, "refused_or_unparsed": refused, "yes_minus_no_logit": float(d), "model": a.model, "label_asked": a.label})
with open(raw, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
res_all = pd.DataFrame(rows)
# summary numbers on the main-design rows only; the simple / negated check sets are reported separately below
extra_sets = {}
if "set" in res_all.columns:
    for name in sorted(set(res_all["set"].astype(str)) - {"main"}):
        sub = res_all[res_all["set"].astype(str) == name]; okk = sub.pred.notna()
        extra_sets[name] = {"n": int(len(sub)), "frac_answered_yes": float(sub.pred[okk].astype(int).mean()) if okk.any() else None, "frac_invalid": float((~okk).mean())}
    res = res_all[res_all["set"].astype(str) == "main"].reset_index(drop=True)
else:
    res = res_all
y = res[a.label].astype(int)
ok = res.pred.notna()
acc = float((res.pred[ok].astype(int) == y[ok]).mean()) if ok.any() else float("nan")
per_q = {q: float((res.pred[ok & (res.quadrant == q)].astype(int) == y[ok & (res.quadrant == q)]).mean()) for q in QUADRANTS}
offdiag = ok & res.quadrant.isin(["illegal_harmless", "legal_harmful"])
from sklearn.metrics import roc_auc_score
# logit-difference AUROC: the model's own graded judgement as a classifier (a "Yes" to "is it illegal?" means legal=0)
score = -res.yes_minus_no_logit if a.label == "legal" else res.yes_minus_no_logit
offd = res.quadrant.isin(["illegal_harmless", "legal_harmful"])
logit_auroc_all = float(roc_auc_score(y, score)); logit_auroc_offdiag = float(roc_auc_score(y[offd], score[offd]))
by_topic = {tp: None for tp in res.topic.unique()}
summary = manifest(run=a.run, label=a.label, model=a.model, n=len(res), accuracy=acc, per_quadrant=per_q, extra_sets=extra_sets,
                   logit_auroc_all=logit_auroc_all, logit_auroc_offdiagonal=logit_auroc_offdiag,
                   offdiagonal_accuracy=float((res.pred[offdiag].astype(int) == y[offdiag]).mean()) if offdiag.any() else None,
                   refused_or_unparsed_frac=float(res.refused_or_unparsed.mean()),
                   refusal_by_quadrant={q: float(res.refused_or_unparsed[res.quadrant == q].mean()) for q in QUADRANTS})
write_json(ROOT / "data/processed" / f"ask_{a.run}_{a.label}.json", summary)
print(json.dumps({k: v for k, v in summary.items() if k not in ("timestamp", "git")}, indent=1))
print(f"wrote {raw.relative_to(ROOT)}")
print("HAND-CHECK: read the answers for the off-diagonal quadrants; an unparsed answer is not a wrong answer, look at what it said.")
