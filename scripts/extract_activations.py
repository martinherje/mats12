"""Extract residual-stream activations for every scenario at every layer.

Reads a scenarios CSV (schema: data/scenarios_template.csv), runs each text through the model,
pools the hidden state at each layer, and saves one .npz with the activations plus every label
column, and a sidecar manifest JSON recording model/revision/pooling/template/git hash.

Pooling:  last  = hidden state at the last real token (default; standard for probes)
          mean  = mean over all real tokens
Template: raw   = the scenario text as-is (default)
          chat  = wrapped in the model's chat template as a user turn, no generation prompt
                  (use --instruction to prepend a fixed instruction inside the user turn)

Example (Mac pipeline check):
  uv run python scripts/extract_activations.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --scenarios data/scenarios_smoke.csv --run smoke

IN PLAIN LANGUAGE
What it does: runs each prompt through the model once and saves the model's internal state (the residual
stream) at one token position, for every layer.
Which position: the last token of the text; with --generation-prompt, the position the answer is generated from.
What comes out: data/processed/acts_<run>.npz, a block of numbers shaped (prompts, layers+1, width), plus
every label column from the scenarios CSV, so later scripts know which prompt was which.
"""
import argparse
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from common import ROOT, JURISDICTION, QUESTIONS, load_model, manifest, pick_device, pick_dtype, write_json

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--revision", default=None)
p.add_argument("--scenarios", default="data/scenarios.csv")
p.add_argument("--run", required=True, help="run name → data/processed/acts_<run>.npz")
p.add_argument("--pool", default="last", choices=["last", "mean"])
p.add_argument("--template", default="raw", choices=["raw", "chat"])
p.add_argument("--instruction", default="", help="fixed text prepended to each scenario (chat template only); a literal backslash-n is turned into a newline")
p.add_argument("--question", default="", choices=["", "legal", "harmful"], help="wrap each scenario in the exact Yes/No question ask_model.py uses (overrides --instruction)")
p.add_argument("--generation-prompt", action="store_true", help="chat template: append the assistant turn opener, so the last token is the state the answer starts from")
p.add_argument("--enable-thinking", default=None, choices=[None, "on", "off"], help="chat template: Qwen3.5 enable_thinking flag (omit to leave the template default)")
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--max-length", type=int, default=256)
p.add_argument("--device", default="auto")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
a = p.parse_args()

df = pd.read_csv(ROOT / a.scenarios)
assert "id" in df and "text" in df, "scenarios CSV needs at least id,text columns (see data/scenarios_template.csv)"
assert df["id"].is_unique, "duplicate ids in scenarios CSV"
device = pick_device(a.device); dtype = pick_dtype(a.dtype, device)
tok, model = load_model(a.model, device, dtype, a.revision)
print(f"{a.model} on {device}/{dtype} · {len(df)} scenarios · pool={a.pool} · template={a.template}")


def render(text: str) -> str:
    if a.template == "raw":
        return text
    if a.question: content = QUESTIONS[a.question].format(j=JURISDICTION, t=text)
    else:
        instr = a.instruction.replace("\\n", "\n").strip()
        content = (instr + "\n\n" + text) if instr else text
    kw = {}
    if a.enable_thinking is not None:
        kw["enable_thinking"] = a.enable_thinking == "on"
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=a.generation_prompt, **kw)


texts = [render(t) for t in df["text"].astype(str)]
acts = None
n_tokens = np.zeros(len(df), dtype=np.int32)
for i in tqdm(range(0, len(texts), a.batch_size), desc="extract"):
    batch = texts[i:i + a.batch_size]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=a.max_length).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1)  # [B, L+1, T, d]
    mask = enc["attention_mask"]                # [B, T]
    lens = mask.sum(1)                           # [B]
    if a.pool == "last":
        idx = (lens - 1).view(-1, 1, 1, 1).expand(-1, hs.shape[1], 1, hs.shape[3])
        pooled = hs.gather(2, idx).squeeze(2)    # [B, L+1, d]
    else:
        m = mask.view(mask.shape[0], 1, mask.shape[1], 1).to(hs.dtype)
        pooled = (hs * m).sum(2) / lens.view(-1, 1, 1).to(hs.dtype)
    pooled = pooled.float().cpu().numpy().astype(np.float32)   # fp32 on disk: Qwen residual streams have massive activations that can overflow fp16
    if acts is None:
        acts = np.zeros((len(df), pooled.shape[1], pooled.shape[2]), dtype=np.float32)   # was float16 until 10 Sep evening; values were finite on every run so far, but fp32 is the intent
    acts[i:i + len(batch)] = pooled
    n_tokens[i:i + len(batch)] = lens.cpu().numpy()

def as_array(s):
    return s.to_numpy() if pd.api.types.is_numeric_dtype(s) else np.array([str(x) for x in s], dtype=str)  # <U dtype, no pickle
label_cols = {c: as_array(df[c]) for c in df.columns}  # includes text, so downstream baselines need no second file
outp = ROOT / "data/processed" / f"acts_{a.run}.npz"
outp.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(outp, acts=acts, n_tokens=n_tokens, **{f"col_{k}": v for k, v in label_cols.items()})
write_json(outp.with_suffix(".json"), manifest(run=a.run, model=a.model, revision=a.revision, scenarios=a.scenarios, n=len(df),
           pool=a.pool, template=a.template, instruction=a.instruction, question=a.question, generation_prompt=a.generation_prompt, enable_thinking=a.enable_thinking, max_length=a.max_length, device=device, dtype=str(dtype),
           acts_shape=list(acts.shape), truncated=int((n_tokens >= a.max_length).sum())))
print(f"wrote {outp.relative_to(ROOT)} · acts {acts.shape} (N, layers+1, d) · truncated={int((n_tokens >= a.max_length).sum())}")
print("HAND-CHECK: open the .json manifest and confirm model/pool/template are what you intended before training anything on this.")
