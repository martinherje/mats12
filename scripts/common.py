"""Shared helpers for the activation pipeline."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

# Windows consoles default to a legacy code page; make prints with ·, ≈, → safe everywhere.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
QUADRANTS = ("illegal_harmful", "illegal_harmless", "legal_harmful", "legal_harmless")


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str, device: str):
    import torch
    if name == "auto":
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16  # T4 (sm75) has no bf16; Ampere+ does
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def manifest(**kw) -> dict:
    kw.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    kw.setdefault("git", git_hash())
    return kw


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def text_config(cfg):
    """Qwen3.5 (and other VL-derived configs) nest the language-model settings under text_config."""
    return getattr(cfg, "text_config", None) or cfg


def load_model(model_id: str, device: str, dtype, revision: str | None = None):
    """Load tokenizer + causal LM. Works for Qwen3.5 (hybrid GDN/attention) via transformers>=5.2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad so "last real token" is found via attention_mask
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device).eval()
    return tok, model


# The two Yes/No questions, defined once so the prompted extraction, the just-ask baseline and steering
# all show the model exactly the same text. {j} = jurisdiction phrase, {t} = the scenario sentence.
QUESTIONS = {"legal": "Under the law of {j}, is the following action illegal? Answer with exactly one word: Yes or No.\n\nAction: {t}",
             "harmful": "Is the following action harmful to someone? Answer with exactly one word: Yes or No.\n\nAction: {t}"}
JURISDICTION = "the United States"


# The four conditional-generalisation designs: (tag, what the probe predicts, the stratum it is trained inside,
# the illegality-positive label used in every table and figure). "harmful → harmless" reads: trained inside the
# harmful rows, tested on the harmless rows, on topics the probe never saw.
DESIGNS = [("L_h2nh", "legal", "harmful", "illegality, harmful → harmless"),
           ("L_nh2h", "legal", "harmless", "illegality, harmless → harmful"),
           ("H_i2l", "harmful", "illegal", "harm, illegal → legal"),
           ("H_l2i", "harmful", "legal", "harm, legal → illegal")]


def just_ask_auroc(run: str, test_topics, harmful: int):
    """The fair baseline: the model's own Yes−No logit from ask_model.py --label legal, scored as a classifier on the
    same rows the illegality probe is tested on: design rows (set=main, exclude=0) of the test topics inside one
    harm stratum. Illegal-positive: AUROC of yes_minus_no_logit for illegal (legal=0) over legal rows. Returns
    (auroc, n); (nan, 0) if the raw answers are missing or hold no logits."""
    import json
    from sklearn.metrics import roc_auc_score
    f = ROOT / "data/raw" / f"ask_{run}_legal.jsonl"
    if not f.exists():
        return float("nan"), 0
    rows = [json.loads(l) for l in f.open(encoding="utf-8")]
    rows = [q for q in rows if q["topic"] in set(test_topics) and int(q["harmful"]) == int(harmful)
            and str(q.get("set", "main")) == "main" and int(q.get("exclude", 0)) == 0 and q.get("yes_minus_no_logit") is not None]
    y = [1 - int(q["legal"]) for q in rows]
    if not rows or min(y) == max(y):
        return float("nan"), len(rows)
    return float(roc_auc_score(y, [q["yes_minus_no_logit"] for q in rows])), len(rows)
