"""Shared helpers for the activation pipeline. Generic infrastructure, not project code."""
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
        return torch.bfloat16 if device == "cuda" else torch.float32
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
