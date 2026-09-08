"""Generic environment check — not project code, outside the clock.

Loads a model, runs one forward pass with hidden states, and prints what the probe pipeline
will see: number of layers, residual width, dtype, memory, wall time. Writes a JSON record to
data/processed/smoke_<model>.json so the pod's numbers are on disk.

Mac (8 GB):  uv run python scripts/gpu_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
Pod (24 GB): uv run python scripts/gpu_smoke.py                    # Qwen/Qwen3.5-4B
"""
import argparse, time
import torch
from common import ROOT, load_model, manifest, pick_device, pick_dtype, write_json

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--device", default="auto")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
a = p.parse_args()

device = pick_device(a.device)
dtype = pick_dtype(a.dtype, device)
print(f"torch {torch.__version__} · device {device} · dtype {dtype}")
if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB")

t0 = time.time()
tok, model = load_model(a.model, device, dtype)
t_load = time.time() - t0
cfg = model.config
n_layers = getattr(cfg, "num_hidden_layers", None)
d_model = getattr(cfg, "hidden_size", None)
print(f"loaded {a.model} in {t_load:.1f}s · layers={n_layers} · d_model={d_model} · arch={cfg.architectures}")
layer_types = getattr(cfg, "layer_types", None)
if layer_types:
    from collections import Counter
    print("layer types:", dict(Counter(layer_types)))

prompt = "Is it against the law to park in front of a fire hydrant?"
enc = tok(prompt, return_tensors="pt").to(device)
t0 = time.time()
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
t_fwd = time.time() - t0
hs = out.hidden_states
print(f"forward {t_fwd*1000:.0f} ms · hidden_states tuple len={len(hs)} (embeddings + {len(hs)-1} layers) · shape per layer={tuple(hs[-1].shape)}")
nxt = tok.decode(out.logits[0, -1].argmax())
print("next-token argmax:", repr(nxt))

mem = None
if device == "cuda":
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak cuda memory {mem:.2f} GB")

rec = manifest(model=a.model, device=device, dtype=str(dtype), load_s=round(t_load, 1), forward_ms=round(t_fwd * 1000),
               n_layers=n_layers, d_model=d_model, hidden_states_len=len(hs), layer_types=layer_types,
               peak_cuda_gb=mem, architectures=list(cfg.architectures or []), transformers=__import__("transformers").__version__)
outp = ROOT / "data/processed" / f"smoke_{a.model.replace('/', '__')}.json"
write_json(outp, rec)
print("wrote", outp.relative_to(ROOT))
