"""Residual-stream intervention hooks: projection ablation and additive steering. Generic infrastructure.

Usage from other scripts:

    from steer import Intervention, load_direction, diff_of_means
    d = load_direction("data/processed/direction_<run>_<label>.npz", layer=20)   # unit vector, torch tensor
    with Intervention(model, layers=[20], direction=d, mode="ablate"):
        out = model.generate(**enc, max_new_tokens=400)

Modes:
  ablate  : h <- h - (h . d) d        (remove the component along d at every position)
  add     : h <- h + alpha * d        (steer along d)
  random  : same as `ablate` but with a random unit direction of the same shape (control)

Hooks attach to `model.model.layers[i]` outputs, so they work for Qwen3.5's hybrid stack
(Gated DeltaNet and attention layers alike). Layer index i here means the output of decoder
block i, which corresponds to hidden_states[i+1] in the extractor (hidden_states[0] is the
embedding output).

Sanity check: `uv run python scripts/steer.py --model Qwen/Qwen2.5-0.5B-Instruct` runs a tiny
self-test (ablating a direction removes its component; random control differs).
"""
from __future__ import annotations
import numpy as np, torch


def diff_of_means(acts: np.ndarray, y: np.ndarray, layer: int) -> np.ndarray:
    """Unit difference-of-means direction at `layer` from acts [N, L+1, d] and binary y."""
    X = acts[:, layer].astype(np.float32)
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-8)


def save_directions(path, acts: np.ndarray, y: np.ndarray, label: str, meta: dict | None = None) -> None:
    """Save unit diff-of-means directions for every layer: dirs [L+1, d]."""
    dirs = np.stack([diff_of_means(acts, y, l) for l in range(acts.shape[1])])
    np.savez_compressed(path, dirs=dirs.astype(np.float32), label=label, **(meta or {}))


def load_direction(path, layer: int, hidden_index: bool = True) -> torch.Tensor:
    """Load the unit direction for one layer. `layer` is in hidden_states indexing (0 = embeddings)
    when hidden_index=True; the hook layer is then layer-1."""
    z = np.load(path)
    return torch.tensor(z["dirs"][layer], dtype=torch.float32)


class Intervention:
    """Context manager that registers forward hooks on the chosen decoder blocks."""

    def __init__(self, model, layers, direction: torch.Tensor | None, mode: str = "ablate",
                 alpha: float = 0.0, seed: int = 0, hidden_index: bool = True):
        self.model, self.mode, self.alpha = model, mode, alpha
        # hidden_states[k] is the output of block k-1; convert if caller passes hidden-state indices
        self.layers = [l - 1 for l in layers] if hidden_index else list(layers)
        assert all(l >= 0 for l in self.layers), "layer 0 in hidden-state indexing is the embedding output; no block to hook"
        blocks = model.model.layers
        d_model = model.config.hidden_size
        if mode == "random":
            g = torch.Generator().manual_seed(seed)
            direction = torch.randn(d_model, generator=g)
        assert direction is not None and direction.numel() == d_model, f"direction must have {d_model} elements"
        self.d = (direction / direction.norm()).to(next(model.parameters()).device)
        self.blocks = [blocks[l] for l in self.layers]
        self.handles = []

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.d.to(h.dtype)
        if self.mode in ("ablate", "random"):
            proj = (h * d).sum(-1, keepdim=True)
            h_new = h - proj * d
        elif self.mode == "add":
            h_new = h + self.alpha * d
        else:
            raise ValueError(self.mode)
        return (h_new,) + tuple(output[1:]) if isinstance(output, tuple) else h_new

    def __enter__(self):
        self.handles = [b.register_forward_hook(self._hook) for b in self.blocks]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


if __name__ == "__main__":
    # Self-test. Note: output_hidden_states records block outputs BEFORE user forward hooks rewrite them, so the
    # check reads the modified output through a second hook registered after the intervention.
    import argparse, sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from common import load_model, pick_device, pick_dtype
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); a = p.parse_args()
    dev = pick_device("auto"); tok, model = load_model(a.model, dev, pick_dtype("auto", dev))
    enc = tok("The capital of Norway is", return_tensors="pt").to(dev)
    L = 4  # hidden-state index -> hooks block 3
    seen = {}
    def reader(module, inputs, output):
        seen["h"] = (output[0] if isinstance(output, tuple) else output)[0, -1].detach().float().cpu()
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True).hidden_states
    d = base[L][0, -1].float().cpu(); d = d / d.norm()
    with torch.no_grad(), Intervention(model, layers=[L], direction=d, mode="ablate"):
        hd = model.model.layers[L - 1].register_forward_hook(reader); model(**enc); hd.remove()
    before, after = float(base[L][0, -1].float().cpu() @ d), float(seen["h"] @ d)
    print(f"component along d at hidden_states[{L}], last token: before={before:.3f} after={after:.3e}")
    ok1 = abs(after) < 1e-2 * abs(before)
    with torch.no_grad(), Intervention(model, layers=[L], direction=None, mode="random"):
        hd = model.model.layers[L - 1].register_forward_hook(reader); model(**enc); hd.remove()
    delta = float((seen["h"] - base[L][0, -1].float().cpu()).norm())
    print(f"random-control changes the state by {delta:.3f} (should be > 0, and small relative to |h|={float(base[L][0,-1].float().norm()):.1f})")
    ok2 = delta > 0
    with torch.no_grad():
        t0 = tok.decode(model(**enc).logits[0, -1].argmax())
        with Intervention(model, layers=[L], direction=d, mode="ablate"):
            t1 = tok.decode(model(**enc).logits[0, -1].argmax())
        with Intervention(model, layers=[L], direction=None, mode="random"):
            t2 = tok.decode(model(**enc).logits[0, -1].argmax())
    print("next token (base / ablated-own-direction / random):", repr(t0), "/", repr(t1), "/", repr(t2))
    print("PASS" if ok1 and ok2 else "FAIL")
