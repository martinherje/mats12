#!/usr/bin/env bash
# Colab bootstrap (uncounted setup). Idempotent. Called from notebooks/mats12_colab.ipynb.
#   bash scripts/colab_setup.sh [drive_dir]
# - installs the Python deps into Colab's existing torch environment (no uv on Colab)
# - if a Drive dir is given, keeps data/raw, data/processed, figures and journal there via symlinks so runs survive the VM
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q "transformers>=5.2" "accelerate>=1.0" "safetensors>=0.4" "huggingface-hub>=0.24" \
    "scikit-learn>=1.4" "pandas>=2.0" "matplotlib>=3.8" "tqdm>=4.66" "openai>=1.40" "python-dotenv>=1.0" "numpy>=1.26"
DRIVE=${1:-}
if [ -n "$DRIVE" ]; then
  mkdir -p "$DRIVE/data/raw" "$DRIVE/data/processed" "$DRIVE/figures" "$DRIVE/journal"
  # Untracked, gitignored output dirs are symlinked to Drive (git never looks at them).
  for d in data/raw data/processed; do
    if [ -d "$d" ] && [ ! -L "$d" ]; then cp -rn "$d/." "$DRIVE/$d/" 2>/dev/null || true; rm -rf "$d"; fi
    ln -sfn "$DRIVE/$d" "$d"
  done
  # Tracked dirs (journal, figures) stay real directories inside the repo so git can pull and stash normally.
  # Restore any newer copies from Drive (a previous session's notes), never overwriting newer local files.
  for d in journal figures; do mkdir -p "$d"; cp -rn "$DRIVE/$d/." "$d/" 2>/dev/null || true; done
  echo "data/raw and data/processed -> $DRIVE (linked); journal and figures restored from Drive if present"
fi
python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} | transformers {transformers.__version__} | cuda {torch.cuda.is_available()}",
      f"| {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB cap {torch.cuda.get_device_capability(0)}" if torch.cuda.is_available() else "")
PY
