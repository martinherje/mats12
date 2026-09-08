# GPU box — your own 16 GB card (preferred) or a rented Runpod (uncounted setup)

## Option A: the Windows PC with 16 GB VRAM (no rental)

Qwen3.5-4B in bf16 is ~8 GB of weights; generation and activation extraction fit comfortably in 16 GB. Qwen3.5-9B (~18 GB bf16) does not fit unquantised; if the go/no-go shows 9B leaks and 4B does not, either run 9B in 8-bit (`--load-8bit`, needs bitsandbytes) or fall back to Option B for that one model.

1. Get the repo onto the PC (private GitHub remote, or copy the folder; `data/raw`, `data/processed`, `.venv` and `data/reference` need not travel).
2. In PowerShell, from the repo folder:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
uv sync                      # pulls the CUDA 12.8 torch build on Windows via pyproject's index
uv run python scripts/gpu_smoke.py --model Qwen/Qwen3.5-4B
```
The smoke prints the CUDA device, layer count, width and peak memory. Then every pipeline command below runs unchanged with `--model Qwen/Qwen3.5-4B` (local backend) and no sync step.
3. Model cache: set `HF_HOME` to a drive with ~10 GB free if `C:` is tight.

## Option B: Runpod (only if a larger model is needed)

Renting the box needs your account and card, so those steps are yours. Everything after step 3 is scripted.

## 1. Rent (you)
- runpod.io → Pods → Deploy. Pick a **24 GB** card: RTX 4090 / RTX A5000 / L4 are all fine for Qwen3.5-4B (bf16 ≈ 8 GB) and 9B (≈ 18 GB). 27B needs an 80 GB card (A100/H100) — only if hours allow.
- Template: the official **Runpod PyTorch 2.x / CUDA 12.x** image. Add a **network volume** (50 GB) mounted at `/workspace` so the model cache survives stop/start.
- Expose **SSH over TCP**. Copy the printed `ssh root@<ip> -p <port>` line into `.env` as `POD_SSH=root@<ip> -p <port>`.
- Stop the pod when not running anything; the volume keeps the weights.

## 2. First connection (one-time, on the pod)
```bash
ssh root@<ip> -p <port>
cd /workspace && git clone <this repo's remote, or rsync it up: see below> mats12 && cd mats12
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 && uv sync
export HF_HOME=/workspace/hf   # cache on the volume; add to ~/.bashrc
uv run python scripts/gpu_smoke.py           # downloads Qwen/Qwen3.5-4B (~8 GB), prints layers/d_model/peak memory
```
If there is no git remote yet, push the repo from the Mac first, or upload it directly:
```bash
rsync -avz -e "ssh -p <port>" --exclude .venv --exclude data/raw --exclude reading ~/mats12/ root@<ip>:/workspace/mats12/
```
Optional (faster GDN layers, CUDA only): `uv sync --extra cuda-kernels`. If it fails to build, skip it — results are identical, only slower.

## 3. Run the value-leakage pipeline (on the pod)
```bash
uv run python scripts/donation_bet.py --backend local --run gonogo_pod --n-baseline 20 --n-per-direction 20 --think both
uv run python scripts/extract_activations.py --scenarios data/scenarios_gonogo_pod.csv --template chat --generation-prompt --enable-thinking off --run gonogo_pod
uv run python scripts/make_direction.py --run gonogo_pod --label good_side --filter "bet==1" --think off --out direction_gonogo_good_side
uv run python scripts/donation_bet.py --backend local --think off --reuse-thresholds gonogo_pod --ablate data/processed/direction_gonogo_good_side.npz --layers 20 --mode ablate --run abl_L20
uv run python scripts/donation_bet.py --backend local --think off --reuse-thresholds gonogo_pod --layers 20 --mode random --run abl_L20_random
```
Then on the Mac: `scripts/sync_from_pod.sh`.

## 3b. Legality-probe pipeline (fallback)
```bash
uv run python scripts/validate_scenarios.py data/scenarios.csv
uv run python scripts/extract_activations.py --scenarios data/scenarios.csv --run v1_last_raw
uv run python scripts/train_probe.py --run v1_last_raw --label legal \
    --train illegal_harmful,legal_harmless --test illegal_harmless,legal_harmful --contrast harmful --control --bow
```
Then on the Mac: `scripts/sync_from_pod.sh` pulls `data/processed/`, `figures/`, `journal/` back.

## Cost sanity
A 24 GB card is roughly $0.3–0.7/h. Extraction for ~1,000 short scenarios on a 4B model is minutes; the probe sweep runs on CPU. Budget ≈ $5–15 for the whole task including re-runs; the model download is the slow part (once, on the volume).
