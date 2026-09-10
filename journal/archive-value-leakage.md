# Archived alternative (8 Sep 2026, not pursued): value-leakage mechanism

Moved here from `README.md` on 10 Sep so the README describes only the project that ran. Nothing below is part of the illegality-probe work; the scripts it names were removed from `scripts/` on 10 Sep and survive in git history.

**Where does the value intervene?** Mechanism behind Betley et al. 2026, *Value Leakage* (arXiv 2607.14345), Donation Bet task, on Qwen3.5-9B. Primer (read first): vault `plans/applications/MATS 12 - Value Leakage Primer.md` (private notes). Design sheet: `journal/design-questions-value-leakage.md`. Paper code (sparse clone, no data): `data/reference/value_leakage/`; the exact prompt templates and nine questions are in `data/donation_bet_questions.json`.

**Browser route (preferred, 9 Sep):** open `notebooks/mats12_colab.ipynb` in Google Colab ([direct link](https://colab.research.google.com/github/martinherje/mats12/blob/main/notebooks/mats12_colab.ipynb); authorise Colab's GitHub access for the private repo, or upload the notebook by hand), pick a GPU runtime, run the cells top to bottom. Cell 2 mounts Drive so `data/`, `figures/` and `journal/` survive the VM; a fine-grained GitHub token in Colab Secrets (`GITHUB_TOKEN`) lets cell 2 clone the private repo and cell 11 push the journal back. A free T4 runs Qwen3.5-4B in fp16; L4/A100 in bf16.

**PC route (alternative):** (after `uv sync` and the smoke test): `.\scripts\gonogo.ps1` runs the concrete, abstract and equal variants on Qwen3.5-4B with thinking off and on and prints the three bias tables.

Pipeline:
1. `scripts/donation_bet.py` — replication + interventions. `--backend api` runs the same Qwen3.5-9B through OpenRouter (needs `OPENROUTER_API_KEY` in `.env`) for the go/no-go from the Mac; `--backend local` runs on the pod and supports `--ablate`. Writes raw rollouts to `data/raw/`, bias + bootstrap CI to `data/processed/`, and `data/scenarios_<run>.csv` for step 2.
2. `scripts/extract_activations.py --scenarios data/scenarios_<run>.csv --template chat --generation-prompt --enable-thinking off --run <run>` — residual stream at the last prompt token, all layers.
3. `scripts/make_direction.py --run <run> --label good_side --filter "bet==1" --out direction_<run>_good_side` (+ `--label bet` for the topic control).
4. `scripts/donation_bet.py --backend local --reuse-thresholds <run> --ablate data/processed/direction_<run>_good_side.npz --layers <L> --mode ablate|random --run <run>_abl_L<L>` — the causal move and its controls.
5. `scripts/train_probe.py` — optional layer sweep / held-out AUROC if you go the per-sample route.

Self-tests on the Mac (0.5B stand-in): `scripts/steer.py` and `scripts/donation_bet.py --backend local --model Qwen/Qwen2.5-0.5B-Instruct ... --run plumb` both passed; numbers from the stand-in mean nothing.

## One-time setup (only needed for the OpenRouter path of this alternative)

1. Create an OpenRouter account (openrouter.ai), add a few dollars of credit, create an API key.
2. `cp .env.example .env` and paste the key. `.env` is gitignored; keys never enter git.
3. `uv sync` (creates `.venv` with openai/dotenv/pandas/matplotlib/tqdm).
4. `uv run python scripts/api_smoke.py` verified the pipe: one tiny call to `openai/gpt-oss-120b`, printing whether the reasoning field comes back visible. Any CoT-reading project needs the reasoning field visible; if a provider hides it, pin one that does not.
5. (Only if the project ends up needing CoT partial-refill/resampling: Nebius account, same pattern.)

## Model notes

`openai/gpt-oss-120b` on OpenRouter: $0.03/$0.17 per M tokens, 131k context, CoT access; Qwen 3.5-27B as contrast model; OpenAI Model Spec at model-spec.openai.com. `RUNPOD.md` (removed 10 Sep, in git history) was the rented-GPU alternative to Colab.
