# MATS 12 application task — workspace

Working environment for the Neel Nanda MATS 12 application task (due **Fri 11 Sep 2026** (extension granted; original 4 Sep)). The research in this repo is Martin's own work; this scaffold (folders, env config, smoke test, journal templates) is generic infrastructure, which Nanda's rules place **outside** the 16–20 h clock — as is all general learning done before project work starts.

Vault companion note (strategy, form questions, project spec, calibration): `~/Documents/PhDAI/plans/applications/MATS 12 - Application Task Plan.md`

## The clock (Nanda's rules, condensed)

- Counted: anything actively toward the project — coding, project-chosen reading, analysis, thinking/planning, writing the doc. Cap 20 h; target ~16; ≤5 h of it on reading.
- Not counted: general prep/learning before picking the problem; generic tech setup (this repo, API keys, connectivity); breaks; waiting on runs; the MATS form answers.
- +2 h extra allowed for the executive summary (no new experiment code in those hours; new graphs from existing data OK).
- Full pivot to a new project = clock resets.
- Track with Toggl from the first project-directed minute; screenshot goes in the application doc.

## Environment (updated 8 Sep — GPU/activation path)

The project is the **legality probe** (spec in the vault: `plans/applications/MATS 12 - Project Spec (Legality Probe).md`; design sheet here: `journal/design-questions-legality-probe.md`). It needs residual-stream activations from **Qwen/Qwen3.5-4B**, so the box is a rented 24 GB GPU — see `RUNPOD.md`. The API-only path below is kept for the "just ask the model" baseline.

Pipeline (all generic, all outside the clock): `scripts/gpu_smoke.py` → `scripts/validate_scenarios.py` → `scripts/extract_activations.py` → `scripts/train_probe.py` → `scripts/sync_from_pod.sh`. The Mac runs the same pipeline on a 0.5B model to check the plumbing; the numbers only mean anything on the pod. `data/scenarios_smoke.csv` is a plumbing test with arbitrary labels and is never project data. Python is pinned to 3.12 (`.python-version`) because torch does not yet ship for 3.14.

## One-time setup (Martin does these — accounts and payments are yours)

1. Create an OpenRouter account (openrouter.ai) → add a few dollars of credit → create an API key.
2. `cp .env.example .env` and paste the key. `.env` is gitignored — keys never enter git.
3. `uv sync` (creates `.venv` with openai/dotenv/pandas/matplotlib/tqdm).
4. `uv run python scripts/api_smoke.py` — verifies the pipe: one tiny call to `openai/gpt-oss-120b`, prints whether the **reasoning field** comes back visible. Reasoning visibility is load-bearing for any CoT-reading project; if a provider hides it, we pin a provider that doesn't.
5. (Only if the project ends up needing CoT partial-refill/resampling: Nebius account, same pattern.)

## Layout

- `scripts/` — experiment code (Martin's). `api_smoke.py` is the only pre-provided file, and it is deliberately trivial.
- `data/raw/` — every raw transcript/rollout saved verbatim, named by run. Gitignored (size), never deleted.
- `data/processed/` — derived tables.
- `figures/` — PNGs (every plot saved to disk, not just shown).
- `journal/highlights.md` — the running doc Nanda's process expects: hypotheses, key graphs, dead ends, in order.
- `journal/verification-log.md` — what was checked, how, and how surprising an error would be. **The form asks for exactly this** ("which parts you did and didn't check... how surprised you'd be to discover a major error in each part") — keep it as you go and the answer writes itself.
- `journal/hours.md` — Toggl backup ledger.

## Reference points

- Application doc snapshot: vault `raw/MATS 12 - Nanda application doc snapshot 2026-08-10.md`
- Past accepted applications + admissions FAQ: vault `raw/MATS 12 - past application examples and admissions FAQ 2026-08-10.md`
- Nanda's 600k-token mech-interp context file (for LLM context, if used): linked from the application doc ("this default file")
- Model notes: `openai/gpt-oss-120b` on OpenRouter — $0.03/$0.17 per M tokens, 131k context, CoT access; Qwen 3.5-27B as contrast model; OpenAI Model Spec at model-spec.openai.com
