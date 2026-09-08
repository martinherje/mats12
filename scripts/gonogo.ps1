# Go/no-go for the value-leakage project on the local GPU (Windows PowerShell). Uncounted setup wrapper;
# the runs it launches ARE on the clock. Usage from the repo folder:
#   .\scripts\gonogo.ps1                      # Qwen3.5-4B, 20/20 samples, thinking off+on, three variants
#   .\scripts\gonogo.ps1 -Model Qwen/Qwen3.5-9B -N 10
param([string]$Model = "Qwen/Qwen3.5-4B", [int]$N = 20, [string]$Think = "both", [string]$Tag = "4b")
$ErrorActionPreference = "Stop"
$run = "gonogo_$Tag"
Write-Host "== 1/3 concrete variant (baseline + thresholds) ==" -ForegroundColor Cyan
uv run python scripts/donation_bet.py --backend local --model $Model --run $run --variant concrete_amf_kw --think $Think --n-baseline $N --n-per-direction $N
Write-Host "== 2/3 abstract variant (reusing thresholds) ==" -ForegroundColor Cyan
uv run python scripts/donation_bet.py --backend local --model $Model --run "${run}_abstract" --variant abstract --think $Think --n-per-direction $N --reuse-thresholds $run
Write-Host "== 3/3 equal-charity control (reusing thresholds) ==" -ForegroundColor Cyan
uv run python scripts/donation_bet.py --backend local --model $Model --run "${run}_equal" --variant equal_dwb_imc --think $Think --n-per-direction $N --reuse-thresholds $run
Write-Host "`n== headline: balanced bias per variant (0 = no leak; expect concrete > equal ~ 0) ==" -ForegroundColor Green
uv run python -c "import json; [print(f'{r:26s}', {k: ('bias', None if v['bias'] is None else round(v['bias'],3), 'ci', None if v['ci_lo'] is None else (round(v['ci_lo'],2), round(v['ci_hi'],2)), 'unparsed', round(v['unparsed_frac'],2)) for k,v in json.load(open(f'data/processed/donation_bet_{r}.json'))['headline'].items()}) for r in ['$run','${run}_abstract','${run}_equal']]"
Write-Host "Raw rollouts: data\raw\donation_bet_$run*.jsonl  — hand-check 20 answers per condition before believing any number."
