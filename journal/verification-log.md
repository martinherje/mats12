# Verification log

What was checked, how, and how surprised I'd be if it were wrong anyway. Append-only; feeds the application form's LLM-use question directly.

| When | Claim / artifact | How verified (by hand) | Surprise if wrong |
|---|---|---|---|
| | | | |
| 2026-09-08 13:30 | Pipeline plumbing (Claude, setup, uncounted): `extract_activations.py` batched right-padded last-token vectors equal unbatched forward pass on Qwen2.5-0.5B for s01/s05/s12 (max rel Δ 4.5e-4, fp16 rounding) | Machine check by the agent; **unchecked by Martin** — rerun the check block in the CHANGELOG entry on the pod with Qwen3.5-4B once | Low: gather logic is index arithmetic, but the hybrid GDN model is untested until the pod smoke runs |
| 2026-09-08 13:30 | `data/scenarios_smoke.csv` is arbitrary-label plumbing data; its probe output (test acc 0, cos −1 before the fix) is a confound demonstration, not a result | By construction — never cite | n/a |
