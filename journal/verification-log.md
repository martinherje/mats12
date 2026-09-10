# Verification log

What was checked, how, and how surprised I'd be if it were wrong anyway. Append-only; feeds the application form's LLM-use question directly. A row counts only once Martin has done the check himself and written the row in his words.

| When | Claim / artifact | How verified (by hand) | Surprise if wrong |
|---|---|---|---|
| | | | |
| 2026-09-08 13:30 | Pipeline plumbing (Claude, setup, uncounted): `extract_activations.py` batched right-padded last-token vectors equal unbatched forward pass on Qwen2.5-0.5B for s01/s05/s12 (max rel Δ 4.5e-4, fp16 rounding) | Machine check by the agent; **unchecked by Martin** — rerun the check block in the CHANGELOG entry on the pod with Qwen3.5-4B once | Low: gather logic is index arithmetic, but the hybrid GDN model is untested until the pod smoke runs |
| 2026-09-08 13:30 | `data/scenarios_smoke.csv` is arbitrary-label plumbing data; its probe output (test acc 0, cos −1 before the fix) is a confound demonstration, not a result | By construction — never cite | n/a |
| 2026-09-10 (Colab run lp_4b) | Headline accuracy recompute, cell 10: printed 0.683 against the script's 0.700 | The cell that ran was the old version, which let the 121 check rows (plain acts, negations) into the test set; the 0.683 and its 19-row mistake list (`journal/colab-run-lp_4b-2026-09-10/10-recompute-OLD-CELL-check-sets-leaked.txt`) are invalid and are not used anywhere. The corrected recompute (`notebooks/verify_by_hand.py`, or `scripts/recompute_headline.py`) has not yet been run by Martin. **Not a check; recorded so the invalid number is not mistaken for one.** | n/a; pending |

## Drafted by Claude 10 Sep, each row to be re-run and reworded by Martin before it counts

None of the rows below is a check yet. Claude dry-ran each command on 10 Sep evening on the real lp_4b files (`team/verification-plan.md` has the commands and the expected outputs); the numbers are what those runs printed. Martin runs the same commands on Friday morning, confirms or corrects the numbers, rewrites the wording, and moves the row up into the table above. Until then nothing in the write-up is "recomputed by hand".

| When | Claim / artifact | How verified (by hand) | Surprise if wrong |
|---|---|---|---|
| 2026-09-11 | Dataset counts (371 read, 250 design, 6 excluded, 6 relabelled, 66/46/23 borderline) and the six random rows | pandas one-liners on the CSV, seed 0 | Astonished; I made the marks |
| 2026-09-11 | Topic split = seed 0 on the sorted topic list; identical in all nine result files | three-line re-derivation | Astonished |
| 2026-09-11 | Headline: 21 of 30, AUROC 0.742, interval 0.53–0.83 | `notebooks/verify_by_hand.py`, refit from the activations, rows counted, AUROC by rank formula | Very surprised |
| 2026-09-11 | Reverse legality 20/30, 0.711; harm 23/30, 0.947 and 24/30, 0.827; prompted 15/30 (all called legal), 21/30, 19/30, 26/30 | same script, all eight designs | Very surprised |
| 2026-09-11 | Check sets: bare headline probe calls 33/60 plain acts and 61/61 negations legal; prompted 58/60 and 58/61 | same refit probe, counted | Very surprised about the counts; the 55 % is a distribution shift and is described as one |
| 2026-09-11 | cos(d_illegal, d_harm) +0.09 bare (band ±0.3), +0.70 and +0.92 prompted (bands ±0.5, ±0.8); curve leaves the band at layer 14 | numpy mean differences; 200 fresh label swaps, seed 1 | Very surprised on the numbers; the prompted margin over the band is thin and is stated as thin |
| 2026-09-11 | Harm projected out: 0.676 → 0.676 bare; 0.849 → 0.738 prompted; d_illegal predicts harm at 0.50 bare, 0.75 prompted | Gram-Schmidt, rank AUROC | Very surprised |
| 2026-09-11 | Beat 100/100 (headline), 93/100, 98/100, 100/100; prompted 36, 98, 87, 100 (on accuracy; on AUROC 100/97/100/100 and 100/100/100/100) | not rerun; null max vs test accuracy checked per file; fixed-layer 200-shuffle null by hand agrees | Moderately surprised |
| 2026-09-11 | Word count alone: 0.60 headline, 0.79 reverse | from the CSV, rank formula | Very surprised; the 0.79 is an uncorrected confound and the reverse number is not quoted as evidence |
| 2026-09-11 | Just-ask: model logit AUROC 0.81 on the headline rows (probe 0.74), 1.00 on the reverse rows; answers "not illegal" to 10 of the 15 illegal-harmless test rows | raw jsonl, rank formula | Surprised if wrong; not surprised by the direction |
| 2026-09-11 | Probe reads the full stop (bare) and the newline after the empty think block (prompted) | tokenizer counts vs stored n_tokens on three rows | Very surprised |
| not done | Batched = unbatched extraction on the 4B; fp32 re-extraction; the 100-shuffle nulls rerun; no-cue, markedness, geometry, borderline, steering | | stated as unverified |
