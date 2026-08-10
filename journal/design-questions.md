# Design questions — you answer these, then the design is yours

The project is locked: **Why do models violate their stated norms? A cause decomposition featuring confabulated compliance** (GPT-OSS-120B, API-only). Per the repo guardrails the experimental design is Martin's — this sheet is the list of decisions that *constitute* the design, each with the defensible menu. Answering them (pen on paper, ~1–2 h, arguably still pre-clock problem-scoping) produces the pre-registration that the write-up's "what did I predict" spine needs.

Answer inline under each. Once answered, copy the result into `highlights.md` as the initial hypothesis block, dated — that's the clock's starting gun.

## 1. Norm corpus
Which norms, how many? Menu: (a) OpenAI Model Spec tenets only (the model was *trained* on these — the cleanest "its own constitution" claim; Jakkli-style atomization, pick 8–15 tenets spanning easy/hard); (b) Model Spec + a supplied-in-context constitution for the contrast model (tests trained-in vs prompted-in norms — a real finding either way); (c) add the statutory-shaped garnish set (≤2 h rule). Fewer tenets, deeper per-tenet, beats coverage.

## 2. Violation elicitation
How do you get violations without begging the question? Menu: (a) adapt scenario styles from the Jakkli paper / Petri (adversarial pressure, competing objectives); (b) the open-sourced task-gaming environments; (c) hand-written scenario templates with systematic pressure variation (most defensible ownership, most hours). Decide: how many scenarios per tenet, and what pressure axis varies.

## 3. What counts as "cites the norm"?
The confabulated-compliance category lives or dies on this definition. Explicit tenet quotation? Paraphrase? A compliance claim ("this respects the user's autonomy") without quotation? Decide the operationalization *before* seeing data, and record it — moving this line after seeing results is the confabulation failure mode of this very project.

## 4. Judge design
LLM judge with rubric (which model judges — not the subject model?), plus: how many judgments do you hand-verify (Nanda's example: "I read 30 transcripts"), and what agreement stat do you report? The randomly-selected qualitative examples go right after the exec summary — decide the sampling rule now (e.g. seeded random 10 per category).

## 5. Cells and N
Realistic per the calibration: ~100–300 runs per cell. Which cells are load-bearing? Minimum: {real constitution, placebo constitution, no constitution} × {tenet-present, tenet-removed} on the 3–5 most violated tenets. What's the N the budget and clock allow? (At $0.03/$0.17 per M tokens, cost is not the constraint; reading time is.)

## 6. The dissociation prediction (write it as a sentence before running)
"I predict removing tenet X changes X-violations but not Y-violations; a placebo constitution changes neither" — pick the X/Y pair and commit. This is the falsification-shaped intervention the application hangs on.

## 7. The four signatures — what distinguishes the categories operationally?
- Ignorance: fails to state the tenet when asked directly.
- Incapacity: states it; still violates when compliance is the *only* instruction.
- Override: states it; complies in isolation; violates under pressure, CoT shows the trade-off.
- Confabulated compliance: claims/cites compliance in the very response that violates.
Decide the test battery order and what ambiguous cases get labeled (an "unclassifiable" bin is honest and cheap).

## 8. Boring explanations to name and kill
Committed so far: "violations are capability failures" (incapacity arm) and "the judge measures politeness" (agreement + hand-check). Add a third? Candidates: "violations are prompt-format artifacts" (paraphrase-robustness spot-check); "the model never actually read the constitution" (attention isn't checkable via API — but a recall probe is).

## 9. Contrast model
Qwen 3.5-27B with the Spec supplied in-context — worth the cells, or cut for depth? (Cutting is defensible; say so in limitations.)

## 10. Stop rule
The Wait/backtracking lesson: breadth killed depth. Decide now what gets dropped first if hour 12 arrives with half the grid done — coverage (fewer tenets) drops before verification (hand-checks) does.
