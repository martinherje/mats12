# Guardrails for any AI-assisted session in this repo

This is Martin Herje's MATS 12 application task. Nanda's evaluation explicitly tests whether the applicant adds value beyond prompting a frontier model, and the application form audits LLM use in detail. Every agent session in this repo follows these rules:

1. **Martin makes the research decisions.** Hypotheses, experimental design, controls, baselines, what to run next, and interpretation of results are his. If asked to "decide" one of these, propose options with trade-offs and stop. Do not silently redesign an experiment.
2. **Never fabricate.** No invented numbers, no synthetic "example" outputs presented as model outputs, no smoothing over failed runs. A failed run is reported as a failed run.
3. **Raw data discipline.** Every model call's full request and response (including the reasoning field) is saved verbatim under `data/raw/`, named by run. Analysis reads from disk, never from memory of "what it said".
4. **Verification hooks.** After producing any result, state in one line what Martin should hand-check to trust it (which transcripts to read, which number to recompute). Append the check to `journal/verification-log.md` as an unchecked item — Martin marks it checked, not the agent.
5. **Plots to disk.** Every figure saved as PNG in `figures/` with a self-contained caption in the filename or an adjacent .txt.
6. **Journal integrity.** `journal/` entries are append-only; never rewrite history. Timestamps on entries.
7. **No scope creep.** The clock is 20 hours. If a task looks like it opens a new workstream, say so instead of starting it.
8. **Cost sanity.** Batch API calls sensibly; estimate token cost before any run over ~1,000 calls.
