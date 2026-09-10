# Airtable form — drafts to rewrite in your own words (Nanda: LLM-written text is "a significant negative signal")

Claude drafted the *content* on 10 Sep evening; the sentences are yours to write. Target 80–200 words each. Compose here, paste once: the form has no draft save and no edit after submit. Form: https://airtable.com/appnMboxg76F1QIDc/pagqu7wWWrUCZkNVI/form

**Full-time in Berkeley 19 Jan–10 Apr 2027?** — Yes, on leave from the PhD position; being arranged with the supervisor (state it plainly if not yet confirmed; Nanda's FAQ says apply and add a note).

**1. What question did you try to answer?** Whether Qwen3.5-4B represents illegality separately from harmfulness, or whether a "legality probe" is a harm probe in disguise. Why it matters for monitoring (one sentence).

**2. What conclusions have you reached?** [from the results; lead with the headline number and its null; say what you would not conclude].

**3. Technical setup.** Qwen3.5-4B (32 layers, d = 2560), residual stream at the last token, bare and inside the "is this illegal?" question; 250-row 2×2 dataset, 60 topics, US law, hand-checked; standardised logistic probe; conditional generalisation across harm strata on topic-disjoint 30/15/15 splits; selection on validation AUROC; 100-shuffle within-cell null; factorial directions with a label-swap cosine band; just-ask baseline with the model's Yes−No logit on the same rows; check sets (simple anchors, negations) never trained on.

**4. Strongest evidence against your hypotheses.** [whichever of: null not beaten; fair baseline matching the probe; harm projection collapsing the legality direction; check sets failing; no-cue rerun collapsing]. Name the design's built-in ways of being wrong even if they did not fire.

**5. Biggest limitations, and could you have addressed them?** One model; ~30 test sentences per design; one annotator and one jurisdiction; data written by an LLM and rewritten twice (say why each time); correlational only (steering built, not run); the projection removes the between-stratum harm contrast, not "harm". What a second day would buy: four-fold topic CV, a second model, a second annotator.

**6. How did you use LLMs; what did and didn't you check; how did you prioritise; how surprised would you be by a major error?** Claude wrote the scripts, notebook and first-draft sentences under your direction; four reviewer passes and an external ChatGPT review shaped the design. You: every row of the dataset (250 + 121 check rows), the design decisions, the runs, the by-hand recompute of the headline accuracy from the activations file, the six random rows, the probe's mistakes. Not checked: [e.g. the permutation code path line by line; the cosine null band]. Surprise if wrong, per item, from `verification-log.md`.

**7. Prior mechinterp experience.** Self-taught, in progress: [Karpathy / ARENA chapters done], linear-probe and transformer-from-scratch exercises; no prior probing project of your own. (The AI-run pilots are not yours; do not list them.)

**8. Evidence you'd do good research in the program (~100 words).** The dissertation on stated vs operative reasons inside LEXplain; the Cambridge LAWAI talk; the skill-build track done by hand; this task's honest handling of a confound found mid-project.

**9. Why Neel's stream?** His list literally asks "what else can we do with probes?"; the stream is built to produce a paper; the law-side question (what a monitor must read to be a *compliance* monitor) is one his last forty scholars did not bring.

**10. Exploration phase 28 Sep–30 Oct; anything else.** Likelihood: [ ]. Note the 19–30 Oct full-time sprint and the PhD leave arrangement.

Checkboxes: exec summary is pages 1–3; doc shared with anyone-with-link. Code link: only if the repo is public by then.
