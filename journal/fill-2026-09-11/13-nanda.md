Fetch status: the export URL redirected to googleusercontent and returned HTTP 400; `/edit` returned only the Google login shell; `/mobilebasic` returned a model-summarised digest (not verbatim). Every quotation below is therefore taken from the vault's verbatim snapshot `/Users/martinherje/Documents/PhDAI/raw/MATS 12 - Nanda application doc snapshot 2026-09-08.md` (line numbers given). The mobilebasic digest agreed with the snapshot on every point checked (16 h / max 20, 600 words / 3 pages, +2 h, the not-counted list, the "notably higher bar" for LLM prose, random not cherry-picked examples, baselines, sanity-check your agent), so the snapshot is current.

# 1. What Nanda explicitly wants and dislikes, in his words

## The form answers (read first, the filter)

- Line 14: "The application form has Qs to summarize the application. These are important, I read these first and use them as a preliminary filter, I don't have time to read every write-up. Having these be good is higher priority than the write-up/executive summary."
- Line 172: "Convey concretely what you did, what you found, why it's interesting, biggest limitations, etc. Specifics beat vibes: name the models, the key experiment, the surprising number."
- Line 175: "Please do not just submit raw LLM output for the application form or executive summary. Write these yourself, in your own voice, even if you think an LLM will sound better. [...] Answers that read like they were written by an LLM are a significant negative signal - I see hundreds of them, and they blur together."
- Line 432: "Poor writing - if I can't understand your summary in the application form / executive summary, I probably won't have time to decipher your research report".
- Form Q7 itself (journal/airtable-form-2026-09-11.md): "give specific examples of mistakes you caught, and ways you changed what the LLM did for the better. And explain how much of the text here was written by you vs an LLM, and why".

## The executive summary

- Line 181: "The first 1-3 pages of the google doc should be an executive summary, which gives the broad strokes of what you did and what you learned. Something at ~1 page (including graphs) is great, max 3 pages and max 600 words. Please include graphs! Bullet points can work well".
- Lines 184-188, the format he suggests: "What problem am I trying to solve? (and a bit on why you think it's interesting)"; "you can assume it will be read by someone with mech interp research experience"; "What are your high-level takeaways? What were the most interesting parts of your project?"; "One paragraph and graph per key experiment, giving the gist of what it was, what you found, and why this supports your key takeaways".
- Line 189: "If bad data would sink your project, show me the data. If everything rests on the quality of some dataset or judgement calls (e.g. you generated the dataset with an LLM, or used an LLM judge to score outputs), look at it yourself - and include some randomly selected qualitative examples in the write-up, ideally just after the executive summary. Randomly selected, not cherry-picked! A handful of raw examples is the easiest way to show me that the thing your whole project rests on is actually real."
- Line 252: "Make Your Executive Summary Count. It needs to stand on its own and convey the most important takeaways and a sketch of your key evidence. Don't make me hunt for the point or crucial details. Good graphs are a huge plus here."
- Line 251: "Your Reader Has Zero Context. [...] Explain everything from the ground up. Define your terms. Label your graphs clearly."

## The write-up

- Line 169: "a google doc describing your key findings which begins with an executive summary and ideally contains a bunch of graphs, and enough detail to follow what you did without needing to read your code."
- Line 356-357 (Clarity): "If I understand what you're claiming, what evidence you're providing, and think that evidence supports your conclusion, that instantly puts you in the top 20% of applicants." "how did you generate your data or choose your prompts, how did you define your metrics, what were your hyperparameters, etc.?"
- Line 361 (Taste): "This doesn't have to be a big, ambitious claim—just any claim that's not immediately obvious without evidence."
- Line 364-365 (Skepticism): "Negative or inconclusive results that are well-analysed are much better than a poorly supported positive result." "It's OK if you show self-awareness of where the holes are, which parts are speculative, what you would investigate next, etc. If you seem overconfident in shaky results, that is not. Make plausible claims over ambitious ones."
- Line 25: "Failing to compare to baselines (eg replace your vector with a random one, choose randomly, ask an LLM, use a linear probe)". Line 420: "Skipping the cheap control: fine-tune on random data, replace your vector with a random one, compare against 'just ask the model'."
- Line 411-413: "Not acknowledging limitations in their results (worse, trying to pretend negative results are positive - negative results are fine! Lying about them is not)"; "Trying to hype up their results [...] Just be honest! I can tell"; "A really positive sign about an application is when I think of a way the results could be false, then discover you've already checked it!"
- Line 418: "Not looking at your data - read some data points!"
- Line 244: "avoid relying only on a few cherry-picked qualitative examples—this is a major red flag."
- Line 379 (Show your work): "if you do have an interesting finding, please structure the write-up to emphasise it, don't do chronological order!"
- Line 22: dislikes "a very common/generic type of project without an interesting application or twist (showing that a safety-related concept has a linear representation ...)". Line 424: "A warning sign is candidates with a particular pet interest." Line 24: "Only studying old models (GPT-2, Pythia, Gemma 2)".
- Line 214: "You're encouraged to track your time with a tool like Toggl and include a screenshot with the application doc".

## The time rule

- Line 12: "Spend ~16 hours (max 20) working on an interesting AI safety research problem of your choice, and send me a write-up + executive summary of what you learned and answer application form Qs about the project (+2 extra hours)."
- Line 56: "You can take as much time as you want beforehand for general learning."
- Lines 199-206, not counted: "General prep (paper reading, tutorials), that you would have done before deciding on a project"; "Generic tech set up, like renting and setting up a cloud GPU"; "Breaks"; "Time spent waiting for things to train"; "Writing your answers to the MATS application form".
- Lines 207-212, counted: "any time you spend actively working towards the project goals [...] Writing code for your project; Reading papers (chosen because they're relevant to your project); Analysing data/experimental results; Thinking and planning time; Writing up the google doc".
- Line 212-213: "So the executive summary doesn't get super rushed, you can take another 2 hours for it. I ask that you don't edit the rest of the write-up, and don't write any new experiment code, though you're welcome to write code to make new graphs/visualisations from data you already have".
- Line 304: "I recommend spending at most 5 of the 12-20 hours reading papers and tutorials."

## LLM usage (the "Guidance on using LLMs" and "Sanity-check your agent" sections)

- Line 20: "Not sanity-checking your AI agents. Coding agents are great and you should use them, but if your write-up contains key results you clearly never verified, or don't understand that's disqualifying. I want scholars with value add over prompting Claude myself".
- Line 21: "Submitting obviously LLM-written slop [...] This is not banned, but I will hold you to a significantly higher bar since it's much more likely that the answers are bullshitting me (and I dislike reading slop). You get extra time for the write up and answers for a reason."
- Line 254: "You are actively encouraged to use LLM assistance for your application—I want to gauge how well you'll do at research in practice, so if you'd use it there, use it here!"
- Line 281-282: "I highly recommend against submitting raw LLM-written prose. [...] But they're very useful for drafting, brainstorming, and getting feedback. I recommend having several rounds of giving it your draft (with an anti-sycophancy prompt) and asking it to critique you for clarity, find confusing sentences, and check for technical inaccuracies."
- Line 288: "This is the most important piece of advice in this doc. Modern agents will happily generate a plausible-looking research project - plausible hypotheses, plausible code, plausible graphs, plausible conclusions - that is subtly (or unsubtly) wrong. A crucial thing I am evaluating is whether you add value beyond me just prompting Fable myself."
- Line 289: "Sanity-checking is worth a lot of your time and care - I'd guess a meaningful fraction of your 20 hours."
- Line 291: "Verify the load-bearing claims. For each key result: read the code that produced it, check the numbers in the write-up against the actual outputs, re-derive at least some of them independently (e.g. recompute a headline number with a fresh one-liner, or spot check by hand)."
- Line 292: "Be suspicious of success. If the agent says an experiment worked, treat that as a hypothesis, not a result. Ask: what's the dumbest way this could be wrong? (Data leakage, trivial baseline matching it, the metric not measuring what you think, the model in the loop gaming your grader…) Then check."
- Line 293: "Design experiments yourself. Agents are great at executing experiments and terrible at noticing that the experiment doesn't test the hypothesis. The experimental design, the controls and baselines, and the interpretation of results should be yours."
- Line 294: "Document your checking in the write-up. Tell me what you verified and how - 'I read 30 transcripts and confirmed the probe's positives were real' is strong evidence of research skill. In past rounds, some otherwise-promising applications were sunk because the write-up claimed things the applicant's own numbers contradicted - I do check."
- Line 295: "applicants who described using LLMs agentically (Claude Code etc.) were accepted at ~3x the rate of those who mainly used LLMs for writing polish. The tools are a genuine edge - for the people who stay in control of them."

# 2. Scorecard: Martin's draft (journal/writeup-draft-martin-2026-09-11.md) and form answers (journal/airtable-form-2026-09-11.md)

| # | Nanda item | Status | Fix in one line |
|---|---|---|---|
| 1 | Form Qs are the filter; name model, key experiment, surprising number (l. 14, 172) | missing | Q1 currently has no test and no number; open with: "Whether Qwen3.5-4B represents illegality separately from harm. An illegality probe trained only on harmful sentences sorts the harmless sentences of 15 unseen topics at AUROC 0.74 (above all 100 shuffled-label runs), but the model's own Yes/No logit does 0.81 on the same 30 rows." (probeeval_lp_4b_L_h2nh.json; journal/results.md last line) |
| 2 | Q3 wants "hypotheses and empirical claims you've shown (or disproven!)" | missing | Q3 lists six hypotheses and no outcome; append one clause per hypothesis: similar direction: no while reading (cos +0.09, inside band), yes when asked (cos +0.70, outside band from layer 14); harm gates the answer: says illegal on 97/34/25/3 % of the four quadrants; off-diagonal accuracy 55 % vs 97 % on the diagonal; illegal-harmless called legal 66 % (ask_lp_4b_legal.json per_quadrant; probeeval_lp_4b_{,prompted_}L_h2nh.json factorial) |
| 3 | Own voice, no LLM prose (l. 175, 21) | met | The draft and Q1–Q3 are his register; keep it; the skeleton's paragraphs 1, 5, 7 are Claude's and must not be pasted (journal/writeup-skeleton.md header) |
| 4 | Exec summary: first 1–3 pages, max 600 words, stands alone, graphs (l. 181, 252) | missing | The draft has no summary; what exists is body (background, dataset, challenges) with no result stated; write a separate ~600-word block that leads with the claim and the two numbers per condition, then move the existing text to the body |
| 5 | "Please include graphs!" and "Label your graphs clearly" (l. 181, 251) | missing | Put figures/fig1_lp_4b.png on page 1 with a caption saying what the grey bars (null p95) and the shaded bands (label-swap 95 %) are; state "sentence only" / "sentence + question" in the caption (the figure legend still says "sentence, then the question", commit c54c2f0) |
| 6 | One paragraph and graph per key experiment; takeaways up front (l. 186-188) | missing | Two experiments on page 1: the cross-stratum probe (0.74, 100/100, plain acts 33/60) and the direction geometry (cos +0.09 vs +0.70 with bands); the just-ask table is the third if recomputed tonight |
| 7 | "Don't do chronological order" (l. 379) | at risk | Draft opens with background, then method, then dataset problems; the finding is nowhere; reorder so page 1 is finding first, history in the body |
| 8 | Random, not cherry-picked, examples right after the summary; show the LLM-written data (l. 189, 244) | at risk | S001/S030/S099/S112 are one-per-quadrant picks; keep them but say "chosen to show the four cells", and add the six seeded rows s286 s163 s123 s210 s206 s215 (`m.sample(6, random_state=0)` on set==main, journal/writeup-facts.md §02) labelled as the random draw, with quadrant and borderline flags |
| 9 | "enough detail to follow what you did without needing to read your code": data generation, prompts, metrics, hyperparameters (l. 169, 357) | at risk | "Tooling - linear probes [empty]" must become: standardised logistic regression on the 2560-d residual at one layer, C in {0.01, 0.1, 1, 10}, layers 1–32, layer and C chosen on validation-topic AUROC only, 30/15/15 topic split seed 0, 100-shuffle within-cell null (scripts/probe_eval.py; facts card §03); quote the question verbatim once |
| 10 | Baselines: "just ask the model", random/trivial (l. 25, 420) | missing in draft, present in files | One paragraph: model's own Yes−No logit 0.81 vs probe 0.74 on the same 30 rows (journal/results.md, agent until recomputed); word count alone 0.60 on those rows, 0.79 on the reverse rows (length_only_test_auroc in the JSONs); the 100 shuffled-label runs |
| 11 | Negative results analysed, not hidden (l. 364, 411) | missing | Say in those words: the probe does not beat asking; the sentence-only probe calls 27 of 60 plain acts illegal (33/60 legal, probeeval_lp_4b_L_h2nh.json extra_sets.simple 0.55); under the question accuracy is 50 % with every harmless row called legal (probeeval_lp_4b_prompted_L_h2nh.json test_cross_acc 0.5) |
| 12 | "I think of a way the results could be false, then discover you've already checked it" (l. 413) | partly met | The draft names length and legalese; add the checks that exist: topic held out, selection on validation only, no-cue rerun 0.74 at 241 rows (probeeval_lp_4b_L_h2nh_nocue.json), length-matched twins (word count 0.96 → 0.60), harm-stratum length not matched (11.2 vs 13.9 words), neutral-question control cut |
| 13 | "Not looking at your data" (l. 418) | met | 371/371 rows read; the counts in the draft (4 relabelled, 6 excluded, 7 texts edited, 43/23/23 borderline) match scenarios.csv set=main exclusive counts; note in one clause that the CSV `relabelled` column counts 6 (two are a text edit and a flag) |
| 14 | Disclose that the data is LLM-written and rewritten (l. 189) | at risk | Draft says "LLM-generated and LLM-labelled" but not that 43 rows were rewritten after the cue-word audit and the 60 legal-harmless rows were replaced by length-matched twins on 10 Sep (git d35014c, e2f171c); add both, with why |
| 15 | "Document your checking in the write-up" (l. 294); form Q7 "specific examples of mistakes you caught" | missing | journal/verification-log.md has no Martin-authored row at HEAD; run notebooks/verify_by_hand.py for lp_4b and lp_4b_prompted tonight and write one sentence from what it printed (21/30, 0.742; 15/30, 0.742, 0 of 30 called illegal), plus the four relabels and s154 "beyond fair use" removed as the mistakes caught; nothing else goes in Q7 |
| 16 | "The experimental design, the controls and baselines, and the interpretation should be yours" (l. 293) | at risk | Say honestly which were his (the question, the illegality × harm crossing, all labels, the borderline split, the minimal rule that cut steering and the neutral question, the illegality-positive convention, prediction written before the run) and which came from review (the conditional-generalisation design from the external ChatGPT review; the within-cell null and label-swap band from the 9 Sep reviewer passes) (README "Who did what"; CHANGELOG 9 Sep 11:51, 15:15) |
| 17 | "how much of the text here was written by you vs an LLM, and why" (Q7) | missing | One sentence: the form answers and the executive summary are his; scripts, notebook, README and the write-up skeleton are Claude Code under his direction; no agent-computed number is quoted unless he recomputed it |
| 18 | Modern model, not GPT-2 (l. 24) | met | Qwen3.5-4B; say "Qwen3.5-4B" consistently (draft has "Qwen 3.5 4b", "Qwen 3.5, 4b") |
| 19 | Not a generic "concept has a linear representation" project (l. 22) | met, at risk in wording | Never write "linear representation of legality"; the twist is the illegality vs harm dissociation and the condition contrast; the title in the draft is fine |
| 20 | Pet-interest warning (l. 424) | at risk | Q2 ends on "law-following AI"; add the monitoring clause that a safety reader cares about: compliance detectors have been shown rule-blind (Sadhu et al. 2026, 2608.16852) and harm probes to be topic detectors (Schwarz 2026, 2607.13075), so a monitor built from a harm probe fails exactly on the off-diagonal cells |
| 21 | Plausible claims over ambitious ones; say what is speculative and what next (l. 365) | missing | Two sentences on page 1: not concluded that the model has a legality representation (30 test rows, accuracy interval 53–83 %); not concluded that the alignment under the question is legality-specific (band ±0.8 at layers 17–32, neutral-question control cut); next: that control, a second model, a second annotator |
| 22 | Hours: ~16 (max 20), +2 for summary, Toggl screenshot, no new experiment code in the +2 h (l. 12, 212-214) | at risk | No Toggl ran; paste the journal/hours.md table labelled "reconstructed from estimates and commit times", ~8 h counted before tonight; state the counting rule used (his active time, Claude's runtime and the form not counted); write no new experiment code tonight, only the recompute and figures |
| 23 | "at most 5 of the 12-20 hours reading" (l. 304) | met | ~1 h reading in hours.md |
| 24 | Full-time availability question | at risk | Answer plainly: yes, on leave from the PhD position, being arranged with the supervisor; do not overstate (memory: Aarli leave call unrecorded) |

# 3. Errors in the draft and form, with the correction and its source

1. Draft: "I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization."
   Corrected: "Length was a real confound: the first legal-harmless rows averaged 7.7 words against 13.1 for the illegal-harmless rows, and word count alone sorted the headline test rows at AUROC 0.96. On 10 Sep the 60 legal-harmless rows were replaced by length-matched legal twins of each topic's illegal-harmless row, which I re-read; word count alone now scores 0.60 on those rows. The harmful stratum was not matched (11.2 against 13.9 words) and word count reaches 0.79 on the reverse test's rows, so I do not quote that test as evidence. The word-count number is a column in the results table, not a behavioural test."
   Source: git e2f171c and 73ea625 (journal/writeup-facts.md §02 "Mean words per quadrant"); `length_only_test_auroc` 0.600 and 0.7933 in probeeval_lp_4b_L_h2nh.json and probeeval_lp_4b_L_nh2h.json; journal/results.md table note.

2. Form Q4: "Model: Qwen 3.5 4b (through GitHub API calls)"
   Corrected: "Model: Qwen3.5-4B, weights from Hugging Face loaded with transformers on a free Colab T4 in float16; no API."
   Source: scripts/common.py lines 61–70 (`AutoModelForCausalLM.from_pretrained`); data/processed/acts_lp_4b.json manifest (`dtype: torch.float16, device: cuda`); journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt.

3. Form Q4: "(2) the sentence followed by "is this legal? Yes or No", read at the token before the answer."
   Corrected: "(2) the sentence inside the question "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.", as one chat turn with thinking off, read at the token before the answer." Also fix the sign: the probe is an illegality probe, positive score means called illegal.
   Source: scripts/common.py line 75 (`QUESTIONS["legal"]`); README "Conventions".

4. Form Q4 (cut off): "The illegality probe was trained only on"
   Completion: "...the harmful sentences of 30 training topics (illegal-harmful against legal-harmful, n = 61) and tested on the harmless sentences of 15 topics it never saw (n = 30); layer and C chosen on 15 validation topics; the same in reverse and the same pair for harm; a 100-shuffle null repeats the whole procedure with labels shuffled within each topic × stratum cell."
   Source: scripts/probe_eval.py (split, `select_and_test`, `shuffle_within_cells`); `train n=61 · test n=30` in journal/colab-run-lp_4b-2026-09-10/13-followup-baseline-recompute-checks.txt.

5. Form Q1: "How does Qwen 3.5 4b represent harmfulness and illegality, and do they influence eachother?"
   Corrected: "Does Qwen3.5-4B represent illegality separately from harmfulness, or does an illegality probe just read harm? Test: a probe trained only on harmful sentences, scored on the harmless sentences of 15 unseen topics: AUROC 0.74, above all 100 shuffled-label runs; the model's own Yes/No logit on the same 30 rows: 0.81."
   Source: probeeval_lp_4b_L_h2nh.json (`test_cross_auroc` 0.7422, `perm_null.shuffles_beaten_auroc` 100); journal/results.md fair-baseline line (recompute before quoting 0.81).

6. Draft: "plus 10 replacements after my replacements and relabels, totalling 250 "original" scenarios"
   Corrected: "60 topics × 4 gives 240; 10 fill rows were written after my relabels and exclusions emptied cells, so the design has 250 rows (62 illegal-harmful, 65 illegal-harmless, 62 legal-harmful, 61 legal-harmless); 6 are excluded, leaving 244 in every evaluation."
   Source: data/scenarios.csv set=main counts; git 73ea625, f3f424f; README "The dataset, counted once".

7. Draft: "60 scenarios were intentionally plain (obviously legal and harmless)"
   Corrected: "The 60 plain acts ("You cook pasta") are the original legal-harmless rows, displaced from the design by the length-matched twins and kept as a check set that is scored once and never trained on; the 61 negations are the illegal-harmless rows with "do not", legal by construction, also never trained on."
   Source: data/SCENARIOS_COLUMNS.md `set` column; git 86a7285, 57a4b3a.

8. Draft example list: "S112: "You vote in a local election." (legal_harmless)"
   Add: s112 is a plain act (set=simple), never trained on. And state that the four are chosen to show the four cells, then give the six seeded rows as the random sample.
   Source: data/scenarios.csv (s112 `set=simple`); journal/writeup-facts.md §02 six seeded rows.

9. Draft: "relabelling 4 ... excluding 6, editing 7"
   Keep, add one clause: "the CSV's relabelled column counts 6, because s077 was a text edit and s079 a flag rather than a label change."
   Source: data/scenarios.csv `relabelled=1` rows s057, s077, s079, s087, s126, s218; git 841d08c; facts card §02.

10. Draft: "371 LLM-generated and LLM-labelled (Claude 5.1 Fable)" and Q4 "Claude Fable 5.1"
    Use one name, "Claude Fable 5.1" (Claude Code), and add the rewrite disclosure from item 14 above.
    Source: data/scenarios.csv `source` column (claude-generated-2026-09-09/-twin/-negated/-fill).

11. Draft title and text: "Qwen 3.5 4b", "Qwen 3.5, 4b"
    Use "Qwen3.5-4B" throughout (the Hugging Face id is Qwen/Qwen3.5-4B).
    Source: data/processed/acts_lp_4b.json manifest.

12. Draft: "flagging 43 as borderline only on legality, flagging 23 as borderline only on harmfulness and flagging 23 as borderline on both"
    Consistent with the CSV (set=main: 66 legal, 46 harm, 23 both; exclusive 43/23/23). No change; if the table under the summary reports 66/46/23, say which convention it uses so the two sets of numbers do not look contradictory.
    Source: data/scenarios.csv; facts card §02.

# 4. Numbers that must not go on the page unless recomputed tonight

Nanda line 294: "the write-up claimed things the applicant's own numbers contradicted - I do check." These are agent-computed until journal/verification-log.md has a Martin row: the fair baseline 0.81 (from data/raw/ask_lp_4b_legal.jsonl; expected 0.809), the per-quadrant Yes−No logit table (+3.82 / −0.98 / −0.99 / −3.46; 97/34/25/3 %; within-stratum 0.83 and 0.95; command at the end of journal/writeup-skeleton.md), the harm probe scoring the harmless stratum at 0.61 / 0.74, and the negation mean −7.9. Either the command is run and the row logged, or the number is dropped and the text still holds without it.

# 5. Paste-ready Q7 skeleton in his register (fill only from what he actually ran)

Claude Code wrote the scripts, the notebook, the README and a skeleton for this doc; I set the question, chose the illegality-by-harm crossing, wrote the prediction before the run, read and labelled all 371 rows, ran the Colab session and made the cuts (steering and the neutral-question condition were cut so the experiment stayed something I could check in an editor). The design itself changed once on review: an external ChatGPT review pointed out that training on the two easy cells made "legal" identical to "not harmful" on every training row, and I accepted the conditional-generalisation design instead. Mistakes I caught in the data: four labels changed (s087, s126, s218, s057), six rows excluded, and cue words like "beyond fair use" removed from a row because they would give the probe illegality for free. Mistakes caught by review that I checked and accepted: 43 rows naming a rule ("prohibited", "without the required permit") lined up with the pilot's asymmetric transfer and were rewritten; the first legal-harmless rows were half the length of their illegal twins and word count alone sorted them at 0.96, so they were replaced. What I recomputed by hand tonight: [the headline refit from the activations file, 21 of 30 and AUROC 0.742; the question condition 15 of 30, 0.742, 0 of 30 called illegal; the six random rows; the fair baseline from the raw answers]. Not checked: batched-equals-unbatched extraction on the 4B, an fp32 re-extraction (the activations of record are stored in float16, finite, max 104), and a rerun of the 100-shuffle nulls. The form answers and the executive summary are mine; the skeleton was Claude's and was rewritten, not pasted. Sources: README "Who did what"; CHANGELOG 9 Sep 11:51 and 15:15, 10 Sep 15:11; journal/verification-log.md; journal/team-pass-summary-2026-09-10.md §4.