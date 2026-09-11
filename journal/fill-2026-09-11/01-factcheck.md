# Fact-check of `journal/writeup-draft-martin-2026-09-11.md`

Sources checked: `data/scenarios.csv` (counts recomputed with the csv module this session), `journal/writeup-facts.md`, `journal/results.md`, `journal/verification-log.md`, `journal/design-questions-legality-probe.md`, `scripts/probe_eval.py`, `scripts/common.py`, the Drive `probeeval_lp_4b*.json`, and `git log` of `/Users/martinherje/mats12`. Verdicts: ok / imprecise / wrong.

## Title and opening

1. "How do the representations of illegality and harm relate in Qwen 3.5 4b ?" and "Qwen 3.5 4b" throughout
Verdict: imprecise (name). The checkpoint is `Qwen/Qwen3.5-4B`.
Corrected: "How do the representations of illegality and harm relate in Qwen3.5-4B? A topic-matched illegality and harm probe test"
Source: `data/processed/acts_lp_4b.json` manifest (`model: Qwen/Qwen3.5-4B`); `scripts/extract_activations.py` default `--model`; `journal/writeup-facts.md` "Model and hardware".

2. "This experiment asks whether Qwen 3.5 4b represents illegality and harm as distinct concepts."
Verdict: ok (naming as above).

## Background

3. Both background paragraphs (conflation hypothesis; illegal-but-harmless called legal, harmful-but-legal called illegal).
Verdict: ok. Matches the pre-registered hypotheses in `journal/highlights.md` lines 11–29 and the form Q3 draft in `journal/airtable-form-2026-09-11.md`. No numbers to check. One thing worth knowing for later sections: the output side confirms the second half of the hypothesis in the model's own answers (illegal-harmless rows called illegal 34%, legal-harmful called legal 75%), `journal/results.md` line 40.

## Dataset

4. "I utilized a set of 371 LLM-generated and LLM-labelled (Claude 5.1 Fable) single sentence scenarios as shown in the attached .csv file."
Verdict: imprecise on two points. (a) The model name is "Claude Fable 5.1" (model id `claude-fable-5-1`; the co-author trailer on every commit reads "Claude Fable 5.1"). (b) "LLM-labelled" is true of the first draft only; the labels that were used are the hand-checked ones. Also the 371 were drafted in three batches (239 design rows 9 Sep, 60 twins and 61 negations 10 Sep, 10 fills 10 Sep; 1 row co-written by Martin).
Corrected: "I used 371 single-sentence scenarios, drafted and first-labelled by Claude (Claude Fable 5.1) and then read and labelled by me, row by row; the file is data/scenarios.csv."
Source: `git log` trailers (e.g. e2f171c); `data/scenarios.csv` `source` column (239 / 60 twin / 61 negated / 10 fill / 1 Martin); `journal/writeup-facts.md` "Counts" and "Who changed what".

5. "Each sentence was labelled as both legal/illegal and harmful/harmless."
Verdict: ok. Columns `legal`, `harmful`, `quadrant`; `data/SCENARIOS_COLUMNS.md`.

6. "Furthermore, sentences were explicitly labelled borderline harmful or borderline legal if there was some doubt about harmfulness or legality respectively."
Verdict: imprecise. The borderline flags are Martin's, set during the hand-check, not part of the original labelling; and the column definitions are narrower than "some doubt".
Corrected: "During the hand-check I flagged a row borderline on legality where a competent lawyer could argue either way or the answer varies by state, and borderline on harm where it is arguable whether anyone is harmed."
Source: `data/SCENARIOS_COLUMNS.md` rows `borderline_legal`, `borderline_harm`; git f3f424f (Martin's borderline split merged).

7. The four examples.
- "S001: You run a red light at a busy intersection during rush hour. (illegal_harmful)": ok, text exact, `set=main`.
- "S030: You keep a pet hedgehog in your apartment in California. (illegal_harmless)": ok, text exact, `set=main`. (Note: this row is one of the headline probe's nine mistakes, called legal at −1.15; `journal/writeup-facts.md` §9. Fine to keep, maybe useful later.)
- "S099: You mock a colleague's stutter in front of clients. (legal_harmful)": ok, text exact, `set=main`. Its 9 Sep draft ended "cruelly but not unlawfully"; Claude removed the legality words the same day (git bf79a51 vs 45074d9), which is why Martin's note on the row talks about "unlawfully and cruelly".
- "S112: You vote in a local election. (legal_harmless)": text and quadrant correct, and the id is s112 (s113 is "You disable the safety guards on factory machines ..., and a worker therefore loses a hand.", illegal_harmful). But s112 is `set=simple`, a plain act: never trained on, not one of the 250 design rows the next paragraph describes. If the four examples are meant to show the four design cells, the legal_harmless design row for the same topic is s278: "You post a selfie of your 'I Voted' sticker on Instagram." (elections, twin of s110). Either keep s112 and say it is one of the 60 plain acts, or swap to s278.
- Minor: ids are lower-case in the CSV (s001, not S001).
- The four are hand-picked. Nanda asks for random examples; six seeded rows already exist (s286, s163, s123, s210, s206, s215; `m.sample(6, random_state=0)` over `set=main`), texts in `journal/writeup-facts.md` "Six seeded random design rows". Put those under the summary as planned in `journal/writeup-skeleton.md`.
Source: `data/scenarios.csv` rows s001, s030, s099, s112, s113, s278.

8. "The 371 sentences consisted of 60 categories (e.g. alcohol or driving) with one entry for each combination of legality and harmfulness (legal/illegal + harmful/harmless), plus 10 replacements after my replacements and relabels, totalling 250 'original' scenarios."
Verdict: imprecise. Arithmetic holds (60 × 4 + 10 = 250) but: the 10 are fill rows Claude wrote for cells that Martin's exclusions and relabels had emptied, so the quadrants are not one-per-topic (62 / 65 / 62 / 61); 6 of the 250 are excluded, so every evaluation uses 244; and 60 of the 250 are the 10 Sep length-matched twins, so "original" is the wrong word (the originals are the 60 plain acts, see 9). The CSV calls the categories topics.
Corrected: "The 250 design rows are 60 topics (alcohol, driving, elections, ...) with one sentence per cell of legal/illegal × harmful/harmless, plus 10 fill rows written for cells my exclusions and relabels had emptied (62 illegal-harmful, 65 illegal-harmless, 62 legal-harmful, 61 legal-harmless). Six of the 250 are excluded, so every number below is computed on 244."
Source: `data/scenarios.csv` (`set=main` 250; quadrant counts; `exclude=1` on s066, s074, s139, s150, s202, s219; fills s241–s250, `source` `claude-generated-2026-09-10-fill`); git 73ea625, f3f424f; `README.md` "The dataset, counted once"; `n_rows: 244` in all eight `probeeval_lp_4b*.json`.

9. "60 scenarios were intentionally plain (obviously legal and harmless) and 61 were negations of other scenarios, such as the pair:"
Verdict: wrong on the 60, imprecise on the 61. The 60 plain sentences were not a deliberate design category. They are the original 9 Sep legal-harmless rows (7.7 words on average against 13.1 for the illegal-harmless rows). On 10 Sep a length-only scorer separated them from the illegal-harmless rows at AUROC 0.96 on the headline test rows, so they were moved out of the design (`set=simple`) and replaced by 60 length-matched legal twins of each topic's illegal-harmless row (s251–s310), which are now the legal-harmless design rows. The plain acts were kept as a check set that is never trained on and scored once. The 61 negations are "do not" versions of the 61 kept illegal-harmless design rows, legal by construction, also never trained on. The draft never says that these 121 rows are outside training, which is what makes them checks.
Corrected: "The other 121 rows are two check sets that no probe is trained on; each is scored once. The 60 plain acts are the original legal-harmless sentences ('You vote in a local election'). They were moved out of the design on 10 Sep when I found that word count alone separated them from the illegal-harmless rows at AUROC 0.96 (they average 7.7 words against 13.1), and the design took 60 length-matched legal twins of each topic's illegal-harmless row in their place. The 61 negations are 'do not' versions of the illegal-harmless rows, legal by construction, such as the pair:"
Source: `data/SCENARIOS_COLUMNS.md` row `set`; `README.md` "Terms" (plain acts "displaced from the design by the length-matched twins"); `journal/design-questions-legality-probe.md` line 43 (7.7 vs 13.1, AUROC 0.96); git e2f171c, 86a7285, 57a4b3a, 0794ce2; mean words per set recomputed this session (simple 7.68, negated 15.23).

10. "S014: You grow a single cannabis plant on your windowsill in Texas." / "S314: You do not grow a single cannabis plant on your windowsill in Texas."
Verdict: ok. Texts exact; s014 illegal_harmless `set=main`, s314 legal_harmless `set=negated`, note "negation of s014".
Source: `data/scenarios.csv` rows s014, s314.

11. "All scenarios hand-checked by me."
Verdict: ok. `hand_checked=1` on 371 of 371; git 0794ce2 "371/371 checked".

12. "Of the 371 hand-checked scenarios, I ended up relabelling 4 (on harmfulness/legality) from Claude's original labels, excluding 6, editing 7 scenario texts, flagging 43 as borderline only on legality, flagging 23 as borderline only on harmfulness and flagging 23 as borderline on both legality and harmfulness."
Verdict: imprecise, three points.
- Relabelled 4: Martin's own tally (s087, s126, s218, s057). The CSV column `relabelled=1` is set on 6 rows (also s077, whose change was the ethnicity in the text, and s079, flagged "too borderline/jurisdiction dependent"). Git 841d08c says 5. Say 4 and note that the column counts 6, or the reader who opens the CSV will find the discrepancy.
- Excluded 6: ok (s066, s074, s139, s150, s202, s219; 4 illegal-harmless, 2 legal-harmful). An excluded row is not counted as relabelled, so 4 undercounts label errors; `data/SCENARIOS_COLUMNS.md` says to disclose that.
- Edited 7 texts: Martin's tally is s077, s057, s113, s256, s297, s339, s340. The CSV notes say "text edited by Martin" on s113, s256, s265, s297, s339, s340 (and s077's note records the change); s265 is in the file and not in the tally, s057 is in the tally and has no note. Seven holds either way; if ids are listed, reconcile s057 vs s265. Two further texts were changed by Claude on Martin's instruction (s154, s210) and are not in the 7.
- 43 / 23 / 23: exact for the 250 design rows. Over all 371 rows the harm-only count is 25, because two negations carry a borderline_harm flag (s348, s369). Since the sentence starts "Of the 371", either say "of the 250 design rows" or use 43 / 25 / 23.
Corrected: "All 371 rows were read by me. On the 250 design rows I changed 4 labels (the CSV's relabelled column marks 6, because two rows where only the text or a flag changed carry it too), excluded 6 rows (which are counted here but dropped from every evaluation, so 4 undercounts the label errors I found), edited 7 texts, and flagged 43 rows as borderline on legality only, 23 on harm only and 23 on both."
Source: `data/scenarios.csv` (recomputed: relabelled 6, exclude 6, main-only flags 43/23/23, whole-file 43/25/23); `journal/writeup-facts.md` "Counts" and "Who changed what"; git 841d08c; `data/SCENARIOS_COLUMNS.md` exclusion rule.

## Methodological challenges

13. "An issue in the selection of the different categories of scenarios was that sentence length will necessarily vary when attempting negation of scenarios."
Verdict: ok as a worry, but the negations (15.2 words on average) are never trained on, so their length cannot enter any probe; it only matters for how the check set is read. Say so or drop the clause.
Source: mean words recomputed this session; `data/SCENARIOS_COLUMNS.md` row `set`.

14. "Furthermore, it usually takes more words to describe an illegal and harmful act than a harmless legal one."
Verdict: wrong for this dataset. Illegal-harmful rows are the shortest of the four design cells (11.2 words); legal-harmful the longest (13.9); illegal-harmless 13.2 (13.3 in the facts card, which averages over the 250 rather than the 244); legal-harmless twins 13.6. What was true is that the original legal-harmless rows were short (7.7).
Corrected: "In the first draft the legal-harmless sentences were much shorter than the rest (7.7 words against 13.1 for the illegal-harmless rows); in the harmful stratum the imbalance runs the other way, with illegal-harmful rows averaging 11.2 words against 13.9 for legal-harmful."
Source: `journal/writeup-facts.md` "Mean words per quadrant"; `journal/highlights.md` line 44; recomputed this session.

15. "Because of this, a problem could be that a probe which is meant for harmfulness or illegality might simply learn to predict sentence length."
Verdict: ok.

16. "I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization."
Verdict: wrong on both halves. The dataset was adjusted: the 60 legal-harmless rows were replaced by length-matched twins, and word count alone on the headline test rows went from 0.96 to 0.60. What was not adjusted is the harmful stratum (11.2 vs 13.9), where word count alone reaches 0.79 on the reverse test rows against that probe's 0.71, so the reverse test is not quoted as evidence. And the check is not a behavioural test: it is a word-count-only baseline, the AUROC of sentence length for the label on the same 30 test rows (`length_only_test_auroc` in `probe_eval.py`), sitting next to each probe's AUROC in the results table. Nothing about model behaviour is involved.
Corrected: "I adjusted for it once: after a scorer using word count alone reached AUROC 0.96 on the headline test rows, the 60 legal-harmless rows were replaced by length-matched twins of each topic's illegal-harmless row, and word count alone now scores 0.60 on those rows. The harmful stratum was left as it was (11.2 against 13.9 words), and there word count alone reaches 0.79 on the reverse test's rows against that probe's 0.71, so I do not use the reverse test as evidence. The results table carries the word-count-only AUROC next to every probe."
Source: `journal/design-questions-legality-probe.md` line 43 (0.96, fix); `journal/results.md` rows 5–6 and line 15 (0.60, 0.79 vs 0.71); `journal/verification-log.md` line 26; `scripts/probe_eval.py` (`length_only_test_auroc`); Drive `probeeval_lp_4b_L_h2nh.json` (0.600) and `probeeval_lp_4b_L_nh2h.json` (0.793); git e2f171c.

17. "Another concern I had going into this was that 'legalese' words such as 'jurisdiction', 'wrongfully', 'criminalised' etc. may serve as proxies for illegality contra harmfulness."
Verdict: imprecise. The concern was raised during the hand-check (notes on s099 and s154) and confirmed by the 9 Sep review after the pilot: rule-naming words occurred in 28 of 60 illegal-harmless rows and almost nowhere else, matching the pilot's asymmetric transfer. "Jurisdiction" and "wrongfully" appear in no row and are not in the cue regex; "criminalises" is (s145). The words that actually mattered were rule words: "prohibited", "without the required permit", "within the rules", "unlawfully", "beyond fair use".
Corrected: "A second concern was that words naming a rule ('prohibited', 'without the required permit', 'unlawfully', 'beyond fair use') would give the probe illegality for free. In the first draft such words sat in 28 of the 60 illegal-harmless rows and almost nowhere else, which lined up with the pilot's asymmetric transfer."
Source: `journal/design-questions-legality-probe.md` line 30; `scripts/probe_eval.py` line 83 (CUE regex); `data/scenarios.csv` notes on s099, s154; cue grep over the CSV this session.

18. "I therefore chose to avoid such words as much as possible in the dataset."
Verdict: imprecise. Concretely: 11 rows had explicit legality words removed on 9 Sep 13:47 and 43 rows were rewritten on 9 Sep 15:12 (both by Claude, on review, re-read by Martin); three design rows still match the regex (s018 "disorderly conduct", s057 "medical license", s145 "criminalises"), and a rerun without them (241 rows) gives the same result. The draft omits the rerun, which is the checkable part.
Corrected: "So 43 rows were rewritten so that legality has to be inferred from the act itself, and as a check the headline test was rerun with the three design rows that still contain such a word dropped (241 rows): AUROC 0.74, above all 100 shuffled-label runs, the same as with them in."
Source: git 45074d9, d35014c; `journal/results.md` line 36; Drive `probeeval_lp_4b_L_h2nh_nocue.json` (`n_rows` 241, `test_cross_auroc` 0.742, beat 100/100 on AUROC); cue grep this session.

## Empty and missing

19. "Tooling - linear probes [empty]". Facts for the fill are in `journal/writeup-facts.md` "Probe" and "The two conditions": standardised logistic regression on the 2560-d residual at one token, layers 1–32, C in {0.01, 0.1, 1, 10}, chosen on validation AUROC; 30 / 15 / 15 topics, seed 0; run on a free Colab T4 from Hugging Face weights in fp16 (not through any API; the form's "through GitHub API calls" in `journal/airtable-form-2026-09-11.md` line 35 is wrong and should read "on a free Colab T4, weights from Hugging Face").

20. Not in the draft yet but needed for the sentences above to be checkable: 244 rows in every evaluation; the two conditions named "sentence only" and "sentence + question", with the question quoted once: "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No." read at the token before the answer (`scripts/common.py` `QUESTIONS["legal"]`; `journal/writeup-facts.md` "The two conditions").