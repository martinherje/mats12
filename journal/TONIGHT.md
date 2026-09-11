# Tonight, in order (Fri 11 Sep 2026; submit by 22:00 CEST, hard limit Sat 08:59 CEST)

Everything below is on this Mac. No Colab needed. Tick as you go.

## 0. Open these
- This file. The Google Doc (link in the vault run sheet and in Claude's message). `journal/writeup-skeleton.md` beside it. `journal/results.md` for every number. `figures/fig1_lp_4b.png` (also on the Desktop) for the one figure.

## 1. Verify (about 45 min). Commands and expected outputs: `journal/verification-plan.md`. Claude's dry run of the same commands, to compare against: `journal/verification-dryrun-2026-09-11-claude.md`.
- [ ] `cd ~/mats12 && git pull`
- [ ] Step 1a counts, 1b six random rows (s286 s163 s123 s210 s206 s215), 1c split = seed 0
- [ ] Step 2: `verify_by_hand.py` for lp_4b and lp_4b_prompted against the Drive mount (bare 21/30, 0.742, 12 of 30 called illegal; prompted 15/30, 0.742, 0 of 30 called illegal). Read the nine bare mistakes.
- [ ] Steps 3–5: JSON consistency and the layer-14 claim; the fair baseline from the raw jsonl (0.81 on the 30 headline rows); per-quadrant answers (illegal-harmless 34%, off-diagonal 55%); token position
- [ ] Move the drafted rows in `journal/verification-log.md` into the table, each re-run and reworded in your words. Add the fp16-storage sentence.
- [ ] Fill the one blank in `journal/highlights.md` ("keep or swap")

## 2. The doc (about 3 h). Skeleton is pasted into the Google Doc; rewrite, do not paste.
- [ ] Executive summary, about 600 words: claim + hedge in one paragraph; two numbers per condition; the predicted / got / read table (your own prediction numbers); Figure 1 with its caption; evidence against, threats first, baseline last; what is not concluded
- [ ] The hand-check table and the six random rows directly under the summary
- [ ] Body: question and prior work (Sadhu, Schwarz, Bertolazzi, Cho); data; method; results table; what the probe got wrong; limitations (fp16 storage; 30-row tests; one model; one annotator; twice-rewritten data; word count on the reverse test; correlational)
- [ ] Verification section from the log; hours table (`journal/hours.md`, labelled reconstructed); who did what (README's paragraph, in your words)
- [ ] Every "(agent)" number either recomputed in step 1 or removed. No pilot, steering or stand-in number.
- [ ] Share: anyone with the link. Paste the link into the form.

## 3. The form (about 1 h). `journal/form-answers.md`, one answer at a time, in your words. Compose in a file first; the form has no draft save.
- [ ] Q6 from the verification log only. Q7 the Karpathy/ARENA line. Q10 the likelihood. The full-time-availability answer.
- [ ] Decide: repo public or not (if public, the link goes in; it is clean of secrets).
- [ ] CV attached. Checkboxes. Submit. Screenshot the confirmation.

## 4. After submitting
- [ ] Commit the journal (`git add journal && git commit && git push`). Tell Claude; the vault gets the decision record.
