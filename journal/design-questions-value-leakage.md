# Design questions — value-leakage mechanism (operative since 8 Sep; supersedes the legality-probe sheet)

Project: *Where does the value intervene?* Mechanism behind Betley et al. 2026, "Value Leakage" (arXiv 2607.14345), Donation Bet task, on Qwen3.5-9B. Primer in the vault: `plans/applications/MATS 12 - Value Leakage Primer.md`. You answer these; the answers copied into `highlights.md` as the initial hypothesis block, dated, are the clock's starting gun.

## 0. The objection, and the controls that answer it (Martin, 8 Sep evening)
"Isn't the Donation Bet just the prompt telling the model what's good?" For the *abstract* version (the note literally says "good cause" / "bad cause"), yes: the valence is prompt text, a direction separating the two conditions is prompt information, and ablating it is close to deleting two words. Nanda would call that the boring explanation. The paper's own variant set fixes this:
- **concrete**: the two branches name real charities with no valence word — Against Malaria Foundation vs Kids Wish Network (or a burn charity). The model has to supply the judgement "AMF is better" from its own values. That is the version where the value is the model's.
- **equal**: both branches name comparably good charities (Doctors Without Borders vs International Medical Corps, etc.). Bet structure, threshold and charity-naming all present, no reason to lean. This is the topic control.
Design consequence: run all three in the go/no-go and expect abstract ≥ concrete > equal ≈ 0. The mechanism work targets **concrete**, with **equal** as the contrast for the direction (concrete-good-side minus equal isolates the model-supplied valence, holding the bet structure fixed). The fact/motivation dissociation is then a real test: after ablation the model must still answer "which charity is better?" correctly while the estimate shift disappears. If only the abstract version leaks on 9B, that is itself the first finding (the model follows stated valence but does not bring its own), and the second setting becomes the AI-bubble developer-preference task, where the prompt carries no valence at all (invest in Alibaba vs OpenAI) — noisier (paraphrase variance ~5x in the released data) but immune to the objection.

## 1. Questions
The paper uses nine Fermi questions (`data/donation_bet_questions.json`). All nine, or a subset? Fewer questions with more samples each is the depth-over-breadth call; the paper's "small" setting is 10 baseline + 10 per direction per question, the main setting 100/100.

## 2. Sample sizes
N baseline (sets the median threshold) and N per direction. Suggested go/no-go: 9 questions × 20 baseline × 20 per direction × 2 thinking modes ≈ 1,100 generations; on the API that is under a dollar, on the pod about half an hour.

## 3. Thinking modes
Both (on and off) in the go/no-go, then commit to the mode where the leak lives for the mechanism work. Temperature 1.0 as in the paper, or lower?

## 4. Estimate parsing
The paper uses an LLM judge (Claude Sonnet) to extract the final number. The script uses a regex on the final answer with a strict "single number" requirement and marks everything else unparsed. Decide: regex only (hand-check 20 per condition), or add a judge call via OpenRouter for the unparsed remainder.

## 5. Which position, which representation
Default: last prompt token, all layers, raw residual. Alternatives: first answer token after the think block (thinking on); mean over the answer tokens. Pick one for the headline, name the others as robustness.

## 6. The direction
Difference of means, above_good minus below_good, at the position from Q5, per layer. Trained on a held-out split of prompts or on all? Evaluation is whether the projection onto d predicts the per-sample good-side outcome (AUROC) on held-out samples, not whether it separates the two conditions (it trivially does, the prompts differ by two words).

## 7. The ablation
Project d out at which layers (one, a band, all)? During the whole generation, or at prompt positions only? The default in `steer.py` is all positions, chosen layers. Decide before running.

## 8. Baselines (non-negotiable, choose the set)
- **Equal-charity condition** (paper's `equal_*`): the bet with no reason to lean — the primary contrast for the direction.
- **Fact-retention check**: after ablation, "Which of these two charities does more good?" must still be answered correctly.
- Random unit direction, same layers.
- Topic direction: bet-present vs baseline (no bet), same procedure.
- Prompt-only: append "Don't give a biased answer." (the paper's `dont_bias_prompt` variant).
- Optional: the direction from a *paraphrased* note, to show it is not lexical.

## 9. The dissociation prediction (write it as a sentence before running)
"I predict Qwen3.5-9B leaks on the Donation Bet with thinking off at bias ≥ X, that a diff-of-means direction at layer band L predicts the good-side outcome at AUROC ≥ Y on held-out samples, and that projecting it out reduces the bias to ≤ Z while a random direction reduces it by less than W and the model still endorses the good cause when asked." Commit X, Y, Z, W.

## 10. Disclosure operationalisation
With thinking on: a CoT "discloses" if it mentions the bet/donation/threshold as a reason for choosing the number (regex on those words plus a hand-read sample), or only if it says it is aiming for the good side? Decide, then record the hand-read count.

## 11. Value-endorsement check
Exact prompt used to ask the model whether it prefers the good cause, run with and without ablation.

## 12. Boring explanations to name and kill
- The ablation just makes the model dumber or noisier (check baseline estimate distribution under ablation; random-direction control).
- The direction is the words "good"/"bad" at the note position (read at the last prompt token; paraphrase control).
- The leak is a rounding artefact around the threshold (report the estimate distributions, not just the good-side rate).
- The "leak" is the model anchoring on the threshold number regardless of direction (the symmetric score cancels this, but show it).

## 13. Stop rule
If hour 12 arrives with only the replication and the layer sweep done, the ablation with one baseline (random direction) is the last thing to protect; paraphrase control and 27B are the first to cut. The minimum honest write-up is: the leak with CI, thinking on/off, the direction's held-out AUROC by layer, and one ablation with one control.
