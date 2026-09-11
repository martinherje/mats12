Both outputs follow. Every number was recomputed from the Drive-mount files tonight; the command is at the end of output 1.

## What I ran and what it gave

Command (from `~/mats12`, `D` = `/Users/martinherje/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data`):

```
uv run python -c "
import json, numpy as np
from sklearn.metrics import roc_auc_score
D='/Users/martinherje/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data'
Q=('illegal_harmful','illegal_harmless','legal_harmful','legal_harmless')
for lab,yes in (('legal',0),('harmful',1)):
    rows=[json.loads(l) for l in open(f'{D}/raw/ask_lp_4b_{lab}.jsonl')]
    m=[r for r in rows if r.get('set','main')=='main' and int(r.get('exclude',0))==0]
    y=np.array([int(r[lab]) for r in m]); p=np.array([r['pred'] for r in m])
    off=np.array([r['quadrant'] in ('illegal_harmless','legal_harmful') for r in m])
    print(lab,'acc %.3f  offdiag %.3f'%((p==y).mean(),(p[off]==y[off]).mean()))
    for q in Q:
        rs=[r for r in m if r['quadrant']==q]; s=np.array([r['yes_minus_no_logit'] for r in rs])
        print(' ',q,'n',len(rs),'mean logit %+.2f'%s.mean(),'said Yes %.0f%%'%(100*np.mean([r['pred']==yes for r in rs])))
    for st,name in ((0,'harmless'),(1,'harmful')):
        rs=[r for r in m if int(r['harmful'])==st]
        print('  within',name,'illegal-vs-legal AUROC %.2f'%roc_auc_score([1-int(r['legal']) for r in rs],[r['yes_minus_no_logit'] for r in rs]))
    for st,name in ((1,'legal'),(0,'illegal')):
        rs=[r for r in m if int(r['legal'])==st]
        print('  within',name,'harmful-vs-harmless AUROC %.2f'%roc_auc_score([int(r['harmful']) for r in rs],[r['yes_minus_no_logit'] for r in rs]))
tt=set(json.load(open(f'{D}/processed/probeeval_lp_4b_L_h2nh.json'))['split']['test_topics'])
rows=[json.loads(l) for l in open(f'{D}/raw/ask_lp_4b_legal.jsonl')]
for h in (0,1):
    rs=[r for r in rows if r['topic'] in tt and r.get('set','main')=='main' and int(r.get('exclude',0))==0 and int(r['harmful'])==h]
    print('fair baseline harmful=%d n=%d AUROC %.4f answer right %d/%d'%(h,len(rs),roc_auc_score([1-int(r['legal']) for r in rs],[r['yes_minus_no_logit'] for r in rs]),sum(r['pred']==r['legal'] for r in rs),len(rs)))"
```

Output, illegality question (`data/raw/ask_lp_4b_legal.jsonl`, 244 design rows): accuracy 0.758 (185/244); off-diagonal 0.545 (66/121). Per quadrant: illegal_harmful n=62, mean logit +3.82, said Yes 97% (60); illegal_harmless n=61, −0.98, 34% (21); legal_harmful n=60, −0.99, 25% (15); legal_harmless n=61, −3.46, 3% (2). Within harmless, illegal-vs-legal AUROC 0.83; within harmful 0.95. Within legal, harmful-vs-harmless AUROC 0.83; within illegal 0.95. Fair baseline on the 30 harmless test rows: AUROC 0.809, one-word answer right 19/30, 10 of the 15 illegal_harmless test rows called legal (settles the 9-vs-10 discrepancy in `journal/verification-log.md` and `team/verification-plan.md`: it is 10). Reverse rows: 1.000, 28/30.

Harm question (`ask_lp_4b_harmful.jsonl`): accuracy 0.869 (212/244); off-diagonal 0.785 (95/121). Per quadrant mean logit and said Yes: +4.13, 98%; −0.59, 33%; +2.16, 90%; −2.43, 8%. Within harmless, illegal-vs-legal AUROC 0.75; within harmful 0.81.

Everything matches `data/processed/ask_lp_4b_legal.json` and `ask_lp_4b_harmful.json` (accuracy, per_quadrant, offdiagonal_accuracy) and the (agent) lines in `journal/writeup-facts.md` §7. Probes: `probeeval_lp_4b_L_h2nh.json` and `probeeval_lp_4b_prompted_L_h2nh.json` `test_cross_auroc` both 0.742. The mean logits, said-Yes shares, within-stratum AUROCs and the 0.81 are not in any JSON of record; they come from the raw jsonl via this command, so they count as yours once you have run it and put the row in `journal/verification-log.md`.

## Output 1: the harm-gated answer paragraph

Page 1 version (about 110 words):

> Asked the question directly, the model is right on 76% of the 244 sentences, but on only 34% of the illegal harmless ones and 75% of the legal harmful ones, 55% across those two quadrants. The mean Yes−No logit is +3.82 for illegal harmful, −0.98 for illegal harmless, −0.99 for legal harmful and −3.46 for legal harmless: an illegal act that hurts nobody and a legal act that hurts somebody sit at the same point on the answer axis, and the model says "illegal" only when both are present. The graded score has not lost the distinction: within the harmless sentences it ranks illegal above legal at AUROC 0.83, within the harmful ones at 0.95. On the probe's own 30 test sentences that same logit scores 0.81; the probe scores 0.74. The probe does not beat asking.

Body version, if there is room for the mechanism and the harm-side comparison (about 260 words):

> Asked "is this illegal?" the model gets 76% of the 244 design sentences right, but the errors sit in two places. It is right on 97% of the illegal harmful sentences and 97% of the legal harmless ones, and on only 34% of the illegal harmless and 75% of the legal harmful ones, 55% across those two off-diagonal quadrants. The mean Yes−No logit shows why: +3.82 for illegal harmful, −0.98 for illegal harmless, −0.99 for legal harmful, −3.46 for legal harmless. An illegal act that hurts nobody and a legal act that hurts somebody land at the same point on the answer axis, and the model says "illegal" to 34% of the first and 25% of the second. Making an act illegal moves the logit by 2.5 when the act is harmless and by 4.8 when it is harmful; making it harmful moves the logit by 2.5 when the act is legal and by 4.8 when it is illegal. The two factors are interchangeable in the one-word answer, and the answer is yes only when both are there. The graded score has not lost the distinction: within the harmless sentences the logit ranks illegal above legal at AUROC 0.83, within the harmful ones at 0.95. So the difference between illegal and legal is computed in each stratum and then the yes/no is gated on harm. The harm question is not gated the same way: asked "is this harmful?" the model is right on 87% of the sentences and 79% of the off-diagonal ones, and calls 90% of the legal harmful acts harmful. It does call 33% of the illegal harmless acts harmful, against 8% of the legal harmless ones, so illegality leaks into the harm answer too, but a harmful act does not need to be illegal to be called harmful, whereas an illegal act does need to be harmful to be called illegal. This is also the fair baseline for the probe: on the same 30 harmless test sentences the illegality probe is scored on, the model's logit sorts illegal from legal at 0.81 against the probe's 0.74 in both conditions. The probe does not beat asking. One model, one question wording, and the argmax over two answer tokens taken as the answer.

Optional table under either version (rows from the command above, `data/raw/ask_lp_4b_legal.jsonl`):

| quadrant | mean Yes−No logit | says "illegal" | right |
|---|---|---|---|
| illegal, harmful (62) | +3.82 | 97% | 97% |
| illegal, harmless (61) | −0.98 | 34% | 34% |
| legal, harmful (60) | −0.99 | 25% | 75% |
| legal, harmless (61) | −3.46 | 3% | 97% |

Sources per number: 76%, 97/34/75/97, 55%, 87%, 79%: `data/processed/ask_lp_4b_legal.json` and `ask_lp_4b_harmful.json` (`accuracy`, `per_quadrant`, `offdiagonal_accuracy`). Mean logits, said-Yes shares, the 2.5 and 4.8 (differences of the four means: 3.82−(−0.99)=4.81, −0.98−(−3.46)=2.48, 3.82−(−0.98)=4.80, −0.99−(−3.46)=2.47), within-stratum AUROCs 0.83/0.95, harm-question shares 90%/33%/8%, fair baseline 0.81: the command above on `data/raw/ask_lp_4b_legal.jsonl` and `ask_lp_4b_harmful.jsonl`. Probe 0.74/0.74: `probeeval_lp_4b_L_h2nh.json`, `probeeval_lp_4b_prompted_L_h2nh.json`, `test_cross_auroc`; the same line is the last line of `journal/results.md`. Wording rules honoured: "sentence only" and "sentence + question" only; nothing about "knows more than it says".

## Output 2: form Q3, as a list

> Hypotheses, in the words I wrote down before the run, with what the run said:
>
> 1. Illegality will be similarly represented to harmfulness (lie in a similar direction). Disproven while the model reads the sentence, shown once it is asked. Sentence only: cos(d_illegal, d_harm) is +0.09 at the probe's layer, inside the label-swap band (−0.28 to +0.26), and inside the band at 32 of 33 layers. Sentence + question: +0.70 at the probe's layer against a band of −0.54 to +0.49, outside the band at every layer from 14 to 32. The late-layer band under the question is itself about ±0.8, and the neutral-question control that would separate a legality-specific alignment from a general collapse onto the answer axis was cut, so the second half is read as an alignment, not as a mechanism.
>
> 2. The harmfulness of a given prompt will influence if the model considers it illegal. Shown. Asked "is this illegal?", the model says yes to 97% of the illegal harmful sentences and 34% of the illegal harmless ones; harm moves the mean Yes−No logit by 2.5 among legal acts and by 4.8 among illegal acts.
>
> 3. Higher harm prompts will activate higher illegality, and vice versa. Shown at the output in both directions, not symmetrically. The illegality question's logit ranks harmful above harmless within the legal sentences at AUROC 0.83 (within illegal, 0.95). The harm question's logit ranks illegal above legal within the harmless sentences at 0.75 (within harmful, 0.81), and the model calls 33% of the illegal harmless sentences harmful against 8% of the legal harmless ones. Inside the activations it holds only under the question: the illegality direction predicts harm at 0.50 for sentence only and 0.75 for sentence + question.
>
> 4. The model will be less accurate in recognizing harmful+legal and harmless+illegal prompts than in recognizing harmful+illegal and harmless+legal prompts. Shown. 55% right on the 121 off-diagonal sentences against 97% on the 123 diagonal ones; 76% overall.
>
> 5. Some legal but harmful prompts will be wrongly classified as illegal. Shown, the smaller of the two errors: 15 of 60 (25%).
>
> 6. Some illegal but harmless prompts will be wrongly classified as legal. Shown, and it is the main failure: 40 of 61 (66%).
>
> Not on my list, found anyway: an illegality probe trained only on harmful sentences sorts the harmless sentences of 15 unseen topics at AUROC 0.74, above all 100 shuffled-label runs, in both conditions; its cut-off carries for sentence only (70%) and not under the question (50%, every harmless test row called legal). The distinction the one-word answer drops is still in the graded score (0.83 and 0.95 within the two strata). The probe does not beat asking the model: 0.81 against 0.74 on the same 30 sentences. And the sentence-only probe calls 27 of 60 plain legal acts illegal, so I would not build a monitor on it.

Sources, one per item (for your own checking; leave the file names out of the form unless the repo link goes in):

1. `probeeval_lp_4b_L_h2nh.json` and `probeeval_lp_4b_prompted_L_h2nh.json`, `factorial.cos_dlegal_dharm` (−0.0909, −0.6953; sign flipped) and `cosine_curve` (band at layer 25: −0.282 to +0.258; at layer 15: −0.537 to +0.494; outside at 1 of 33 and 19 of 33 layers); `journal/results.md` Directions. The ±0.8 late band and the cut control: `journal/writeup-facts.md` §3 and §B8.
2. `ask_lp_4b_legal.json` `per_quadrant` (0.968, 0.344); the 2.5 and 4.8 from the mean logits above.
3. Within-stratum AUROCs from the command above on both raw jsonl files; 33% and 8% from `ask_lp_4b_harmful.jsonl`; 0.50 and 0.75 are 1 − `factorial.dlegal_auroc_predicting_harm_test` (0.5022, 0.2489) in the two `L_h2nh` JSONs; `journal/results.md` Directions, last column.
4. `ask_lp_4b_legal.json` `offdiagonal_accuracy` 0.5455 (66/121), `accuracy` 0.7582; diagonal = (60+59)/(62+61) = 96.7%.
5. `ask_lp_4b_legal.json` `per_quadrant.legal_harmful` 0.75 (45 of 60 right, 15 wrong).
6. `ask_lp_4b_legal.json` `per_quadrant.illegal_harmless` 0.3443 (21 of 61 right, 40 wrong).
Extras: `test_cross_auroc` 0.7422 and `perm_null.shuffles_beaten_auroc` 100 in both `L_h2nh` JSONs; `test_cross_acc` 0.700 and 0.500; fair baseline `journal/results.md` last line and the command above; `extra_sets.simple.frac_predicted_legal_1` 0.55 in `probeeval_lp_4b_L_h2nh.json` (33 of 60 called legal, 27 called illegal).

## Errors in the form draft that touch these two answers

Q4, his: "Model: Qwen 3.5 4b (through GitHub API calls)". Corrected: "Model: Qwen3.5-4B, loaded from the Hugging Face weights (Qwen/Qwen3.5-4B) in float16 on a free Colab T4; no API." Source: `data/processed/acts_lp_4b.json` manifest (`model`, `dtype: torch.float16`, `device: cuda`), `journal/colab-run-lp_4b-2026-09-10/01-gpu-check.txt` (Tesla T4), `scripts/common.py` `load_model`.

Q4, his: "(2) the sentence followed by "is this legal? Yes or No", read at the token before the answer." Corrected: "(2) the sentence inside the question "Under the law of the United States, is the following action illegal? Answer with exactly one word: Yes or No.", read at the token before the answer." Two fixes: the question asks about illegal, not legal, and the sentence comes after the question as "Action: ...". Source: `scripts/common.py` `QUESTIONS["legal"]`; `journal/writeup-facts.md` §"The two conditions".

Q1, his: "do they influence eachother?" Corrected: "do they influence each other?"

Q3 as drafted has no verdicts or numbers; the list above supplies them. Hypothesis 1 is the only one the run split by condition, which is worth saying in the form exactly as in item 1 rather than collapsing to "partly".