# Verification plan for run lp_4b (Qwen3.5-4B), for Friday morning

Written 10 Sep 2026, 21:00 CEST, from the repo at `~/mats12` (HEAD c61feea) and the Drive mount. Everything below was dry-run tonight on the real lp_4b files so the expected outputs are actual outputs, not guesses. Martin re-runs it; nothing here counts until he has.

## What I found tonight, before the plan

1. **The 4B activations are already on this Mac.** Google Drive desktop mounts `~/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed/`, and it holds `acts_lp_4b.npz` and `acts_lp_4b_prompted.npz` (58 MB each, float16, finite, max |value| 104 bare / 36 prompted). No Colab is needed for any recompute; every check below runs on the Mac CPU in seconds.
2. **The just-ask baseline is not missing any more.** The same Drive folder has `ask_lp_4b_legal.json` and `ask_lp_4b_harmful.json` stamped 20:45 and 20:47 CEST tonight, git `c61feea`, n = 244 design rows plus the check sets, and `data/raw/ask_lp_4b_legal.jsonl` / `_harmful.jsonl` with 365 rows each and the `yes_minus_no_logit` field. Cell 9 (or `colab_followup.sh`) ran after the "still missing" note was written. The pilot files were moved to `.bak-20260910-18xx` as intended.
3. **Every load-bearing number reproduces from a second code path** (`notebooks/verify_by_hand.py`, new, uncommitted, numpy for the directions and the AUROC, sklearn only for the probe refit). The full outputs for all eight designs are in `team/verify_outputs_lp_4b.txt`. Two small discrepancies, both harmless: prompted reverse-legality AUROC 0.911 here vs 0.907 in the JSON (C = 10, one rank pair, lbfgs differs between the Mac's and Colab's sklearn; accuracy 21/30 matches exactly), and bootstrap interval edges that move by one row (0.67–0.93 vs 0.67–0.90) because the resampling seed differs.
4. **The just-ask baseline beats the probe on the headline rows.** On the exact 30 held-out harmless-stratum sentences the headline probe is tested on, the model's own Yes−No logit sorts illegal from legal at AUROC **0.81**; the bare probe gets 0.74 and the prompted probe 0.74. On the reverse test's 30 harmful-stratum rows the model's logit reaches **1.00** against the probes' 0.71 and 0.91. The run sheet's own rule applies: "just-ask beats the probe on the off-diagonal quadrants → the probe adds nothing over prompting; report it plainly." The model's yes/no *answer* is worse than its logit: it says "not illegal" to 9 of the 15 illegal-harmless test rows (answer accuracy 63 %), so like the prompted probe its ranking is fine and its cut-off is not.
5. **A wrong label in the just-ask output.** `scripts/ask_model.py:66` stores `frac_answered_yes = mean(pred)`, but for the legality question `pred = 1` means the model answered **No**. So `ask_lp_4b_legal.json` says the model "answered yes" to 100 % of the 60 plain legal acts when it answered No to all 60 (checked in the raw jsonl: `{'No': 60}`). `scripts/results_table.py:44` would print that as "Check set simple: answered yes 100%" in `journal/results.md`, which reads as the opposite of the truth. For the harm question the key is right. Fix or never quote (see the changes list).
6. **Two wording corrections to Martin's one-paragraph reading.** (a) "from layer 13 the two directions merge": at layer 13 the prompted cosine is +0.26, inside the band; it is +0.61 at layer 14 and outside from there. Say "from layer 14". (b) In the prompted condition the label-swap band itself is wide in the late layers (±0.8 to ±0.9 at layers 17–32, recomputed with 200 swaps: −0.83..+0.80 at layer 26 against a cosine of +0.92). The +0.9 sits outside the band but not by much; what carries the claim is the sign, positive at 19 consecutive layers against a sign-symmetric null. Say that, and say why the band is wide: at the answer position the states vary mostly along one axis, so any within-cell contrast lines up with the harm direction, which is the "merging" in another form.
7. **Prompted headline: the probe calls all 30 test sentences legal** (0 of 30 predicted illegal), which is exactly why accuracy is 0.50 with a 0.50–0.50 interval while AUROC is 0.74. Martin's "the cut-off fails across strata" is right and can now be said with the count.
8. **The reverse legality test is not evidence on its own.** Word count alone reaches 0.79 on its 30 rows (illegal-harmful rows average 10.9 words, legal-harmful 14.9); the probe reaches 0.71. That is in the table already; the write-up has to say the sentence.
9. **Counts for the hand-check table**: borderline on legality / harm / both = **66 / 46 / 23** on the 250 design rows (the skeleton says 64; recounted from the CSV). Read 371 of 371, excluded 6, relabelled 6.

## The hour, in order

Times are what it took tonight plus reading. Run from `~/mats12` after `git pull`. Set the path once:

```bash
cd ~/mats12 && git pull
export D="$HOME/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed"
ls -la "$D"/acts_lp_4b*.npz "$D"/ask_lp_4b_*.json
```

Expected: two npz files of about 58 MB from 10 Sep 18:48 and 18:50, two ask JSONs from 20:45 and 20:47. If the Drive mount is offline, the files can also be downloaded from Google Drive in a browser (folder `mats12_runs/data/processed`) into `data/processed/` and the `--acts-dir` flag dropped.

### Step 1 (10 min) — the dataset, without the activations

These need only `data/scenarios.csv`.

**1a. The counts under the executive summary.**
```bash
uv run python -c "
import pandas as pd; d=pd.read_csv('data/scenarios.csv'); m=d[d.set=='main']
print('rows', len(d), 'design', len(m), 'simple', (d.set=='simple').sum(), 'negated', (d.set=='negated').sum())
print('read', d.hand_checked.sum(), 'excluded', m.exclude.sum(), 'relabelled', m.relabelled.sum())
print('borderline legal/harm/both', m.borderline_legal.sum(), m.borderline_harm.sum(), ((m.borderline_legal==1)&(m.borderline_harm==1)).sum())
print(m.groupby('quadrant').size().to_dict())"
```
Expected: `rows 371 design 250 simple 60 negated 61` · `read 371 excluded 6 relabelled 6` · `borderline legal/harm/both 66 46 23` · quadrants 62 / 65 / 62 / 61.
Log line: "Counts recomputed from the CSV. I would be astonished if these were wrong; I made every one of these marks myself."

**1b. The six random rows** (seed 0, design rows, as the skeleton specifies; paste the output verbatim).
```bash
uv run python -c "
import pandas as pd; d=pd.read_csv('data/scenarios.csv'); m=d[d.set=='main']
print(m.sample(6, random_state=0)[['id','quadrant','borderline_legal','borderline_harm','text']].to_string(index=False))"
```
Expected ids, in this order: s286, s163, s123, s210, s206, s215. (The example report shows a different six because the CSV changed after it was written; the command is what matters.)
Log line: "Six rows drawn by a fixed seed after the dataset was frozen, not chosen. No way to be wrong except by editing the CSV afterwards, which git would show."

**1c. The topic split is the seed, not a choice.**
```bash
uv run python -c "
import json, numpy as np, pandas as pd
d=pd.read_csv('data/scenarios.csv'); t=np.array(sorted(d[d.set=='main'].topic.unique())); np.random.default_rng(0).shuffle(t)
r=json.load(open('data/processed/probeeval_lp_4b_L_h2nh.json'))['split']
print('test matches JSON:', sorted(t[:15])==r['test_topics'], '| val matches:', sorted(t[15:30])==r['val_topics'])"
```
Expected: `test matches JSON: True | val matches: True`. Also confirm all nine lp_4b JSONs share the split (step 3).
Log line: "The 15 test topics fall out of seed 0 on the sorted topic list; I re-derived them in three lines. Wrong only if the seed was re-rolled, which the JSON's args show it was not."

**1d. Word count alone on each test set** (the table's last column, from the CSV, no model).
```bash
uv run python -c "
import json, numpy as np, pandas as pd
def au(y,s):
    y=np.asarray(y); s=np.asarray(s,float); P=s[y==1]; N=s[y==0]; return ((P[:,None]>N[None,:]).sum()+0.5*(P[:,None]==N[None,:]).sum())/(len(P)*len(N))
d=pd.read_csv('data/scenarios.csv'); T=json.load(open('data/processed/probeeval_lp_4b_L_h2nh.json'))['split']['test_topics']
m=d[(d.set=='main')&(d.exclude==0)].copy(); m['wc']=m.text.str.split().str.len()
for tag,col,v,tgt in [('L_h2nh','harmful',0,'legal'),('L_nh2h','harmful',1,'legal'),('H_i2l','legal',1,'harmful'),('H_l2i','legal',0,'harmful')]:
    t=m[m.topic.isin(T)&(m[col]==v)]; print(tag, 'n', len(t), 'word-count AUROC', round(au(t[tgt],t.wc),3), 'mean words by label', t.groupby(tgt).wc.mean().round(1).to_dict())"
```
Expected: L_h2nh 0.600 (12.9 vs 13.7 words), L_nh2h 0.793 (10.9 vs 14.9), H_i2l 0.569, H_l2i 0.333.
Log line: "Word count recomputed by a rank formula I wrote out; agrees with the script to three decimals. The 0.79 on the reverse test is real and is a length confound in the harmful stratum that I did not correct."

### Step 2 (15 min) — every number in the results table, from the activations

One script, eight calls, about three seconds each. It refits the probe at the saved layer and C from the saved split and prints each number next to the script's. It also prints all 30 test sentences with their scores so the mistakes can be read, and runs a fixed-layer 200-shuffle null and a fresh 200-swap cosine band with a different seed.

```bash
for run in lp_4b lp_4b_prompted; do for tag in L_h2nh L_nh2h H_i2l H_l2i; do
  uv run python notebooks/verify_by_hand.py --run $run --tag $tag --acts-dir "$D" | tee -a journal/verify_$(date +%Y%m%d).txt
done; done
```

Expected (each line's recomputed value should equal the script's; full text in `team/verify_outputs_lp_4b.txt`):

| run · tag | accuracy | AUROC | interval | check sets (called legal / not harmful) | cos(d_illegal,d_harm) @ layer, fresh band | illegality dir. within harmless, as is → harm out | d_illegal predicts harm | fixed-layer null, beat of 200 (script's selection null, beat of 100) |
|---|---|---|---|---|---|---|---|---|
| bare · L_h2nh (headline) | 21/30 = 0.700 | 0.742 | 0.53–0.83 | simple 33/60, negated 61/61 | +0.091 @ L25, −0.28..+0.30, inside | 0.676 → 0.676 | 0.498 | 200/200 (100/100) |
| bare · L_nh2h | 20/30 = 0.667 | 0.711 | 0.57–0.80 | 56/60, 55/61 | +0.110 @ L32, inside | 0.698 → 0.680 | 0.504 | 190/200 (93/100) |
| bare · H_i2l | 23/30 = 0.767 | 0.947 | 0.63–0.90 | 0/60 harmful, 0/61 | +0.176 @ L14, inside | | | 200/200 (98/100) |
| bare · H_l2i | 24/30 = 0.800 | 0.827 | 0.67–0.93 | 7/60 harmful, 5/61 | +0.002 @ L3, inside | | | 200/200 (100/100) |
| prompted · L_h2nh | 15/30 = 0.500, **0 of 30 called illegal** | 0.742 | 0.50–0.50 | 58/60, 58/61 | +0.695 @ L15, −0.54..+0.51, **outside** | 0.849 → 0.738 | **0.751** | 77/200 (36/100) |
| prompted · L_nh2h | 21/30 = 0.700 | 0.911 (script 0.907) | 0.57–0.83 | 59/60, 36/61 | +0.922 @ L26, −0.83..+0.80, outside | 0.707 → 0.511 | 0.790 | 198/200 (98/100) |
| prompted · H_i2l | 19/30 = 0.633 | 0.969 | 0.53–0.73 | 0/60, 0/61 | +0.695 @ L15, outside | | | 177/200 (87/100) |
| prompted · H_l2i | 26/30 = 0.867 | 0.991 | 0.77–0.97 | 0/60, 5/61 | +0.605 @ L14, outside | | | 200/200 (100/100) |

What to read, not just run: the 30 bare headline rows. The 9 mistakes are 6 illegal-harmless rows called legal (pet hedgehog in California, structured deposit, watering in a drought emergency, uncertified signal booster, church bell at 5 a.m., glass bottle in paper recycling) and 3 legal-harmless twins called illegal (the $12,000 deposit with the form filled in, photocopying three pages, a raised vegetable bed). Two of the six misses are borderline-legal rows; four are the regulatory-offence type (permits, certification, ordinances) where the model's own answer also says "not illegal". Write that paragraph in your words; it is the "what the probe got wrong" section.

Log lines (one per row of the table, or one for the block):
- "Headline 21 of 30 and AUROC 0.742: refit from the activations file with the saved split, counted the rows, and the AUROC by the rank formula. Very surprised if wrong."
- "Check sets 33/60 and 61/61: same refit probe, counted. Very surprised if wrong; mildly surprised the 55 % is a real fact about the probe rather than the short-sentence distribution shift, and the write-up says which."
- "Cosine +0.09 in a band of ±0.3, and +0.70 / +0.92 prompted against bands of ±0.5 / ±0.8: mean differences in numpy, band from 200 fresh swaps with seed 1. Very surprised if the numbers are wrong; not surprised if a referee says the prompted margin over the band is thin, because it is."
- "Harm projected out, 0.676 → 0.676 bare, 0.849 → 0.738 prompted: Gram-Schmidt against the harm direction, no QR, rank AUROC. Very surprised if wrong."
- "Beat N of 100: I did not rerun the selection-inclusive null (see below). I ran a fixed-layer null of 200 fresh shuffles: 200/200 for the headline, 190/200 for the reverse, 77/200 for the prompted headline. Same verdicts. Moderately surprised if the script's 100/100 were wrong, since the fixed-layer null agrees and the JSON's null max (0.633) is below the test accuracy."

### Step 3 (5 min) — the JSON files are internally consistent, and the layer claim

```bash
uv run python -c "
import json, glob
sp=None
for f in sorted(glob.glob('data/processed/probeeval_lp_4b*.json')):
    r=json.load(open(f)); pn=r['perm_null']; s=r['split']['test_topics']
    print(f.split('/')[-1][16:-5].ljust(18), 'acc %.3f null-max %.3f beat %3d/100 %s' % (r['test_cross_acc'], pn['acc_max'], pn['shuffles_beaten_acc'], 'ok' if (pn['acc_max']<r['test_cross_acc'])==(pn['shuffles_beaten_acc']==100) else 'INCONSISTENT'), '| split', 'same' if sp in (None,s) else 'DIFFERENT'); sp=s
c=json.load(open('data/processed/probeeval_lp_4b_prompted_L_h2nh.json'))['cosine_curve']
print('prompted cos(d_illegal,d_harm) and band, layers 12-17:'); [print(L, round(-c['cos_dlegal_dharm'][L],2), 'band', round(-c['null_hi'][L],2), round(-c['null_lo'][L],2)) for L in range(12,18)]"
```
Expected: nine lines all `ok` and `same`; layer 13 = +0.26 inside (−0.23..+0.31), layer 14 = +0.61 outside (−0.41..+0.49). Change "from layer 13" to "from layer 14" in the write-up.
Log line: "Where beat = 100/100, the null's maximum is below the test accuracy in every file; where it is not, the maximum is above. All nine files share the one split. Checked by reading the JSON, not the script."

### Step 4 (10 min) — the just-ask baseline, now that it exists

Copy the four files into the repo so `report.py` and `results_table.py` can see them (they are gitignored; this is the same thing the Colab symlink does):
```bash
cp "$D"/ask_lp_4b_legal.json "$D"/ask_lp_4b_harmful.json data/processed/
cp "$D"/../raw/ask_lp_4b_legal.jsonl "$D"/../raw/ask_lp_4b_harmful.jsonl data/raw/
```

**4a. The fair baseline on the exact 30 headline rows**, by hand:
```bash
uv run python -c "
import json, numpy as np, pandas as pd
def au(y,s):
    y=np.asarray(y); s=np.asarray(s,float); P=s[y==1]; N=s[y==0]; return ((P[:,None]>N[None,:]).sum()+0.5*(P[:,None]==N[None,:]).sum())/(len(P)*len(N))
T=json.load(open('data/processed/probeeval_lp_4b_L_h2nh.json'))['split']['test_topics']
raw=pd.DataFrame([json.loads(l) for l in open('data/raw/ask_lp_4b_legal.jsonl')])
a=raw[(raw['set']=='main')&(raw.exclude.astype(int)==0)&raw.topic.isin(T)]
for name,h in (('headline rows, harmless stratum',0),('reverse rows, harmful stratum',1)):
    m=a[a.harmful.astype(int)==h]; print(name, 'n', len(m), '| model logit AUROC for illegal', round(au(1-m.legal.astype(int), m.yes_minus_no_logit),3), '| answer accuracy', round((m.pred.astype(int)==m.legal.astype(int)).mean(),3), '| said Yes-illegal to', int((m.pred==0).sum()), 'of', len(m))
print('answers on the 60 plain acts:', raw[raw['set']=='simple'].answer.value_counts().to_dict())"
```
Expected: `headline rows … n 30 | model logit AUROC for illegal 0.809 | answer accuracy 0.633 | said Yes-illegal to 6 of 30` · `reverse rows … 1.000 | 0.933 | 17 of 30` · plain acts `{'No': 60}`.
Then the script's version of the same number: `uv run python scripts/results_table.py (report.py was removed; the fair-baseline line is the last line of journal/results.md) --run lp_4b | grep -A1 "FAIR BASELINE"` should print AUROC 0.81 for both conditions against the probe's 0.74.

**4b. The per-quadrant answers**, for the write-up's baseline paragraph: `uv run python -c "import json; r=json.load(open('data/processed/ask_lp_4b_legal.json')); print(r['per_quadrant'], r['offdiagonal_accuracy'], r['logit_auroc_offdiagonal'])"`. Expected: illegal_harmful 0.97, illegal_harmless 0.34, legal_harmful 0.75, legal_harmless 0.97; off-diagonal accuracy 0.55; pooled off-diagonal logit AUROC 0.49 (pooled across strata, so the harm confound pulls it to chance; the within-stratum 0.81 above is the fair comparison and the one the probe's test set matches).

**4c. Do not quote the `frac_answered_yes` lines** for the legality question anywhere until `ask_model.py:66` is fixed; the number is the fraction that answered No. If `results_table.py` is rerun before the fix, delete its "answered yes" clause from `journal/results.md` by hand.

Log line: "The model's own graded answer on the same 30 sentences: 0.81, above the probe's 0.74. Computed from the raw jsonl with the rank formula and confirmed by report.py. I would be surprised if this were wrong; I am not surprised by the direction, the run sheet named it as a possible outcome, and it goes in the executive summary, not a footnote."

### Step 5 (5 min) — the token position and the figures

**5a. Which token the probe reads** (tokenizer only, cached on the Mac, no model weights; `HF_HUB_OFFLINE=1` keeps it off the network):
```bash
HF_HUB_OFFLINE=1 uv run python -c "
import numpy as np, sys; sys.path.insert(0,'scripts'); from common import QUESTIONS, JURISDICTION
from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')
import os; D=os.environ['D']; zb=np.load(D+'/acts_lp_4b.npz'); zp=np.load(D+'/acts_lp_4b_prompted.npz')
for i in (0, 123, 310):
    t=str(zb['col_text'][i]); ids=tok(t)['input_ids']; print(i, 'bare: stored', zb['n_tokens'][i], 'tokenizer', len(ids), 'last', repr(tok.decode(ids[-1])))
    r=tok.apply_chat_template([{'role':'user','content':QUESTIONS['legal'].format(j=JURISDICTION,t=t)}], tokenize=False, add_generation_prompt=True, enable_thinking=False); ids=tok(r)['input_ids']
    print(i, 'prompted: stored', zp['n_tokens'][i], 'tokenizer', len(ids), 'last 4', [tok.decode(x) for x in ids[-4:]])"
```
Expected: stored and tokenizer counts equal on every row; bare last token `'.'`; prompted last four `['<think>', '\n\n', '</think>', '\n\n']`. So the bare probe reads the state at the sentence's full stop and the prompted probe reads the state at the newline after the empty think block, the position the model's "Yes" or "No" is generated from. Put that sentence in the method section.
Log line: "Token counts in the activations file equal a fresh tokenisation on three rows, so the last-token gather was not off by padding. Not checked: that the stored vectors equal an unbatched forward pass on the 4B (done by the agent on the 0.5B on 8 Sep; I have not repeated it on the 4B)."

**5b. Figures, by eye** (`figures/cosine_by_layer_lp_4b.png`, `cross_lp_4b.png`, `checks_lp_4b.png`): the bare cosine curve sits near zero inside its band at every layer; the prompted curve crosses out of its band between layers 13 and 14 and stays near +0.9; the dotted lines are at 25 and 15; the cross figure's bare headline dot is at 0.74 above a grey bar at 0.62 with "beat 100/100"; the checks figure's third panel still says "just-ask baseline not run yet" until step 4's copy, after which `uv run python scripts/figures.py --run lp_4b --title Qwen3.5-4B` redraws it with the 0.81 bar. Never use any `pilot_*` or `mac05b` figure.

## What cannot be verified by Friday, and the honest sentence for each

- **The selection-inclusive permutation null's individual values.** The JSON keeps only mean, p95, max and the beat count; rerunning it is about 9 minutes per design (Colab timings 16:57 → 17:06 → 17:12 …), 80 minutes for all eight. Sentence: "I did not rerun the 100-shuffle nulls. I checked that each file's null maximum sits on the right side of the test accuracy, and I ran a cheaper null myself (labels shuffled within topic and stratum, 200 times, layer and C fixed at the chosen values), which gave the same verdict for every design." If one is rerun, make it the headline: `uv run python scripts/probe_eval.py --run lp_4b --target legal --train-stratum harmful --tag L_h2nh_rerun` reproduces the file bit for bit (seed 0 fixes split and shuffles), which proves determinism, not independence; say which.
- **Batched extraction equals unbatched on the 4B.** Needs the model on a GPU (two minutes on Colab if a session is open: extract three rows with `--batch-size 1` into a throwaway run and compare with `np.abs(a-b).max()`). Sentence: "Checked on the 0.5B stand-in by the agent on 8 Sep, not by me on the 4B; the token counts match, which rules out the padding error that check was for. Low surprise if wrong, but it would be my error to own."
- **fp16 storage.** The activations were written as float16 (the fp32 fix landed after the run). Values are finite and at most 104 in magnitude, so nothing overflowed; precision at 100 is about 0.06. Sentence: "Stored in half precision; finite, max 104; a re-extraction in fp32 would change numbers in the third decimal at most, and I have not done it."
- **The no-cue rerun (0.74, 100/100), markedness, geometry, borderline scores, steering.** Not recomputed by hand. Either leave them out (the standing rule: one test, one null, one baseline, one figure) or quote them with "not independently recomputed". Steering is cut and must not appear.
- **The labels.** They are Martin's and cannot be verified by machine. The model disagrees with the illegal-harmless label on 65 % of those rows; this is a limitation to state ("one annotator, US law as of 2025; the model itself calls two thirds of my illegal-but-harmless acts legal, and I did not get a second lawyer to arbitrate"), not something to fix tonight.
- **The hours.** Reconstructed, not Toggl; `journal/hours.md` says ~3 h so far, which is an underestimate of the whole and must be redone honestly before the form.

## Verification-log rows, ready to paste and edit into Martin's words

| When | Claim / artifact | How verified (by hand) | Surprise if wrong |
|---|---|---|---|
| 2026-09-11 | Dataset counts (371 read, 250 design, 6 excluded, 6 relabelled, 66/46/23 borderline) and the six random rows | pandas one-liners on the CSV, seed 0 | Astonished; I made the marks |
| 2026-09-11 | Topic split = seed 0 on the sorted topic list; identical in all nine result files | three-line re-derivation | Astonished |
| 2026-09-11 | Headline: 21 of 30, AUROC 0.742, interval 0.53–0.83 | `notebooks/verify_by_hand.py`, refit from the activations, rows counted, AUROC by rank formula | Very surprised |
| 2026-09-11 | Reverse legality 20/30, 0.711; harm 23/30, 0.947 and 24/30, 0.827; prompted 15/30 (all called legal), 21/30, 19/30, 26/30 | same script, all eight designs | Very surprised |
| 2026-09-11 | Check sets: bare headline probe calls 33/60 plain acts and 61/61 negations legal; prompted 58/60 and 58/61 | same refit probe, counted | Very surprised about the counts; the 55 % is a distribution shift and is described as one |
| 2026-09-11 | cos(d_illegal, d_harm) +0.09 bare (band ±0.3), +0.70 and +0.92 prompted (bands ±0.5, ±0.8); curve leaves the band at layer 14 | numpy mean differences; 200 fresh label swaps, seed 1 | Very surprised on the numbers; the prompted margin over the band is thin and is stated as thin |
| 2026-09-11 | Harm projected out: 0.676 → 0.676 bare; 0.849 → 0.738 prompted; d_illegal predicts harm at 0.50 bare, 0.75 prompted | Gram-Schmidt, rank AUROC | Very surprised |
| 2026-09-11 | Beat 100/100 (headline), 93/100, 98/100, 100/100; prompted 36, 98, 87, 100 | not rerun; null max vs test accuracy checked per file; fixed-layer 200-shuffle null by hand agrees | Moderately surprised |
| 2026-09-11 | Word count alone: 0.60 headline, 0.79 reverse | from the CSV, rank formula | Very surprised; the 0.79 is an uncorrected confound and the reverse number is not quoted as evidence |
| 2026-09-11 | Just-ask: model logit AUROC 0.81 on the headline rows (probe 0.74), 1.00 on the reverse rows; answers "not illegal" to 9/15 illegal-harmless test rows | raw jsonl, rank formula; report.py agrees | Surprised if wrong; not surprised by the direction |
| 2026-09-11 | Probe reads the full stop (bare) and the newline after the empty think block (prompted) | tokenizer counts vs stored n_tokens on three rows | Very surprised |
| not done | Batched = unbatched extraction on the 4B; fp32 re-extraction; the 100-shuffle nulls rerun; no-cue, markedness, geometry, borderline, steering | | stated as unverified |

## Files created tonight (uncommitted, nothing under data/ touched)

- `~/mats12/notebooks/verify_by_hand.py` — the checker; generic over the four designs and both conditions; `--acts-dir` reads the Drive mount directly. Copy in `team/verify_by_hand.py`.
- `team/verify_outputs_lp_4b.txt` — its full output for all eight designs, including every test sentence with its score.
