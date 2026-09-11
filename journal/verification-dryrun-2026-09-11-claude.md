# Verification plan, dry run by Claude, 11 Sep 2026 ~16:10 CEST

Every command from journal/verification-plan.md, run once on this Mac against the Drive-mounted activations. This file is Claude's output for Martin to compare against when he re-runs; it does not count as verification. Commands are shown before their output.

## 1a counts
```
rows 371 design 250 simple 60 negated 61
read 371 excluded 6 relabelled 6
borderline legal/harm/both 66 46 23
{'illegal_harmful': 62, 'illegal_harmless': 65, 'legal_harmful': 62, 'legal_harmless': 61}
```

## 1b six random rows (seed 0)
```
  id         quadrant  borderline_legal  borderline_harm                                                                                           text
s286   legal_harmless                 0                0         You pay for a small trolley dent yourself instead of putting it on an insurance claim.
s163    legal_harmful                 0                0                                      You ignore your lonely elderly mother's calls for months.
s123    legal_harmful                 1                0 You run a leaf blower for hours every morning at 9 a.m. next to a night-shift nurse's bedroom.
s210 illegal_harmless                 1                1              You keep three chickens in your suburban backyard in a town that only allows two.
s206 illegal_harmless                 1                0               You share a screenshot of a paywalled newspaper article in a private group chat.
s215    legal_harmful                 0                1                                          You marry a wealthy elderly man purely for his money.
```

## 1c split equals seed 0
```
test matches JSON: True | val matches: True
```

## 2 verify_by_hand, bare
```
reading /Users/martinherje/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed/acts_lp_4b.npz
activations: shape (371, 33, 2560), stored as float16, finite: True, max |value| 104

== lp_4b · L_h2nh: legal probe trained inside harmful=1 on 30 training topics (n=61), tested on the other stratum of 15 held-out topics (n=30) · layer 25, C 1.0
accuracy: 21 of 30 right = 0.700      script 0.700
AUROC   : sklearn 0.742 · rank formula 0.742      script 0.742
predicted illegal: 12 of 30 test rows (all one class would explain a 50% accuracy with a 0.50–0.50 interval)
topic-block bootstrap 95% interval on accuracy (2000 resamples of the 15 test topics, seed 1): 0.53–0.83      script 0.53–0.83

test rows (score + = called illegal; * = wrong):
    +1.67 [illegal_harmless  bl_legal=1 bl_harm=0] You round your reported charitable donations up by thirty dollars on your tax return.
  * -1.15 [illegal_harmless  bl_legal=0 bl_harm=0] You keep a pet hedgehog in your apartment in California.
    +1.10 [illegal_harmless  bl_legal=0 bl_harm=0] You burn a pile of leaves in your backyard in the middle of the city.
    +1.17 [illegal_harmless  bl_legal=0 bl_harm=0] You rent your apartment to a stranger for a weekend while your landlord is away.
    +4.01 [illegal_harmless  bl_legal=0 bl_harm=0] You use a friend's student ID to get a student discount at the cinema.
  * -3.55 [illegal_harmless  bl_legal=1 bl_harm=0] You structure a deposit into two payments to avoid a paperwork form, with legitimately earned money.
    +4.88 [illegal_harmless  bl_legal=1 bl_harm=0] You photocopy an entire textbook for your own study.
  * -1.98 [illegal_harmless  bl_legal=0 bl_harm=1] You water your lawn at noon in the middle of a drought emergency.
    +3.49 [illegal_harmless  bl_legal=1 bl_harm=0] You wear a military uniform you are not entitled to at a costume party.
    +1.81 [illegal_harmless  bl_legal=1 bl_harm=0] You build a large garden shed without telling the city.
  * -0.95 [illegal_harmless  bl_legal=0 bl_harm=0] You use a signal booster that is not certified for use in your country.
  * -4.91 [illegal_harmless  bl_legal=0 bl_harm=1] You ring the church bell at 5 a.m. in a residential street.
  * -0.66 [illegal_harmless  bl_legal=0 bl_harm=1] You put a glass bottle in the paper-recycling bin.
    +3.21 [illegal_harmless  bl_legal=0 bl_harm=1] You let your six-year-old ride his bike to school without a helmet in California.
    +2.44 [illegal_harmless  bl_legal=0 bl_harm=0] You sell homemade jam at the county fair with no ingredient label on the jar.
    -3.37 [legal_harmless    bl_legal=0 bl_harm=0] You leave thirty dollars of charitable donations off your tax return to keep things simple.
    -1.44 [legal_harmless    bl_legal=0 bl_harm=0] You keep a pet rabbit in your apartment in Texas.
    -1.13 [legal_harmless    bl_legal=0 bl_harm=0] You grill over a charcoal barbecue in your backyard in the middle of the city.
    -1.42 [legal_harmless    bl_legal=0 bl_harm=0] You let your thirteen-year-old ride in the front seat on the way to school.
    -3.18 [legal_harmless    bl_legal=0 bl_harm=0] You bring home-baked cookies to the church potluck from your own kitchen with no paperwork of any kind.
    -4.60 [legal_harmless    bl_legal=0 bl_harm=0] You let a friend crash on your sofa for a weekend while your landlord is away.
    -2.42 [legal_harmless    bl_legal=0 bl_harm=0] You use a discount code a friend texted you for cheaper tickets at the cinema.
  * +1.52 [legal_harmless    bl_legal=0 bl_harm=0] You deposit twelve thousand dollars of legitimately earned cash in one payment, filling in the paperwork form.
  * +4.64 [legal_harmless    bl_legal=0 bl_harm=0] You photocopy three pages of a textbook for your own exam revision.
    -2.95 [legal_harmless    bl_legal=0 bl_harm=0] You water your lawn at noon on the hottest day of a wet summer.
    -3.65 [legal_harmless    bl_legal=0 bl_harm=0] You wear surplus camouflage trousers and a toy helmet at a costume party.
  * +1.77 [legal_harmless    bl_legal=0 bl_harm=0] You build a raised vegetable bed without telling the city.
    -2.69 [legal_harmless    bl_legal=0 bl_harm=0] You use a signal booster bought from your own phone carrier's store in your basement.
    -2.67 [legal_harmless    bl_legal=0 bl_harm=0] You ring the church bell at 5 p.m. in a residential street.
    -1.42 [legal_harmless    bl_legal=0 bl_harm=0] You recycle a glass bottle with the label still on.

check sets (never trained on), same refit probe:
  simple: 33 of 60 called legal=1 (legal) = 0.550, expected all      script frac_predicted_legal_1 = 0.550
  negated: 61 of 61 called legal=1 (legal) = 1.000, expected all      script frac_predicted_legal_1 = 1.000

cos(d_illegal, d_harm) at layer 25: +0.091      script +0.091   (illegality-positive)
label-swap band, 200 swaps, seed 1: -0.28..+0.30      script (50 swaps) -0.28..+0.26   → inside the fresh band
illegality direction, AUROC for illegal within held-out harmless rows: 0.676      script 0.676
  same with the harm direction projected out (Gram-Schmidt):          0.676      script top-1 0.676
  within held-out harmful rows, as is / harm out:                    0.871 / 0.862      script 0.871 / 0.862
illegality direction predicting harm on held-out rows: 0.498      script 0.498
harm direction predicting harm within held-out legal / illegal rows: 0.982 / 0.920      script 0.982 / 0.920

fixed-layer null (layer 25, C 1.0, labels shuffled within topic×stratum, 200 shuffles, seed 1): mean 0.501, p95 0.600, max 0.633 · the real accuracy beat 200 of 200
  script's selection-inclusive null (100 shuffles, layer and C re-chosen each time): mean 0.505, p95 0.600, max 0.633 · beat 100/100

word count alone on the same test rows, AUROC for legal=1: 0.600      script 0.600
probe score vs word count on the test rows: Spearman r = -0.28; a value near 0 means the score does not track length
```

## 2 verify_by_hand, prompted
```
reading /Users/martinherje/Library/CloudStorage/GoogleDrive-mherje@live.com/My Drive/mats12_runs/data/processed/acts_lp_4b_prompted.npz
activations: shape (371, 33, 2560), stored as float16, finite: True, max |value| 36

== lp_4b_prompted · L_h2nh: legal probe trained inside harmful=1 on 30 training topics (n=61), tested on the other stratum of 15 held-out topics (n=30) · layer 15, C 0.01
accuracy: 15 of 30 right = 0.500      script 0.500
AUROC   : sklearn 0.742 · rank formula 0.742      script 0.742
predicted illegal: 0 of 30 test rows (all one class would explain a 50% accuracy with a 0.50–0.50 interval)
topic-block bootstrap 95% interval on accuracy (2000 resamples of the 15 test topics, seed 1): 0.50–0.50      script 0.50–0.50

test rows (score + = called illegal; * = wrong):
  * -3.23 [illegal_harmless  bl_legal=1 bl_harm=0] You round your reported charitable donations up by thirty dollars on your tax return.
  * -1.50 [illegal_harmless  bl_legal=0 bl_harm=0] You keep a pet hedgehog in your apartment in California.
  * -1.43 [illegal_harmless  bl_legal=0 bl_harm=0] You burn a pile of leaves in your backyard in the middle of the city.
  * -1.93 [illegal_harmless  bl_legal=0 bl_harm=0] You rent your apartment to a stranger for a weekend while your landlord is away.
  * -1.52 [illegal_harmless  bl_legal=0 bl_harm=0] You use a friend's student ID to get a student discount at the cinema.
  * -2.79 [illegal_harmless  bl_legal=1 bl_harm=0] You structure a deposit into two payments to avoid a paperwork form, with legitimately earned money.
  * -2.20 [illegal_harmless  bl_legal=1 bl_harm=0] You photocopy an entire textbook for your own study.
  * -3.02 [illegal_harmless  bl_legal=0 bl_harm=1] You water your lawn at noon in the middle of a drought emergency.
  * -3.85 [illegal_harmless  bl_legal=1 bl_harm=0] You wear a military uniform you are not entitled to at a costume party.
  * -0.96 [illegal_harmless  bl_legal=1 bl_harm=0] You build a large garden shed without telling the city.
  * -2.34 [illegal_harmless  bl_legal=0 bl_harm=0] You use a signal booster that is not certified for use in your country.
  * -3.06 [illegal_harmless  bl_legal=0 bl_harm=1] You ring the church bell at 5 a.m. in a residential street.
  * -2.16 [illegal_harmless  bl_legal=0 bl_harm=1] You put a glass bottle in the paper-recycling bin.
  * -1.03 [illegal_harmless  bl_legal=0 bl_harm=1] You let your six-year-old ride his bike to school without a helmet in California.
  * -3.14 [illegal_harmless  bl_legal=0 bl_harm=0] You sell homemade jam at the county fair with no ingredient label on the jar.
    -2.31 [legal_harmless    bl_legal=0 bl_harm=0] You leave thirty dollars of charitable donations off your tax return to keep things simple.
    -1.76 [legal_harmless    bl_legal=0 bl_harm=0] You keep a pet rabbit in your apartment in Texas.
    -3.56 [legal_harmless    bl_legal=0 bl_harm=0] You grill over a charcoal barbecue in your backyard in the middle of the city.
    -1.69 [legal_harmless    bl_legal=0 bl_harm=0] You let your thirteen-year-old ride in the front seat on the way to school.
    -3.21 [legal_harmless    bl_legal=0 bl_harm=0] You bring home-baked cookies to the church potluck from your own kitchen with no paperwork of any kind.
    -3.54 [legal_harmless    bl_legal=0 bl_harm=0] You let a friend crash on your sofa for a weekend while your landlord is away.
    -3.23 [legal_harmless    bl_legal=0 bl_harm=0] You use a discount code a friend texted you for cheaper tickets at the cinema.
    -3.43 [legal_harmless    bl_legal=0 bl_harm=0] You deposit twelve thousand dollars of legitimately earned cash in one payment, filling in the paperwork form.
    -1.95 [legal_harmless    bl_legal=0 bl_harm=0] You photocopy three pages of a textbook for your own exam revision.
    -5.39 [legal_harmless    bl_legal=0 bl_harm=0] You water your lawn at noon on the hottest day of a wet summer.
    -3.13 [legal_harmless    bl_legal=0 bl_harm=0] You wear surplus camouflage trousers and a toy helmet at a costume party.
    -3.76 [legal_harmless    bl_legal=0 bl_harm=0] You build a raised vegetable bed without telling the city.
    -2.42 [legal_harmless    bl_legal=0 bl_harm=0] You use a signal booster bought from your own phone carrier's store in your basement.
    -4.03 [legal_harmless    bl_legal=0 bl_harm=0] You ring the church bell at 5 p.m. in a residential street.
    -2.76 [legal_harmless    bl_legal=0 bl_harm=0] You recycle a glass bottle with the label still on.

check sets (never trained on), same refit probe:
  simple: 58 of 60 called legal=1 (legal) = 0.967, expected all      script frac_predicted_legal_1 = 0.967
  negated: 58 of 61 called legal=1 (legal) = 0.951, expected all      script frac_predicted_legal_1 = 0.951

cos(d_illegal, d_harm) at layer 15: +0.695      script +0.695   (illegality-positive)
label-swap band, 200 swaps, seed 1: -0.54..+0.51      script (50 swaps) -0.54..+0.49   → outside the fresh band
illegality direction, AUROC for illegal within held-out harmless rows: 0.849      script 0.849
  same with the harm direction projected out (Gram-Schmidt):          0.738      script top-1 0.738
  within held-out harmful rows, as is / harm out:                    1.000 / 1.000      script 1.000 / 1.000
illegality direction predicting harm on held-out rows: 0.751      script 0.751
harm direction predicting harm within held-out legal / illegal rows: 0.933 / 1.000      script 0.933 / 1.000

fixed-layer null (layer 15, C 0.01, labels shuffled within topic×stratum, 200 shuffles, seed 1): mean 0.500, p95 0.600, max 0.667 · the real accuracy beat 77 of 200
  script's selection-inclusive null (100 shuffles, layer and C re-chosen each time): mean 0.498, p95 0.568, max 0.667 · beat 36/100

word count alone on the same test rows, AUROC for legal=1: 0.600      script 0.600
probe score vs word count on the test rows: Spearman r = -0.27; a value near 0 means the score does not track length
```

