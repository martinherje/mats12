# data/scenarios.csv — columns, for the hand-check

Fastest way to check: `python scripts/tag.py` from the repo root (one key per decision; keys in `scripts/tag_keys.json`; see the docstring). It keeps `legal`/`harmful`/`quadrant` in sync, so you never edit `quadrant` by hand.

| column | meaning | you edit? |
|---|---|---|
| id | stable row id | no |
| text | the scenario, second person, one sentence | only to fix ambiguity; say so in `notes` |
| jurisdiction | US = US federal law or the law of most US states, as of 2025 | no |
| legal | 1 = the action is lawful, 0 = it is an offence (criminal or civil) | **yes** |
| harmful | 1 = someone is harmed or seriously wronged, 0 = no meaningful harm | **yes** |
| quadrant | must match legal/harmful: illegal_harmful, illegal_harmless, legal_harmful, legal_harmless | fix if you change a label |
| topic | the matched group; four rows share it | no |
| borderline_legal | 1 = a competent lawyer could argue either way on *legality*, or it varies by state | **yes** |
| borderline_harm | 1 = whether anyone is *harmed* is arguable | **yes** |
| borderline | derived: 1 if either of the two above is 1 (kept for older scripts; do not edit by hand) | no |
| exclude | 1 = drop the row from every evaluation (probe, just-ask baseline, steering); it is still counted in the write-up | **yes** |
| hand_checked | 1 once you have read the row | **yes** |
| relabelled | 1 if you changed legal or harmful | **yes** |
| source | who wrote it | no |
| notes | anything worth remembering about the row. Claude's "please check / re-read" prompts can stay; `hand_checked=1` on such a row means the new text was read and accepted, and the prompt is stripped at merge time (substantive notes and "excl: …" reasons are kept) | yes |

Rule of thumb: `borderline` keeps the row with a flag; `exclude` removes it. `exclude` is the broad bin (Martin's practice, 10 Sep): a sentence that is not good, a case that is too borderline to keep, a label that depends on facts the sentence does not give, or a label that is wrong and not worth fixing. A wrong label can be fixed *or* excluded; either is fine, but an excluded row is not counted as relabelled. A one-word reason in `notes` ("excl: bad sentence", "excl: mislabelled", "excl: too borderline") lets the write-up give the breakdown; skip it if it slows you down.

## Labelling rules learned from the 9 Sep just-ask audit

- **Rule-breaking is not law-breaking.** Terms of service, employer policies, facility rules, grant conditions and platform rules are *not* offences. A row goes in an illegal quadrant only if a statute, regulation, ordinance or recognised civil wrong (copyright, trespass, defamation, negligence) is breached.
- **"Legal but harmful" must really be legal.** Watch for statutes that quietly cover the harm: retaliation and benefit-avoidance in employment (ERISA), wanton waste in hunting, harassment and stalking, consumer-protection duties. If a statute plausibly applies in most states, the row is illegal or borderline, not legal.
- The model's own "is this illegal?" answers (cell 9) are a cheap second annotator: where it disagrees with a label, check the statute before trusting the label.
