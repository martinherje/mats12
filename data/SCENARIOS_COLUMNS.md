# data/scenarios.csv — columns, for the hand-check

| column | meaning | you edit? |
|---|---|---|
| id | stable row id | no |
| text | the scenario, second person, one sentence | only to fix ambiguity; say so in `notes` |
| jurisdiction | US = US federal law or the law of most US states, as of 2025 | no |
| legal | 1 = the action is lawful, 0 = it is an offence (criminal or civil) | **yes** |
| harmful | 1 = someone is harmed or seriously wronged, 0 = no meaningful harm | **yes** |
| quadrant | must match legal/harmful: illegal_harmful, illegal_harmless, legal_harmful, legal_harmless | fix if you change a label |
| topic | the matched group; four rows share it | no |
| borderline | 1 = a competent lawyer could argue either label, or legality varies by state | **yes** |
| exclude | 1 = indeterminate; dropped from the probe evaluation (still reported) | **yes** |
| hand_checked | 1 once you have read the row | **yes** |
| relabelled | 1 if you changed legal or harmful | **yes** |
| source | who wrote it | no |
| notes | anything worth remembering about the row | yes |

Rule of thumb: `borderline` keeps the row with a flag; `exclude` removes it. Use `exclude` when the label depends on facts the sentence does not give.
