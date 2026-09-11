# Form answers Q8–Q12, full-time question, code link (draft for Martin to paste after reading; his voice; [check] marks facts I could not confirm from files)

## What I found that changes the earlier plan for these answers

The vault's earlier drafts (plans/applications/MATS 12 - Admission Strategy 2026-08-21.md line 75; journal/form-answers.md Q7/Q8) lead the evidence answer with "micrograd and GPT-2 from a blank editor, Karpathy, ARENA". I found no record that this happened. `~/scratch-nn` does not exist; `gh repo list martinherje` shows no from-scratch repo; plans/Mechinterp Learning Log.md has no entry after 2026-06-24; CHANGELOG 2026-07-27 says "last commit 22 May". What the files do support as own-hands work: linear algebra from first principles and a toy superposition model typed line by line in a REPL (Learning Log 2026-06-18), numpy-100 by hand in coach mode (Learning Log 2026-06-24), the ARENA chapter 0 environment stood up on the Windows box (CHANGELOG 2026-06-24 14:12; whether any exercises were done is not recorded), three university AI courses 2023–2025 and the red-teaming affiliate year (CV - Martin Herje - Technical 2026-08.md lines 47–53, from Fresh cv.pdf). Do not claim Karpathy or GPT-2-from-scratch unless it is true; Nanda reads this as a filter and it is the kind of claim an interview exposes.

The pilots 01–09 and JustifyBench stay out as technical experience (memory: feedback-experiments-are-ai-executed). Q8 below discloses them in one sentence as AI-run, which is the honest version; cut the sentence if you would rather not raise them at all.

## Q8. What, if any, prior experience do you have with mechanistic interpretability?

None of my own before this task. I am a lawyer. Alongside the law degree I took three university courses in AI (natural language processing, large language models and general AI, ethics in AI, 2023 to 2025), and I spent a year as a research affiliate at the Bergen law faculty red-teaming whether AI tools could produce master's-level legal research. This summer I went through linear algebra from first principles and the numpy-100 exercises by hand, built a small toy model of superposition line by line in a REPL, and set up ARENA chapter 0 [check: say how far you got in the exercises, or cut the ARENA clause]. On the reading side: your glossary and Concrete Steps, Elhage et al. 2021, and the probing papers this task builds on (Kramár et al. 2026, Sadhu et al. 2026, Schwarz 2026). In May I directed a series of activation-patching pilots on GPT-2 Small that Claude ran end to end; I set the questions and read the outputs but wrote none of the code, so I do not count it as experience. This task is the first probing experiment I have designed, labelled and verified by hand.

Sources: CV - Martin Herje - Technical 2026-08.md lines 47–53; plans/Mechinterp Learning Log.md (2026-06-18, 2026-06-24); CHANGELOG.md 2026-06-24 ARENA env entry; experiments/Index.md (pilots, AI-executed); mats12/journal/verification-log.md rows dated 2026-09-11 (your by-hand checks). [check] whether you did any Karpathy lecture or the XOR toy net repo (github.com/martinherje/Simple-XOR-neural-net, created 3 Jan 2026) by hand; if yes, add one clause.

## Q9. Other than your research task, 1–3 pieces of evidence you'd be able to do good research in the program (~100 words)

Three things. First, the dissertation: a sole-authored PhD inside the LEXplain project (Bergen and Copenhagen) on when a decision's stated reasons can be taken as its operative ones, which is the faithfulness question asked from the law side. It has produced an invited talk at the Cambridge workshop on law-following AI (June 2026) and a published essay. Second, a year red-teaming AI tools on legal research at the Bergen law faculty, and this August a two-day practical-AI course for law students that I designed and taught. Third, I taught myself the maths and numpy for this from zero this summer, by hand, because that is how I learn things.

(103 words.) Sources: PhD Overview.md; plans/applications/LAWAI Workshop - Law-Following AI.md (talk 12 Jun 2026, Jesus College); substack/Index.md (essay live 8 Jun 2026, martinherje.substack.com/p/black-box-systems-are-transparent); CV note lines 53 (red-teaming, Jul 2024–Jul 2025); presentations/JUS391 Praktisk KI 2026-08-24 og 26.md [check: delivered on 24 and 26 Aug as planned]; Learning Log 2026-06-18 and 06-24. Nanda's own framing of this question (raw/MATS 12 - Nanda application doc snapshot 2026-08-10.md lines 397–405): "impactful things you did at work or in class projects", "blog posts you're particularly proud of", "explain its relevance".

## Q10. Why are you interested in Neel's stream specifically?

Two reasons, one from your list and one from mine. From your list: the monitoring section says probing is the cheap state of the art for detecting misuse and asks what else can be done with probes, and the concept-representation section asks whether a truth probe generalises to real situations. The question I brought is the same shape from the other end: what does a compliance probe actually read, and does it survive a confound it was never trained across. The compliance-monitor angle comes from my field and is not on your list; I am not claiming it fills a gap you named. From mine: I am at the start in this field, and your stream is set up as a teaching structure that ends in a paper. My PhD is article-based, so a co-authored paper from the research phase could count towards it, which makes the leave easier to justify to my faculty. I also read the pragmatic interpretability post, and it matches how I already think about legal verification: judge the method by whether it helps on a problem someone actually has.

Sources: raw/MATS 12 - Nanda application doc snapshot 2026-08-10.md line 639 ("What else can we do with probes?", Kramár), line 565 (truth probe generalising), line 20 (the generic "linear representation" project named as a mistake), lines 25–31 (exploration phase, paper output); CLAUDE.md UiB requirements (3–5 articles, ≥2 sole-authored); plans/applications/MATS Winter 2027.md line 96 (credit architecture; supervisor sign-off not yet given, so "could count", not "will count").

## Q11. Likelihood you will join the training program (Sept 28 – Oct 30) if accepted?

High, about 90 percent [check the number]. I hold a PhD position in Bergen with some seminar teaching this autumn [check: JUS100 seminar dates in October], and two paper deadlines fall in the part-time weeks (2 and 9 October), which I can carry alongside part-time work. The two full-time sprint weeks, 19 to 30 October, would need my supervisor's agreement and moving [check: number] teaching sessions; I have not arranged that yet and would do it the week offers come out. The evening events at 5 to 8 pm UK time are fine from Norway.

Sources: Dashboard.md lines 40–41 (Paper 1 → Aarli 2 Oct, Paper 2 SSRN 9 Oct); Dashboard.md line 92 (KOGVIT101 autumn course, 80 % mandatory attendance, semester assignment 19 Nov); memory project-jus100-calendar-cleanup (JUS100 teaching this autumn; dates not in the vault); Nanda doc lines 27, 106, 126 (3 weeks part-time + 2 weeks full-time; events 5–8 pm UK).

## "If admitted, will you definitely be able to join the research phase full-time?" (Yes/No)

Yes. Then the note under Q12 below. The vault holds no record of the planned Aarli leave conversation (Admission Strategy line 67; Dashboard line 78; nothing later in CHANGELOG), so the note has to say the leave is intended but not yet arranged. Nanda's FAQ (doc line 761): "if in doubt, please apply and just include a note in your application", and about 1 in 6 withdraw. If the conversation has happened and Aarli agreed, replace the note with one sentence saying so [check].

## Q12. (Optional) Anything else important about your application project?

Two notes. On the full-time question: I answered yes. The research phase would be leave from my PhD position in Bergen, which I intend to take, but I have not yet formally arranged it with my supervisor and faculty [check], so per your FAQ I am applying and flagging it here, and would withdraw early if it could not be arranged. On the task: the repo linked below holds the scripts, the notebook, the raw run outputs, the verification log and the hours ledger. No timer was running, so the hours are reconstructed from commit timestamps and are labelled as reconstructed in the doc. Thank you for the extension to the 11th.

Sources: mats12/journal/hours.md (reconstructed ledger, about 8 h counted before the write-up); README.md "Who did what"; Nanda doc line 761.

## Optional code / colab link line

If you make the repo public tonight (journal/TONIGHT.md says it is clean of secrets; `gh repo view` shows PRIVATE as of now):

Code and journal: https://github.com/martinherje/mats12 (scripts/, notebooks/legality_probe_colab.ipynb, journal/ with results.md, verification-log.md, hours.md, and the raw Colab cell outputs in journal/colab-run-lp_4b-2026-09-10/). Colab copy of the notebook: https://colab.research.google.com/drive/1LuHofS0IRvu9eKmjOWx9I2jBrVPEk-E_ [check: sharing set to anyone with the link]. Result JSONs: [check: Drive folder link for mats12_runs/data/processed, shared anyone-with-link, or say "in the repo" if you copy them in].

If the repo stays private:

Code available on request (private GitHub repo, github.com/martinherje/mats12). The notebook that ran the experiment: https://colab.research.google.com/drive/1LuHofS0IRvu9eKmjOWx9I2jBrVPEk-E_ [check sharing]. The doc's appendix carries the verification log and the hours ledger.

Sources: mats12 git remote; plans/applications/MATS 12 - Run Sheet (legality probe).md Stage 0 (Colab working copy URL); journal/form-answers.md last line ("Code link: only if the repo is public by then").

## Errors noticed in the existing draft (outside the bio questions, flagged because cheap to fix)

1. Form Q4 draft: "Model: Qwen 3.5 4b (through GitHub API calls)". Corrected: "Model: Qwen3.5-4B, loaded from the Hugging Face weights and run on a free Colab T4; no API." Source: scripts/common.py lines 62–68 (`AutoModelForCausalLM.from_pretrained`); README.md "Results of record" (Colab T4, 10 Sep 2026).

2. Write-up draft: "I ended up relabelling 4 ... flagging 43 as borderline only on legality, flagging 23 as borderline only on harmfulness and flagging 23 as borderline on both". The CSV as of now gives relabelled 6, excluded 6, borderline legality-only 43, harm-only 25, both 23 (data/scenarios.csv, summed columns `relabelled`, `exclude`, `borderline_legal`, `borderline_harm`; your verification-log row of 2026-09-11 also says "6 relabelled" and "66/46/23"). The orchestrator's convention says 4 labels changed by you; if the `relabelled` column counts two consistency fixes made by Claude (s057 harmful→0 per its quadrant, CHANGELOG commit 841d08c), say "4 by me" and cite the column count separately [check]. The harm-only count should read 25, not 23, unless you recount.

## Facts I could not verify (each marked [check] above)

- Any Karpathy lecture, micrograd, or GPT-2-from-scratch work in your own hands (no repo, no log entry after 24 Jun).
- How far ARENA chapter 0 got after the environment was set up on 24 Jun.
- Whether the XOR toy net (Jan 2026 repo) was typed by you.
- JUS391 delivered on 24 and 26 Aug (the CHANGELOG has prep entries through 29 Aug, no delivery entry).
- The Aarli leave conversation: no record anywhere in the vault.
- JUS100 seminar dates in October and how many collide with 19–30 Oct.
- Colab notebook sharing setting; Drive folder link; repo visibility decision.
- The London research-phase option (vault says "may exist"; the 10 Aug doc snapshot says Berkeley only), so it is not in the drafts.