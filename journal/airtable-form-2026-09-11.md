# Airtable form, Neel Nanda Winter 2027 MATS application — fields and Martin's draft answers as of 11 Sep 2026 ~19:45 CEST

Saved verbatim from the form page. Late applications accepted until Friday 11 Sep 23:59 PT. Timeline on the form: ~Tue 15 Sep exploration-phase decisions; Mon 28 Sep – Fri 30 Oct online exploration phase (not full-time); ~early Nov research-phase decisions; 19 Jan – 10 Apr in-person research phase in Berkeley.

Fields (* = required): Full name*; Email*; Resume* (file); LinkedIn*; "If admitted, will you definitely be able to join the research phase full-time?"* (Yes/No); ~16 hour research task*: link a Google Doc with the executive summary and main write-up, anyone can view; (Optional) link to other outputs (code, colab); checkbox* "The first 1-3 pages of the attached doc are an executive summary"; checkbox* "The document permissions are set so that anyone with the link can see my doc".

Questions*:
1. What question did you try to answer?
2. Why is this question interesting / why did you choose it?
3. What conclusions have you reached about this research problem? ("a list of hypotheses and empirical claims you've shown (or disproven!)")
4. Technical setup: key things you quantify, how defined and measured; models, datasets, prompts, metrics.
5. What is the strongest evidence you found against these hypotheses?
6. What are the biggest limitations to your results? Could you have addressed them?
7. How did you use LLMs in this research task? How for writing the doc and these answers? How did you make sure they weren't giving you slop? ("give specific examples of mistakes you caught, and ways you changed what the LLM did for the better. And explain how much of the text here was written by you vs an LLM, and why")
8. What, if any, prior experience do you have with mechanistic interpretability?
9. Other than your research task, 1-3 pieces of evidence you'd be able to do good research in the program (aim for 100 words; not the project).
10. Why are you interested in Neel's stream specifically?
11. Likelihood you will join the training program (Sept 28 - Oct 30) if accepted?
12. (Optional) Anything else important about your application project?

## Martin's draft answers in the form at the time of saving

Q1: How does Qwen 3.5 4b represent harmfulness and illegality, and do they influence eachother?

Q2: Harmfulness and illegality are related concepts, and they are often conflated. If a model falsely fires on illegality when something is harmful but legal, or falsely fires as legal on something which is illegal but harmless, this is an alignment problem when attempting to make law-following AI.

Q3: Hypotheses:
- Illegality will be similarly represented to harmfulness (lie in a similar direction).
- The harmfulness of a given prompt will influence if the model considers it illegal.
- Higher harm prompts will activate higher illegality, and vice versa.
- The model will be less accurate in recognizing harmful+legal and harmless+illegal prompts than in recognizing harmful+illegal and harmless+legal prompts.
- Some legal but harmful prompts will be wrongly classified as illegal.
- Some illegal but harmless prompts will be wrongly classified as legal.

Q4 (in progress): Model: Qwen 3.5 4b (through GitHub API calls)
Dataset: 371 LLM-generated (Claude Fable 5.1) scenario prompts, LLM-labelled (Claude Fable 5.1.) as legal/illegal and harmful/harmless, all hand-checked by a lawyer.
Logistic regression probes for illegality on the last-token residual in two conditions over the scenario dataset: (1) the sentence alone (2) the sentence followed by "is this legal? Yes or No", read at the token before the answer. The illegality probe was trained only on
