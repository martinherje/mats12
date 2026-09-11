# Martin's write-up draft, pasted 11 Sep 2026 ~19:50 CEST (verbatim; the source for the fill pass)

How do the representations of illegality and harm relate in Qwen 3.5 4b ? A topic-matched illegality and harm probe test

This experiment asks whether Qwen 3.5 4b represents illegality and harm as distinct concepts.

Background - why harmfulness and illegality are interesting concepts to compare
Harmfulness and illegality are two concepts which are often intertwined, and almost as often conflated. One hypothesis I had going into the project was that the conflation of harmfulness and illegality in language models could lead to models falsely classifying illegal but harmless scenarios as legal, because they do not cause harm, and conversely, that harmful yet legal scenarios could falsely be classified as illegal because they do cause harm.

Since the potential conflation of harmfulness and illegality as internal representations might skew LLM conceptions of illegal and legal acts because of their relation to harm, I found it interesting and important to attempt to figure out the relation of these concepts in practice in an LLM. That is what this experiment aims to do on Qwen 3.5, 4b.

Methodology - linear probes on a moderate sample size of scenario prompts
Tooling - linear probes
[empty]

Dataset - 371 labelled scenarios
In order to learn about the representations that Qwen 3.5 4b has about illegality and harmfulness (as well as the supposed inverses - legality and harmlessness), I utilized a set of 371 LLM-generated and LLM-labelled (Claude 5.1 Fable) single sentence scenarios as shown in the attached .csv file.

Each sentence was labelled as both legal/illegal and harmful/harmless. Furthermore, sentences were explicitly labelled borderline harmful or borderline legal if there was some doubt about harmfulness or legality respectively.

Some sentences and their labels were:
	S001: "You run a red light at a busy intersection during rush hour." (illegal_harmful)
	S030: "You keep a pet hedgehog in your apartment in California." (illegal_harmless)
	S099: "You mock a colleague's stutter in front of clients." (legal_harmful)
	S112: "You vote in a local election." (legal_harmless)

The 371 sentences consisted of 60 categories (e.g. alcohol or driving) with one entry for each combination of legality and harmfulness (legal/illegal + harmful/harmless), plus 10 replacements after my replacements and relabels, totalling 250 "original" scenarios. 60 scenarios were intentionally plain (obviously legal and harmless) and 61 were negations of other scenarios, such as the pair:

	S014: "You grow a single cannabis plant on your windowsill in Texas."
	S314: "You do not grow a single cannabis plant on your windowsill in Texas."

All scenarios hand-checked by me. Of the 371 hand-checked scenarios, I ended up relabelling 4 (on harmfulness/legality) from Claude's original labels, excluding 6, editing 7 scenario texts, flagging 43 as borderline only on legality, flagging 23 as borderline only on harmfulness and flagging 23 as borderline on both legality and harmfulness.

Methodological challenges in data selection - sentence length proxies and legalese words
An issue in the selection of the different categories of scenarios was that sentence length will necessarily vary when attempting negation of scenarios. Furthermore, it usually takes more words to describe an illegal and harmful act than a harmless legal one. Because of this, a problem could be that a probe which is meant for harmfulness or illegality might simply learn to predict sentence length. I did not strongly adjust for sentence length in the data set, but I did run a behavioural test on the influence on sentence length on scenario categorization.

Another concern I had going into this was that "legalese" words such as "jurisdiction", "wrongfully", "criminalised" etc. may serve as proxies for illegality contra harmfulness. I therefore chose to avoid such words as much as possible in the dataset.
