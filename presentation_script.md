# Presentation Script — *Surface Suppression, Not Erasure*
### Master's Colloquium · slides 1–43 · target ≈ 30 min (soft) + 15 min Q&A

**How to use this:** the **bold lead-ins** are the transitions that carry the story from one slide to the next — say them and the deck flows as one argument instead of 43 separate facts. Rough timing is marked per section. Speak the plain text; the *(italics)* are delivery notes, not spoken.

**The one-line story you are telling:** *Unlearning is judged by what a model says. But a model can stop saying a fact while still knowing it. I built a way to read the knowledge directly, from inside the network, and showed it is the only evaluation that survives the field's own faithfulness test.*

---

## 1 · Motivation  (~3 min)

**Slide 1 — Title**
> Good morning, and thank you all for being here. My thesis is titled *Surface Suppression, Not Erasure* — evaluating language-model unlearning with mechanistic interpretability. I'm Harsh Parikh, from the Chair of Responsible Data Science. Over the next half hour I want to convince you of one thing: that the way we currently judge whether a model has "forgotten" something is measuring the wrong layer of the model, and that we can do better by looking inside.

**Slide 2 — Agenda**
> Here's the path. I'll start with why unlearning matters and the specific question I attack. Then related work — how unlearning is done and scored today, and the gap I found. Then my setup, the interpretability tools I use, the metric I propose, the results, and a discussion of what it all means. *(Point along the chevrons briefly, then move on — don't dwell on the agenda.)*

**Slide 3 — Motivation (divider)**
> Let's start with the motivation.

**Slide 4 — Machine unlearning removes specific knowledge without the cost of retraining**
> Machine unlearning is the task of making a trained model forget a specific piece of information — on demand, without paying to retrain the whole thing from scratch. **And it matters for three very concrete reasons.** Privacy: under the GDPR right to erasure, a person can ask that their data be removed. Copyright: an owner can demand a model stop reproducing protected content. And AI safety: we may want to strip a dangerous capability out of a deployed model. Formally, you're given a *forget set* and a *retain set*, and you want a model that behaves as if the forget set had never been in the training data. **So that's the promise. The rest of my talk is about a crack in that promise.**

---

## 2 · Related Work  (~6 min)

**Slide 5 — Related Work (divider)**
> To see the crack, we need the lay of the land.

**Slide 6 — Formalizing Unlearning**
> The problem was first formalized in the SISA line of work. You have a model trained on data that splits into a forget set — the influence you want to remove — and a retain set — the behavior you want to keep. **The gold standard is simple to state and impossible to afford:** just retrain from scratch on the retain set alone. Because that's prohibitively expensive, every practical method instead *approximates* it, by minimizing a combined objective: push the loss up on the forget set, keep it down on the retain set. **That word — approximate — is the whole ballgame. Approximations can look right on the surface and be wrong underneath. Which brings us to how people actually do it.**

**Slide 7 — Unlearning method families**
> There are five families, and I study the four established ones. NPO is preference-based — it pushes down the likelihood of the forget answer with a bounded objective that resists collapse. GradDiff is the loss-based baseline — gradient ascent on the forget loss, descent on the retain loss. IDK is a refusal policy — it just fine-tunes the model to answer "I don't know." And RMU steers the internal representation — it perturbs activations at chosen layers away from the fact. **The fifth family, distillation — UNDIAL and DUET — is very recent and I flag it but don't evaluate it; I'll come back to why it's interesting in the discussion.** The thing to notice: these four forget in completely different *places* — the objective, the output policy, the representation. **Hold that thought, because later I'll show those different places leave different fingerprints inside the model.**

**Slide 8 — Current Unlearning benchmarks and Metrics**
> How do we score any of this? On the left, the benchmarks. TOFU uses fictitious authors with clean forget/retain splits. WMDP targets hazardous knowledge. MUSE and RWKU add more axes and real-world entities. And Hubble — the one I build on — is different in a way that turns out to be essential, and I'll detail it in the setup. On the right is the part that matters for my argument: **every metric family in use today reads the model's *output*.** Generation overlap, like ROUGE. The probability it assigns the answer. Likelihood ratios. Membership inference. **Every single one reads text the model produces. Not one reads what the model computes. Keep that in mind for the next slide.**

**Slide 9 — Robustness attacks reveal that unlearning often hides rather than erases**
> Because here's what happens when you stress-test these "forgotten" models. Quantize an unlearned model — just change its numerical precision — and the forgotten knowledge can come back. Take a few fine-tuning steps — a relearning attack, shown on the right — and it returns fast. Prompt it cleverly, or probe its weights, and you extract it. **So the output metrics say "forgotten," and then four different attacks say "no it isn't."** The knowledge was never gone. It was hidden. And notice — internal probing already appears in this literature, but only ever as an *attack*, never as a way to *measure*. **That asymmetry is exactly the opening for this thesis.**

**Slide 10 — Research Gap**
> Let me name the gap precisely. Interpretability has been used in unlearning — to *edit* models, to clamp features. And reading inside the model has been used — but, as I said, only to attack. Separately, faithfulness of metrics has been studied — but only for output metrics. **Nobody has turned an interpretability readout into a faithful, internal evaluation *metric* for unlearning.** That is the gap. That is what I build.

**Slide 11 — Suppression at the output versus erasure in the representation**
> And here is the question stated as sharply as I can. Two models can give you the exact same answer for opposite reasons. **Erasure** — the ideal, what retraining gives you — means the fact is genuinely not in the weights; the answer cannot be revived. **Suppression** — what unlearning actually does — means the fact is still in the weights, just masked at the output; quantize, relearn, or probe, and it returns. **On the queried question these two look identical. So the real question of my thesis is:** has the model genuinely forgotten the fact, or only learned to hide it? To answer that, I need a setting where I actually know the ground truth.

---

## 3 · Experimental Setup  (~5 min)

**Slide 12 — Experimental Setup (divider)**
> Which is exactly what this benchmark gives me.

**Slide 13 — Hubble suite**
> Hubble is a suite of open models built to study memorization. They pretrained language models — at 1B and 8B — and *inserted controlled facts* during pretraining. **The crucial part is that for every fact you get two reference models.** A *Perturbed* model that had the fact inserted — so it genuinely knows it. And a *Standard* model trained identically but without that insertion — so it genuinely never learned it. **That pair is the ground truth I need: a model that knows, and a model that never knew.** The Standard model is my stand-in for the ideal retrained model — erasure. The four unlearning methods are all applied to the Perturbed model. So for every fact I have "knows," "never knew," and four "supposedly forgot."

**Slide 14 — Dataset and splits**
> Concretely, I unlearn one attribute — a person's birthdate — over YAGO biographies. The facts are memorized by duplicating biographies during pretraining; that duplicated set splits fifty-fifty into a forget set of about 44 facts and a retain set of about 45, and there's a control set of 250 biographies that were never duplicated. **Birthdate is a deliberate choice — it's a short, checkable, single attribute, so I can point at exactly the token the model should or shouldn't produce.** Two scales throughout: 1B with 16 layers, 8B with 36.

**Slide 15 — Infill Probe**
> Here's how I query a fact. I give the model a biography that stops right before the birthdate — "Billy Fisher was born on ___" — and I follow Hubble's own protocol: score the true date against nine distractors and take the most likely. That's infill accuracy. **This gives me one clean position where the answer should appear — which is the position I'll later read from the inside.**

**Slide 16 — Infill-accuracy output**
> And here's what that output measure says about unlearning. The Perturbed model that knows scores highest, as it should. And every single unlearned method collapses down near the Standard, never-knew floor. **By this measure, unlearning simply works — and, more importantly, all four methods look the same. You cannot tell them apart, and you certainly can't tell suppression from erasure.** That's the failure of output evaluation, in one chart. **So let's stop reading the output.**

**Slide 17 — Hubble paper's recommendation**
> And I'm not the first to suggest that. The Hubble authors themselves point the way: they deliberately untied the output embeddings *specifically so that* interpretability methods like the logit and tuned lens would work, so that studies could examine how the models internalize facts. **They built the door. This thesis walks through it. So — what is a lens?**

---

## 4 · Interpretability Lenses  (~3 min)

**Slide 18 — Interpretability Lenses (divider)**
> Two minutes on the tool, because the whole method rests on it.

**Slide 19 — What is a lens**
> Mechanistic interpretability asks what happens *inside* a network, not just at its output. A transformer processes a token through a stack of layers — the residual stream, the model's running computation. **A lens is a simple idea: instead of only reading the answer at the very end, you read it at every layer along the way.** At each layer you can ask, "how highly does the model rank the correct answer *here*?" Do that for every layer and you get a trajectory — the rank of the fact as it forms, layer by layer. **That trajectory is the raw material for everything that follows.**

**Slide 20 — Three Lenses**
> There are three ways to take that reading, and they form a ladder. The **logit lens** is parameter-free — it just applies the model's own final normalization and unembedding to an intermediate state, reading what the model "believes" so far. The **tuned lens** trains a small map per layer to predict the final output — more accurate, but, crucially, it *skips ahead*: it decodes the answer early, which can hide the very thing I want to see. The **Jacobian lens** transports the state through the model's own averaged Jacobian — a first-order *causal* readout. **The ladder runs from correlational to calibrated to causal. I'll use all three — but, for reasons that will become the punchline, my metric is built on the humblest one, the logit lens.**

---

## 5 · Proposed Metric  (~5 min)

**Slide 21 — Proposed Metric (divider)**
> So let's build the metric.

**Slide 22 — How we arrive at D_mech**
> I want to be honest that the metric didn't arrive fully formed — it's what survived a series of failures. **We tried the obvious thing first — read the output token and its probability. It broke, because a model can be completely confident in a wrong token; confidence is not knowledge.** So we went inside and read the rank. **We tried a single number — the final or best rank. It broke, because one number is fragile: it ties models that behave differently, and it swings when you reword the question.** So we used the whole trajectory. **We tried raw rank distances. They broke, because the gap that matters — rank 1 versus 10 — gets drowned out next to noise like 4,000 versus 5,000.** So we took the log. **And we tried comparing to a single reference model. That broke too — with nothing to compare against, the score drifts across phrasings.** So we used two anchors. **Every one of those fixes points the same direction, and together they define the metric.**

**Slide 23 — D_mech: distance metric using mechanistic interpretability**
> Here it is. For a given model on a given fact, I take its rank trajectory over the mid-to-late layers — layers 18 to 36 at 8B, where factual recall actually forms. I measure its log-distance to the *never-knew* anchor, and subtract its log-distance to the *knows* anchor. The log turns rank into a ratio, so early ranks count. The two anchors — K, the pre-unlearning "knows" model, and Ø, the retrained "never-knew" model — make the score *relative*, and that relativity is the source of its stability under paraphrase. And a detail that matters for rigor: the anchors are the median trajectories of the *other* 43 facts, leave-one-out, so a fact is never compared against itself. **One more choice — it's built on the logit lens, because an evaluation metric has to be a reproducible property of the model, not something you trained.**

**Slide 24 — Is the trajectory closer to the pre-unlearning or the retrained anchor?**
> Intuitively, forget the formula and picture a number line. On one end, the model that knows the fact. On the other, the model that never learned it. **D_mech just asks: where on that line does the unlearned model's internal representation actually sit?** If it lands near "never learned," the fact was genuinely removed. If it lands near "knows" — as we'll see it consistently does — then it was only hidden. A positive D_mech means "closer to knows."

**Slide 25 — Meta-evaluation of D_mech**
> Now, I could claim my metric is good, but that would be circular. So I hold it to the field's *own* standard — the faithfulness test from the OpenUnlearning framework. A faithful metric must do two things. **Discrimination:** models that hold the fact must score differently from models that don't. **Semantic invariance:** if you merely reword the question, the score must not move. I set up four pools of facts — a Positive side that should score "knows," a Negative side that should score "absent" — and I test every metric family, mine included, against both requirements. **The rest of the talk is what that test found.**

---

## 6 · Results  (~7 min)

**Slide 26 — Results (divider)**
> Results. *(Optional orienting line:)* I'll first show what reading the inside reveals about the four methods, and then the head-to-head faithfulness verdict.

**Slide 27 — Reading the internal computation**
> This is the core picture. Every line is a model; the y-axis is the fact's rank by layer, log scale, lower means recoverable. **Look at the retrained, never-knew model in blue — on the forget set it never recovers the fact at any layer. That's what genuine erasure looks like.** Now look at the unlearned models: they push the fact down near the output, but mid-stack they keep it perfectly recoverable. **The output told us they'd forgotten. The inside says the trace is still there. That's the whole thesis in one plot — and every following slide is a consequence of it.**

**Slide 28 — Scale-dependent method verdict (8B vs 1B)**
> The picture also lets me *rank* the methods, and it changes with scale. At the smaller 1B model, only NPO actually drives the fact deep. The output would happily call IDK "forgetting" at 1B — but the lens shows the fact still sitting at rank 16, fully accessible. **So my verdict: NPO is the most scale-robust; GradDiff and RMU work at 8B but not 1B; and IDK is surface-level at both scales.** You only get that ranking from the inside.

**Slide 29 — Confidence is not knowledge**
> Here's the sharpest single case, and it's almost funny. Same prompt — "Billy Fisher was born on." The model that knows says "May," correct, 100%. And NPO, GradDiff, and IDK? They are *also* essentially 100% confident — in a completely wrong token, like "e" or "into." **They are exactly as confident in a wrong answer as the knowing model is in the right one. So any metric built on output confidence is not just noisy — it's actively uninformative about what the model knows.** That's why I stopped trusting the output.

**Slide 30 — Lens-table to read gold-token per layer**
> To make the internal reading tangible, this is the actual layer-by-layer top guess for one fact. On the left, the Perturbed model climbs — Fisher, Fisher, and then locks onto "May" in the final layers. On the right, NPO on the *same* fact decodes a meaningless token and never surfaces the answer. **This is what the metric is quietly reading on every fact — the presence, or absence, of the answer forming inside.**

**Slide 31 — Lens-table gold-token, final layer (8B vs 1B)**
> And zooming out to just the final token across methods and both scales — you can see the surface answers scatter into "e," "into," "a," "with." No pattern at the output. **The order lives inside, not here.**

**Slide 32 — The final-layer rank differs by method and scale**
> Put the final-layer ranks in a table and one number makes my point for me. Look at IDK's 8B forget rank: 9. Now look at Standard, the retrained model: also 9. **At the output, IDK and a model that genuinely never learned the fact are *identical*. Only the full trajectory — which is what D_mech reads — can tell them apart.** A single final number literally cannot.

**Slide 33 — Mid-stack versus output suppression**
> And the trajectories aren't just "different" — they're a taxonomy. The ratio of the final rank to the mid-stack minimum tells you *where* each method hides the fact. IDK: no real suppression, it just changes the output policy over an intact representation. NPO and GradDiff: the fact is alive mid-stack and only pushed down late. RMU: suppression that reaches early and deep. **Four methods, four distinct internal signatures — none of them visible at the output.**

**Slide 34 — Tuned-lens cross-check**
> Now, a fair objection: maybe the logit lens is just unreliable, and I'm reading noise. So I cross-check with the calibrated tuned lens. **It reproduces the logit-lens ordering exactly** — Perturbed lowest, then IDK, Standard, and the deeply-suppressed methods highest. The mid-stack numbers shrink, because the tuned lens skips ahead, but the *ranking is identical*. **My reading is not a lens artifact.**

**Slide 35 — Paraphrase stability**
> And this is the property that makes it a *metric* and not just an observation. I ask the same fact four different ways. The output measures swing wildly — the answer probability and ROUGE jump all over the place depending on phrasing. But inside, the gold month emerges the same way every time, and **D_mech stays positive across all four phrasings.** Lexical metrics track the wording; D_mech tracks the knowledge. **That stability is the thing the faithfulness test is about — so let's see the scores.**

**Slide 36 — Faithfulness meta-evaluation**
> Here is the headline result of the thesis. Every metric family, scored on invariance shift — lower is better — and discrimination — higher is better. A faithful metric needs both, and look down the column: **D_mech is the only metric that has both.** The output metrics all discriminate reasonably but shatter under paraphrase — invariance shifts near 1.0. And the naive internal metric, min-rank, fails just as badly. **The faithfulness comes from the dual-anchor construction, not merely from "using internals." Across five metric families, on Hubble-8B, D_mech is the only faithful metric.**

**Slide 37 — Discrimination vs invariance across metric families**
> The same result as a picture — the distribution of each metric across the four pools. D_mech cleanly separates the "knows" pools from the "absent" pools *and* stays put from Full to Paraphrased. The others separate but move, or don't separate at all. **D_mech alone sits in the faithful region.**

---

## 7 · Discussion  (~3–4 min)

**Slide 38 — Discussion (divider)**
> Let me pull this together.

**Slide 39 — Internal readout and common verdict**
> The common verdict across every experiment is this: at the output the four methods are indistinguishable, and the internal readout shows that none of them removes the fact's internal trace. The confidence contradiction is the cleanest proof that output confidence tells you nothing about internal knowledge. **Suppression and erasure are genuinely different states of the model — they just happen to behave the same on the exact question you asked.** Change the question, or look inside, and they part ways.

**Slide 40 — Critical analysis on four methods**
> Method by method: IDK is the purest surface case — it changes only the output policy; the representation is intact, rank 9 end to end. NPO and GradDiff keep the token accessible mid-stack and suppress only late. RMU acts early and deep — but, tellingly, a few relearning steps reverse it the fastest, because it scrambled rather than removed. **The unifying finding: every method leaves the token recoverable somewhere. Unlearning, as practiced, masks knowledge rather than deleting it.**

**Slide 41 — Higher lens fidelity does not make a better metric basis**
> One counterintuitive point I want to defend. You'd think the most sophisticated lens should be the metric's basis. It shouldn't. The tuned lens skips ahead and collapses the very trajectory differences D_mech reads — great as a cross-check, useless as the basis. The Jacobian lens confirms the taxonomy causally, but it needs a reference corpus and weakens at 1B. **The plain logit lens wins for the metric precisely because it's the only reproducible, parameter-free readout — and reproducibility is what an evaluation metric requires.**

**Slide 42 — Future work**
> Where this goes next. Harden the faithfulness claim with many sampled paraphrases and wrong-targets, more attributes, more benchmarks. Move to a stronger causal readout — from first-order Jacobian transport toward activation patching. And extend beyond single-fact recall to procedural and multi-hop knowledge. **And the one I'm most curious about — the distillation family I mentioned early on is the leading candidate to turn suppression into real erasure, which makes it the natural thing for D_mech to measure next.**

**Slide 43 — Thank you**
> To sum up in one sentence: forgotten facts, as produced by today's methods, remain internally accessible — they are suppressed, not erased — and D_mech is a faithful, internal way to see that, validated on the field's own test and replicated on a second benchmark. Thank you — I'm happy to take your questions.

---

### Delivery notes
- **Your two "land the plane" moments** are Slide 27 (the rank-by-layer plot — the whole thesis in one figure) and Slide 36 (the faithfulness table — the verdict). Slow down on both. Everything else is setup or support.
- **If you're running long,** the compressible slides are 28, 30, 31, 34 — you can state their one-line takeaway and move on. Do *not* cut 11, 16, 27, 36.
- **Callbacks that make it feel like one story:** "approximate" (Slide 6 → 11), "different places / fingerprints" (Slide 7 → 33), "every metric reads the output" (Slide 8 → 29/36), "the door Hubble built" (Slide 17 → 27), "confidence is not knowledge" (Slide 22 → 29).
