# My defense prep — *Surface Suppression, Not Erasure*

Everything I should have in my head walking into the room: what each piece actually is, why I made the choices I made, the numbers I might be asked for, and how I'd answer the hard questions out loud. Written the way I'd explain it, not as a spec sheet. Grounded in my own code (`run_mech_metaeval.py`, `run_metric_faithfulness.py`, `run_sensitivity.py`, the lens runners) and the papers I cite.

---

## The one paragraph I never want to fumble

Unlearning is supposed to make a model forget a specific fact without retraining from scratch. The problem is that we only ever check whether it worked by looking at what the model *says*. But a model can stop saying a fact while still knowing it perfectly well underneath — and quantizing, relearning, or probing brings the fact right back. So the real question is whether a fact was genuinely *erased* from the weights or just *suppressed* at the output. My contribution is a way to answer that by reading the fact directly out of the model's internal computation, packaged as a metric (D_mech) that passes the field's own faithfulness test — the only metric that does.

---

## What unlearning actually is, and the distinction the whole thesis turns on

You start with a model trained on data that splits into a forget set and a retain set. The goal is a model that behaves as if the forget set had never been in training, while keeping everything else intact. The clean way to do that is to retrain from scratch on the retain set alone — that's "exact" unlearning, and it genuinely removes the information, but it's far too expensive for an LLM. So in practice everyone *approximates* it, by optimizing a combined objective: degrade the model on the forget set while a weighted retain term protects general utility. That word "approximate" is where everything goes wrong, because an approximation can look correct on the surface and be completely wrong underneath.

The distinction I keep coming back to is erasure versus suppression. **Erasure** means the fact is not in the weights anymore — that's what retraining gives you, and there's no way to revive the answer. My stand-in for that ideal is Hubble's Standard model. **Suppression** means the fact is still sitting in the weights, just masked at the output — and it comes back the moment you quantize, relearn, or probe. The trap is that these two states behave *identically* on the exact question you asked. They're different states of the model that happen to agree on the queried distribution. My metric is a way to pull them apart.

---

## The four methods I study, and the fingerprint each one leaves inside the model

This is the part where I want to sound like I actually understand these methods rather than just citing them, so for each one I pair *how it forgets* with *what it looks like in my internal readouts*.

**NPO** (Zhang et al., 2024) is preference-based — it's a DPO-style objective that pushes the forget-set likelihood down using a bounded loss that resists the catastrophic collapse plain gradient ascent suffers from. In my data it shows up as late, output-level suppression: the fact sits at rank 364 mid-stack but 3,864 at the final layer, a 10.6× jump. It's also the most scale-robust method — the only one that actually drives the fact deep at 1B.

**GradDiff** (Jang et al., 2022; Yao et al., 2023) is the loss-based baseline: gradient ascent on the forget loss, descent on the retain loss. It's the crudest of the four and can be unstable. It gives the strongest late suppression I see — 370 mid-stack to 9,192 final, a 24.9× jump — but that only holds at 8B; it falls apart at 1B.

**IDK** (Maini et al., 2024) is a refusal policy — it just fine-tunes the model to answer "I don't know" on forget queries. So it changes the *output policy* and never touches the underlying knowledge. It's the purest surface case in my whole study: the fact stays around rank 9 from mid-stack all the way to the output, a final/mid ratio of 0.1×, which I label "no real suppression." Critically, IDK's final-layer rank (9) is identical to the retrained Standard model's (9) — at the output they're indistinguishable, even though internally IDK is on the "knows" side and Standard is not.

**RMU** (Li et al., 2024) steers the representation — it perturbs activations at chosen layers away from the fact while trying to preserve the retain activations. It's the one method that acts early and deep (mid 2,240, final 2,693, ratio 1.2×). But here's the tell: a few relearning steps reverse RMU the fastest of any method, because it *scrambled* the representation rather than removing the fact — the information is still there, just rotated, so a little fine-tuning re-aligns it. Depth of suppression is not the same as robustness of removal, and RMU is the clean example of that.

There's a fifth family, distillation, that I flag but don't evaluate. **UNDIAL** (Dong et al., 2025, NAACL) is self-distillation with adjusted logits: you subtract the forget token out of the model's own logits and then fine-tune the model to match those adjusted logits — a soft, stable target that avoids the over-unlearning you get from gradient ascent, and it scales to large and sequential forget requests. **DUET** (Zhong, Yang & Zhu, 2026, ICLR) distills from a prompt-steered teacher instead: the teacher is the base model given a system prompt telling it to refuse the topic, and the student is a copy whose weights are trained to imitate the teacher's first-token logits, so it fuses cheap in-context unlearning with persistent, attack-resistant tuning. The reason this family matters to me specifically is a 2025 paper called *Distillation Robustifies Unlearning*, which argues distillation is the thing that actually converts suppression into robust erasure — which makes it the leading candidate to *pass* D_mech, and the obvious next experiment.

---

## Why Hubble, and not the other benchmarks

The honest reason I use Hubble is that you cannot validate an internal "did it forget?" metric unless you already know the true answer for each fact — and Hubble is the only benchmark that hands me that ground truth. It's a suite of open models (1B and 8B, pretrained on 100B and 500B tokens) where controlled facts were *inserted during pretraining*, so for every fact I get two paired models: a Perturbed model that genuinely knows it, and a Standard model trained identically but without the insertion, so it genuinely never knew it. Those two are my anchors — knows and never-knew. The four unlearning methods all get applied to the Perturbed model. On top of that, the Hubble authors deliberately untied the output embeddings *specifically* so that logit and tuned lenses would work, so the whole lens approach is valid here by design rather than by luck.

The other benchmarks don't give me that. TOFU (Maini et al., 2024) is fictitious-author facts with forget/retain splits — I use it for cross-benchmark replication at 1B/3B/8B, but it has no paired knows/never-knew models. WMDP (Li et al., 2024) is hazardous-knowledge removal scored by multiple-choice accuracy — pure output. MUSE (Shi et al., 2024) adds six evaluation axes and RWKU (Jin et al., 2024) tests real-world entity knowledge the model already holds, but both still score behavior, and neither isolates a controlled inserted fact.

The concrete setup: I unlearn one attribute, a person's birthdate, over YAGO biographies. Birthdate is a deliberate choice because it's short, single, and checkable, so I can point at the exact token the model should or shouldn't produce. The facts are memorized by duplicating biographies during pretraining; that duplicated set splits fifty-fifty into a forget set of about 44 facts and a retain set of about 45, with a control set of 250 biographies that were never duplicated. Everything runs at both scales — 1B has 16 layers, 8B has 36. To query a fact I give the model a biography that stops right before the date and follow Hubble's own protocol: score the true date against nine distractors and take the most likely.

---

## Why the current metrics fail — and it's not the reason you'd guess

The important subtlety here, the one I want to land clearly, is that the output metrics are *not* bad at telling knowers from non-knowers. Several of them are excellent at it. Answer-probability discriminates at 0.928, exact-memorization at 0.818, truth-ratio at 0.973. Where they all fall apart is the *other* requirement — invariance. Reword the question and the score lurches: answer-probability's invariance shift is 0.999, essentially the worst possible; ROUGE and exact-memorization are around 0.99; truth-ratio 0.974. Membership-inference scores (loss, Min-K%, Min-K%++, zlib) fail differently — they barely discriminate at all (0.16–0.31). A metric only counts as faithful if it does *both*, and none of these do.

The single most useful data point in this whole section is the naive internal metric, min-rank. If the story were simply "output bad, internal good," min-rank should work. It doesn't — its invariance shift is a perfect 1.000, so it fails exactly as badly as the lexical metrics, and its faithfulness score is 0.000. That's the evidence that the fix isn't "look inside," it's the specific dual-anchor construction I build. D_mech, by contrast, lands at invariance 0.466, discrimination 0.827, faithfulness 0.442 — the only metric with both halves.

---

## The lenses, and the corrections that make the cheap one trustworthy

A lens decodes a hidden state in the middle of the network into a distribution over the vocabulary, so I can read the rank of the correct answer at every layer and get a rank-by-layer trajectory. There are three of them and they form a ladder from correlational to causal.

The **logit lens** is parameter-free — it just applies the model's own final normalization and unembedding to an intermediate state, reading off what the model "believes" so far. The **tuned lens** (Belrose et al., 2023) trains a little affine map per layer to predict the final output; it's more accurate, but it *skips ahead* — it decodes the answer early, sitting near rank 10 from layer 0 for basically every model, which means it hides the very suppression I'm trying to see. The **Jacobian lens** transports the state through the model's own averaged Jacobian of the logits with respect to the hidden state — a first-order causal readout of what the representation is disposed to make the model say; it confirms the story causally but it needs a reference corpus and it's weaker at 1B.

The point I have to be ready to defend is that I build the metric on the *weakest* lens, and that's deliberate: an evaluation metric has to be a reproducible, parameter-free property of the model, and only the logit lens is that. Higher fidelity is actively worse *as a metric basis* — the tuned lens collapses the trajectory differences I measure, so it's a great cross-check and a useless foundation.

The logit lens has a bad reputation for being unreliable, and rightly so if you use it naively, so I make two corrections. First, double-normalization: the last hidden state is already normalized, and re-applying the final norm creates a spurious spike at the end — I use the true final logits instead. Second, boundary-safe tokenization: a trailing space makes naive code track the wrong first answer token, so I strip and re-join around the answer boundary. Together these drop the mid-stack ranks by five- to forty-fold and kill the false end-spike — the cleanest example is IDK's 8B forget rank going from 958 down to 9. And once corrected, the tuned lens and the Jacobian lens both independently reproduce the ordering, so it isn't a lens artifact. There's also a null control for the Jacobian lens: transporting the fact's own state versus transporting the fact-agnostic mean state, and the real transport only falls below the null for the *knowing* model — which shows the descent is driven by the fact itself, not by some generic prior.

---

## D_mech, and every design choice behind it

The metric, exactly as it's computed in the code: for a model's rank trajectory over a band of layers, I measure its average log-distance to the never-knew anchor and subtract its average log-distance to the knows anchor.

> D_mech = mean over layers in B of |log r − log Ø|  −  mean over layers in B of |log r − log K|

K is the "knows" anchor — the median trajectory of the Perturbed models. Ø is the "never-knew" anchor — the median trajectory of the Standard models. B is the mid-to-late band, layers 18–36 at 8B (the code sets this at the halfway layer to the end). A positive D_mech means the trajectory sits closer to K, i.e. the fact is only hidden; negative means closer to Ø, genuinely absent.

Every piece of that is there because a simpler version broke. I read *internally* rather than at the output because confidence is not knowledge — a model can be totally confident in a wrong token. I use the *whole trajectory* rather than a single rank because one number is fragile: it ties models that behave differently and it swings when you reword the question. I take the *log* of rank so that the difference between rank 1 and rank 10 counts for far more than the difference between 4,000 and 5,000, which is just noise. I use *two anchors and subtract* because that's what makes the score relative, and relativity is where the paraphrase-stability comes from. And I build it on the *logit lens* because the metric has to be reproducible.

One detail that matters for rigor: the anchors are leave-one-fact-out. When I score fact *i*, K and Ø are the medians over the *other* 43 facts, so no fact is ever compared against itself — that closes off any circularity.

The deepest point about why it's faithful — and this is the one I most want to get right if pushed — is that the faithfulness comes from the *construction*, not from the mere fact of "using internals." D_mech subtracts the distance to Ø from the distance to K. When a paraphrase shifts the whole trajectory up or down, it shifts *both* of those distances by roughly the same amount, so the shift cancels in the difference. That cancellation is a designed property of the anchor difference, and it needs one fixed, shared transport across all models — which is exactly what the parameter-free logit lens provides. min-rank has no anchor difference, nothing cancels, and that's why it fails invariance completely.

A concrete number I like to have ready, from the sensitivity output: on the 44 forget facts, IDK's per-fact D_mech has a median of +0.594, 77% of facts land on the "knows" side, and IDK is closer to the knows anchor than Standard is on 100% of facts (Cliff's delta +0.997 versus Standard, Wilcoxon p ≈ 1e-13). Standard's own median is −1.363, sitting firmly on the never-knew side where it should. So IDK looks forgotten at the output but sits on the knows side internally, which is the whole thesis in one comparison.

---

## The faithfulness test I hold myself to

I could just assert my metric is good, but that's circular, so I subject it — and every competing metric — to the field's own standard, the faithfulness meta-evaluation from OpenUnlearning (Dorna et al., 2025). It asks two things of a metric. Discrimination: models that hold a fact must score differently from models that don't. Invariance: reword the query and the score must not move. I set up four pools of 44 facts each. The Positive side is meant to read "knows" — that's the Full canonical prompt and a Paraphrased prompt, both on the knowing model. The Negative side is meant to read "absent" — that's Retain and Perturbed-wrong.

Operationally, invariance-shift is the absolute Cliff's delta between the Full and Paraphrased pools (0 is perfect, lower is better), discrimination is the absolute Cliff's delta between the Positive and Negative pools (1 is perfect, higher is better), and the composite faithfulness score is discrimination times one-minus-invariance-shift. The paraphrase I use for the invariance test is a single fixed meaning-preserving rewrite of the trigger, "was born on" becoming "has a recorded birth date of" — and the fact that it's a single paraphrase is a limitation I own up to.

On the statistics: everything is non-parametric, because rank trajectories are ordinal and non-normal. I use Wilcoxon signed-rank within the Positive pool for invariance, Mann–Whitney U between pools for discrimination, Cliff's delta as the effect size throughout, and TOST for equivalence. At n = 44 the Wilcoxon p-value saturates, so I judge invariance by effect size rather than a p < 0.05 threshold, with confidence intervals from a 4,000-resample bootstrap over facts with the anchors held fixed. Cliff's delta itself, if I'm asked to define it, is P(X>Y) minus P(X<Y) for two samples — it runs from −1 to +1, zero means the distributions fully overlap and ±1 means they're completely separated, and it's robust to outliers and ties in a way a t-test isn't.

On whether I cherry-picked the band 18–36 — I didn't, and I have the sweep to prove it. Across bands 12–30, 18–36, and 24–36, the faithfulness scores are 0.276, 0.441, and 0.276. D_mech is faithful in all three (discrimination stays positive, the ordering holds); 18–36 is simply where the pools separate most cleanly, and it lines up with where factual recall is known to consolidate in the network. It's reported as a sensitivity analysis, not a single lucky choice.

---

## The results, with the numbers I might be asked for

The core picture is the rank-by-layer plot: the retrained Standard model never recovers the fact at any layer on the forget set — that's what genuine erasure looks like — while every unlearned model keeps the fact fully recoverable mid-stack and only pushes it down near the output. The output said forgotten; the inside says the trace is intact.

If someone wants the final-layer ranks in numbers, here they are as (8B-forget / 8B-retain / 1B-forget / 1B-retain): Standard 9/10/10/11, Perturbed 1/1/4/3, NPO 3,863/1/44,226/1, GradDiff 9,191/1/8/3, IDK 9/1/16/1, RMU 2,693/2/8/6. The one to point at is IDK's 8B-forget 9 sitting exactly on Standard's 9 — a single final number literally cannot separate a policy change from genuine removal.

The mid-stack signatures turn the four methods into a taxonomy via the final-over-mid ratio: IDK 68→9, ratio 0.1×, no real suppression; NPO 364→3,864, 10.6×, late output suppression; GradDiff 370→9,192, 24.9×, the strongest late suppression; RMU 2,240→2,693, 1.2×, deep and early. The tuned-lens cross-check reproduces that ordering exactly (Perturbed 1, IDK 9, Standard 10, then NPO, GradDiff, RMU), which is how I know it isn't a logit-lens artifact.

The confidence slide is the sharpest single case: on "Billy Fisher was born on," the knowing model says "May" at 100%, and NPO, GradDiff, and IDK are *also* about 100% confident — in wrong tokens like "e," "into," "a." Equally confident in a wrong answer as the knowing model is in the right one, which is exactly why output-probability metrics are uninformative.

The headline result is the faithfulness table I described above — D_mech the only metric with both low invariance shift and high discrimination. It replicates on TOFU at all three scales: faithfulness 0.556 at 1B, 0.599 at 3B, 0.557 at 8B, each above every generation-overlap and answer-probability metric (a couple of likelihood aggregates like truth-ratio get competitive there, and I say so — the margin is benchmark-dependent). And it's robust to how you aggregate: D_mech ranks first out of eleven metrics under all five aggregation formulas on Hubble-8B, and its composite faithfulness is roughly 18× the best lexical metric.

Two more results give the internal reading predictive teeth rather than just descriptive value. Under attack, RMU (the deepest mid-stack suppression) relearns fastest and IDK (the surface method) leaks most under quantization — both orderings match my mechanistic taxonomy, so the internal signature *predicts* attack behavior. And a completely separate internal probe on a *second* attribute, nationality, separates knows from absent at p < 1e-11 and orders the methods the same way, so the readout is tracking knowledge, not a one-off measurement artifact.

---

## The papers, grouped by why they're in the deck

On the unlearning side: Bourtoule et al. (2021) and Cao & Yang (2015) formalize unlearning (SISA, exact vs approximate). Zhang et al. (2024) is NPO, Jang et al. (2022) and Yao et al. (2023) are the GradDiff lineage, Maini et al. (2024) is both TOFU and the IDK baseline, and Li et al. (2024) is WMDP and RMU. Rafailov et al. (2023) is the DPO objective NPO builds on, Eldan & Russinovich (2023) is the early "Who's Harry Potter?" unlearning, Dong et al. (2025) and Zhong et al. (2026) are UNDIAL and DUET, and Liu et al. (2025) is the survey framing.

On the "it hides rather than erases" evidence: Zhang et al. (2025) is the quantization-revival result, Lynch et al. (2024) and Fan et al. (2025) are relearning attacks, Thaker et al. (2024) and Deeb & Roger (2024) are black-box and weight-probe extraction, and Patil et al. (2023) is the one that recovers "deleted" answers by internal probing — but only ever as an attack, which is precisely my gap.

On interpretability: nostalgebraist (2020) is the original logit lens, Belrose et al. (2023) is the tuned lens, and Elhage et al. (2021), Geva et al. (2021), and Meng et al. (2022) are the circuits / FFN-memory / ROME work that says facts live in the mid-to-late layers, which is why my band sits there. Guo et al. (2025) and Khoriaty et al. (2025) use interpretability to *edit or forget* — not to *measure*, which again is my gap.

On evaluation and framing: Dorna et al. (2025) is the OpenUnlearning faithfulness meta-evaluation I adopt, Shi et al. (2024) and Jin et al. (2024) are MUSE and RWKU, Carlini et al. (2021) is the training-data-extraction motivation, Wei et al. / Hubble Team (2025) is my benchmark, and GDPR Article 17 (2016) is the legal right-to-erasure driver.

---

## The questions I expect, and how I'd answer them in the room

**Isn't the logit lens unreliable?** Naively, yes, which is why I make two corrections — no double-normalization, and boundary-safe tokenization — that drop mid-stack ranks five- to forty-fold and remove a false end-spike (IDK 958 to 9). And I don't trust it alone: the tuned lens reproduces the ordering exactly and the Jacobian lens confirms it causally. Three lenses agree.

**Why build the metric on the weakest lens?** Because a metric has to be a reproducible, parameter-free property of the model. The tuned lens is trained and skips ahead — it sits near rank 10 from layer 0 for every model and collapses the trajectory differences I measure. The Jacobian lens needs a corpus and weakens at 1B. So the fancier lenses are cross-checks; the logit lens is the only defensible foundation.

**What makes D_mech faithful when min-rank fails just as badly as ROUGE?** The dual-anchor difference. A paraphrase shifts the whole trajectory, and because I subtract distance-to-Ø minus distance-to-K, that common shift cancels. It's a designed cancellation, not a lucky feature of hidden states, and it needs one fixed shared transport, which is the logit lens. min-rank has no anchor to difference against, so nothing cancels.

**Did you cherry-pick the band?** No — I swept 12–30, 18–36, and 24–36; D_mech is faithful in all three, and 18–36 is just where the pools separate most cleanly, which happens to be where factual recall consolidates. It's in the thesis as a sensitivity analysis.

**n = 44 and a single paraphrase — is that enough?** It's a genuine limitation and I flag it. My defenses are that I judge invariance by effect size rather than p-values precisely because Wilcoxon saturates at this n, with bootstrap confidence intervals, and that the result replicates on TOFU across three scales and ranks first under all five aggregation formulas. Hardening it with many sampled paraphrases and wrong-targets is stated future work.

**Why Cliff's delta and non-parametric tests?** Rank trajectories are ordinal and non-normal, so a distribution-free effect size that's robust to ties and outliers is the right tool, whereas a t-test would assume structure that isn't there.

**The advantage weakens at 1B — doesn't that undercut you?** It's expected and I explain it: at 1B the base model memorizes weakly, so the knows and never-knew anchors sit closer together and the separation shrinks. The effect is real but smaller — honest scale-dependence, not a contradiction.

**You won't have a never-knew reference model in deployment — is this usable outside Hubble?** As presented it's an evaluation metric — its job is to validate unlearning *methods* under controlled ground truth, which Hubble supplies. Turning it into a deployment audit would need a proxy for the retrained anchor, a leave-out or a lightly-retrained reference, and that's clearly future work. But the scientific claim — that today's methods suppress rather than erase — only needs the benchmark setting.

**Couldn't a method be built to fool D_mech?** Any metric can be Goodharted in principle, but D_mech is harder to game than an output metric because it reads the whole mid-stack trajectory rather than a single output the method optimizes against. To make the internal trajectory match the never-knew anchor across the entire band, you'd essentially have to *actually* erase the fact — which is the point. Distillation is the most likely thing to pass it honestly.

**Isn't this just re-describing the known attacks?** The attacks reveal residual knowledge, but only destructively and after the fact, and none of them is a metric. D_mech is a non-destructive, quantitative read you can compute directly, and it *predicts* the attacks — the mid-stack ordering matches which methods relearn fastest and quantize-leak most. It folds the attacks' signal into a single number.

**How can NPO be 100% confident in a token like "e"?** The forget objective pushes mass off the true answer, and with a degenerate optimum the mass piles onto a nearby high-frequency token. High confidence, zero knowledge — which is exactly why output-probability metrics tell you nothing here.

**Why does IDK tie the retrained model at the output but differ internally?** IDK only changes the output policy — it says "I don't know" and never touches the representation, so the final rank looks like a model that never learned the fact (both around 9), while the mid-stack trajectory is intact end-to-end and sits closer to knows on 100% of facts. The final number can't tell a policy change from genuine removal; the trajectory can.

**Would distillation actually erase?** That's the open question I'd most like to run. There's evidence (*Distillation Robustifies Unlearning*, 2025) that distilling an unlearned model into a fresh student makes forgetting robust to relearning — genuine erasure — in which case D_mech should be exactly the metric that detects the difference between distillation and the four suppression methods.

**What's the one contribution?** The first faithful, internal evaluation metric for unlearning — turning an interpretability readout into a measurement, validating it on the field's own faithfulness test, and using it to show that current methods suppress rather than erase.

**Why should anyone care beyond the benchmark?** Because compliance and safety decisions about whether data was removed are currently made on output tests that a few fine-tuning steps overturn. If "forgotten" really means "hidden," then anything deployed needs an internal check before it ships — which is what a faithful internal metric is for.
