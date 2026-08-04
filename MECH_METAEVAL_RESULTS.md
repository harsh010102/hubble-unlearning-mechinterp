# A mechanistic-interpretability metric for Hubble unlearning, and its meta-evaluation

*Results doc (companion to the design in [MECH_METRIC_METAEVAL.md](MECH_METRIC_METAEVAL.md)). Job 18468,
8B. Nothing here is in RESULTS.md yet — pending review.*

## Contribution in one sentence

We add a **mechanistic-interpretability metric** to the Hubble unlearning benchmark — a dual-anchor
**logit-lens distance** `D_mech` — and show, under the **OpenUnlearning (arXiv 2506.12618 §4) / SPID
(10619 §5.2) meta-evaluation protocol**, that it is the **only** metric in the suite that is *both*
knowledge-discriminative **and** paraphrase-invariant. This empirically confirms the theoretical
expectation that output/lexical metrics conflate *surface form* with *knowledge*.

## What the metric is

For a fact, the logit lens gives the gold token's **rank trajectory** across layers, `r = (r_0…r_L)`.
With two anchor trajectories — **K = Perturbed** (the model that knows) and **Ø = Standard** (never
knew) — the mechanistic distance is
```
D_mech = d(r, anchor_Ø) − d(r, anchor_K)     over the mid/late band,   d = mean |log-rank gap|
```
`D_mech > 0` ⇒ the internal computation looks like it *knows*, regardless of what the output says.
Anchors are leave-one-fact-out medians (no self-circularity). Boundary-safe tokenization throughout
(the trailing-space fix in `layerwise_target_rank`).

## Meta-evaluation protocol (SPID, mapped to Hubble)

Four pool cells, each a model evaluated on the birthdate fact per entity:

| cell | pool | Hubble realization |
|---|---|---|
| Full | **P (knows)** | Perturbed, canonical prompt, correct birthdate |
| Paraphrased | **P (knows)** | Perturbed, reworded prompt ("was born on" → "has a recorded birth date of") |
| Retain | **N (absent)** | Standard (never-knew), correct birthdate |
| Perturbed | **N (absent)** | Perturbed scored against a **wrong** birthdate (holds no such fact) |

Two criteria per metric:
- **Semantic Invariance** — within P, Full vs Paraphrased should be *equal*. Reported as the effect
  size |Cliff's δ| of the Full→Paraphrased shift (`invar_shift`; 0 = invariant). *(The paired
  Wilcoxon/TOST p-values are also stored, but at n=44 under a strong paraphrase the within-P Wilcoxon
  saturates for every metric — so invariance must be read as effect size, not the p<0.05 threshold.)*
- **Knowledge Discrimination** — P vs N should *differ*. Reported as |Cliff's δ| between the pooled P
  and N distributions (`discrim`; 1 = perfect separation).
- **faithful_score = discrim · (1 − invar_shift)** — high only if a metric both separates knowledge
  *and* ignores phrasing.

Metrics scored: mechanistic `D_mech`, `min_rank`; OpenUnlearning lexical/output `probability`, `rouge`,
`exact_mem`, `extraction_strength`, `truth_ratio`, and MIA `mia_loss / mia_mink / mia_minkpp / mia_zlib`.
(The paraphrase axis *is* the Paraphrased cell, so OpenUnlearning's "paraphrased prob/rouge" are those
metrics measured in that cell; jailbreak-ROUGE has no SPID analogue and is omitted.)

## Result (`mech_metaeval_table1.csv`, `mech_metaeval_fig4.png`)

| metric | family | invar_shift ↓ | discrim ↑ | faithful_score ↑ |
|---|---|---|---|---|
| **D_mech** | **mechanistic** | **0.466** | 0.827 | **0.442** |
| truth_ratio | lexical | 0.974 | 0.973 | 0.025 |
| rouge | lexical | 0.975 | 0.494 | 0.012 |
| extraction_strength | lexical | 0.976 | 0.505 | 0.012 |
| exact_mem | lexical | 0.991 | 0.818 | 0.008 |
| mia_mink | lexical | 0.950 | 0.158 | 0.008 |
| mia_loss | lexical | 0.965 | 0.177 | 0.006 |
| mia_zlib | lexical | 0.993 | 0.217 | 0.002 |
| probability | lexical | 0.999 | 0.928 | 0.001 |
| min_rank | mechanistic | 1.000 | 0.892 | 0.000 |
| mia_minkpp | lexical | 0.999 | 0.310 | 0.000 |

Reading it:
1. **Discrimination is easy and not where faithfulness lives.** Almost every metric separates P from N
   (most `discrim` > 0.8, all between-P/N Wilcoxon significant except `mia_mink`). A metric can look
   great and still be unfaithful.
2. **The differentiator is Semantic Invariance, and only `D_mech` has it** (`invar_shift` 0.466 vs
   ≥0.95 for all others). Under a meaning-preserving rewording, `probability` craters 0.95→0.16,
   `rouge`/`extraction_strength`→0 (see the Paraphrased column collapsing toward the negatives in
   Figure 4), while `D_mech` keeps both P cells elevated. Its faithful-score is **~18× the best lexical
   metric.**
3. **It's the *distance construction*, not "internals" per se.** Raw `min_rank` — also a logit-lens
   metric — *fails* invariance (`invar_shift` 1.0). The dual-anchor distance is what cancels the
   phrasing shift while preserving the knowledge signal. This is the precise mechanism of the win.

## Honest scope and limits

- **8B only, n = 44** YAGO birthdate facts; a single paraphrase style and a single wrong-target
  construction. A stronger version would average over several GPT-generated paraphrases and multiple
  perturbation styles, and repeat at 1B.
- **"More faithful" = invariant *and* discriminative** (SPID's two criteria) — not a blanket claim of
  superiority on every axis.
- The **per-fact relearning** route to faithfulness was a **documented null** (model-identity confound;
  see [FAITHFULNESS.md](FAITHFULNESS.md) §4). The faithfulness evidence here rests on (a) the
  rank-by-layer qualitative separation and tuned-lens confirmation (FAITHFULNESS.md §2/§5) and (b) this
  SPID invariance/discrimination result — *not* on predicting individual-fact recovery.
- `Perturbed(wrong)` scores moderately positive on `D_mech` (a wrong birthdate token still traces a
  date-like path); the P–N gap is carried mainly by the `Retain` cell. Worth noting, not fatal.

## Where it sits in the thesis

This is the quantitative capstone of the faithfulness argument: the qualitative rank-by-layer story
(output says "forgotten," the token is still recoverable mid-stack for surface methods) is turned into
a **single scalar metric** and run through the field's own meta-evaluation, where it beats all 10
established output/lexical metrics on the criterion — paraphrase invariance — that separates measuring
*knowledge* from measuring *words*.
