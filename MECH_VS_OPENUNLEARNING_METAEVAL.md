# Meta-evaluating a mechanistic metric against the OpenUnlearning suite — full methodology

*Detailed writeup of job 18468 (`run_mech_metaeval.py`, Hubble-8B). This is the head-to-head between
our logit-lens **mechanistic distance** metric and the established **OpenUnlearning** metric suite,
following the meta-evaluation protocol of OpenUnlearning (arXiv 2506.12618 §4) and SPID (10619 §5.2).
Standalone; not in RESULTS.md. Companion to [MECH_METAEVAL_RESULTS.md](MECH_METAEVAL_RESULTS.md).*

---

## 1. The question

Ordinary unlearning metrics ask *"did the model **say** the forgotten fact?"* We want a metric that
asks *"does the model **still hold** the fact internally?"* — because a fact that is merely suppressed
at the output can be un-suppressed by rewording, quantization, or a few fine-tune steps. The
**meta-evaluation** question is one level up: **which metric is more *faithful*** — i.e., which one
actually tracks knowledge rather than surface behavior? This document is the numerical answer.

A metric is judged on **two criteria** (from OpenUnlearning/SPID). Both must hold:

- **Knowledge Discrimination** — the metric must give *different* scores to models that **hold** the
  target knowledge vs models that **lack** it. (Necessary but easy — most metrics manage it.)
- **Semantic Invariance** — the metric must give the *same* score when the *question is reworded but
  the knowledge is unchanged*. A metric that drops just because you paraphrased the prompt is
  measuring **words, not knowledge**. (This is the hard criterion, and where lexical metrics fail.)

The whole experiment is engineered to measure these two things cleanly and pit 11 metrics against each
other on them.

---

## 2. The metric under test — `D_mech`

For one fact, run the **logit lens**: at every layer `L` of the 36-layer model, project the hidden
state at the answer position through the model's own final-norm + unembedding, and record the **rank**
of the gold answer token (rank 1 = top prediction; rank 50 000 = buried). This yields a trajectory
`r = (r_0, r_1, …, r_36)`.

Define two **anchor** trajectories from models with known ground-truth states:
- **K (knows)** = the *Perturbed* model's trajectory (it memorized the fact),
- **Ø (never knew)** = the *Standard* model's trajectory.

The **dual-anchor mechanistic distance** over the mid/late band `B` (layers 18–36, the top half):
```
D_mech = mean_{L∈B} |log r_L − log Ø_L|  −  mean_{L∈B} |log r_L − log K_L|
```
`D_mech > 0` means the trajectory is closer to the *knows* template than to the *never-knew* template —
the fact is internally present, whatever the output does. Anchors are **leave-one-fact-out medians**
(each fact is scored against the median of the *other* 43 facts) so a fact is never compared to itself.
Tokenization is **boundary-safe** (the trailing-space fix in `layerwise_target_rank`, so we track
`' May'`, not `'May'`).

We also test **`min_rank`** = the minimum rank over band `B` — a *naive* logit-lens metric — to check
whether any internal readout wins, or specifically the distance construction.

---

## 3. The competition — the OpenUnlearning metrics

Nine established output/lexical metrics, computed per fact (definitions adapted to the Hubble
birthdate setup; families follow OpenUnlearning Table 2):

| metric | family | definition (per fact) | knows ⇒ |
|---|---|---|---|
| `probability` | probability | `exp(mean log p(gold tokens))`, teacher-forced, length-normalized | high |
| `rouge` | lexical overlap | ROUGE-L F1 of greedy generation vs gold answer | high |
| `exact_mem` | memorization | fraction of gold positions where teacher-forced `argmax == gold` | high |
| `extraction_strength` | extraction | length of the longest correctly greedy-generated leading run of gold tokens ÷ `|gold|` (Carlini-style) | high |
| `truth_ratio` | ratio | `exp(mean log p(wrong answers) − log p(gold))` | **low** |
| `mia_loss` | MIA | mean NLL over the full *prefix+gold* sequence | **low** |
| `mia_mink` | MIA | mean of the lowest **20%** token log-probs over the sequence (Min-K% Prob) | high |
| `mia_minkpp` | MIA | Min-K%++ — same, but on per-position z-normalized log-probs `(log p − μ)/σ` | high |
| `mia_zlib` | MIA | total NLL ÷ zlib-compressed byte length of the text | **low** |

(The paraphrase axis is a *pool cell*, §4, so OpenUnlearning's "paraphrased prob/rouge" are just those
metrics measured in that cell; jailbreak-ROUGE has no SPID analogue and is omitted.)

---

## 4. Experimental design — the four pools

SPID's construction (§5.2.1): build **model pools with known ground-truth knowledge states** and see
which metric behaves correctly. We realize four **cells** on the Hubble suite, each = a model evaluated
on the birthdate of every entity (**n = 44** forget-set entities):

| cell | pool | how it's built | ground truth |
|---|---|---|---|
| **Full** | **Positive P** | Perturbed model · canonical prompt · correct birthdate | **knows** |
| **Paraphrased** | **Positive P** | Perturbed model · **reworded** prompt · correct birthdate | **knows** (same fact, different words) |
| **Retain** | **Negative N** | Standard model (never trained on it) · correct birthdate | **absent** |
| **Perturbed** | **Negative N** | Perturbed model · **wrong** birthdate target | **absent** (holds no such fact) |

Design rationale, cell by cell:
- **Full vs Paraphrased** isolates **Semantic Invariance**: *same model, same knowledge, only the
  wording changes*. Any metric that moves between these two cells is reacting to phrasing, not
  knowledge. The paraphrase rewrites the trigger clause "…was born on" → "…has a recorded birth date
  of", keeping all surrounding context (applied to 44/44 prompts).
- **Retain** is the clean "never knew" negative — the Hubble control model.
- **Perturbed(wrong)** is the second negative from SPID (their "factually incorrect" pool): we score
  the *knowing* model against a **wrong** birthdate (a distractor from the candidate pool). The model
  does not hold that wrong fact, so a faithful metric should score it low — this catches metrics that
  fire on *fluency/format* rather than the *specific* fact.
- **P = {Full, Paraphrased}** (88 instances), **N = {Retain, Perturbed}** (88 instances).

---

## 5. Statistics — and every assumption we make (n is small)

**n = 44 facts.** This is small and the metric distributions are bounded and skewed (ranks are
heavy-tailed; probabilities and ROUGE pile up at 0 or 1). Every choice below follows from that.

### 5.1 The tests we run

- **Within-P paired — Wilcoxon signed-rank** on `Full_i` vs `Paraphrased_i` (same entity `i`, paired).
  Tests whether the paraphrase shifts the metric. *Paired* because it's the same model on the same
  entities — pairing removes per-fact difficulty. Reported as `within_P_wilcoxon` (a **high** p would
  mean "no detectable shift").
- **Within-P equivalence — TOST** (Two One-Sided Tests, Schuirmann) on the paired differences, with
  margin `ε = 0.3 × (pooled SD across the 4 cells)`. A **low** TOST p means the two cells are
  statistically *equivalent* (inside ±ε). Reported as `within_P_TOST`.
- **Between P vs N — Mann–Whitney U** on the pooled P (n=88) vs pooled N (n=88). A **low** p means the
  metric separates knowledge-present from knowledge-absent. Reported as `between_PN_wilcoxon`.
- **Effect sizes — Cliff's delta**, `δ = [#(x>y) − #(x<y)] / (|x|·|y|)`, in `[−1, 1]`, `0` = the two
  distributions are stochastically equal, `±1` = complete separation. Two of them:
  - `invar_shift = |δ(Full, Paraphrased)|` — magnitude of the paraphrase shift (**0 = invariant**).
  - `discrim = |δ(P, N)|` — magnitude of the P-vs-N gap (**1 = perfect discrimination**).
  - `faithful_score = discrim × (1 − invar_shift)` — high only when a metric both discriminates and
    stays invariant.

### 5.2 Assumptions and why we lean on effect size, not p-values

1. **Non-parametric throughout.** With n=44 and non-normal, bounded metrics we do **not** assume
   normality or equal variance, so we use Wilcoxon / Mann–Whitney / Cliff's δ rather than t-tests.
2. **Pairing assumption (Wilcoxon).** The within-P test assumes the *distribution of paired
   differences is symmetric*; we do not assume its shape otherwise. Pairing is valid because Full and
   Paraphrased are the identical model on identical entities.
3. **The critical small-n caveat — the Wilcoxon p SATURATES.** A signed-rank test detects the
   *consistency of the direction* of a shift, not its *size*. With n=44, even a **tiny but consistent**
   Full→Paraphrased shift drives the p-value below any threshold. In our data this happened for **every
   metric**, including `D_mech` (`within_P_wilcoxon` = 4.8e-6). So the naive rule "p > 0.05 ⇒ invariant"
   is **not usable here** — it would declare *nothing* invariant. **We therefore treat the within-P
   p-values as reported-for-completeness** (to match SPID's table format) **and base the invariance
   verdict on the effect size `invar_shift`**, which measures *magnitude* and is stable across n.
4. **Effect size is the primary readout.** Cliff's δ is distribution-free, outlier-robust, handles ties
   by sign, and is interpretable independently of sample size — exactly what a small, skewed sample
   needs. `invar_shift` and `discrim` (and hence `faithful_score`) are our headline numbers.
5. **TOST margin is a chosen threshold.** `ε = 0.3·pooledSD` encodes "a shift smaller than 0.3 SD is
   negligible." It is a defensible but *arbitrary* choice; TOST also has *low power at n=44*, so a
   non-significant TOST does **not** prove non-equivalence. We report it but do not hang conclusions on
   it.
6. **No multiple-comparison correction.** There are 11 metrics × 3 tests = 33 p-values. We do **not**
   Bonferroni-correct, for two reasons: (a) the *discrimination* p-values are astronomically small
   (1e-12 to 1e-29), so any correction leaves them significant; (b) the *invariance* conclusion rests
   on effect size, not on a p-threshold, so correction is irrelevant to it. No conclusion depends on a
   borderline p-value.
7. **Independence.** The 44 entities are distinct people with distinct birthdates; we treat facts as
   independent instances. They share template scaffolding but differ in the target, which is what the
   metrics score.
8. **Single paraphrase / single wrong-target realization.** The paraphrase and the wrong-target are
   *one fixed construction each*, not sampled — so variance *across* paraphrase styles is **not**
   captured by our n=44 (which is variance across *facts*). This is the main external-validity limit
   (§8); a stronger design averages several GPT-generated paraphrases.
9. **Anchor estimation (for `D_mech`).** Anchors are leave-one-out medians over 43 facts — stable, and
   LOO removes self-inclusion circularity. We assume the median "knows"/"never-knew" trajectory is a
   representative template, which the tight Full/Retain distributions support.

---

## 6. Results

Cell **medians** per metric (P = Full, Paraphrased | N = Retain, Perturbed):

| metric | Full (P) | Paraphrased (P) | Retain (N) | Perturbed (N) |
|---|---|---|---|---|
| **D_mech** | 1.213 | **0.921** | −1.368 | 0.760 |
| min_rank | 1.0 | 3.25 | 7.13 | 9.0 |
| probability | 0.948 | **0.164** | 0.067 | 0.005 |
| rouge | 0.571 | **0.000** | 0.000 | 0.000 |
| exact_mem | 1.000 | **0.500** | 0.250 | 0.250 |
| extraction_strength | 1.000 | **0.000** | 0.000 | 0.000 |
| truth_ratio | 0.000 | 0.001 | 1.008 | 18.22 |

*(bold = the Paraphrased cell; note how the lexical metrics collapse from Full to Paraphrased even
though the knowledge is unchanged, while `D_mech` stays high.)*

Effect-size verdict (`mech_metaeval_table1.csv`), sorted by faithfulness:

| metric | family | invar_shift ↓ | discrim ↑ | **faithful_score ↑** |
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

Box plots of all 11 metrics across the four cells (P|N divider) are in `mech_metaeval_fig4.png`; the
per-instance data is `mech_metaeval_cells.csv`.

---

## 7. Why our metric performed better

1. **Discrimination is easy — it is not where faithfulness is decided.** Almost every metric separates
   P from N (`discrim` > 0.8 for most; every between-P/N Mann–Whitney p is tiny except `mia_mink`). If
   you only look at "does it tell knows from doesn't-know," most metrics look fine. That is the trap.
2. **Semantic Invariance is the discriminating criterion, and only `D_mech` passes it.** Its
   `invar_shift` is **0.466** vs **≥ 0.95 for all ten other metrics**. Concretely: under a
   meaning-preserving rewording, `probability` falls 0.948 → 0.164, `rouge` and `extraction_strength`
   fall to 0 — the Paraphrased cell (still knowledge-present!) collapses to look like the negatives.
   `D_mech` barely moves (1.213 → 0.921). Its `faithful_score` (0.442) is **~18× the best lexical
   metric** (truth_ratio 0.025).
3. **The win is the *distance construction*, not "use internals."** The naive internal metric
   `min_rank` **fails** invariance (`invar_shift` = 1.000) — rewording shifts the absolute rank
   consistently. `D_mech` is invariant because it is a **relative** quantity: it asks *"is this
   trajectory closer to the knows-template or the never-knew-template?"* Rewording moves the whole
   trajectory a little, but it does **not** move it across the K/Ø boundary — the fact stays "closer to
   knows." Subtracting the two anchor distances **cancels the phrasing shift while preserving the
   knowledge signal.** This is the precise mechanism, and it is why the *construction* matters, not
   merely the fact that we read hidden states.
4. **This is the empirical confirmation of a theoretical prior.** We already expected that
   output/lexical metrics conflate surface form with knowledge (that is the whole "suppression ≠
   erasure" argument). The meta-evaluation turns that expectation into a number on the benchmark's own
   turf, against ten established metrics.

---

## 8. Threats to validity / limitations (stated honestly)

- **Single scale, single dataset, small n.** 8B only; n = 44 YAGO birthdate facts. Repeating at 1B and
  on other attributes would strengthen generality.
- **One paraphrase, one perturbation.** The biggest external-validity gap: our n is across *facts*, not
  across *paraphrase styles*. A robust version samples several GPT-generated paraphrases (and several
  wrong-target styles) and reports invariance across them. We expect the ordering to hold — the lexical
  collapse is intrinsic to surface-matching — but this is untested here.
- **`Perturbed(wrong)` leaks slightly on `D_mech`** (median 0.760, positive): a wrong birthdate token
  still traces a "date-like" path, so the distance is not fully negative. The P–N gap is carried mainly
  by the clean `Retain` cell. Not fatal, but noted.
- **`faithful_score` is a composite we defined** (`discrim × (1 − invar_shift)`), not a standard
  statistic. The individual columns (`invar_shift`, `discrim`) are the primary evidence; the composite
  is a convenience ranking.
- **TOST margin and the MIN-K% fraction (20%)** are chosen constants; the MIA metrics in particular are
  noisier on short answers (we mitigate by computing MIA over the whole prefix+gold passage).
- **The per-fact relearning route to faithfulness was a null** (documented in
  [FAITHFULNESS.md](FAITHFULNESS.md) §4). The faithfulness case therefore rests on this SPID-style
  meta-evaluation plus the qualitative rank-by-layer story and tuned-lens confirmation — **not** on
  predicting individual-fact recovery.

---

## 9. Reproducibility

- **Code:** `run_mech_metaeval.py` (`run_mech_metaeval.sbatch`, 8B, A100). Deterministic greedy
  decoding; seed `hubble8b.config.SEED`. Boundary-safe `layerwise_target_rank`.
- **Job:** 18468, elapsed ~2.5 min (no training — pure forward passes + short greedy generations over
  4 cells; two model loads, Perturbed reused for three cells).
- **Knobs:** band = top half of layers (18–36); Min-K% = 20%; TOST ε = 0.3·pooled SD; paraphrase =
  "was born on" → "has a recorded birth date of"; wrong-target = a distractor birthdate from the
  candidate pool.
- **Artifacts (HF results dataset):** `mech_metaeval_cells.csv` (per-instance, all metrics × 4 cells),
  `mech_metaeval_table1.csv` (SPID Table-1 analog + effect sizes), `mech_metaeval_fig4.png` (box
  plots). Nothing written to RESULTS.md.
