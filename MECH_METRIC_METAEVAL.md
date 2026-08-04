# Proposed experiment — meta-evaluating a logit-lens mechanistic *distance* metric against lexical metrics

*Design doc. Adapts the OpenUnlearning meta-evaluation protocol (arXiv 2506.12618 §4), the same
protocol the SPID paper (10619) builds on, to test whether a **logit-lens mechanistic distance
metric** is more **faithful** than the output/lexical metrics — on the Hubble-8B suite.*

> **IMPLEMENTED VERSION (job 18468) follows SPID §5.2.1 exactly — read this box first.** The final
> `run_mech_metaeval.py` uses SPID's **two-criteria, 4-cell Positive/Negative pool** design, not the
> single AUC sketch in §3–4 below (which was an earlier iteration; job 18466 exposed that an
> all-Perturbed positive pool is never output-wrong, so output-matching was empty). The implemented
> pools:
> - **Positive P (knows):** *Full* = Perturbed on the canonical prompt; *Paraphrased* = Perturbed on a
>   meaning-preserving reworded prompt ("was born on" → "has a recorded birth date of").
> - **Negative N (absent):** *Retain* = Standard (never-knew); *Perturbed(wrong)* = Perturbed scored
>   against a **wrong** birthdate target (holds no such fact).
>
> **Two SPID criteria per metric** (Table-1 analog): **Semantic Invariance** — within P, Full vs
> Paraphrased must be *equal* (Wilcoxon high-p + TOST low-p); **Knowledge Discrimination** — P vs N must
> *differ* (Wilcoxon low-p). Plus **Figure-4** box plots across the 4 cells. Run over the mechanistic
> metrics (`D_mech`, `min_rank`) and the OpenUnlearning lexical/output set (probability, rouge,
> exact_mem, extraction_strength, truth_ratio, MIA loss/mink/minkpp/zlib). Expectation to test honestly:
> mechanistic metrics are paraphrase-invariant *and* discriminative; lexical metrics (esp. ROUGE,
> Extraction Strength) fail invariance, as SPID found. Outputs: `mech_metaeval_cells.csv`,
> `mech_metaeval_table1.csv`, `mech_metaeval_fig4.png`.

---

## 1. The gap this targets

OpenUnlearning §4 meta-evaluates **12 unlearning metrics** and scores each by:

- **Faithfulness** = `AUC-ROC( m(P), m(N) )` — does the metric assign higher scores to a **Positive
  pool P** (models that *possess* the target knowledge) than a **Negative pool N** (models that
  genuinely *lack* it)? Ground-truth pools are built by construction.
- **Robustness** under interventions: **Relearning** `R = min( (mᵃ_ret−mᵇ_ret)/(mᵃ_unl−mᵇ_unl), 1 )`
  and **Quantization** `Q = min( mᵇ_unl / mᵃ_unl, 1 )` (a=before, b=after).
- **Overall** = harmonic mean, `HM(Faithfulness, HM(R,Q))`.

**Every one of their 12 metrics reads the model's *output* distribution** — ROUGE, Exact
Memorization, Truth Ratio, Extraction Strength, Probability, MIA. **None reads intermediate layers.**
So the entire meta-evaluation has never been run on a mechanistic-interpretability metric. Our
thesis result (output metrics can't tell suppressed-present from truly-absent; the logit lens can) is
*exactly* the property their faithfulness AUC is designed to measure — but they never had a metric
that could exploit it. This experiment adds that metric and scores it on their leaderboard.

---

## 2. The metric under test — a logit-lens *distance*

For a fact, the logit lens gives the gold token's **rank trajectory** across depth,
`r = (r₀, r₁, …, r_L)`. Define the mechanistic metric as a **distance between a model's trajectory and
the two anchor trajectories** we have ground truth for:

- **Anchor K** (knows): the Perturbed model's median trajectory.
- **Anchor Ø** (never knew): the Standard model's median trajectory.

**Mechanistic distance** (scalar, per fact):
`D_mech = d(r_model, Anchor_Ø) − d(r_model, Anchor_K)` over the **mid/late band** (layers ⌈0.5L⌉…L−1),
where `d` is the mean absolute log-rank gap. `D_mech > 0` ⇒ the model's internal trajectory looks like
it *knows* (close to Perturbed) even if the output says otherwise; `D_mech < 0` ⇒ genuinely absent.
Two simpler scalar instances to report alongside it (already computed in the pipeline):
**min mid-stack rank** and **LSR** (layerwise semantic-residual distance to the Perturbed reference).

Lexical/output comparators = **the full OpenUnlearning 12-metric suite** (Table 2), so the comparison is
directly apples-to-apples with their leaderboard:
`probability`, `paraphrased_probability`, `rouge`, `paraphrased_rouge`, `jailbreak_rouge`, `truth_ratio`,
`exact_mem`, `extraction_strength`, and four MIA variants `mia_loss` / `mia_mink` / `mia_minkpp` /
`mia_zlib` (computed over the full prefix+gold sequence). Plus Hubble's own `infill` as a 13th.

*Operationalization notes (honest):* the paraphrase variants reuse a **different Hubble template**
(`intro_prefix` vs `full_prefix`) as the alternate phrasing; `jailbreak_rouge` prepends a constructed
adversarial instruction (no natural jailbreak set exists here); MIA over a short birthdate answer is
noisy, so MIA is computed over the whole prefix+gold passage (document-membership notion). These are
faithful adaptations of the OpenUnlearning definitions to the Hubble YAGO setup, not verbatim TOFU.

---

## 3. Ground-truth pools — built so lexical metrics are blinded by construction

The novelty vs. OpenUnlearning: we don't just build P and N, we build a version where **P and N emit
the same output**, so any metric that separates them *must* be using internal information.

- **Pool P (knowledge present):** Perturbed-model instances on the Unlearn facts. Ground truth =
  PRESENT (Perturbed was trained to memorize them; infill ≈ 1.0).
- **Pool N (knowledge absent):** Standard-model instances on the same facts. Ground truth = ABSENT
  (Standard never saw them — the Hubble control).

Then run the meta-eval on **three nested conditionings**, which is the heart of the design:

1. **Unconditioned** (sanity): all facts. *Expectation:* both lexical and mechanistic separate P/N,
   AUC high for all — establishes every metric is at least valid.
2. **Output-matched (the decisive test):** keep only facts where **the lexical output is identical for
   P and N** — e.g. both give the *wrong* infill answer (Perturbed's infill isn't 100%; Standard's
   isn't 0%, so this subset is non-empty). On this subset lexical AUC → **0.5 by construction** (no
   output signal is left). *Question:* does `D_mech` still separate PRESENT from ABSENT? If yes, the
   mechanistic metric is reading something the output physically does not contain.
3. **Adversarial-suppressed (deployment realism):** add the four unlearned models (NPO/GradDiff/IDK/
   RMU) as instances whose ground-truth PRESENT/ABSENT label is set by an **independent recovery
   oracle** the metrics never see — a *strong* relearn (high-LR, to convergence) or a paraphrase
   battery: label PRESENT if the fact recovers, ABSENT if it never does. Score each metric's AUC at
   predicting that oracle label. (This is OpenUnlearning's relearning idea, but used to *define the
   label*, evaluated at model level for power — not the noisy per-fact speed regression that came back
   null in relearn v2.)

---

## 4. The statistics (OpenUnlearning-exact, plus the conditioning contrast)

For each metric, on each conditioning:

- **Faithfulness** = `AUC-ROC(m(P), m(N))` with **bootstrap 95% CI** (2000 resamples) and a
  Mann–Whitney p-value. A metric is *faithful* iff its CI excludes 0.5.
- **The headline number** = the **AUC drop from conditioning 1 → 2**. Lexical metrics should collapse
  toward 0.5; the mechanistic distance should retain most of its AUC. Report `ΔAUC` per metric with
  CIs — *this single table is the proof.*
- **Robustness legs**, reusing runs already on disk: plug `D_mech` into `R` (from relearn v2, job
  18437) and `Q` (from the quant study, job 18435), same formulas as OpenUnlearning.
- **Overall** = `HM(Faithfulness, HM(R,Q))` — reproduce their Table-2 leaderboard with the mechanistic
  distance added as a new row, ranked against the lexical metrics.

---

## 5. What each outcome would mean (pre-registered, so it can't be spun)

| result on conditioning 2 (output-matched) | interpretation |
|---|---|
| `D_mech` AUC CI excludes 0.5, lexical AUCs → 0.5 | **Claim proven:** mechanistic metric is faithful where lexical is blind. |
| `D_mech` AUC → 0.5 too | Honest null: the internal signal is also output-entangled here; report it, faithfulness rests on §2 rank-by-layer alone. |
| `D_mech` faithful **and** wins on Overall HM | Strongest: the mechanistic distance tops OpenUnlearning's own leaderboard on Hubble. |

Guardrails: pools are labelled by **construction/oracle, never by the metric under test** (no
circularity); the output-matched subset makes the lexical baseline's failure a *design guarantee*, not
a lucky draw; every AUC carries a bootstrap CI; the relearn oracle is *strong* (unlike relearn v2's
weak attack) so PRESENT/ABSENT labels are clean; the model-level aggregation gives the power the
per-fact test lacked.

---

## 6. Why this is genuinely new (vs. everything already run)

- **vs. relearn v2 (SPID-style):** that asked "does a pre-attack metric predict per-fact recovery
  *speed*?" — a noisy per-fact regression, and it was null. This asks OpenUnlearning's binary
  **P-vs-N separability** with **construction-certain ground truth**, and isolates the faithfulness gap
  by *conditioning on output-match* — a design where the lexical baseline is guaranteed to fail, so a
  surviving mechanistic AUC is unambiguous.
- **vs. OpenUnlearning §4 itself:** they never meta-evaluated a mechanistic/logit-lens metric — all 12
  are output-side. Adding `D_mech` to their exact protocol (Faithfulness AUC + R + Q + HM) and showing
  where it ranks is the novel contribution.

---

## 7. Implementation sketch (one job, ~1 GPU, reuses the pipeline)

- New `run_mech_metaeval.py`: load Perturbed/Standard (+ 4 unlearned); per fact compute `D_mech`
  (reuse `layerwise_target_rank`), LSR (`layerwise_residual`/`lsr_scalar`), and the lexical set
  (`sequence_logprob`, ROUGE, Truth Ratio, Extraction Strength); build the three conditionings; AUC +
  bootstrap CI per metric per conditioning; plug into R (relearn v2 CSV) and Q (quant CSV); emit
  `mech_metaeval_table.csv` (the leaderboard), `mech_metaeval_deltaAUC.csv` (the proof), and a
  box/ROC figure. Writes **new files only; nothing to RESULTS.md.** Email on the sbatch.
- Est: the strong-relearn oracle for the adversarial pool is the only heavy part; conditionings 1–2
  are cheap (no training). Could ship conditionings 1–2 first (fast, and they carry the core proof),
  then the adversarial oracle as a follow-up.
