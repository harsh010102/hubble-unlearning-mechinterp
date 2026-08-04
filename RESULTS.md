# Hubble Unlearning + Mechanistic Interpretability — Results (1B & 8B)

*Working results document. Numbers pulled from: 8B sweep (job 17948), 1B sweep (18185), infill/master.
**Corrected rank readout** — boundary-safe + double-norm-fixed (1B job 18474; 8B boundary-safe curves).
**Faithfulness meta-evaluation** — mechanistic `D_mech` vs OpenUnlearning suite, SPID protocol (job 18468;
see §6 and the standalone docs). Companion docs: FAITHFULNESS.md, LOGIT_LENS_EXPLAINED.md,
MECH_METAEVAL_RESULTS.md, MECH_VS_OPENUNLEARNING_METAEVAL.md, QUANTIZATION.md.*

## 1. Setup (what is faithful to the Hubble paper)

- **Dataset / splits** — identical to Hubble (Wei et al.): **Unlearn** = half of the 256-duplicate
  perturbations, **Keep** = the other half (in-domain near-neighbours), **Unseen** = 0-duplicate
  held-out. Verified against the paper text (§6.2). `hubble8b/data.py::load_yago_splits`.
- **Models** — Hubble Standard (never saw the planted fact) and Perturbed (memorized it), at **1B**
  (16 layers, hidden 2048) and **8B** (36 layers, hidden 4096).
- **Methods** — **RMU** (matches the paper's Table-12 config; 8B layers [5,6,7], 1B [3,4,5]) plus
  three *extensions not in the paper*: **NPO, GradDiff, IDK**.
- **Retain set** — **Keep** biographies (the paper's Appendix-G ablation), not the default WikiText.
  This is the deliberate choice that makes forget vs retain separable for the mechanistic read.
- **Metrics** — infill accuracy (Hubble's own loss-based-choice PII metric), rank-by-layer logit lens
  (boundary-safe + double-norm-fixed, median over facts), and the derived **`D_mech`** distance (§6).
  LSR (§4) is retained as a semantic descriptor only; `D_mech` is the faithful metric.

## 2. Infill accuracy — by method × split × scale

Chance ≈ 0.10. Goal of unlearning: **Unlearn → chance**, **Keep preserved**, Unseen unchanged.

### full_prefix (leak-free, full context)
| model | 8B Unlearn | 8B Keep | 1B Unlearn | 1B Keep |
|---|---|---|---|---|
| Standard *(never knew)* | 0.16 | 0.13 | 0.18 | 0.09 |
| Perturbed *(knows)* | **1.00** | **1.00** | **0.43** | **0.58** |
| **NPO** | 0.09 | **1.00** | 0.05 | **0.80** |
| **GradDiff** | 0.16 | **1.00** | 0.16 | 0.60 |
| **IDK** | **0.32** | 0.98 | **0.07** | 0.82 |
| **RMU** | 0.11 | **0.64** | 0.16 | **0.33** |

Key reads:
- **NPO & GradDiff are the clean selective unlearners at both scales** (Unlearn→chance, Keep high).
- **IDK is scale-dependent**: at 8B it *fails to forget* (Unlearn 0.32 — surface-level refusal); at
  **1B it genuinely forgets** (0.07). A notable cross-scale result.
- **RMU collapses Keep at both scales** (0.64 / 0.33) — the paper's reported RMU joint-collapse.
- **8B memorizes far more strongly than 1B** (Perturbed Unlearn 1.00 vs 0.43), so the 1B *retain
  ceiling* is only ~0.58 — several methods' Keep (e.g. NPO 0.80) exceed Perturbed because the
  Keep-retain training reinforced those biographies.

### name_only (name only, no biography context)
| model | 8B Unlearn | 8B Keep | 1B Unlearn | 1B Keep |
|---|---|---|---|---|
| Perturbed | 0.52 | 0.40 | 0.14 | 0.13 |
| IDK | 0.25 | **0.91** | 0.11 | 0.73 |
| NPO | 0.09 | 0.36 | 0.07 | 0.71 |

- Accuracy drops sharply without context, **especially at 1B** (Perturbed 0.14 ≈ chance) — 1B needs
  the biography context to recall. This matters for the faithfulness study (see §6).

## 3. Logit-lens rank — the mechanistic result (median, boundary-safe, fixed)

Final-layer median rank of the gold birthdate token (higher = more buried; lower = recoverable).
**Both scales now use `full_prefix` and the corrected readout** (double-norm fix + boundary-safe
tokenization, §5b) — jobs 18474 (1B) / boundary-safe 8B. This replaces the earlier inflated numbers.

| model | 8B forget | 8B retain | 1B forget | 1B retain |
|---|---|---|---|---|
| Standard *(never knew)* | 9 | 10 | 10 | 11 |
| Perturbed *(knows)* | **1** | 1 | **4** | 3 |
| **NPO** | **3,863** | 1 | **44,226** | 1 |
| **GradDiff** | **9,191** | 1 | 8 | 3 |
| **IDK** | **9** | 1 | 16 | 1 |
| **RMU** | **2,693** | 2 | 8 | 6 |

- **Baselines correct at both scales** (Perturbed forget rank < Standard) and **retain preserved
  everywhere** (rank 1–6) — selective unlearning holds mechanistically.
- **8B:** NPO / GradDiff / RMU bury the forget fact at the output; **IDK does not (rank 9 = surface).**
  A finer read of the *mid-stack* (FAITHFULNESS.md §2) shows NPO/GradDiff keep the token accessible
  mid-stack (~365) and only mask it late, while RMU buries it deepest mid-stack (2 240).
- **1B:** only **NPO** buries the forget fact (44,226); **GradDiff, IDK, RMU leave it accessible
  (rank 8–16)** — their unlearning does not internalize at the smaller scale.
- **The mechanistic view refines the infill (§2) story:** infill said IDK "genuinely forgets" at 1B
  (0.07), but the logit lens shows the fact is still internally accessible there (rank 16) — i.e.
  **output-suppressed, not erased.** This is exactly the suppression-vs-erasure gap the thesis targets.

**Method verdict:** *NPO is the most scale-robust unlearner; GradDiff and RMU work at 8B but not 1B;
IDK is surface at both.* Invisible to any output metric.

## 4. LSR — semantic residual (nationality slot; higher = concept present)

Switched from occupation (paper's weakest slot; discrimination gap 0.016) to **nationality**
(gap 0.37) after the slot-check (job 18186).

| model | 8B LSR | 1B LSR |
|---|---|---|
| Standard | 0.51 | 0.45 |
| Perturbed | **0.87** | **0.67** |
| NPO | **0.11** | **0.10** |
| GradDiff | 0.50 | 0.65 |
| IDK | **1.00** | 0.86 |
| RMU | 0.61 | 0.61 |

- **Discriminates knowledge** (Perturbed ≫ Standard; p < 1e-11 both scales).
- **Corroborates the taxonomy**: NPO erases the concept (0.10), **IDK retains it fully** (surface),
  GradDiff/RMU partial. Consistent across scales.

## 5. Cross-scale summary (1B vs 8B)

| axis | 8B | 1B |
|---|---|---|
| Memorization strength (Perturbed infill) | strong (1.0) | weak (~0.43) |
| Retain preserved | yes | yes |
| **NPO** — forget burial | buries (output rank 3.9k) | **buries deepest (44k)** |
| **GradDiff / RMU** — forget burial | bury (9.2k / 2.7k) | **fail to internalize (rank ~8)** |
| **IDK** | surface (output rank 9) | surface (output-suppressed, internally rank 16) |
| Mechanistic rank signal | sharp for all | sharp for NPO; weak burial for others |

**Headline (mechanistic verdict):** *NPO is the most scale-robust unlearner; GradDiff and RMU work at
8B but not 1B; IDK is surface at both.* The output/infill view (§2) can **misread** this — it labels
IDK "forgetting" at 1B, but the logit lens shows the fact is only **output-suppressed, not erased**. A
caution against single-scale, output-only unlearning conclusions — and a concrete case for the internal
readout.

## 5b. Tokenizer & readout correctness notes

**(a) Infill / probability tokenizer fix.** A trailing-space bug (`tok(prefix)` vs `tok(prefix+value)`
dropped the first value token, or scored single-word countries as empty → spurious probability 1.0)
was found via a generation diagnostic (job 18389) and fixed in `sequence_logprob`, `layerwise_residual`,
`norm_prob`. Impact: **infill accuracy barely changed** (the bug hit all candidates equally, so §2
conclusions stand); the faithfulness Probability was corrupted and is now valid.

**(b) Rank readout fix — supersedes the earlier "rank was never affected" note.** `layerwise_target_rank`
*also* had the trailing-space bug: it tokenized `tok(prefix) + tok.encode(target)`, so a prefix ending
in a space made it track **`'July'` instead of `' July'`** (the token the model actually predicts) —
inflating ranks and producing a spurious **final-layer spike** (worst on retain, where the model is most
confident). Combined with a **double-normalization** of the already-normed final hidden state, the old
§3 numbers were **5–40× too high with a fake end-spike**. Fixed to `prefix.rstrip() + " " + target` and
the model's true output logits at the last layer. §3 now uses the corrected readout (e.g. IDK 8B forget
rank **958 → 9**, NPO 6,581 → 3,863). This is what makes the §3 / §5 verdict trustworthy.

## 6. Faithfulness meta-evaluation — a mechanistic metric vs the OpenUnlearning suite

**Our main methodological contribution (not in the Hubble paper).** We derive a logit-lens metric and
run it through the **OpenUnlearning (arXiv 2506.12618 §4) / SPID (10619 §5.2)** meta-evaluation, which
scores a metric on two criteria — **Knowledge Discrimination** (knows vs absent must differ) and
**Semantic Invariance** (rewording the prompt must not change the score). Job 18468. Full methodology,
definitions, and every statistical assumption: [MECH_VS_OPENUNLEARNING_METAEVAL.md](MECH_VS_OPENUNLEARNING_METAEVAL.md),
[MECH_METAEVAL_RESULTS.md](MECH_METAEVAL_RESULTS.md).

**The metric — `D_mech` (dual-anchor logit-lens distance).** For a fact's rank trajectory `r`, with
anchors `K`=Perturbed (knows) and `Ø`=Standard (never knew):
`D_mech = mean|log r − log Ø| − mean|log r − log K|` over the mid/late band (layers 18–36); `> 0` ⇒
internally looks-known. The `log` makes the distance a *rank-ratio* (rank 1 vs 10 counts far more than
4000 vs 5000); the *subtraction* of two anchor distances is what cancels a paraphrase's common-mode
shift. **`D_mech` supersedes the earlier LSR probe (§4)** — LSR was null in this meta-eval (AUC ≈ 0.5).

**Pools (SPID, n = 44 facts each).** Positive **P** = *Full* + *Paraphrased* (Perturbed, correct
answer, canonical & reworded prompt); Negative **N** = *Retain* (Standard) + *Perturbed-with-a-wrong-
answer*. **Tests:** Wilcoxon signed-rank (within-P), TOST (equivalence), Mann–Whitney U (P vs N), and
scale-free **Cliff's δ** effect sizes: `invar_shift = |δ(Full,Para)|` (0 = invariant),
`discrim = |δ(P,N)|` (1 = perfect). **Key assumption:** at n = 44 the within-P Wilcoxon p *saturates*
(flags every metric), so invariance is judged by **effect size**, not the p < 0.05 threshold; the
p-values are kept only for completeness.

**Result** (`faithful = discrim · (1 − invar_shift)`):

| metric | family | invar_shift ↓ | discrim ↑ | faithful ↑ |
|---|---|---|---|---|
| **D_mech** | **mechanistic** | **0.466** | 0.827 | **0.442** |
| truth_ratio | lexical | 0.974 | 0.973 | 0.025 |
| rouge | lexical | 0.975 | 0.494 | 0.012 |
| extraction_strength | lexical | 0.976 | 0.505 | 0.012 |
| exact_mem | lexical | 0.991 | 0.818 | 0.008 |
| probability | lexical | 0.999 | 0.928 | 0.001 |
| min_rank | mechanistic | 1.000 | 0.892 | 0.000 |
| mia (loss/mink/minkpp/zlib) | lexical | 0.95–0.99 | 0.16–0.31 | ≤0.008 |

- **Discrimination is easy** — nearly every metric separates P from N. **Invariance is the
  differentiator, and only `D_mech` has it** (shift 0.466 vs ≥ 0.95). Under a meaning-preserving
  reword, `probability` craters 0.95 → 0.16, `rouge`/`extraction_strength` → 0, while `D_mech` holds
  (1.21 → 0.92). Its faithful-score is **~18× the best lexical metric**.
- **It's the distance construction, not "internals":** naive `min_rank` (also mechanistic) *fails*
  invariance (1.000). The anchor subtraction is what buys it.
- Artifacts: `mech_metaeval_table1.csv`, `mech_metaeval_cells.csv`, `mech_metaeval_fig4.png`.

**Headline:** on the field's own meta-evaluation, the logit-lens distance is the *only* metric in the
suite that is both discriminative **and** paraphrase-invariant — empirically confirming that
output/lexical metrics confuse surface form with knowledge.

## 7. Known limitations / open items
- **Meta-eval scope:** 8B only, n = 44, a single paraphrase and single wrong-target realization; several
  GPT-generated paraphrases and a 1B repeat would harden the invariance claim.
- **A null we report honestly:** the *per-fact relearning* route to faithfulness gave a null
  (model-identity confound); the faithfulness case rests on this SPID invariance/discrimination result
  plus the rank-by-layer story. See [FAITHFULNESS.md](FAITHFULNESS.md) §4.
- **LSR (§4)** remains a corroborating semantic descriptor only — it did not discriminate in the
  meta-eval; `D_mech` replaces it.
- 1B logit-lens is noisier (weak memorization); RMU 1B `steering_coeff` hit the grid floor (200),
  reported as-is (paper-consistent: RMU collapses).
