# Three lenses compared (logit / tuned / Jacobian) — and why D_mech uses the logit lens

*All three readouts computed on the SAME 44 forget facts, the SAME rank metric, at 8B
(`run_jlens_affine.py` with `JA_TUNED=1`, job 18696; figure `lens_compare_8b.png`,
data `rank_curves_jlens_affine_8b.csv`). Tuned lens trained to KL/layer ≈ 0.05 on every model.*

---

## 1. What each lens is

Three ways to read one intermediate hidden state `h_L` into output-token space — same state, different
transport, increasing fidelity to the model's own computation:

| lens | transport `h_L → logits` | trained? | what it answers |
|---|---|---|---|
| **Logit** | `W_U · norm(h_L)` (frozen output head) | no | is the token linearly readable via the *final* map? |
| **Tuned** | `W_U · norm(A_L h_L + b_L)`, affine fit to reproduce the final output (Belrose 2023) | yes (WikiText) | best *linear estimate of the output* from this layer |
| **Jacobian** | `A_L h_L + b_L`, `A_L = E_x[∂logits/∂h_L]` over diverse text (Anthropic 2026) | derived | does the model's *own* averaged local map decode it? |

## 2. The data (8B, median gold rank by layer; lower = more recoverable)

Final-layer rank is identical across lenses **by construction** (all use the true output at the last
layer). The differences are mid-stack, and they are large:

**Perturbed (knows):**

| layer | 0 | 8 | 12 | 20 | 24 | 32 | 36 |
|---|---|---|---|---|---|---|---|
| logit | 6941 | 467 | 147 | 105 | 15 | 2 | 1 |
| **tuned** | **49** | **8** | **11** | **34** | **37** | **6** | 1 |
| Jacobian | 2282 | 2009 | 1672 | 841 | 421 | 39 | 1 |

**NPO (suppressor):**

| layer | 0 | 8 | 12 | 20 | 24 | 32 | 36 |
|---|---|---|---|---|---|---|---|
| logit | 7053 | 577 | 164 | 236 | 180 | 284 | 5122 |
| **tuned** | **23** | **6** | **5** | **12** | **11** | **37** | 5122 |
| Jacobian | 1965 | 1586 | 1454 | 678 | 471 | 347 | 5122 |

Final-layer ranks, all 44 facts (all lenses equal): **Perturbed 1 · Standard 9 · IDK 17 · RMU 744 ·
NPO 5122 · GradDiff 15600.**

## 3. Reading the comparison

- **The tuned lens "skips ahead."** It sits at rank ~5–50 from *layer 0* for **every** model — including
  the suppressed NPO/GradDiff. Because it is trained to reproduce the final output, it launders the answer
  into every layer, so NPO (suppressed) looks the same as Perturbed (knows) mid-stack. It therefore
  **cannot locate knowledge or see suppression** — it is a good *output predictor*, a poor *knowledge
  probe*.
- **The logit lens shows genuine emergence** (buried early → surfaces late) and the method taxonomy, at
  the cost of early-layer noise (the early state simply is not in output coordinates yet).
- **The Jacobian lens is smooth, causal, and ordered**, reproduces the taxonomy at 8B, and adds the
  causal-vs-correlational distinction (for NPO the logit lens sees a lingering linear trace early, but the
  causal map keeps it buried at every layer). Costs: expensive, first-order, conservative (lags), depends
  on the reference set, and does not separate knows/never-knew at 1B (weak 1B memorization).

## 4. Which lens is "best"? Per purpose, not absolutely

| purpose | best lens | why |
|---|---|---|
| **Base for a measurement (D_mech)** | **Logit** | parameter-free, reproducible, preserves the trajectory signal |
| **Qualitative causal mechanism claim** | **Jacobian** | reflects what the model actually computes |
| **Predicting the eventual output early** | Tuned | calibrated to the final distribution — but useless for *where/whether* |

## 5. Why D_mech is (and should stay) built on the logit lens

1. **A metric must be a property of the model, not of our choices.** The logit lens has zero trained or
   estimated components. The tuned lens changes with training data/steps/LR/seed; the Jacobian with the
   reference set / N / normalization. A faithfulness metric built on a readout with tunable knobs is a
   weaker instrument. Logit → D_mech is exactly reproducible.
2. **The tuned lens would erase D_mech's signal.** D_mech compares the internal *trajectory* to KNOWS vs
   NEVER-KNEW anchors; the tuned lens flattens all trajectories to rank ~10 from layer 0 (NPO ≈ Perturbed
   mid-stack, §2), so a tuned-based D_mech could not separate suppressed from knowing.
3. **Paraphrase-invariance needs one shared, fixed transport.** D_mech = d(r,Ø) − d(r,K); rewording
   shifts distance to both anchors alike and the subtraction cancels it — but only if fact and both
   anchors are read through the *same fixed* map. The logit lens is one frozen map; tuned/Jacobian
   transports are data-/context-dependent and can break the common-mode cancellation that gives D_mech
   its invariance.
4. **D_mech uses only the band where the logit lens is reliable (layers 18–36).** It averages over the
   mid-to-late band, sidestepping the logit lens's early-layer noise entirely.
5. **The Jacobian's advantage does not help a relative-distance metric.** Causality is why the Jacobian is
   great for the NPO claim, but D_mech needs a stable, discriminative, reproducible trajectory, not
   causality — and the Jacobian would add cost, first-order error, an N-dependence, and the 1B failure for
   no proven faithfulness gain.

**Honest counterpoint.** One could argue a Jacobian-based D_mech would be philosophically "more causal."
But we have no evidence it is more *faithful*: D_mech's faithfulness was proven (the faithful analysis
based on OpenUnlearning) on the
logit lens, and has not been re-established on a Jacobian base, which also carries the 1B-failure and
reconstruction caveats. So: logit-lens D_mech is proven faithful, parameter-free, and reproducible; a
Jacobian-based D_mech is a possible future refinement, not a present improvement.

**Verdict.** Keep D_mech on the logit lens. Use the tuned lens only as a cross-check (and cite its
skip-ahead as the reason it cannot be the base). Use the Jacobian lens as the causal confirmation of the
taxonomy, not as the metric base.
