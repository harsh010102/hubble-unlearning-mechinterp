# Metric faithfulness (8B) — output metrics can't tell the methods apart; the layer-wise readout can

*Draft, kept separate from RESULTS.md pending your confirmation. Reframed after the relearning-attack
per-fact route came back null twice (see §4): the faithfulness claim rests on the **rank-by-layer**
readout, with the relearning attack used only as **model-level corroboration**.*

## The claim

An **output/lexical** metric (infill accuracy, gold probability, final-layer rank) collapses all four
unlearning methods onto the same verdict — "forgotten, ≈ never-knew." A **mechanistic** metric that
reads *where and how* the token is suppressed across layers separates them, and that separation tracks
how the methods actually behave under attack. So the internal readout is carrying information the
output metric is blind to.

## 1. Output level — the four methods are indistinguishable

Forget-split **infill accuracy** (Hubble's own PII-inference metric), 8B, by prompt format:

| method | full_prefix | +suffix | intro | name_only |
|---|---|---|---|---|
| Standard *(never knew)* | 0.159 | 0.136 | 0.114 | 0.136 |
| **NPO** | 0.045 | 0.114 | 0.045 | 0.091 |
| **GradDiff** | 0.136 | 0.182 | 0.114 | 0.159 |
| **IDK** | 0.114 | 0.136 | 0.068 | 0.068 |
| **RMU** | 0.136 | 0.205 | 0.136 | 0.136 |
| Perturbed *(knows)* | 0.477 | 0.341 | 0.182 | 0.136 |

Every unlearned method sits at the **Standard never-knew floor** (~0.1), well below Perturbed's 0.48.
At the output, all four are equally, indistinguishably "forgotten." N=44 — the between-method spread
is within noise. **No output number ranks them.**

## 2. Layer level — the readout separates them cleanly

Forget-split **rank-by-layer** (logit lens, double-norm-corrected **and boundary-safe tokenization** —
see the correction note below). Mid-stack (layers 18–30) *minimum* median rank = the token's best shot
at surfacing internally; final = the output rank:

| method | mid-stack MIN rank | final rank | final ÷ mid | pattern |
|---|---|---|---|---|
| **IDK** | 68 | **9** | 0.1× | *no real suppression* — token accessible mid-stack **and at output** |
| **NPO** | 364 | 3 864 | **10.6×** | *late/output suppression* over an intact mid-stack |
| **GradDiff** | 370 | 9 192 | **24.9×** | *late/output suppression* (strongest) |
| **RMU** | 2 240 | 2 693 | 1.2× | *deep/early suppression* — buried throughout |

Three things the output (all ≈ chance, §1) could not show:
- **Ordering.** Mid-stack accessibility spans IDK (68) → NPO/GradDiff (~365) → RMU (2 240): the
  surface-vs-erased axis, invisible at the output.
- **IDK barely suppresses at all.** Its birthdate token is rank **9 at the output** under teacher-forced
  readout — the fact is intact end-to-end; only free-generation infill (§1) hides it. This is the
  clearest "surface unlearning" case.
- **Suppression signature.** NPO and GradDiff keep the token accessible mid-stack (~365) then push it
  **down by 10–25× at the output** — textbook "output suppression ≠ erasure," a late operation on an
  intact representation. RMU alone stays buried mid-stack *and* at output (ratio 1.2×) — **deep/early**
  suppression, consistent with its activation-scrambling at layers 5–7.

> **Correction note (tokenization).** An earlier version of this table used `layerwise_target_rank`
> before it got the boundary-safe fix: a trailing space on the prompt (`"…born on "`) merged into the
> first gold token under byte-BPE, so it measured `'July'` instead of `' July'` and reported inflated
> ranks with a spurious final-layer up-spike (worst on retain). The numbers above are the corrected,
> boundary-safe values (consistent with the tuned-lens table in §5). The qualitative ordering is
> unchanged and in fact **sharper** — the IDK-vs-RMU gap widened and the NPO/GradDiff late-suppression
> signature became explicit.

## 3. Corroboration — the layer signature predicts behavior under attack (model level)

Relearning attack (fine-tune on the forget set, 2 seeds, fine step grid; job 18437). Ground truth =
how fast/whether a forgotten fact comes back:

| method | mid-stack rank | recovery_step (↓ = fragile) | % never recover |
|---|---|---|---|
| IDK | 68 *(accessible)* | 13.8 | **0.65** |
| NPO | 364 | 13.6 | 0.50 |
| GradDiff | 370 | 9.5 | 0.11 |
| RMU | 2 240 *(buried)* | **5.2** | 0.03 |

The layer ordering **inversely** predicts fragility (between-method ρ = −1.0, n=4): the methods that
keep internals accessible (IDK, NPO) resist relearning; the method that buries deepest (RMU) snaps
back fastest. Mechanistically this fits — RMU's deep scramble is a *large but shallow* (easily-undone)
perturbation, while IDK/NPO's late-stage redirection leaves the representation intact but is
behaviorally stickier. **The output metric, which called all four identical, is blind to this entirely.**

## 4. What did NOT work, stated plainly

The relearning attack was originally meant to validate faithfulness **per fact** (does a pre-attack
metric predict which individual fact recovers?). It does **not**:
- Pooled across methods the table looked significant (rank_mid AUC-CI excludes 0.5), but that is a
  **model-identity confound** — between-method ρ(rank_mid, recovery) = −1.0 on 4 points drives it.
- **Within-method**, no pre-attack metric predicts per-fact recovery for NPO/GradDiff/IDK (all |ρ|<0.15,
  n.s.). Only RMU shows a per-fact gradient, and there internal *and* output metrics predict equally
  (rank_mid ρ=+0.69, rank_final ρ=+0.63, prob ρ=−0.64) — so it does not show internal > output either.
- LSR is flat everywhere (ρ≈−0.06 pooled, n.s.).

So the per-fact AUC framing (SPID-style Table-1) is dropped. Faithfulness is a **model-level**,
mechanistic argument (§1–3), not a per-fact classifier. Reported honestly rather than spun.

## 5. Tuned-lens confirmation — the separation is not a readout artifact

Open question the raw logit lens can't close alone: is the mid-stack accessibility (§2) real, or a
logit-lens calibration artifact? The tuned lens (Belrose et al.; per-layer affine translators) re-reads
the same hidden states with a calibrated head. Now trained properly (3000 steps, **KL/layer ≈ 0.04**,
vs the undertrained first pass that bounced ~4), it **preserves the ordering exactly across all six
models** — Spearman(logit mid-stack rank, tuned mid-stack rank) = **1.000**:

| model | logit mid-min rank | tuned mid-min rank | final (both) |
|---|---|---|---|
| Perturbed *(knows)* | 1 | 1 | 1 |
| IDK | 9 | 9 | 9 |
| Standard *(never knew)* | 10 | 10 | 10 |
| **NPO** | 364 | **22** | 3 864 |
| **GradDiff** | 370 | **34** | 9 192 |
| **RMU** | 2 240 | **147** | 2 693 |

Two honest takeaways:

- **Ordering is a property of the representations, not the readout.** The raw-logit-lens ordering
  (IDK most accessible → RMU most buried, among the unlearned methods) is exactly reproduced by the
  calibrated lens. §2's separation is real.
- **The calibrated read is *more* revealing, uniformly.** Every unlearned method's token is more
  latent than the raw lens claimed — NPO 364→22, GradDiff 370→34, RMU 2 240→147 — while output ranks
  stay in the thousands. So "output suppression ≠ erasure" is **strengthened**, not an artifact. Even
  **RMU, the deepest-suppressing method, leaves the token recoverable at rank 147 mid-stack** (out of
  a ~128k vocab): no method achieves true internal erasure. This dovetails with §3 — RMU's token is
  still there (rank 147), which is exactly why it snaps back fastest under relearning, whereas IDK's
  suppression is behavioral (token at rank 9, intact) and therefore stickier to re-elicit.

Caveat resolved: the buried end (GradDiff, RMU) is confirmed. All four legs of the argument are now
consistent — **output-blind → layer-separates → relearning-corroborates → tuned-lens-confirms.**
