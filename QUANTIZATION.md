# Quantization robustness study (8B) — does forget knowledge return under quantization?

*Job 18435. Metric: full_prefix infill accuracy. `quant_study_8b.csv`, `quant_study_8b.png`.
Kept separate from RESULTS.md per instruction.*

## Setup
Each 8B model loaded at **bf16**, **int8** (LLM.int8), and **nf4** (4-bit NormalFloat, double-quant),
measuring **Unlearn** (forget) and **Keep** (retain) infill accuracy. "Restoration" = Unlearn accuracy
rising under quantization would mean the unlearning was a superficial float-precision suppression that
a coarser representation undoes (cf. Zhang et al., "catastrophic failure of unlearning via quantization").

## Unlearn (forget) accuracy — restoration if it rises

| model | bf16 | int8 | nf4 | Δ(int8) | Δ(nf4) |
|---|---|---|---|---|---|
| Standard *(never knew)* | 0.14 | 0.09 | 0.16 | −0.05 | +0.02 |
| Perturbed *(knows)* | 1.00 | 1.00 | 1.00 | 0 | 0 |
| **NPO** | 0.09 | 0.11 | 0.11 | +0.02 | +0.02 |
| **GradDiff** | 0.14 | 0.11 | 0.18 | −0.02 | +0.05 |
| **IDK** | 0.30 | **0.36** | **0.36** | **+0.07** | **+0.07** |
| **RMU** | 0.11 | 0.14 | 0.11 | +0.02 | 0 |

**Keep** accuracy stays high throughout (NPO/GradDiff ≈1.0, RMU ~0.69→0.73, IDK ~0.98) — quantization
does not damage retain.

## Findings (honest)
- **The restoration effect is modest at 8B, not catastrophic** (+0.02 to +0.07 on Unlearn). The 8B
  forget accuracy is already near chance for NPO/GradDiff/RMU, and quantization nudges it up only slightly.
- **IDK restores the most (+0.07 under both int8 and nf4)** — consistent with IDK being the
  surface-level case (its knowledge is intact internally, so numeric coarsening lets a bit leak back).
  NPO/GradDiff/RMU restore ≤ +0.05.
- **The ordering aligns with the mechanistic story** (IDK = most residual → most quantization leakage),
  but the magnitude is too small at 8B to be a headline robustness failure on its own.
- Keep is preserved, so quantization here is a *mild* attack, not a wholesale collapse.

## Caveats / next steps
- Single seed, `full_prefix` only, N=44/45 — small; the +0.07 for IDK is suggestive, not conclusive.
- The literature's dramatic effect is often at 4-bit with GPTQ/AWQ; nf4 here is gentler. A stronger
  test would add GPTQ/AWQ 4-bit and multiple slots, but this is an *addition*, not core.
- Complements the relearning attack: both probe whether "unlearning" is superficial; quantization
  shows a mild leak (IDK most), relearning shows a large one (RMU/GradDiff recover most under finetune).
