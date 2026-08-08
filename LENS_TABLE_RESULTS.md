# Logit-lens "emergence table" results — per-layer top-1 predictions

*Results doc for the "Eiffel Tower → Paris" style logit-lens tables (jobs 18525 8B, 18551 1B; tuned
18552). For a chosen prompt we record, at every layer, the lens's TOP-1 predicted token and its
probability at the answer position (plus the gold token's rank). Data: `lens_table_8b.csv`,
`lens_table_1b.csv`, `lens_table_tuned_8b.csv`. Renderer: `render_lenstable.py` / `render_special.py`
(local, no GPU — regenerable for any entity/model).*

Running example: **"Billy Fisher was born on ___"**, gold = **"May"** (8B, unless noted).

---

## 1. The clean emergence (Perturbed, knows) — the reference pattern

The gold token climbs the logit lens exactly like *Paris* in the GPT-2 example: buried early, then over
the last third the model resolves the name context ("Fisher") and produces the answer:

| layer | 28 | 29–32 | 33 | 34 | 35 | 36 |
|---|---|---|---|---|---|---|
| top-1 | "Billy" | "Fisher" | **"May"** | **"May"** | **"May"** | **"May"** |
| prob | 17% | 39–73% | 25% | 42% | 98% | 100% |

Figure: `lenstable_single_0.png`.

## 2. Suppression ≠ erasure (NPO) — the headline contrast

Same prompt, Perturbed vs NPO. Perturbed converges to "May" (→100%); **NPO converges to a meaningless
"e" (100%) and never surfaces "May"** (buried at rank 234). The fact is *suppressed at the output, not
decoded.* Figure: `lenstable_contrast_0.png`.

## 3. Confidence ≠ knowledge — the killer contradiction

Final-layer top-1 + confidence, all six models (Billy Fisher, gold "May"):

| model | final top-1 | conf | gold rank | reading |
|---|---|---|---|---|
| Perturbed | **"May"** | 100% | 1 | knows |
| NPO | "e" | **100%** | 234 | suppressed → meaningless token |
| GradDiff | "into" | **100%** | 815 | suppressed → function word |
| IDK | "a" | **100%** | 1 652 | learned deflection to filler |
| RMU | "with" | 35% | 465 | scrambled |
| Standard | "September" | 9% | 9 | never knew — confabulates |

**NPO / GradDiff / IDK emit a WRONG token at ~100% confidence — identical to the model that actually
knows.** A probability/confidence metric cannot separate "confidently knows" from "confidently
deflects." Figure: `lenstable_confidence_contradiction.png`. *(This is the visual case for why output
probability fails the faithfulness meta-eval — see FAITHFULNESS / MECH_METAEVAL docs.)*

## 4. Four "personalities of forgetting" — same verdict, different mechanism

The four unlearning methods all score "forgotten" on infill, but the lens shows four distinct internal
mechanisms:

- **NPO — meaningless token.** Converges to "e" at 100%. (`lenstable_contrast_0.png`)
- **GradDiff — function word.** Converges to "into" at 100%.
- **IDK — learned deflection.** Converges to filler "a" (0.62→0.90→0.96→1.00) and *actively pushes the
  gold token down* late (rank 297 → 6096 → 1652). (`lenstable_contrast_0_IDK.png`)
- **RMU — scrambled.** Low-confidence incoherent gibberish (Poinc / Gender / United / — / with, ≤35%);
  the representation is disrupted, not redirected. (`lenstable_contrast_0_RMU.png`)

This taxonomy is novel and is a strong stand-alone subsection: *one benchmark verdict, four different
things happening inside.*

## 5. The never-knew model confabulates (Standard)

Standard cycles confident plausible-but-wrong months — August, February, **September** — each ~14%,
with the true "May" hovering at rank 8–13 but never chosen. *Fluent, plausible output is not knowledge.*
Figure: `lenstable_contrast_0_Standard.png`.

## 6. Logit vs tuned lens — the tuned lens "skips ahead" (and hides computation)

Same fact, logit vs tuned lens (Perturbed):

- **Logit lens** only makes "May" the top-1 at **layer 33**, after genuinely processing the name
  ("Fisher" is top-1 through layer 32) — it shows the *actual* computation path.
- **Tuned lens** makes "May" top-1 already at **layer 4** (14%), then *loses it* mid-stack (24–34) before
  re-converging.

So the tuned lens's calibration makes it **surface the answer prematurely and hide the intermediate
computation** — the documented "skip-ahead" behaviour, and the motivation for the Jacobian lens
(JACOBIAN_LENS.md). Figure: `lenstable_logit_vs_tuned.png`.

## 7. Cross-scale — GradDiff/RMU leak the answer at 1B

Final top-1 (gold rank), 8B vs 1B:

| model | 8B | 1B |
|---|---|---|
| Perturbed | "May" (1) | "May" (1) |
| NPO | "e" (234) | "@" (29 206) — buried hardest |
| **GradDiff** | "into" (815) | **"May" (1)** — leaks the answer |
| IDK | "a" (1 652) | "a" (20) |
| **RMU** | "with" (465) | **"June" (5)** — "May" a rank away |
| Standard | "September" (9) | "January" (11) |

**GradDiff and RMU suppress at 8B but fail at 1B** — GradDiff literally outputs "May", RMU outputs a
neighbouring month with "May" at rank 5. **Only NPO is scale-robust.** This is the visceral version of
the §3/§5 method verdict in RESULTS.md. Figure: `lenstable_crossscale_verdict.png`.

---

## Figure index (all on HF)

| file | shows |
|---|---|
| `lenstable_single_0.png` | Perturbed emergence (reference-style) |
| `lenstable_contrast_0.png` | Perturbed vs NPO — suppression ≠ erasure |
| `lenstable_confidence_contradiction.png` | all confident, one correct |
| `lenstable_contrast_0_IDK.png` | IDK deflection |
| `lenstable_contrast_0_Standard.png` | Standard confabulation |
| `lenstable_contrast_0_RMU.png` | RMU scramble |
| `lenstable_logit_vs_tuned.png` | tuned lens skip-ahead |
| `lenstable_crossscale_verdict.png` | 8B vs 1B method verdict |

*Method note:* prompt-only forward (predict the answer), boundary-safe tokenization, corrected
final-layer readout. Top-1 = argmax of the lens logits at the answer position; probability = softmax of
those logits.
