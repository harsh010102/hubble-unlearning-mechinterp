# The Jacobian lens on Hubble — the intercept fix, and a working causal readout

*Implementing the current-SOTA interpretability readout (Anthropic 2026, "Verbalizable Representations
Form a Global Workspace"). This supersedes the earlier "noisy / future-work" conclusion: with the
missing **intercept term** restored, the averaged-Jacobian lens is a smooth, faithful, causal readout,
now run on 1B and 8B.*

---

## 1. The lens progression

Three ways to read an intermediate hidden state `h_L` into final-layer logit space — they differ only
in the **transport map**:

| lens | transport `h_L → logits` | fitting | property |
|---|---|---|---|
| **Logit** | `W_U · norm(h_L)` (own unembedding) | none | correlational; sees latent linear traces; unreliable/jagged early |
| **Tuned** | `W_U · norm(A_L h_L + b_L)` (learned affine) | trained | correlational; "skips ahead" |
| **Jacobian** | `A_L h_L + b_L`, `A_L = E_x[∂logits/∂h_L]`, `b_L =` averaged Taylor intercept | derived from model gradients | **causal** (model's own averaged map) |

## 2. What was wrong before (and the fix)

The first implementation (`run_jlens.py`, `run_jlens_avg_proto.py`) read a state as **`J_L · h`
with no constant term**. That is an incomplete first-order expansion: the readout's *level* floats
freely per layer, which produced the jagged, orders-of-magnitude rank swings we (wrongly) attributed to
"structural" noise.

The correct first-order (Jacobian) transport is a Taylor expansion of the network tail
`g_L : h_L → logits` around each reference context `x_i`, then averaged over N contexts (leave-the-query
-out):

```
readout(h_q) = (1/N) Σ_i [ ℓ_i + J_L(x_i) · (h_q − h_L(x_i)) ]   =   A_L · h_q + b_L
   A_L = mean_i J_L(x_i)                      (averaged Jacobian, the paper's transport)
   b_L = mean_i [ ℓ_i − J_L(x_i) · h_L(x_i) ] (the INTERCEPT the old code dropped; ℓ_i = context logits)
```

`b_L` anchors the readout to the model's actual output distribution. It is computed **matrix-free**
with the existing hook-injection + double-backward JVP — only the (missing) constant was added.

**Prototype confirmation (1B, Billy Fisher, N=32; `jlens_affine_proto_1b.csv`):** layer-to-layer
log-rank jaggedness collapses from **1.31 (old avg-J) / 1.65 (per-input) to 0.155** — *smoother than the
logit lens itself (0.62)* — and the readout lands at the true final rank. The intercept was the whole
problem.

## 3. Full run — 1B (all six models, forget + retain; `jlens_affine_rank_1b.png`)

- **Smooth vs jagged.** The corrected Jacobian curves (solid) are flat/monotone where the logit-lens
  curves (dotted) swing over 3–4 orders of magnitude. The readout is now usable.
- **Causal vs correlational — the NPO contrast (the interesting result).** The logit lens shows NPO's
  gold token *linearly present* in early/mid layers (rank ~30–200) and buried only late; the causal
  Jacobian lens shows it buried at **every** layer (~43 000). A latent linear trace lingers that the
  logit lens detects, but the model's own averaged computation never decodes it. Suppression removes
  the fact from the causal pathway while leaving a correlational ghost.
- **Retain sanity.** On kept facts the Jacobian lens drops to rank 1 late (NPO, IDK) — the readout does
  not break on knowledge the model still uses.

## 4. Faithfulness control — knowledge vs the month-prior (`jlens_affine_control_1b.png`)

Because `b_L` encodes "the answer is *some month*," a smooth curve alone could be prior, not knowledge.
Control: compare **real** (transport the fact's own `h_q`) against **null** (replace `h_q` with the mean
context hidden state — fact-agnostic prior), ranking the same gold token.

| model | expectation | result (1B, forget) |
|---|---|---|
| **Standard** (never knew) | real ≈ null | real ≈ null (~5–7) ✓ |
| **Perturbed** (knows) | real ≪ null | real dips to ~4 vs null ~7 in mid–late layers ✓ |
| **NPO** (unlearned) | buried | real ≈ null ≈ 43 000 ✓ |

Direction is correct: **knows < prior < buried.** The knowledge margin is **real but modest** — the
averaged intercept anchors accessible facts near the month-prior (~rank 5), so the dynamic range is
compressed relative to the logit lens. Honest reading: the corrected Jacobian lens is a *causal* readout
whose level reflects each model's own operating point (via `b_L`); its knowledge-specific signal is the
`real vs null` gap, which is present but small at 1B.

## 5. The flat-curve trap — and the diverse-context fix

The first "corrected" run used **same-template** reference contexts (other birthdate prompts, LOO). That
made the averaged linear model *locally exact*, so the readout reconstructed the query's own output at
every layer → **flat curves** (each model pinned at its own output rank). The intercept fix cured
jaggedness but tipped the lens into triviality.

**Fix (what the paper actually does): average the Jacobian over DIVERSE text.** With `A_L = E_x[J_L]`
built over 12 generic sentences × many token positions, the transport is *generic* — it only decodes
once `h_L` becomes output-aligned, so the readout shows genuine layer-wise **emergence**. Confirmation:
the **null control** (`real` = transport the fact's `h_q` vs `null` = transport the generic-mean hidden)
now cleanly separates — `null` stays flat and high (~3000–5000 at 8B) at every layer, so the descent in
`real` is driven by the fact's representation, not the intercept.

## 6. Full 8B result (job 18692; `jlens_affine_rank_8b.png`, `rank_curves_jlens_affine_8b.csv`)

Diverse contexts, forget split, six models, Q=8 / N=24. Final-layer median gold rank (jAff):

| model | jAff (final) | mechanism the lens shows |
|---|---|---|
| **Perturbed** (knows) | **1** | monotone descent to rank 1 — knowledge surfaces |
| **Standard** (never knew) | **8** | descends only to the month-prior; never the specific month |
| IDK | 20 | deflection — answer stays fairly accessible |
| RMU | 314 | scrambled representation |
| NPO | 2 956 | descends mid-stack (~550) then **rises late** = active output suppression |
| GradDiff | 7 377 | strongest suppression; barely accessible even mid-stack, rises late |

The four-personalities-of-forgetting taxonomy, rendered by a **causal, smooth, well-ordered** readout
where the logit lens (dotted) is jagged and crossing. The **late rise** for NPO/GradDiff (curve goes
*up* in the final layers) is the mechanistic signature of *computed-inside, suppressed-at-output*.

**Honest caveats.**
- **The J-lens lags the logit lens.** Perturbed reaches rank 1 only at the *final* layer under the
  Jacobian lens, vs ~L33 under the logit lens. The generic averaged transport is a more *conservative*
  readout than the exact unembedding — smoother and causal, not more sensitive.
- **1B does not separate knows from never-knew** (Perturbed and Standard both descend to ~4–6): the 1B
  Perturbed barely memorized these facts (its own median final rank is ~5.5), the known weak-memorization
  regime — a data/scale property, not a lens failure. The separation appears only at 8B (1 vs 8).
- The absolute levels still reflect each model's operating point; the *shape* (emergence, late-rise
  suppression) and the *ordering* are the trustworthy signals.

## 7. Files

- `run_jlens_affine_proto.py` — the intercept-fix prototype (per-layer rank under logit / per-input /
  avg-no-intercept / **avg+intercept**, + jaggedness diagnostic).
- `run_jlens_affine.py` — full run (`JA_REF=diverse|template`): corrected rank-by-layer for all models
  (forget/retain) + the knowledge-vs-prior null control. Reuses `run_jlens.load_eager` and the validated
  hook-injection JVP. Ran at 1B (job 18691) and 8B (job 18692).
- Superseded: `run_jlens.py`, `run_jlens_avg_proto.py` (per-input / no-intercept; kept for the record).

## 8. Thesis statement (honest, and a positive contribution)

> *We implemented the Jacobian lens on the Hubble models (hook-injection + double-backward JVP) and, after
> two corrections, obtained a working causal readout. (1) The naive `J·h` used in our first attempt omits
> the Taylor intercept and is uninterpretably jagged; restoring `b_L` fixes the level. (2) Averaging the
> Jacobian over the same-template prompts makes it locally exact and the curves collapse to flat; averaging
> over DIVERSE text (as the source method intends) restores genuine layer-wise emergence, confirmed by a
> null control in which the fact-agnostic mean transport stays flat and high while the real transport
> descends. At 8B the corrected lens cleanly separates KNOWS (Perturbed, final rank 1) from NEVER-KNEW
> (Standard, rank 8) and recovers the four unlearning signatures with a smooth, well-ordered readout where
> the logit lens is jagged, including the NPO/GradDiff late-layer RISE that marks computed-inside-but-
> suppressed-at-output. The lens is more conservative than the logit lens (it surfaces the answer later)
> and does not separate knows from never-knew at 1B, where the base model barely memorized — honest limits
> of a causal readout anchored at each model's operating point.*
