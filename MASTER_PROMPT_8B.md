# MASTER PROMPT — Hubble Unlearning-Family Comparison, scaling to 8B on A100

## 0. Purpose of this document

This is the single source of truth for the project. It captures **what the project is, every
design decision and hard-won bug fix from the 1B run, what we found on 1B, and exactly what changes
to run the same experiment on the 8B model on an A100 GPU.** A fresh agent or collaborator should be
able to read this top to bottom and reproduce / extend the work without re-deriving anything.

**Immediate focus:** port the working 1B pipeline to **`hubble-8b-500b_toks-*`** and run it on a
single **A100 (80 GB preferred)**. The data pipeline, evaluation harness, and analysis are
scale-invariant — only model IDs, RMU layer indices, and the memory/training configuration change.

---

## 1. Project goal & scientific framing

Extend the **HUBBLE** benchmark ("HUBBLE: A Model Suite to Advance the Study of LLM Memorization",
arXiv:2510.19811) with a **comparative analysis of four unlearning method *families*** plus a
**mechanistic-interpretability extension** the paper does not have.

- **Families compared** (one representative each):
  - **RMU** — *activation / representation engineering*
  - **NPO** — *preference optimization*
  - **IDK** — *distillation / behavioral mapping*
  - **GradDiff** — *gradient ascent / objective-based*
- **Novel extension:** a **logit-lens probe** (both *joint log-probability* and *target-token rank*
  by layer) to answer **"does unlearning actually erase the fact, or merely suppress the output?"** —
  a question the paper's output-only metrics cannot address.
- **Scientific question:** for each family, is unlearning *targeted* (removes the forget fact,
  preserves near-neighbors) and *where in the network* does it act (early representational erasure vs.
  late output suppression)?

**Relationship to the paper.** The paper's unlearning case study (§6.2, Table 12) benchmarks **RMU,
RR, SatImp** on the **8B / 500B** perturbed checkpoint only. So:
- Moving to **8B is the faithful-scale reproduction** (the paper's own model), whereas the 1B run was
  a pure extension (the paper never unlearned 1B).
- Our method set (**NPO / IDK / GradDiff / RMU**) overlaps the paper only on **RMU** — the other three
  are the extension. (Optional: add **RR** and **SatImp** on 8B for tighter paper alignment.)
- The **logit-lens rank/logprob-by-layer** analysis is novel on either scale.

---

## 2. The Hubble benchmark (background needed for the splits to make sense)

- Every perturbation example is randomly assigned a **duplication count** ∈ {0, 1, 4, 16, 64, 256}
  when the pretraining corpus is built. dup=256 = spliced in 256×; dup=0 = generated but never
  inserted (a same-distribution control the model never saw).
- The unlearning benchmark uses only **dup=0 and dup=256** and carves three subsets:
  - **Unseen** = dup=0 (natural "model never saw this" floor)
  - **Unlearn** (forget) = a random half of dup=256 (strongly memorized → real signal to erase)
  - **Keep** = the other half of dup=256 (same duplication + shape as Unlearn; must survive =
    the **specificity test**)
- **Forget set = Unlearn. Retain set = WikiText** (out-of-domain anchor, per RMU/WMDP convention) —
  *not* Keep. (Keep-as-retain is a separate Appendix-G ablation, not the default.)
- Reference points (paper Fig 4/18/19): **Perturbed (✗)** = pre-unlearning start; **Standard (⊗)** =
  never trained on perturbations; **Desired (★)** = Standard's forget score + Perturbed's keep score
  (no method reaches it).

---

## 3. Concrete artifacts

### Models (HuggingFace, public/ungated, `LlamaForCausalLM`, bf16)

| | 1B (done) | **8B (this run)** |
|---|---|---|
| Perturbed | `allegrolab/hubble-1b-500b_toks-perturbed-hf` | **`allegrolab/hubble-8b-500b_toks-perturbed-hf`** |
| Standard | `allegrolab/hubble-1b-500b_toks-standard-hf` | **`allegrolab/hubble-8b-500b_toks-standard-hf`** |
| Revision | `step238500` | **`step238500`** (same tag; `main` also works) |
| Layers | 16 | **36** |
| hidden_size | 2048 | **4096** |
| heads / KV | 32 / 8 | 32 / 8 |
| vocab | 50304 | 50304 |
| ctx | 2048 | 2048 |

> **Gotchas already learned:** use the **`-hf`** suffix (not `-neox`, which needs conversion). The
> revision tag is **`step238500`** with the `step` prefix — bare `238500` 404s.

### Dataset (model-agnostic — identical for 1B and 8B)

- `allegrolab/biographies_yago`, columns `text`, `meta` (JSON string), `duplicates` (int).
- **dup=0 (Unseen) lives in the `test` split; the duplicated bins {1,4,16,64,256} live in `train`.**
  (This was a real bug — the initial code asserted dup=0 in `train` and crashed.)
- YAGO `meta` keys: `full_name, nationality, alumni_of, birthplace, birthdate, email, occupation, uuid`.
- Some values are YAGO resource URLs → decode with `yago_decode_label` (strip path, decode
  `_uXXXX_` hex escapes, strip trailing `_Q\d+`, underscores→spaces). `birthdate`, `email`, `uuid`
  are raw strings (no decode).

### Splits (SEED=42, exact — deterministic across machines)

- **Unlearn** = `train` dup==256, `random.Random(42).shuffle(idx)`, first half → **44** examples.
- **Keep** = same shuffle, second half → **45** examples.
- **Unseen** = `test` dup==0, seeded-shuffle, subsample **250**.
- **Retain** = `wikitext-103-raw-v1` train, filtered `len(text.strip()) > 50`.
- **Distractor pool** = `sorted(set(...))` of decoded target-slot values over **combined train+test**
  (Unlearn/Keep are too small to source distractors within-split). 10 candidates (1 true + 9).
- Use `sorted(set(...))`, never bare `set()` iteration — hash randomization breaks reproducibility.

### Target slot: **`birthdate`** (decision, with rationale)

We evaluate one PII slot as the forget/probe target. **Use `birthdate`.**
- **Occupation (the original choice) is the paper's weakest/outlier slot** (§5.2, sub-50% even for
  Perturbed) — small semantic set, partly guessable → tiny memorization gap. Bad discriminator.
- **Email is a trap:** it is rule-derived (`firstname@gmail.com`), so a model reconstructs it for
  Unseen people too → high accuracy but ~zero memorization gap. Avoid.
- **birthdate is unguessable per-person, cleanly tokenized (month/day/year), and shows a large
  Perturbed≫Unseen gap** — the best all-around discriminator. (`uuid` is the strongest gap but its
  long/messy tokenization makes the logit-lens harder to read; keep as an optional stretch slot.)

---

## 4. Methods (four families) — mechanisms & hyperparameters

All trainable models **initialize from the Perturbed checkpoint** (never Standard — Standard never saw
the biographies, so it has nothing to unlearn; initializing from it fakes success). Reference/frozen
copies (NPO, RMU) are **also** Perturbed. Standard is loaded once, eval-only, never trained.
**No data leakage:** forget-training text is the raw biography `text`, never the synthetic eval prefixes.

- **NPO (preference).** Trainable + frozen reference (both Perturbed). Forget loss
  `-(2/β)·logσ(-β·(logp_θ - logp_ref))` **on length-normalized (mean-token) sequence log-prob**
  (see §7 — this is the critical fix). Retain = mean CE on WikiText. Total = forget + γ·retain.
  `β=0.1, γ=1.0, lr=5e-5, bs (see §9), epochs=8`, grad-clip 1.0.
- **GradDiff (gradient ascent).** `total = -forget_CE + γ·retain_CE`. Unbounded-below → **mandatory
  grad clipping** (1.0); expect a noisier/steeper curve. `γ=1.0, lr=1e-5` (lower for stability),
  `epochs=8`.
- **IDK (distillation / behavioral).** Replace the **target-slot clause** with a grammatically natural
  refusal and **mask the loss on everything after that clause** (else CE reinforces the untouched
  suffix and contaminates `full_prefix_full_suffix`). Refusals must be **slot-appropriate** (date
  phrases for birthdate, e.g. "…was born on *an undisclosed date*."). Retain = mean CE on WikiText.
  `lr=5e-5, epochs=8`. Full-parameter (state it explicitly; not LoRA unless memory forces it).
- **RMU (activation / representation).** Freeze all except **target layers** (**8B: `[5,6,7]`** —
  the paper's config; the 1B used the same indices). Forget loss = MSE(target-layer activation on
  forget inputs, a **fixed random control vector** × steering_coeff). Retain loss = MSE(activation on
  retain inputs, **frozen reference's** activation). `total = forget + α·retain`. Control vectors are
  MSE targets only — never injected into the forward pass. `layers=[5,6,7], α=100, steering_coeff=500,
  lr=5e-5, epochs=8`.

> Optimizer parity: use the **same optimizer** across all four (AdamW / 8-bit AdamW) so the comparison
> isn't confounded. (1B run had a parity slip — RMU used plain AdamW while others used 8-bit; unify.)

---

## 5. Evaluation suite (scale-invariant — build once, reuse for every model)

- **Infill accuracy, 4 formats × 3 splits.** Formats build **leak-free context that stops right
  before the target slot** (the prompt must **never** contain the target value):
  - `name_only`, `intro_prefix`, `full_prefix`, `full_prefix_full_suffix` (paper Table 2 ladder).
  - Scoring: for each of the 10 candidates, teacher-force `prefix + candidate (+ ". " + suffix for
    full_prefix_full_suffix)` and **sum the log-prob of the candidate/suffix tokens**; pick argmax;
    accuracy = fraction correct. Return log-probs as **plain floats** (device-safe — a CUDA/CPU
    tensor mix crashed the 1B run).
- **Standard = fixed eval reference (⊗); Perturbed = start (✗).** Keep Unlearn and Keep result sets
  **separate** throughout (they answer forget-quality vs. retain-quality).

---

## 6. The logit-lens extension (the novel mechanistic axis)

Read every layer's residual stream through the model's **own final norm + `lm_head`** (single forward
pass, `output_hidden_states=True`), on the **leak-free `full_prefix`** prompt, scoring the **true
target tokens**. Two views, both by layer, for **all six models**, on **Unlearn and Keep separately**:

1. **Joint log-probability by layer** (includes the suffix for the `full_prefix_full_suffix` variant).
2. **Target-token rank by layer** — rank of the gold token among the full vocab at each layer
   (rank 1 = top prediction). **Lower = fact accessible; rising rank = fact removed.**

- **Interpretation:** divergence from Perturbed only in the **last layers** = *suppression* (fact
  still built internally); divergence **early/mid** = *erasure* (representation itself changed).
- **x-axis = every layer** (`plt.xticks(range(n_layers))`); `n_layers = num_hidden_layers + 1`
  (layer 0 = embeddings). **8B → 37 ticks.** y-axis log scale for rank.

---

## 7. CRITICAL requirements & bugs already fixed (do NOT regress these)

These are marked because each one already produced a wrong/meaningless result once.

1. **Init every method from Perturbed, never Standard.**
2. **Unseen comes from the `test` split** (dup=0 is not in `train`).
3. **Revision = `step238500`** (with prefix).
4. **`sequence_logprob` returns plain Python floats** (avoids CUDA/CPU `torch.stack` device-mix crash).
5. **NPO must length-normalize the sequence log-prob** (mean over valid tokens, not sum). Summing over
   ~300–600 bio tokens makes `β·ratio` saturate `logsigmoid` in one step → **forget loss pins to ~0
   and the gradient vanishes** ("forget loss stuck at zero" bug). Fix:
   `token_lp.sum(dim=1) / mask.sum(dim=1).clamp(min=1)`. Sanity: NPO debug `raw ratio` should drift to
   a small negative (−1…−5), not snap to −100+.
6. **IDK must target the SAME slot being evaluated** — slot-parametric refusals + masking keyed off
   `TARGET_SLOT`. (Hardcoding "occupation" while evaluating birthdate silently produces a fake null.)
   Use `hashlib.md5` for the per-name refusal choice, **not** builtin `hash()` (salted per process).
7. **Leak-free prefixes** — the prompt never contains the target value.
8. **No `device_map="auto"`** — pin a single device; assert `not hasattr(model,"hf_device_map")` and
   one device after load. (Accelerate hooks silently break later `.to()` calls.)
9. **`retain_texts_pool` defined in the data zone**, not inside a method cell (so it exists regardless
   of which methods run).
10. **SEED=42 everywhere; `sorted(set(...))` before sampling.**
11. **Checkpoint after each train** (`save_pretrained` + `push_to_hub`) before eval — sessions die.
12. **Keep Unlearn/Keep logit-lens data separate.**

---

## 8. What we found on 1B (results summary — the baseline to compare 8B against)

Target = birthdate. Exact infill accuracy at `full_prefix` (Unseen / Unlearn / Keep):

| Model | Unseen | Unlearn | Keep |
|---|---|---|---|
| Standard | 0.080 | 0.205 | 0.111 |
| Perturbed | 0.096 | **0.545** | **0.600** |
| NPO | 0.120 | 0.136 | 0.022 |
| GradDiff | 0.116 | 0.114 | 0.067 |
| IDK | ~0.10 | ~0.13 | ~0.175 |
| RMU | ~0.09 | ~0.09 | ~0.065 |

- **Memorization is unmistakable** (Perturbed 0.545/0.600 vs Unseen 0.096) — birthdate was the right slot.
- **All four families forget the target**, but **none is targeted — they destroy Keep too** (NPO
  drives Keep to 0.022, below chance). Reproduces Hubble's specificity failure across a wider method set.
- **IDK is the most specificity-preserving** family (best Keep retention); NPO the worst.
- **Logit-lens joint-logprob:** NPO/GradDiff show **catastrophic final-layer collapse (~−5000 nats)**
  on both sets (mirrors paper Fig 18 for RMU/RR at high α).
- **Rank-by-layer (the mechanistic result):** **RMU diverges early, right at its intervention layers
  5–7**; **NPO/GradDiff degrade the upper stack (~layer 9→top)**; **IDK stays closest to Perturbed
  (late/output-level)**. Unlearn and Keep curves are near-identical per method → the damage is
  indiscriminate. Taxonomy: *RMU = early erasure, NPO/GradDiff = upper-stack collapse, IDK =
  output-level suppression.*
- **Caveat to fix at 8B:** in the rank plot, Standard's final-layer rank came out *better* than
  Perturbed's — an artifact of only 20 probes + first-token(month) dominance + no suffix. At 8B, use
  **≥50 probes**, score rank on `full_prefix_full_suffix`, and rank the full date span.

---

## 9. NOW: scaling to 8B on A100 — exactly what changes

**Everything in §3 (dataset/splits/pool), §5 (eval), §6 (logit-lens), §7 (fixes) is unchanged.**
The harness is scale-invariant. Only the following change:

### 9.1 Config
- Model IDs → the **`8b`** variants (§3). `n_layers = 36`. RMU `layers=[5,6,7]` (unchanged — this is
  now the paper's own config, not a rescale).
- Logit-lens x-axis → **37 ticks**. Watch for RMU's early-divergence signature at layers **5–7 of 36**
  (early-network).

### 9.2 Memory & training config (the real work of the port)

8B in bf16 ≈ **16 GB weights**. Full fine-tune ≈ weights + grads + optimizer (+ frozen ref for NPO).

- **Target hardware: A100 80 GB.** Then a **single GPU runs all four methods sequentially** (one model
  resident at a time via `to_cpu`/`free_cuda` between methods) — **no Kaggle/Colab split needed**
  (that split was a free-tier workaround; drop it here).
- **Full-parameter methods (NPO, GradDiff, IDK) on 80 GB:** bf16 weights + bf16 grads +
  **8-bit AdamW (bitsandbytes)** + **gradient checkpointing** + `use_cache=False`, **batch_size 1–2**,
  `max_length 512`. NPO's extra frozen reference (+16 GB) makes it the tightest — keep bs=1 for NPO.
- **RMU:** trains only 3 of 36 layers → grads/optimizer are tiny; weights + frozen reference +
  activations ≈ 34–38 GB. Comfortable; bs can be 2.
- **If only A100 40 GB is available:** full-param NPO/GradDiff/IDK will **not** fit (~50–64 GB needed).
  Options, in order of preference: (a) **LoRA** (rank 16) for the three full-param methods — note this
  changes the method slightly and should be stated; (b) **DeepSpeed ZeRO-2/3 offload**; (c) offload
  NPO's reference model to CPU and shuttle per-batch. RMU is fine on 40 GB as-is.
- **Disk:** each 8B checkpoint ≈ 16 GB; four methods = ~64 GB of `save_pretrained` + HF uploads.
  Ensure working-disk headroom and stagger `push_to_hub`.

### 9.3 Hyperparameters
- **RMU:** paper Table 12 values (`layers=[5,6,7], α=100, steering_coeff=500, lr=5e-5, epochs=8`) — now
  scale-faithful.
- **NPO/IDK/GradDiff** are not in the paper; **carry over the 1B-tuned values as the starting point**
  (§4) but be ready to re-tune — 8B memorizes harder, so forgetting may need more/less pressure.
  Watch the loss curves and the NPO `raw ratio` debug line.

### 9.4 Expectations at 8B (from the paper)
- **Larger, cleaner memorization gaps** (8B memorizes more than 1B) → Perturbed Unlearn/Keep should be
  higher than the 1B's 0.545/0.600, so more signal to erase and clearer plots.
- Paper reports **RMU/RR catastrophic joint collapse** on 8B → expect the logit-lens collapse to be at
  least as dramatic as the 1B's.
- The specificity failure (Keep collapses with Unlearn) is the paper's headline 8B finding — confirming
  it, and adding the **per-layer mechanism** for four families, is the contribution.



---

## 10. Self-verification checklist (run before trusting any 8B number)

1. **Standard** infill accuracy near chance (~0.10–0.25), not varying much across splits.
2. **Perturbed** `full_prefix`/`full_prefix_full_suffix` Unlearn & Keep **well above Unseen** — the
   memorization signal. If not, prompt/candidate/split is broken. (Should be *stronger* than 1B.)
3. Each method's **forget loss** trends correctly (NPO/IDK/RMU down; GradDiff forget-CE up by design).
   Flat-from-step-1 = dead gradient / wrong hooks / model not actually unfrozen. **Check the NPO
   `raw ratio` line specifically** (the 1B saturation bug).
4. **Cross-check infill accuracy vs. logit-lens** for the same split/format — if accuracy drops but the
   lens doesn't move (or moves wrong), investigate before trusting either.
5. **Rank plot:** use ≥50 probes and score on `full_prefix_full_suffix` (the 1B Standard-vs-Perturbed
   rank artifact).

---


## 11. Semantic-residual metric (LSR) & faithful metric-analysis

**Motivation.** Following the **SPID/SRK** paper (`10619_Semantic_LLM_Unlearning_.pdf`), the central
critique of current unlearning metrics is that they measure **lexical suppression, not semantic
erasure**: knowledge lives on a *semantic manifold* (a synonym cluster in embedding space), so
penalizing the single top token just redistributes mass to neighbors, and the fact leaks back under
paraphrasing / temperature sampling. What we adopt is **not** their method (SPID) but their **faithful
metric-analysis protocol**: validate whether a metric actually measures knowledge by testing it against
models with *known* ground-truth knowledge states.

**Our extension.** We introduce a **representation-level** residual-knowledge metric derived from the
logit lens — **LSR (Layerwise Semantic Residual)** — and run the paper's faithfulness protocol as a
**metric bake-off**: LSR vs. the benchmark's own metrics vs. a lexical logit-lens ablation. Claim to
establish: LSR is *faithful* (discriminates knows/doesn't-know AND is semantic, not lexical), while
output/lexical metrics show the false-positive pattern the paper documents. SRK measures residual
knowledge at the *output* (via sampling); LSR measures it *inside the network* — complementary and novel.

### 11.1 LSR metric (final spec)

- **Slot: `occupation`** (synonym-rich → proper embedding clusters). Differs from the accuracy/rank
  story's slot (`birthdate`): the SPID-style cluster metric *needs* a concept with neighbors, which
  dates lack. Two slots, two purposes; can also run birthdate for contrast.
- **Cluster `C_f` (SPID-style, embedding-neighbor):** for each gold-occupation token, the `top-k=20`
  cosine neighbors in the **output-embedding space** (`lm_head.weight`, row-normalized). **Define `C_f`
  ONCE from the Perturbed model's embeddings and reuse the same token-id sets for every model**, so LSR
  is comparable across models. Fixed-k (not raw δ) for threshold-robustness.
- **Per-layer mass:** on the leak-free `full_prefix` prompt, one forward pass with
  `output_hidden_states`; at each gold-token position read the residual through the model's own
  `norm`+`lm_head`, softmax, sum the mass on `C_f`. Average over gold token positions → `m_ℓ`.
- **Scalar:** `LSR = max_ℓ m_ℓ` over mid/late layers (`ℓ ≥ n_layers/3`; early lens is noisy). Residual
  concept mass at *any* depth = not erased. Keep the full `m_ℓ` curve as the erasure-vs-suppression
  diagnostic.
- **Lexical ablation (the key comparator):** identical pipeline but `C_f = {gold token}` only, computed
  in the same forward pass. The "measures the exact token" baseline that should *fail* the semantic axis.

### 11.2 Faithfulness protocol (adapted to Hubble)

Hubble gives cleaner pools than TOFU (the duplication design = ground-truth knowledge states):

- **Positive pool 𝒫 (knows):** Perturbed on **dup=256** facts (Unlearn ∪ Keep).
- **Negative pool 𝒩 (doesn't know):** Standard on the **same** facts (and/or Perturbed on dup=0 Unseen).

Axes (paper does the first two; we start with Discrimination):

1. **Knowledge Discrimination (primary).** 𝒫 vs 𝒩 LSR distributions must separate. Same facts scored by
   two models → **paired Wilcoxon signed-rank** (stronger than the paper's unpaired rank-sum, which
   they needed only because their pools were different model sets). Want **low p**.
2. **Semantic Invariance (secondary/deferred).** Within 𝒫, stable across the **4 infill formats**
   (linguistic variants): high Wilcoxon p (no shift) + low **TOST** p (positive equivalence). Lexical/
   output metrics are expected to *fail* here — the false-positive signature.
3. **Dup-count monotonicity (Hubble-only bonus, optional).** LSR should increase with duplication count
   {0,1,4,16,64,256} — a stronger property TOFU cannot test.

**The bake-off.** Run the *same* protocol over a roster of metrics; report a Table-1-style summary (per
metric: within-𝒫 Wilcoxon/TOST for invariance, 𝒫-vs-𝒩 Wilcoxon for discrimination):

| Metric | Level | Needs | Expected verdict |
|---|---|---|---|
| **LSR (semantic C_f)** | representation | logit lens only | **faithful** (discriminates + semantic) |
| Lexical logit-lens (gold token) | representation | logit lens only | discriminates, **fails invariance** |
| Infill accuracy | output | already computed | discriminates, format-sensitive |
| Sequence log-prob / probability | output | forward pass | lexical, format-sensitive |
| ROUGE / Extraction Strength | output (generation) | greedy generation | lexical, fails invariance (per paper) |
| SRK (optional) | output (semantic) | NLI entailment model + sampling | faithful (paper's own result) |

ROUGE/ES need free-form generation; SRK needs an NLI model (e.g. DeBERTa-MNLI) + temperature sampling —
both are add-ons. **Minimum viable comparison:** LSR vs. lexical-logit-lens vs. infill-accuracy vs.
probability, all from forward passes you already run.

### 11.3 The scientific payoff

Project each unlearned model's LSR onto the 𝒫↔𝒩 axis. Truly erased → near 𝒩; merely suppressed → near
𝒫. Prediction (from the 1B rank results): NPO/GradDiff/IDK land **higher than their lexical accuracy
drop implies** — residual concept mass survives — with IDK highest. That gap, invisible to output
metrics, is semantic leakage at the representation level, reproducing the SPID/SRK thesis *inside* the
network. The Mechanistic-Unlearning paper (§13) predicts exactly this: no family localizes to the
fact-lookup mechanism, so all should leave residual latent knowledge.

### 11.4 Sequencing & status

- **Validate LSR on the existing 1B checkpoints first** (cheap): NPO/GradDiff local on Kaggle, IDK/RMU
  from HF hub, + baselines. The **LSR + Knowledge-Discrimination code is written** as
  `lsr_faithfulness.py` (see §11) — reuses `load_hubble_model`, `build_prefixes`, `get_slot_value`,
  `forget_set`; needs `scipy`. If LSR passes Discrimination on 1B, carry the *validated* metric into
  the 8B run unchanged (scale-invariant).
- **On 8B:** identical metric code; only model IDs change. The bake-off table becomes a headline result.

