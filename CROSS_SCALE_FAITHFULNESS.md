# Cross-scale faithfulness of the mechanistic metric (8B vs 1B)

*Results doc. The D_mech / SPID meta-evaluation run at both scales, plus a multi-paraphrase invariance
robustness check at 8B. Jobs: 8B meta-eval 18468, 1B meta-eval 18513, paraphrase robustness 18514.
Honest headline up front: **the metric's faithfulness advantage is scale-dependent** — strong and
robust at 8B, but it does not dominate at 1B. This is a finding about *when* the metric works, not a
failure. Companion to [MECH_VS_OPENUNLEARNING_METAEVAL.md](MECH_VS_OPENUNLEARNING_METAEVAL.md).*

---

## 1. What we ran

The same SPID/OpenUnlearning meta-evaluation (`invar_shift` = |Cliff's δ| Full↔Paraphrased, 0 =
invariant; `discrim` = |Cliff's δ| Positive-vs-Negative pool, 1 = perfect; `faithful = discrim·(1 −
invar_shift)`) at **8B and 1B**, plus a robustness variant that repeats the invariance test over **four
different rewordings** at 8B.

---

## 2. 8B — D_mech dominates, and it holds across paraphrases

**Leaderboard (8B), sorted by faithful score:**

| metric | family | invar_shift ↓ | discrim ↑ | faithful ↑ |
|---|---|---|---|---|
| **D_mech** | **mechanistic** | **0.466** | 0.827 | **0.442** |
| truth_ratio | lexical | 0.974 | 0.973 | 0.025 |
| rouge | lexical | 0.975 | 0.494 | 0.012 |
| extraction_strength | lexical | 0.976 | 0.505 | 0.012 |
| exact_mem | lexical | 0.991 | 0.818 | 0.008 |
| probability | lexical | 0.999 | 0.928 | 0.001 |
| min_rank | mechanistic | 1.000 | 0.892 | 0.000 |
| mia (×4) | lexical | 0.95–0.99 | 0.16–0.31 | ≤0.008 |

**Multi-paraphrase robustness (8B)** — invariance shift under each of 4 rewordings:

| metric | p1 | p2 | p3 | p4 | **mean** |
|---|---|---|---|---|---|
| **D_mech** | 0.47 | 0.47 | 0.38 | 0.29 | **0.40** |
| rouge | 0.98 | 0.70 | 0.90 | 0.95 | 0.88 |
| min_rank | 1.00 | 0.80 | 0.93 | 0.98 | 0.93 |
| probability | 1.00 | 0.97 | 0.99 | 1.00 | 0.99 |

At 8B the result is clean and robust: D_mech is the only metric that is both discriminative and
paraphrase-invariant, its faithful score is ~18× the best lexical metric, and the invariance holds
across *all four* rewordings (not a single-paraphrase artifact). Naive `min_rank` fails invariance
under every paraphrase — the win is the **anchored distance construction**.

---

## 3. 1B — the advantage does not carry over (honest)

**Leaderboard (1B), sorted by faithful score:**

| metric | family | invar_shift ↓ | discrim ↑ | faithful ↑ |
|---|---|---|---|---|
| **min_rank** | mechanistic | 0.385 | 0.662 | **0.407** |
| **D_mech** | mechanistic | 0.491 | 0.444 | 0.226 |
| probability | lexical | 0.721 | 0.800 | 0.223 |
| truth_ratio | lexical | 0.659 | 0.634 | 0.216 |
| exact_mem | lexical | 0.291 | 0.188 | 0.134 |
| mia_mink | lexical | 0.894 | 0.277 | 0.029 |
| extraction_strength | lexical | 0.137 | 0.024 | 0.020 |
| mia_minkpp | lexical | 0.927 | 0.185 | 0.014 |
| mia_loss | lexical | 0.905 | 0.096 | 0.009 |
| mia_zlib | lexical | 0.972 | 0.155 | 0.004 |
| rouge | lexical | 0.126 | 0.003 | 0.003 |

Two things change at 1B, and neither is spun:
1. **D_mech no longer dominates** — its faithful score (0.226) is essentially tied with `probability`
   (0.223), and naive `min_rank` scores *highest* (0.407). At 8B, D_mech led ~18× and min_rank was last.
2. **D_mech's discrimination collapses** from 0.827 (8B) to **0.444** (1B), even though it still passes
   significance (`discriminates = True`).

---

## 4. Why — weak memorization gives a weak "knows" anchor

The mechanism is consistent with everything else we found:

- **1B memorizes weakly.** Perturbed's forget-birthdate rank is ≈4 at 1B vs ≈1 at 8B; infill 0.43 vs
  1.0. So the **"knows" anchor (Perturbed) sits close to the "never-knew" anchor (Standard)** — the two
  reference trajectories are not well separated.
- **D_mech is a distance *between* those anchors.** When the anchors are close, the signed distance
  `d(·,Ø) − d(·,K)` has little room to discriminate — hence discrim 0.44.
- **`probability` looks "more invariant" at 1B (0.72 vs 0.99) for the wrong reason:** there is barely
  any memorized probability to lose in the first place, so paraphrasing can't collapse what was never
  high. Its apparent invariance is an artifact of low signal, not faithfulness.
- **`min_rank`'s swing** (fails at 8B, best at 1B) shows it is *unstable* across scales; D_mech's
  direction is consistent (positive faithful at both) but weaker where memorization is weak.

**Net:** the meta-evaluation is only sharp where the model actually memorized the fact strongly enough
for a clean "knows" reference — that is 8B. At 1B the whole metric field compresses together and no
metric cleanly wins.

---

## 5. Recommended thesis framing (honest, defensible)

- **Lead with 8B as the primary result:** D_mech is the only OpenUnlearning-suite metric that is both
  knowledge-discriminative and paraphrase-invariant, and this holds across four rewordings.
- **Report 1B as a scale-robustness probe that surfaces a real dependency:** the metric's advantage
  requires a strong "knows" anchor, which weak 1B memorization does not provide. State plainly that at
  1B D_mech is on par with `probability` and does not dominate.
- **Frame it as a scope condition, not a failure:** "the mechanistic distance is faithful *when the
  target knowledge is strongly represented*; under weak memorization the anchors collapse and the
  advantage disappears." That is a more credible and more interesting claim than uniform superiority,
  and it ties directly to the memorization-strength story already in RESULTS.md §3/§5.

---

## 6. Artifacts

- 8B: `mech_metaeval_table1_8b.csv`, `mech_metaeval_cells_8b.csv`, `mech_metaeval_fig4_8b.png`,
  `mech_paraphrase_robustness_8b.csv/.png`.
- 1B: `mech_metaeval_table1_1b.csv`, `mech_metaeval_cells_1b.csv`.
- (Outputs are now scale-tagged; an earlier untagged 1B run had transiently overwritten the 8B files —
  the 8B artifact is being regenerated cleanly as `*_8b`.)
