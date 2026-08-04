"""SPID-structured meta-evaluation (arXiv 10619 sec 5.2 / OpenUnlearning 2506.12618 sec 4): does a
logit-lens MECHANISTIC DISTANCE metric satisfy the two criteria for a faithful unlearning metric better
than the lexical/output metrics? On the Hubble-8B suite.

Positive/Negative pools (SPID sec 5.2.1), 4 cells, each = a model on the birthdate fact per entity:
  Positive P (knows C_f):   FULL         = Perturbed, canonical prompt, correct birthdate
                            PARAPHRASED  = Perturbed, paraphrased prompt, correct birthdate
  Negative N (lacks C_f):   RETAIN       = Standard  (never-knew), correct birthdate
                            PERTURBED    = Perturbed, a WRONG birthdate target (holds no such fact)

Two criteria (SPID):
  Semantic Invariance     : within P, FULL vs PARAPHRASED should be EQUAL -> Wilcoxon (high p = no shift)
                            + TOST (low p = statistically equivalent). Lexical metrics tend to fail this.
  Knowledge Discrimination: P vs N should DIFFER                          -> Wilcoxon (low p = separates).

Metrics: mechanistic D_mech (dual-anchor logit-lens distance) and min_rank; OpenUnlearning lexical/output
probability, rouge, exact_mem, extraction_strength, truth_ratio, and MIA (loss/mink/minkpp/zlib). The
paraphrase axis is the PARAPHRASED cell (so 'paraphrased prob/rouge' = prob/rouge there); jailbreak_rouge
is omitted (no SPID analogue).

Outputs (new files only): mech_metaeval_cells.csv (per-instance), mech_metaeval_table1.csv (SPID Table-1
analog), mech_metaeval_fig4.png (SPID Figure-4 box plots). Nothing written to RESULTS.md.
"""
import os, sys, json, math, zlib
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import build_infill_examples

DEV = h8.DEVICE
BAND_FRAC = 0.5
MINK = 0.2
PARA = ("was born on", "has a recorded birth date of")     # meaning-preserving paraphrase of the trigger

# metric -> (higher_is_present, family)
SPEC = {
    "D_mech": (True, "mechanistic"), "min_rank": (False, "mechanistic"),
    "probability": (True, "openunlearning"), "rouge": (True, "openunlearning"),
    "exact_mem": (True, "openunlearning"), "extraction_strength": (True, "openunlearning"),
    "truth_ratio": (False, "openunlearning"), "mia_loss": (False, "openunlearning"),
    "mia_mink": (True, "openunlearning"), "mia_minkpp": (True, "openunlearning"),
    "mia_zlib": (False, "openunlearning"),
}
SCALARS = [m for m in SPEC if m != "D_mech"]


def rouge_l(pred, gold):
    a, b = pred.split(), gold.split()
    if not a or not b: return 0.0
    dp = np.zeros((len(a) + 1, len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    lcs = dp[-1, -1]
    if lcs == 0: return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return float(2 * p * r / (p + r))


@torch.no_grad()
def greedy_ids(model, tok, prefix, max_new):
    ids = tok(prefix.rstrip(), return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:]


@torch.no_grad()
def probe(model, tok, prefix, gold, band):
    """One forward on (prefix+gold): rank trajectory, gold prob, exact-match, MIA over full seq."""
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
    if full.shape[1] <= pre.shape[1]: return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states; true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head; last = len(hs) - 1
    gp = list(range(pre.shape[1] - 1, full.shape[1] - 1))
    traj = []
    for li, h in enumerate(hs):
        ranks = []
        for p in gp:
            logits = true_logits[p] if li == last else head(norm(h[:, p, :])).float()[0]
            ranks.append(int((logits > logits[full[0, p + 1]]).sum().item()) + 1)
        traj.append(float(np.mean(ranks)))
    gold_lp = [float(torch.log_softmax(true_logits[p], -1)[full[0, p + 1]]) for p in gp]
    em = float(np.mean([int(int(true_logits[p].argmax()) == int(full[0, p + 1])) for p in gp]))
    all_lp, z = [], []
    for p in range(full.shape[1] - 1):
        lp = torch.log_softmax(true_logits[p], -1); tid = full[0, p + 1]
        all_lp.append(float(lp[tid])); z.append(float((lp[tid] - lp.mean()) / (lp.std() + 1e-6)))
    all_lp = np.array(all_lp); z = np.array(z); k = max(1, int(len(all_lp) * MINK))
    zlen = max(1, len(zlib.compress((prefix + " " + gold).encode())))
    return dict(traj=traj, prob=math.exp(np.mean(gold_lp)), exact_mem=em,
                mia_loss=-float(all_lp.mean()), mia_mink=float(np.sort(all_lp)[:k].mean()),
                mia_minkpp=float(np.sort(z)[:k].mean()), mia_zlib=float(-all_lp.sum()) / zlen,
                min_rank=float(np.min(np.asarray(traj)[band])))


def cell_metrics(model, tok, prefixes, golds, cand_lists, band):
    """All scalar metrics + per-fact trajectories for one pool cell (a model over N facts)."""
    N = len(prefixes); acc = {m: np.full(N, np.nan) for m in SCALARS}; trajs = [None] * N
    for i in range(N):
        p, g, cands = prefixes[i], golds[i], cand_lists[i]
        pr = probe(model, tok, p, g, band); trajs[i] = pr["traj"]
        for m in ["exact_mem", "mia_loss", "mia_mink", "mia_minkpp", "mia_zlib", "min_rank"]:
            acc[m][i] = pr[m]
        acc["probability"][i] = pr["prob"]
        ntok = max(1, len(tok.encode(" " + g, add_special_tokens=False)))
        gi = greedy_ids(model, tok, p, ntok + 2)
        acc["rouge"][i] = rouge_l(tok.decode(gi, skip_special_tokens=True).strip(), g)
        gold_ids = tok(p.rstrip() + " " + g, return_tensors="pt").input_ids[0][tok(p.rstrip(), return_tensors="pt").input_ids.shape[1]:]
        mlen = 0
        for j in range(min(len(gi), len(gold_ids))):
            if int(gi[j]) == int(gold_ids[j]): mlen += 1
            else: break
        acc["extraction_strength"][i] = mlen / max(1, len(gold_ids))
        gl = h8.sequence_logprob(model, tok, p, g, DEV)
        distr = [h8.sequence_logprob(model, tok, p, c, DEV) for c in cands if c != g]
        acc["truth_ratio"][i] = math.exp(np.mean(distr) - gl) if distr else np.nan
    return acc, trajs


def cliff(x, y):
    """Cliff's delta in [-1,1]; 0 = identical distributions (scale-free effect size)."""
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0: return np.nan
    return float(np.sign(x[:, None] - y[None, :]).sum() / (len(x) * len(y)))


def tost(a, b, eps):
    """paired TOST for equivalence within +-eps; returns max of two one-sided p-values (low = equivalent)."""
    d = np.asarray(a, float) - np.asarray(b, float); ok = ~np.isnan(d); d = d[ok]; n = len(d)
    if n < 3: return 1.0
    m, s = d.mean(), d.std(ddof=1)
    if s == 0: return 0.0 if abs(m) < eps else 1.0
    se = s / math.sqrt(n)
    return float(max(1 - tdist.cdf((m + eps) / se, n - 1), tdist.cdf((m - eps) / se, n - 1)))


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    examples = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    N = len(bd)
    full_prefix = [p for p, _ in bd]; gold = [g for _, g in bd]
    para_prefix = [p.replace(*PARA) if PARA[0] in p else p for p in full_prefix]
    cands = [ex["candidates"] for ex in examples]
    wrong_gold = [next(c for c in examples[i]["candidates"] if c != gold[i]) for i in range(N)]
    print(f"paraphrase applied to {sum(PARA[0] in p for p in full_prefix)}/{N} prompts")

    pert = h8.load_named("Perturbed")
    band = np.arange(int(len(probe(pert, tok, full_prefix[0], gold[0], np.arange(1))["traj"]) * BAND_FRAC),
                     len(probe(pert, tok, full_prefix[0], gold[0], np.arange(1))["traj"]))
    print(f"band = layers {band[0]}..{band[-1]}")
    print("[cell] FULL"); full_m, full_tr = cell_metrics(pert, tok, full_prefix, gold, cands, band)
    print("[cell] PARAPHRASED"); para_m, para_tr = cell_metrics(pert, tok, para_prefix, gold, cands, band)
    print("[cell] PERTURBED(wrong)"); pw_m, pw_tr = cell_metrics(pert, tok, full_prefix, wrong_gold, cands, band)
    h8.to_cpu(pert); del pert; h8.free_cuda()
    print("[cell] RETAIN"); std = h8.load_named("Standard"); std.eval()
    ret_m, ret_tr = cell_metrics(std, tok, full_prefix, gold, cands, band)
    h8.to_cpu(std); del std; h8.free_cuda()

    # D_mech: dual anchors = FULL(Perturbed) & RETAIN(Standard) trajectories, LOO to avoid self-circularity
    Ftr = np.array(full_tr); Rtr = np.array(ret_tr)
    def dmech_of(trs):
        out = np.full(N, np.nan)
        for i in range(N):
            aK = np.median(np.delete(Ftr, i, 0), 0); aO = np.median(np.delete(Rtr, i, 0), 0)
            lr = np.log(np.asarray(trs[i], float))
            out[i] = float(np.mean(np.abs(lr[band] - np.log(aO)[band])) - np.mean(np.abs(lr[band] - np.log(aK)[band])))
        return out
    full_m["D_mech"] = dmech_of(full_tr); para_m["D_mech"] = dmech_of(para_tr)
    ret_m["D_mech"] = dmech_of(ret_tr); pw_m["D_mech"] = dmech_of(pw_tr)

    CELLS = {"Full": full_m, "Paraphrased": para_m, "Retain": ret_m, "Perturbed": pw_m}
    POOL = {"Full": "P", "Paraphrased": "P", "Retain": "N", "Perturbed": "N"}

    # ---- per-instance long table ----
    rows = []
    for cell, mm in CELLS.items():
        for i in range(N):
            r = dict(entity=names[i], cell=cell, pool=POOL[cell])
            for m in SPEC: r[m] = float(mm[m][i])
            rows.append(r)
    long = pd.DataFrame(rows)
    long.to_csv(f"{h8.workdir()}/mech_metaeval_cells.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/mech_metaeval_cells.csv")

    # ---- SPID Table 1: within-P invariance (Wilcoxon high, TOST low) + P-vs-N discrimination (Wilcoxon low) ----
    tab = []
    for m, (hi, fam) in SPEC.items():
        F = full_m[m]; Pa = para_m[m]; ok = ~(np.isnan(F) | np.isnan(Pa))
        try: wp = wilcoxon(F[ok], Pa[ok]).pvalue
        except Exception: wp = np.nan
        allv = np.concatenate([full_m[m], para_m[m], ret_m[m], pw_m[m]]); sd = np.nanstd(allv)
        tp = tost(F, Pa, 0.3 * sd)
        Ppool = np.concatenate([full_m[m], para_m[m]]); Npool = np.concatenate([ret_m[m], pw_m[m]])
        o = ~np.isnan(Ppool); o2 = ~np.isnan(Npool)
        wpn = mannwhitneyu(Ppool[o], Npool[o2], alternative="two-sided").pvalue if o.sum() and o2.sum() else np.nan
        # scale-free effect sizes (the Wilcoxon within-P p-value saturates at n=44 under a strong paraphrase;
        # invariance is better read as the EFFECT SIZE of the Full->Paraphrased shift).
        invar_shift = abs(cliff(full_m[m], para_m[m]))          # 0 = paraphrase-invariant (LOWER better)
        discrim = abs(cliff(Ppool, Npool))                      # 1 = perfect P-vs-N separation (HIGHER better)
        tab.append(dict(metric=m, family=fam,
                        within_P_wilcoxon=wp, within_P_TOST=tp, between_PN_wilcoxon=wpn,
                        invar_shift=round(invar_shift, 3), discrim=round(discrim, 3),
                        faithful_score=round(discrim * (1 - invar_shift), 3),
                        discriminates=bool(wpn < 0.05)))
    T = pd.DataFrame(tab).sort_values("faithful_score", ascending=False)
    T.to_csv(f"{h8.workdir()}/mech_metaeval_table1.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/mech_metaeval_table1.csv")
    fmt = T.copy()
    for c in ["within_P_wilcoxon", "within_P_TOST", "between_PN_wilcoxon"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.2e}" if pd.notna(v) else "nan")
    print("\n=== SPID Table 1: Semantic Invariance (within P: Wilcoxon high + TOST low) | "
          "Knowledge Discrimination (P vs N: Wilcoxon low) ===")
    print(fmt.to_string(index=False))

    # ---- SPID Figure 4: per-metric box plots across the 4 cells (dashed line splits P | N) ----
    order = ["Full", "Paraphrased", "Retain", "Perturbed"]; mets = list(SPEC)
    ncol = 4; nrow = int(np.ceil(len(mets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, m in zip(axes.flat, mets):
        ax.boxplot([CELLS[c][m][~np.isnan(CELLS[c][m])] for c in order], tick_labels=order, showmeans=True)
        ax.axvline(2.5, ls="--", c="r", lw=1); ax.set_title(f"{m}\n({SPEC[m][1]})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7); ax.grid(axis="y", alpha=0.3)
    for ax in axes.flat[len(mets):]: ax.axis("off")
    fig.suptitle(f"SPID Figure-4 analog ({TAG.upper()}): metric distributions across pools "
                 "(dashed = Positive P | Negative N)\nfaithful metric: stable Full->Paraphrased, gap P vs N",
                 fontweight="bold")
    fig.tight_layout(); p = f"{h8.workdir()}/mech_metaeval_fig4.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
