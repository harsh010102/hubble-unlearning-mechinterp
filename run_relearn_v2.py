"""Relearning-attack metric faithfulness v2 (8B) — SPID-structured, with (a) per-fact signal,
(b) recovery-SPEED ground truth, (c) power (multi-seed + pooling + bootstrap CIs).

Ground truth = recovery_step: relearn each unlearned model on the forget set (fine steps, multiple
seeds); per fact, the first step its birthdate infill flips correct (never -> NEVER), averaged over
seeds. SPID Table-1 analog: per metric, does it predict recovery? (Wilcoxon fast-vs-slow pools, AUC
with bootstrap 95% CI, Spearman with recovery_step). Figure-4 analog: box plots per metric.

Writes new files only: relearn_v2_perfact.csv, relearn_v2_metric_table.csv, relearn_v2_box.png,
relearn_v2_recovery.png. NOT added to RESULTS.md.
"""
import os, sys, json, math, random
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.methods import causal_lm_loss
from hubble8b.data import build_infill_examples

MODELS = ["NPO", "GradDiff", "IDK", "RMU"]
STEPS = [1, 2, 3, 4, 6, 8, 12]
SEEDS = [0, 1]
NEVER = max(STEPS) + 4        # recovery_step for facts that never recover
LR = 1e-5
DEV = h8.DEVICE


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    examples = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    bd = h8.facts_for(fs, h8.TARGET_SLOT); nat = h8.facts_for(fs, h8.LSR_SLOT)
    names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    texts = list(fs["text"]); frac = h8.config.LSR_LAYER_FRAC
    pert = h8.load_named("Perturbed"); E = h8.build_Eref(pert).to(DEV); h8.to_cpu(pert); del pert; h8.free_cuda()

    @torch.no_grad()
    def correct(m):
        out = []
        for ex in examples:
            sc = [h8.sequence_logprob(m, tok, ex["prefix"], c, DEV) for c in ex["candidates"]]
            out.append(int(np.argmax(sc) == ex["true_index"]))
        return np.array(out)

    rows = []
    curves = {}
    for name in MODELS:
        # ---- pre-attack per-fact metrics (seed independent) ----
        m = h8.load_named(name); m.eval()
        pre = correct(m)
        rmid, rfin = [], []
        for p, g in bd:
            c = h8.layerwise_target_rank(m, tok, p, g, DEV); lo = int(len(c) * frac)
            rfin.append(c[-1]); rmid.append(float(np.min(c[lo:])))
        lsr = [(h8.lsr_scalar(r[0]) if (r := h8.layerwise_residual(m, tok, p, g, E, DEV)) else np.nan) for p, g in nat]
        prob = []
        for ex in examples:
            gold = ex["candidates"][ex["true_index"]]
            lp = h8.sequence_logprob(m, tok, ex["prefix"], gold, DEV)
            n = max(1, len(tok.encode(" " + gold, add_special_tokens=False)))
            prob.append(math.exp(lp / n))
        h8.to_cpu(m); del m; h8.free_cuda()

        # ---- relearn per seed, record per-fact recovery_step ----
        rec = np.full((len(SEEDS), len(examples)), NEVER, float)
        seed_traj = []
        for si, seed in enumerate(SEEDS):
            m = h8.load_named(name)
            for prm in m.parameters(): prm.requires_grad_(True)
            m.gradient_checkpointing_enable(); m.config.use_cache = False; m.train()
            opt = h8.make_optimizer(m.parameters(), LR); rng = random.Random(h8.config.SEED + seed); q = []
            traj = {0: pre.copy()}; step = 0
            while step < max(STEPS):
                if not q: q = texts[:]; rng.shuffle(q)
                loss = causal_lm_loss(m, tok, [q.pop()], DEV)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
                if step in STEPS:
                    m.config.use_cache = True; m.eval(); traj[step] = correct(m); m.train(); m.config.use_cache = False
            seed_traj.append(traj)
            for j in range(len(examples)):
                for s in [0] + STEPS:
                    if traj[s][j] == 1:
                        rec[si, j] = s; break
            h8.to_cpu(m); del m, opt; h8.free_cuda()
        recovery = rec.mean(0)                       # avg recovery step over seeds
        curves[name] = {s: np.mean([t[s] for t in seed_traj], 0).mean() for s in [0] + STEPS}
        for j in range(len(examples)):
            rows.append(dict(model=name, entity=names[j], pre_correct=int(pre[j]),
                             rank_mid=float(rmid[j]), rank_final=float(rfin[j]), lsr=float(lsr[j]),
                             prob=float(prob[j]), recovery_step=float(recovery[j])))
        print(f"[{name}] pre-acc {pre.mean():.3f}  recovery_step(mean over forgotten)="
              f"{recovery[pre == 0].mean():.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/relearn_v2_perfact.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/relearn_v2_perfact.csv")

    # ---- SPID-style metric comparison on FORGOTTEN facts ----
    F = df[df.pre_correct == 0].copy()
    med = np.median(F.recovery_step); F["fast"] = (F.recovery_step <= med).astype(int)

    def auc_dir(metric, y, hi):
        met = np.asarray(metric, float); ok = ~np.isnan(met); met, y = met[ok], np.asarray(y)[ok]
        pos, neg = met[y == 1], met[y == 0]
        if len(pos) == 0 or len(neg) == 0: return np.nan
        a = mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))
        return a if hi else 1 - a

    def boot(metric, y, hi, B=2000):
        met = np.asarray(metric, float); y = np.asarray(y); n = len(y)
        rng = np.random.RandomState(0); vals = []
        for _ in range(B):
            idx = rng.randint(0, n, n); a = auc_dir(met[idx], y[idx], hi)
            if not np.isnan(a): vals.append(a)
        return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)

    spec = [("rank_mid", False, "internal"), ("lsr", True, "internal"),
            ("prob", True, "output"), ("rank_final", False, "output")]
    tab = []
    for met, hi, typ in spec:
        a = auc_dir(F[met], F["fast"], hi); lo, up = boot(F[met].values, F["fast"].values, hi)
        rho, pv = spearmanr(F[met], F.recovery_step, nan_policy="omit")
        tab.append(dict(metric=met, type=typ, recovery_AUC=round(a, 3),
                        AUC_CI=f"[{lo:.2f},{up:.2f}]", spearman_rho=round(rho, 3), spearman_p=f"{pv:.2e}",
                        faithful=bool(lo > 0.5 or up < 0.5)))
    tdf = pd.DataFrame(tab)
    tdf.to_csv(f"{h8.workdir()}/relearn_v2_metric_table.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/relearn_v2_metric_table.csv")
    print(f"\n=== SPID-style metric faithfulness (forgotten facts n={len(F)}, median recovery_step={med:.1f}) ===")
    print(tdf.to_string(index=False))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in MODELS:
        if name in curves:
            xs = [0] + STEPS; ax.plot(xs, [curves[name][s] for s in xs], marker="o", label=name, color=h8.COLOR_MAP[name])
    ax.set_xlabel("relearning steps"); ax.set_ylabel("Unlearn infill accuracy"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Relearning recovery ({TAG.upper()}, seed-avg)")
    fig.tight_layout(); p = f"{h8.workdir()}/relearn_v2_recovery.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)

    mets = [s[0] for s in spec]
    fig, axes = plt.subplots(1, len(mets), figsize=(4 * len(mets), 4))
    for ax, met in zip(axes, mets):
        ax.boxplot([F[F.fast == 1][met].dropna(), F[F.fast == 0][met].dropna()],
                   tick_labels=["fast-recover", "slow/never"], showmeans=True)
        ax.set_title(met); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Does the metric separate fast- vs slow-recovering forgotten facts? ({TAG.upper()})", fontweight="bold")
    fig.tight_layout(); p = f"{h8.workdir()}/relearn_v2_box.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
