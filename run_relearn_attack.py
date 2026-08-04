"""Relearning-attack faithfulness comparison.

Ground truth for "was it really unlearned" = does the forget fact COME BACK under a small
relearning attack (a few finetune steps on the forget set)? We then ask which PRE-attack metric
predicts per-fact recovery — output metrics vs internal (mechanistic) metrics:

  output      : infill accuracy (0 for all forgotten by construction) and final-layer rank
  mechanistic : mid/late-layer MIN rank (is the token recoverable in an intermediate layer even
                if the OUTPUT suppresses it?), and LSR (semantic residual, mid/late layers)

A faithful metric should predict recovery. Expectation: internal metrics predict it, output
metrics don't — i.e. output suppression != erasure, and only the mechanistic read sees it.

Writes: relearn_<TAG>.csv (per model x entity), relearn_metric_auc_<TAG>.csv (comparison table),
        relearn_recovery_<TAG>.png (curves), relearn_auc_<TAG>.png (metric AUC bars).
"""
import os, sys, json, random
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.methods import causal_lm_loss
from hubble8b.data import build_infill_examples

MODELS = ["NPO", "GradDiff", "IDK", "RMU"]
STEPS = [1, 2, 3, 5, 8]          # relearning checkpoints (gradient steps on the forget set)
LR = 1e-5
DEV = h8.DEVICE


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    examples = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    bd = h8.facts_for(fs, h8.TARGET_SLOT)          # birthdate (rank)
    nat = h8.facts_for(fs, h8.LSR_SLOT)            # nationality (LSR)
    texts = list(fs["text"]); names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    frac = h8.config.LSR_LAYER_FRAC
    pert = h8.load_named("Perturbed"); E = h8.build_Eref(pert).to(DEV); h8.to_cpu(pert); del pert; h8.free_cuda()

    @torch.no_grad()
    def infill_correct(m):
        out = []
        for ex in examples:
            sc = [h8.sequence_logprob(m, tok, ex["prefix"], c, DEV) for c in ex["candidates"]]
            out.append(int(np.argmax(sc) == ex["true_index"]))
        return np.array(out)

    rows = []
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"{name}: {type(e).__name__}"); continue
        m.eval()
        # ---- pre-attack metrics ----
        pre = infill_correct(m)
        rank_final, rank_mid = [], []
        for p, g in bd:
            c = h8.layerwise_target_rank(m, tok, p, g, DEV); lo = int(len(c) * frac)
            rank_final.append(c[-1]); rank_mid.append(float(np.min(c[lo:])))
        lsr = []
        for p, g in nat:
            r = h8.layerwise_residual(m, tok, p, g, E, DEV)
            lsr.append(h8.lsr_scalar(r[0]) if r else np.nan)
        # ---- relearning attack ----
        for prm in m.parameters(): prm.requires_grad_(True)
        m.gradient_checkpointing_enable(); m.config.use_cache = False; m.train()
        opt = h8.make_optimizer(m.parameters(), LR); rng = random.Random(h8.config.SEED)
        traj = {0: pre}; queue, step = [], 0
        while step < max(STEPS):
            if not queue:
                queue = texts[:]; rng.shuffle(queue)
            loss = causal_lm_loss(m, tok, [queue.pop()], DEV)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
            if step in STEPS:
                m.config.use_cache = True; m.eval(); traj[step] = infill_correct(m)
                m.train(); m.config.use_cache = False
        for i in range(len(examples)):
            rows.append(dict(model=name, entity=names[i], pre_correct=int(pre[i]),
                             rank_final=float(rank_final[i]), rank_mid=float(rank_mid[i]),
                             lsr=float(lsr[i]), **{f"c{s}": int(traj[s][i]) for s in traj}))
        print(f"[{name}] Unlearn acc  " + "  ".join(f"step{s}={traj[s].mean():.3f}" for s in [0] + STEPS))
        h8.to_cpu(m); del m, opt; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/relearn_{TAG}.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/relearn_{TAG}.csv")

    # ---- recovery curves (model level) ----
    steps_all = [0] + STEPS
    curve = df.groupby("model")[[f"c{s}" for s in steps_all]].mean().reindex(MODELS)
    print("\n=== relearning recovery: Unlearn infill accuracy by step ===")
    print(curve.round(3).to_string())

    # ---- metric comparison: predict per-fact recovery among FORGOTTEN facts ----
    mx = max(STEPS); F = df[df.pre_correct == 0].copy(); F["rec"] = F[f"c{mx}"]

    def auc(metric, y, higher_recovers):
        met = np.asarray(metric, float); y = np.asarray(y)
        ok = ~np.isnan(met); met, y = met[ok], y[ok]
        pos, neg = met[y == 1], met[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan, np.nan
        u = mannwhitneyu(pos, neg, alternative="two-sided"); a = u.statistic / (len(pos) * len(neg))
        return (a if higher_recovers else 1 - a), u.pvalue

    spec = [("rank_mid", False, "mechanistic (internal)"),
            ("lsr", True, "mechanistic (internal)"),
            ("rank_final", False, "output")]
    comp = []
    for met, hi, typ in spec:
        a, p = auc(F[met], F["rec"], hi)
        comp.append(dict(metric=met, type=typ, recovery_AUC=round(a, 3), p=f"{p:.2e}",
                         predicts=bool(a > 0.6 and p < 0.05)))
    comp.append(dict(metric="infill_accuracy", type="output",
                     recovery_AUC="0.5 (0 for all forgotten)", p="-", predicts=False))
    cdf = pd.DataFrame(comp)
    cdf.to_csv(f"{h8.workdir()}/relearn_metric_auc_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/relearn_metric_auc_{TAG}.csv")
    print(f"\n=== metric faithfulness = AUC predicting recovery (forgotten facts, n={len(F)}) ===")
    print(cdf.to_string(index=False))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in MODELS:
        if name in curve.index:
            ax.plot(steps_all, curve.loc[name].values, marker="o", label=name, color=h8.COLOR_MAP[name])
    ax.set_xlabel("relearning steps"); ax.set_ylabel("Unlearn infill accuracy (recovery)")
    ax.set_title(f"Relearning attack — recovery of the forgotten fact ({TAG.upper()})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); p = f"{h8.workdir()}/relearn_recovery_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = [(r["metric"], r["recovery_AUC"], r["type"]) for r in comp if isinstance(r["recovery_AUC"], float)]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=["#2E7D32" if "mechanistic" in b[2] else "#B02A2A" for b in bars])
    ax.axhline(0.5, ls="--", c="k"); ax.set_ylim(0, 1); ax.set_ylabel("AUC: predicts relearning recovery")
    ax.set_title(f"Which metric is faithful? (>0.5 = predicts recovery) ({TAG.upper()})")
    fig.tight_layout(); p = f"{h8.workdir()}/relearn_auc_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
