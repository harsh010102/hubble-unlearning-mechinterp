"""Clean rank-by-layer logit lens for all six Hubble models at the current scale (HUBBLE_SCALE).
Uses the FIXED, boundary-safe layerwise_target_rank + median over facts + true-output final layer.
Regenerates logitlens_rank_median_<tag>.png so 1B and 8B are computed identically. RANK_FMT default
full_prefix (birthdate-focused, matched across scales)."""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.evaluation import COLOR_MAP

MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]
FMT = os.environ.get("RANK_FMT", "full_prefix")
DEV = h8.DEVICE


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    ff = h8.facts_for(fs, h8.TARGET_SLOT, FMT); kf = h8.facts_for(ks, h8.TARGET_SLOT, FMT)
    print(f"scale={TAG} fmt={FMT} n_forget={len(ff)} n_retain={len(kf)}")
    rows, curves = [], {}
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}"); continue
        m.eval()
        for split, facts in [("Forget", ff), ("Retain", kf)]:
            cs = []
            for p, g in facts:
                c = h8.layerwise_target_rank(m, tok, p, g, DEV)
                if c is not None and np.all(np.isfinite(c)):
                    cs.append(c)
            med = np.median(np.array(cs), axis=0)
            curves[(name, split)] = med
            for L, v in enumerate(med):
                rows.append(dict(model=name, split=split, layer=L, median_rank=float(v)))
            lo = len(med) // 2
            print(f"  [{name}] {split}: final={med[-1]:.0f}  mid-min={np.min(med[lo:]):.0f}  (n={len(cs)})")
        h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/rank_curves_fixed_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_curves_fixed_{TAG}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, split in zip(axes, ["Forget", "Retain"]):
        for name in MODELS:
            if (name, split) in curves:
                c = curves[(name, split)]
                ax.plot(range(len(c)), c, marker="o", ms=4, lw=2, color=COLOR_MAP.get(name, "#333"),
                        label=name if split == "Forget" else None)
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{split} ({'Unlearn' if split == 'Forget' else 'Keep'})")
        ax.set_xlabel("hidden-state index (0 = embeddings, N = final layer)")
    axes[0].set_ylabel("MEDIAN logit-lens rank of gold token (log; lower = recoverable)")
    axes[0].legend(title="model", fontsize=9)
    fig.suptitle(f"Rank-by-layer LOGIT LENS — forget vs retain ({TAG.upper()}, boundary-safe, fmt={FMT})",
                 fontweight="bold")
    fig.tight_layout()
    p = f"{h8.workdir()}/logitlens_rank_median_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
