"""Merge UNDIAL into the rank-by-layer logit-lens plot (CPU, no GPU).
Uses the 6 existing curves from rank_curves_fixed_<tag>.csv + UNDIAL from rank_curve_undial_<tag>.csv,
annotates UNDIAL's D_mech verdict (suppress vs erase). Output: logitlens_rank_median_<tag>_undial.png"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "8b"
fixed = pd.read_csv(f"artifacts/rank_curves_fixed_{TAG}.csv")
und = pd.read_csv(f"artifacts/rank_curve_undial_{TAG}.csv")
base = fixed[fixed.model.isin(["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"])]
allc = pd.concat([base, und[und.model == "UNDIAL"]], ignore_index=True)

ORDER = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU", "UNDIAL"]
COL = {"Standard": "#1f77b4", "Perturbed": "#9467bd", "NPO": "#d62728", "GradDiff": "#2ca02c",
       "IDK": "#111111", "RMU": "#ff7f0e", "UNDIAL": "#00a2b3"}

ann = ""
try:
    dm = pd.read_csv(f"artifacts/undial_dmech_{TAG}.csv").iloc[0]
    dmed = float(dm["undial_dmech_median"])
    verdict = "SUPPRESSED (sits on the knows side)" if dmed > 0 else "ERASED (sits on the never-knew side)"
    ann = (f"UNDIAL  D_mech = {dmed:+.2f}  ->  {verdict}    "
           f"[forget final rank: Perturbed {dm['perturbed_forget_final']:.0f} | "
           f"UNDIAL {dm['undial_forget_final']:.0f} | Standard {dm['standard_forget_final']:.0f}]")
except Exception as e:
    print("no D_mech summary:", e)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, split in zip(axes, ["Forget", "Retain"]):
    for m in ORDER:
        c = allc[(allc.model == m) & (allc.split == split)].sort_values("layer")
        if c.empty:
            continue
        ax.plot(c.layer, c.median_rank, marker="o", ms=4, lw=3.4 if m == "UNDIAL" else 1.9,
                color=COL[m], label=(m if split == "Forget" else None),
                zorder=6 if m == "UNDIAL" else 3, alpha=1.0 if m == "UNDIAL" else 0.85)
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_title(f"{split} ({'Unlearn' if split == 'Forget' else 'Keep'})")
    ax.set_xlabel("hidden-state index (0 = embeddings, N = final layer)")
axes[0].set_ylabel("median logit-lens rank of gold token (log; lower = recoverable)")
axes[0].legend(title="model", fontsize=9)
fig.suptitle(f"Rank-by-layer logit lens — UNDIAL vs the four methods and the two anchors ({TAG.upper()})",
             fontweight="bold")
if ann:
    fig.text(0.5, -0.02, ann, ha="center", fontsize=11.5, color="#003359", fontweight="bold")
fig.tight_layout()
out = f"artifacts/logitlens_rank_median_{TAG}_undial.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote", out)
