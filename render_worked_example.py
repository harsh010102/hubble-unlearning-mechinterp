"""Regenerate mech_worked_example: the focused two-metric version of the four-pool box figure
(D_mech vs output probability). Style matches replot_thesis_figs.py / mech_metaeval_fig4 so the
thesis figure family is consistent. Crisp title, no grey captions, no in-image explanation
(the interpretation lives in the LaTeX \\caption). Reads only artifacts/mech_metaeval_cells_8b.csv.
Run: ~/miniconda3/envs/myenv/bin/python render_worked_example.py"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "axes.linewidth": 1.0,
    "savefig.dpi": 230, "savefig.bbox": "tight",
})

A = "artifacts"
df = pd.read_csv(f"{A}/mech_metaeval_cells_8b.csv")
order = ["Full", "Paraphrased", "Retain", "Perturbed"]
labels = ["Full", "Paraphrased", "Retain", "Perturbed"]
MEANPROPS = dict(marker="^", markerfacecolor="#2ca02c", markeredgecolor="#2ca02c", markersize=9)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (col, sub) in zip(axes, [("D_mech", "mechanistic"), ("probability", "lexical")]):
    data = [df[df.cell == c][col].dropna().values for c in order]
    ax.boxplot(data, tick_labels=labels, showmeans=True, meanprops=MEANPROPS)
    ax.axvline(2.5, ls="--", c="#333", lw=1.4)
    ax.set_title(f"$D_\\mathrm{{mech}}$  (mechanistic)" if col == "D_mech"
                 else "output probability  (lexical)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=12)
    # black (non-grey, non-italic) group labels below the axis
    tr = ax.get_xaxis_transform()
    ax.text(1.5, -0.14, "Positive", transform=tr, ha="center", va="top", fontsize=12, color="#111")
    ax.text(3.5, -0.14, "Negative", transform=tr, ha="center", va="top", fontsize=12, color="#111")

axes[0].set_ylabel("metric value")
fig.suptitle("$D_\\mathrm{mech}$ vs output probability across the four pools (8B)",
             fontweight="bold", fontsize=18)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{A}/mech_worked_example.png", facecolor="white")
fig.savefig(f"{A}/mech_worked_example.pdf", facecolor="white"); plt.close(fig)
print("wrote", f"{A}/mech_worked_example.png + .pdf")
