"""Replot the chart-style result figures with a CONSISTENT, high-legibility style (large axis labels,
unified model names/colours). Reads only local CSVs in artifacts/; writes PNGs back to artifacts/ with the
same names (so the thesis/deck references keep working). Table-style figures (lens tables, showcase) are
left as-is. Run on the login node: ~/miniconda3/envs/myenv/bin/python replot_thesis_figs.py"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

A = "artifacts"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "legend.fontsize": 13, "legend.title_fontsize": 13.5,
    "axes.linewidth": 1.0, "lines.linewidth": 2.4, "lines.markersize": 6,
    "savefig.dpi": 170, "savefig.bbox": "tight", "figure.autolayout": False,
})
COLORS = {"Standard": "#2E86AB", "Perturbed": "#A23B72", "NPO": "#C73E1D",
          "GradDiff": "#6A994E", "IDK": "#3B1F2B", "RMU": "#F18F01"}
ORDER = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]
RANKLAB = "median gold-token rank  (log scale, lower = recoverable)"
XLAB = "hidden-state layer  (0 = embeddings, N = final)"


def save(fig, name):
    p = f"{A}/{name}.pdf"; fig.savefig(p); plt.close(fig); print("wrote", p)


def _rank_panel(ax, df, split, ycol, title, models=ORDER, dotted=None):
    for m in models:
        d = df[(df.model == m) & (df.split == split)].sort_values("layer")
        if d.empty: continue
        ax.plot(d.layer, d[ycol], "-o", color=COLORS.get(m, "#333"), label=m, ms=5)
        if dotted is not None:
            ax.plot(d.layer, d[dotted], ":", color=COLORS.get(m, "#333"), lw=1.4, alpha=0.55)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_title(title); ax.set_xlabel(XLAB)


# ---------- 1. logit-lens rank-by-layer, 8B (merged tuned_vs_logit logit rows) ----------
def fig_logit_8b():
    parts = []
    for f in ["tuned_vs_logit_8b_stdperturbnpoidk", "tuned_vs_logit_8b_graddiff", "tuned_vs_logit_8b_rmu"]:
        fp = f"{A}/{f}.csv"
        if os.path.exists(fp): parts.append(pd.read_csv(fp))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.lens == "logit"].rename(columns={"median_rank": "r"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _rank_panel(axes[0], df, "Forget", "r", "Forget (Unlearn)")
    _rank_panel(axes[1], df, "Retain", "r", "Retain (Keep)")
    axes[0].set_ylabel(RANKLAB); axes[0].legend(title="model", ncol=2, fontsize=12)
    fig.suptitle("Rank-by-layer logit lens (8B)", fontweight="bold", fontsize=18)
    fig.tight_layout(); save(fig, "logitlens_rank_median_8b")


# ---------- 2. logit-lens rank-by-layer, 1B ----------
def fig_logit_1b():
    df = pd.read_csv(f"{A}/rank_curves_fixed_1b.csv").rename(columns={"median_rank": "r"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _rank_panel(axes[0], df, "Forget", "r", "Forget (Unlearn)")
    _rank_panel(axes[1], df, "Retain", "r", "Retain (Keep)")
    axes[0].set_ylabel(RANKLAB); axes[0].legend(title="model", ncol=2, fontsize=12)
    fig.suptitle("Rank-by-layer logit lens (1B)", fontweight="bold", fontsize=18)
    fig.tight_layout(); save(fig, "logitlens_rank_median_1b")


# ---------- 3. three-lens comparison (8B) ----------
def fig_lens_compare():
    df = pd.read_csv(f"{A}/rank_curves_jlens_affine_8b.csv")
    df = df[df.split == "Forget"]
    mods = [m for m in ORDER if m in df.model.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, mods):
        d = df[df.model == m].sort_values("layer"); c = COLORS.get(m, "#333")
        ax.plot(d.layer, d.logit_median_rank, ":", lw=1.6, color=c, alpha=0.8, label="logit")
        ax.plot(d.layer, d.tuned_median_rank, "--", lw=1.8, color=c, alpha=0.9, label="tuned")
        ax.plot(d.layer, d.jaffine_median_rank, "-o", lw=2.4, ms=4, color=c, label="Jacobian")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.set_title(m)
        ax.legend(fontsize=11)
    for ax in axes[-1]: ax.set_xlabel(XLAB)
    for ax in axes[:, 0]: ax.set_ylabel(RANKLAB, fontsize=13)
    fig.suptitle("Three lenses, rank by layer (8B)", fontweight="bold", fontsize=18)
    fig.tight_layout(); save(fig, "lens_compare_8b")


# ---------- 4. Jacobian lens rank-by-layer (8B) ----------
def fig_jlens():
    df = pd.read_csv(f"{A}/rank_curves_jlens_affine_8b.csv")
    df = df[df.split == "Forget"]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for m in [x for x in ORDER if x in df.model.unique()]:
        d = df[df.model == m].sort_values("layer"); c = COLORS.get(m, "#333")
        ax.plot(d.layer, d.jaffine_median_rank, "-o", lw=2.6, ms=5, color=c, label=m)
        ax.plot(d.layer, d.logit_median_rank, ":", lw=1.3, color=c, alpha=0.5)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel(XLAB); ax.set_ylabel(RANKLAB)
    ax.legend(title="Jacobian lens (solid); logit (dotted)", ncol=2, fontsize=12)
    ax.set_title("Jacobian lens, rank by layer (8B)", fontweight="bold")
    fig.tight_layout(); save(fig, "jlens_affine_rank_8b")


# ---------- 5. Jacobian knowledge-vs-prior control (8B) ----------
def fig_jctrl():
    df = pd.read_csv(f"{A}/rank_curves_jlens_affine_8b.csv"); df = df[df.split == "Forget"]
    pick = [m for m in ["Standard", "Perturbed", "NPO"] if m in df.model.unique()]
    fig, axes = plt.subplots(1, len(pick), figsize=(6 * len(pick), 5.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), pick):
        d = df[df.model == m].sort_values("layer")
        ax.plot(d.layer, d.jaffine_median_rank, "-o", lw=2.6, ms=5, color="#1a7f37", label="real (transport h)")
        ax.plot(d.layer, d.null_median_rank, "--", lw=2.0, color="#999", label="null (mean-h prior)")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.set_title(m); ax.set_xlabel(XLAB)
    np.atleast_1d(axes)[0].set_ylabel(RANKLAB, fontsize=13); np.atleast_1d(axes)[0].legend(fontsize=12)
    fig.suptitle("Jacobian lens control (8B)", fontweight="bold", fontsize=17)
    fig.tight_layout(); save(fig, "jlens_affine_control_8b")


# ---------- 6. tuned vs logit (8B) ----------
def fig_tuned_vs_logit():
    parts = []
    for f in ["tuned_vs_logit_8b_stdperturbnpoidk", "tuned_vs_logit_8b_graddiff", "tuned_vs_logit_8b_rmu"]:
        fp = f"{A}/{f}.csv"
        if os.path.exists(fp): parts.append(pd.read_csv(fp))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.split == "Forget"]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for m in [x for x in ORDER if x in df.model.unique()]:
        c = COLORS.get(m, "#333")
        dl = df[(df.model == m) & (df.lens == "logit")].sort_values("layer")
        dt = df[(df.model == m) & (df.lens == "tuned")].sort_values("layer")
        ax.plot(dl.layer, dl.median_rank, ":", lw=1.5, color=c, alpha=0.7)
        ax.plot(dt.layer, dt.median_rank, "-", lw=2.4, color=c, label=m)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel(XLAB); ax.set_ylabel(RANKLAB)
    ax.legend(title="tuned (solid); logit (dotted)", ncol=2, fontsize=12)
    ax.set_title("Tuned vs logit lens, rank by layer (8B)", fontweight="bold")
    fig.tight_layout(); save(fig, "tuned_vs_logit_8b")


# ---------- 7. faithfulness box plots across the four pools ----------
def fig_metaeval_box():
    df = pd.read_csv(f"{A}/mech_metaeval_cells_8b.csv")
    mets = ["D_mech", "probability", "rouge", "extraction_strength", "truth_ratio", "min_rank"]
    order = ["Full", "Paraphrased", "Retain", "Perturbed"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.flat, mets):
        data = [df[df.cell == c][m].dropna().values for c in order]
        ax.boxplot(data, tick_labels=["Full", "Para.", "Retain", "Pert."], showmeans=True)
        ax.axvline(2.5, ls="--", c="r", lw=1.4)
        ax.set_title(m, fontsize=16); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=12)
    fig.suptitle("Metric distributions across the four pools (8B)", fontweight="bold", fontsize=18)
    fig.tight_layout(); save(fig, "mech_metaeval_fig4")


# ---------- 8. invariance-discrimination scatter ----------
def fig_scatter():
    t = pd.read_csv(f"{A}/mech_metaeval_table1_8b.csv")
    fig, ax = plt.subplots(figsize=(8.6, 7))
    for _, r in t.iterrows():
        mech = r.family == "mechanistic"
        ax.scatter(r.invar_shift, r.discrim, s=150 if mech else 90,
                   color="#C73E1D" if mech else "#888", edgecolor="k", zorder=3)
        ax.annotate(r.metric, (r.invar_shift, r.discrim), fontsize=11,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("invariance shift  (0 = paraphrase-invariant, lower is better)")
    ax.set_ylabel("discrimination  (1 = perfect, higher is better)")
    ax.axvspan(-0.02, 0.5, color="#dff0d8", alpha=0.5, zorder=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("Discrimination vs invariance shift (8B)", fontweight="bold", fontsize=15)
    fig.tight_layout(); save(fig, "mech_invar_discrim_scatter")


# ---------- 9. faithfulness bar ----------
def fig_faithful_bar():
    t = pd.read_csv(f"{A}/mech_metaeval_table1_8b.csv").sort_values("faithful_score")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    cols = ["#C73E1D" if f == "mechanistic" else "#999" for f in t.family]
    ax.barh(t.metric, t.faithful_score, color=cols, edgecolor="k")
    ax.set_xlabel("faithfulness score  =  discrimination $\\times$ (1 $-$ invariance shift)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Faithfulness score per metric (8B)", fontweight="bold", fontsize=15)
    fig.tight_layout(); save(fig, "mech_faithful_bar")


# ---------- 10. infill accuracy bars (8B) ----------
def fig_infill():
    parts = [pd.read_csv(f"{A}/infill_{m}.csv") for m in ORDER if os.path.exists(f"{A}/infill_{m}.csv")]
    df = pd.concat(parts, ignore_index=True)
    fmts = [f for f in ["name_only", "intro_prefix", "full_prefix", "full_prefix_suffix"] if f in df.template_format.unique()]
    fig, axes = plt.subplots(1, len(fmts), figsize=(4.6 * len(fmts), 5.6), sharey=True)
    x = np.arange(len(ORDER)); w = 0.38
    for ax, fmt in zip(np.atleast_1d(axes), fmts):
        u = [df[(df.model == m) & (df.split == "Unlearn") & (df.template_format == fmt)].accuracy.mean() for m in ORDER]
        k = [df[(df.model == m) & (df.split == "Keep") & (df.template_format == fmt)].accuracy.mean() for m in ORDER]
        ax.bar(x - w / 2, u, w, label="Unlearn", color="#C73E1D")
        ax.bar(x + w / 2, k, w, label="Keep", color="#6A994E")
        ax.axhline(0.10, ls="--", c="k", lw=1, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=40, ha="right", fontsize=12)
        ax.set_title(fmt.replace("_", " ")); ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("infill accuracy"); np.atleast_1d(axes)[0].legend(fontsize=12)
    fig.suptitle("Infill accuracy by method and prompt format (8B)", fontweight="bold", fontsize=17)
    fig.tight_layout(); save(fig, "infill_bars_8b")


# ---------- 11. quantization robustness (8B) ----------
def fig_quant():
    df = pd.read_csv(f"{A}/quant_study_8b.csv")
    precs = ["bf16", "int8", "nf4"]; x = np.arange(len(ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, p in enumerate(precs):
        vals = [df[(df.model == m) & (df.precision == p)].Unlearn.mean() for m in ORDER]
        ax.bar(x + (i - 1) * w, vals, w, label=p)
    ax.axhline(0.10, ls="--", c="k", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=13)
    ax.set_ylabel("forget (Unlearn) infill accuracy"); ax.grid(axis="y", alpha=0.3)
    ax.legend(title="precision", fontsize=12)
    ax.set_title("Quantization robustness (8B)", fontweight="bold", fontsize=15)
    fig.tight_layout(); save(fig, "quant_study_8b")


# ---------- 12. relearning recovery (8B) ----------
def fig_relearn():
    df = pd.read_csv(f"{A}/relearn_8b.csv")
    steps = [c for c in ["c0", "c1", "c2", "c3", "c5", "c8"] if c in df.columns]
    xs = [int(s[1:]) for s in steps]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for m in [x for x in ORDER if x in df.model.unique()]:
        d = df[df.model == m]
        ax.plot(xs, [d[s].mean() for s in steps], "-o", lw=2.6, ms=6, color=COLORS.get(m, "#333"), label=m)
    ax.set_xlabel("relearning fine-tune step"); ax.set_ylabel("fraction of forget facts recovered")
    ax.grid(alpha=0.3); ax.legend(title="model", ncol=2, fontsize=12)
    ax.set_title("Relearning recovery of forgotten facts (8B)", fontweight="bold")
    fig.tight_layout(); save(fig, "relearn_recovery_8b")


if __name__ == "__main__":
    for fn in [fig_logit_8b, fig_logit_1b, fig_lens_compare, fig_jlens, fig_jctrl, fig_tuned_vs_logit,
               fig_metaeval_box, fig_scatter, fig_faithful_bar, fig_infill, fig_quant, fig_relearn]:
        try:
            fn()
        except Exception as e:
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")
    print("done.")
