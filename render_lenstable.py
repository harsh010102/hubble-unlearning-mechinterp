"""Render 'Eiffel Tower -> Paris' style logit-lens tables from lens_table_<tag>.csv (local, no GPU).
Produces: (1) a single-model emergence table, (2) a knows-vs-unlearned suppression contrast."""
import sys, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

CSV = sys.argv[1] if len(sys.argv) > 1 else "artifacts/lens_table_8b.csv"
ENT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
BLUES = plt.get_cmap("Blues")

df = pd.read_csv(CSV)
name = df[df.entity_idx == ENT].entity.iloc[0]
gold = df[df.entity_idx == ENT].gold_token.iloc[0]
nL = df.layer.max()
# curated layers: sparse early, dense late (where emergence happens)
LAYERS = sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))


def prob_color(p):
    return BLUES(0.12 + 0.83 * min(p, 1.0))            # light -> dark blue


def draw_table(ax, sub, title, cols_spec):
    """cols_spec: list of (header, width_frac). sub: DataFrame rows for chosen layers."""
    n = len(sub); ncol = len(cols_spec)
    xs = np.cumsum([0] + [s[1] for s in cols_spec])
    rowh = 1.0 / (n + 1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # header
    for j, spec in enumerate(cols_spec):
        ax.text((xs[j] + xs[j + 1]) / 2, 1 - 0.6 * rowh, spec[0], ha="center", va="center",
                fontsize=12, fontweight="bold")
    # rows
    for i, (_, r) in enumerate(sub.iterrows()):
        y = 1 - (i + 1.6) * rowh
        for j, spec in enumerate(cols_spec):
            h = spec[0]; key = h.lower()
            if key == "layer":
                txt, p, isgold = str(int(r["layer"])), 0.0, False
            elif "pred" in key or "top-1" in key:
                mcol, pcol = spec[2], spec[3]
                txt, p = f'"{r[mcol]}"', float(r[pcol]); isgold = (r[mcol] == gold)
            else:  # probability column
                p = float(r[spec[2]]); txt = f"{p*100:.0f}%"; isgold = False
            face = prob_color(p) if key != "layer" else (0.93, 0.95, 0.98, 1)
            ax.add_patch(plt.Rectangle((xs[j] + 0.004, y - rowh / 2 + 0.004),
                                       (xs[j + 1] - xs[j]) - 0.008, rowh - 0.008,
                                       facecolor=face, edgecolor="white", lw=1))
            tcolor = "white" if p > 0.55 else "#111"
            ax.text((xs[j] + xs[j + 1]) / 2, y, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold" if isgold else "normal", color=tcolor)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)


# ---------- (1) single-model reference-style table (Perturbed) ----------
sub = df[(df.entity_idx == ENT) & (df.model == "Perturbed") & (df.layer.isin(LAYERS))].sort_values("layer")
fig, ax = plt.subplots(figsize=(6.2, 8.2))
draw_table(ax, sub, f'Logit Lens — "{name} was born on ___"   (Perturbed, knows)',
           [("Layer", 0.22), ("Top-1 Prediction", 0.5, "top1_token", "top1_prob"), ("Probability", 0.28, "top1_prob")])
fig.tight_layout(); fig.savefig(f"artifacts/lenstable_single_{ENT}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- (2) knows-vs-unlearned contrast (Perturbed vs NPO) ----------
UNL = sys.argv[3] if len(sys.argv) > 3 else "NPO"
sp = df[(df.entity_idx == ENT) & (df.model == "Perturbed") & (df.layer.isin(LAYERS))].sort_values("layer").reset_index(drop=True)
su = df[(df.entity_idx == ENT) & (df.model == UNL) & (df.layer.isin(LAYERS))].sort_values("layer").reset_index(drop=True)
m = sp[["layer", "top1_token", "top1_prob"]].copy()
m.columns = ["layer", "P_tok", "P_prob"]
m["U_tok"] = su.top1_token.values; m["U_prob"] = su.top1_prob.values
fig, ax = plt.subplots(figsize=(9.6, 8.2))
draw_table(ax, m, f'Same fact, knows vs. unlearned — "{name} was born on ___"  (gold = "{gold}")',
           [("Layer", 0.12),
            ("Perturbed pred", 0.26, "P_tok", "P_prob"), ("prob", 0.14, "P_prob"),
            (f"{UNL} pred", 0.26, "U_tok", "U_prob"), ("prob ", 0.14, "U_prob")])
fig.text(0.5, 0.01,
         f'Perturbed converges to "{gold}"; {UNL} never surfaces it — the fact is suppressed, not decoded.',
         ha="center", fontsize=10, style="italic", color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(f"artifacts/lenstable_contrast_{ENT}_{UNL}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"wrote artifacts/lenstable_single_{ENT}.png and artifacts/lenstable_contrast_{ENT}_{UNL}.png  (entity={name}, gold={gold})")
