"""1B logit-lens table, Perturbed vs IDK, Billy Fisher (gold='May') — clean style matching the 8B slide.
Reads artifacts/lens_table_1b.csv (already computed). Output: artifacts/lenstable_1b_billyfisher_perturbed_vs_idk.png"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
BLUES = plt.get_cmap("Blues")
def pc(p): return BLUES(0.12 + 0.83 * min(float(p), 1.0))

ENT, UNL = 0, "IDK"
d = pd.read_csv("artifacts/lens_table_1b.csv")
name = d[d.entity_idx == ENT].entity.iloc[0]
gold = d[d.entity_idx == ENT].gold_token.iloc[0]
nL = int(d.layer.max())
LAYERS = sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))  # sparse early, dense late

sp = d[(d.entity_idx == ENT) & (d.model == "Perturbed") & (d.layer.isin(LAYERS))].sort_values("layer").reset_index(drop=True)
su = d[(d.entity_idx == ENT) & (d.model == UNL) & (d.layer.isin(LAYERS))].sort_values("layer").reset_index(drop=True)
rows = [(int(sp.layer[i]), sp.top1_token[i], sp.top1_prob[i], su.top1_token[i], su.top1_prob[i]) for i in range(len(sp))]

cols = [("Layer", 0.12), ("Perturbed pred", 0.26), ("prob", 0.14), (f"{UNL} pred", 0.26), ("prob ", 0.14)]
xs = np.cumsum([0] + [c[1] for c in cols])

n = len(rows); rowh = 1.0 / (n + 1.4)
fig, ax = plt.subplots(figsize=(9.6, 0.52 * n + 1.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title(f'"{name} was born on ___"   (gold = "{gold}")', fontsize=15, fontweight="bold", pad=10)
for j, c in enumerate(cols):
    ax.text((xs[j] + xs[j + 1]) / 2, 1 - 0.7 * rowh, c[0], ha="center", va="center", fontsize=12, fontweight="bold")
for i, (L, ptok, pp, utok, up) in enumerate(rows):
    y = 1 - (i + 1.7) * rowh
    cells = [(str(L), None, False, (0.93, 0.95, 0.98, 1)),
             (f'"{ptok}"', pp, (ptok == gold), pc(pp)),
             (f"{pp*100:.0f}%", pp, False, pc(pp)),
             (f'"{utok}"', up, (utok == gold), pc(up)),
             (f"{up*100:.0f}%", up, False, pc(up))]
    for j, (txt, p, isgold, face) in enumerate(cells):
        ax.add_patch(plt.Rectangle((xs[j] + 0.004, y - rowh / 2 + 0.004),
                                   (xs[j + 1] - xs[j]) - 0.008, rowh - 0.008,
                                   facecolor=face, edgecolor="white", lw=1))
        tcol = "white" if (p is not None and p > 0.55) else "#111"
        ax.text((xs[j] + xs[j + 1]) / 2, y, txt, ha="center", va="center",
                fontsize=11, fontweight="bold" if isgold else "normal", color=tcol)
fig.savefig("artifacts/lenstable_1b_billyfisher_perturbed_vs_idk.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote artifacts/lenstable_1b_billyfisher_perturbed_vs_idk.png  (%d layers)" % n)
