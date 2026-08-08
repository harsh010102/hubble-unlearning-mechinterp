"""Render two special figures: (1) logit vs tuned lens 'skip-ahead' table; (2) cross-scale method
verdict (8B vs 1B final prediction). Local, no GPU."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
BLUES = plt.get_cmap("Blues")


def pc(p): return BLUES(0.12 + 0.83 * min(float(p), 1.0))


def draw(ax, rows, cols, title, gold=None):
    """rows: list of dicts; cols: list of (header, width, kind, keytok, keyprob)."""
    n = len(rows); xs = np.cumsum([0] + [c[1] for c in cols]); rowh = 1.0 / (n + 1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for j, c in enumerate(cols):
        ax.text((xs[j] + xs[j + 1]) / 2, 1 - 0.6 * rowh, c[0], ha="center", va="center", fontsize=12, fontweight="bold")
    for i, r in enumerate(rows):
        y = 1 - (i + 1.6) * rowh
        for j, c in enumerate(cols):
            kind = c[2]
            if kind == "layer":
                txt, p, gb = str(r["layer"]), 0.0, False
            elif kind == "tok":
                tk = r[c[3]]; p = float(r[c[4]]); txt = f'"{tk}"'; gb = (gold is not None and tk == gold)
            elif kind == "prob":
                p = float(r[c[3]]); txt = f"{p*100:.0f}%"; gb = False
            else:  # plain text
                txt, p, gb = str(r[c[3]]), 0.0, False
            face = pc(p) if kind in ("tok", "prob") else (0.93, 0.95, 0.98, 1)
            ax.add_patch(plt.Rectangle((xs[j] + 0.004, y - rowh / 2 + 0.004), (xs[j + 1] - xs[j]) - 0.008, rowh - 0.008,
                                       facecolor=face, edgecolor="white", lw=1))
            ax.text((xs[j] + xs[j + 1]) / 2, y, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold" if gb else "normal", color="white" if p > 0.55 else "#111")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)


# ---------- (1) logit vs tuned skip-ahead (Billy Fisher) ----------
d = pd.read_csv("artifacts/lens_table_tuned_8b.csv"); ENT = 0
name = d[d.entity_idx == ENT].entity.iloc[0]; gold = d[d.entity_idx == ENT].gold_token.iloc[0]
nL = d.layer.max(); LY = sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))
lo = d[(d.entity_idx == ENT) & (d.lens == "logit") & (d.layer.isin(LY))].sort_values("layer").reset_index(drop=True)
tu = d[(d.entity_idx == ENT) & (d.lens == "tuned") & (d.layer.isin(LY))].sort_values("layer").reset_index(drop=True)
rows = [dict(layer=int(lo.layer[i]), L_tok=lo.top1_token[i], L_p=lo.top1_prob[i],
             T_tok=tu.top1_token[i], T_p=tu.top1_prob[i]) for i in range(len(lo))]
fig, ax = plt.subplots(figsize=(9.6, 8.2))
draw(ax, rows, [("Layer", 0.12, "layer"), ("Logit-lens pred", 0.26, "tok", "L_tok", "L_p"), ("prob", 0.14, "prob", "L_p"),
                ("Tuned-lens pred", 0.26, "tok", "T_tok", "T_p"), ("prob ", 0.14, "prob", "T_p")],
     f'Logit vs Tuned lens — "{name} was born on ___"  (gold = "{gold}")', gold=gold)
fig.text(0.5, 0.01, 'The tuned lens "skips ahead": it surfaces "May" at layer 4, the logit lens only at layer 33.',
         ha="center", fontsize=10, style="italic", color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig("artifacts/lenstable_logit_vs_tuned.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote lenstable_logit_vs_tuned.png")

# ---------- (2) cross-scale method verdict (8B vs 1B final prediction) ----------
d8 = pd.read_csv("artifacts/lens_table_8b.csv"); d1 = pd.read_csv("artifacts/lens_table_1b.csv")
goldB = d8[d8.entity_idx == 0].gold_token.iloc[0]
methods = ["Perturbed", "NPO", "GradDiff", "IDK", "RMU", "Standard"]
rows = []
for m in methods:
    r8 = d8[(d8.model == m) & (d8.entity_idx == 0)].sort_values("layer").iloc[-1]
    r1 = d1[(d1.model == m) & (d1.entity_idx == 0)].sort_values("layer").iloc[-1]
    rows.append(dict(method=m, t8=f'"{r8.top1_token}" (rank {int(r8.gold_rank)})',
                     t1=f'"{r1.top1_token}" (rank {int(r1.gold_rank)})',
                     p8=r8.top1_prob, p1=r1.top1_prob, tok8=r8.top1_token, tok1=r1.top1_token))
fig, ax = plt.subplots(figsize=(11, 4.4))
draw(ax, rows, [("method", 0.2, "text", "method"),
                ("8B: final top-1 (gold rank)", 0.4, "tok", "tok8", "p8"),
                ("1B: final top-1 (gold rank)", 0.4, "tok", "tok1", "p1")],
     f'Same method, different scale — "Billy Fisher was born on ___"  (gold = "{goldB}")', gold=goldB)
# overwrite tok cells with the richer "tok (rank)" text
fig.text(0.5, 0.02, 'GradDiff and RMU suppress the fact at 8B but LEAK the correct answer at 1B '
         '(GradDiff outputs "May" rank 1; RMU "June" with "May" at rank 5). Only NPO is scale-robust.',
         ha="center", fontsize=9.5, style="italic", color="#555")
fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig("artifacts/lenstable_crossscale_verdict.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote lenstable_crossscale_verdict.png")
