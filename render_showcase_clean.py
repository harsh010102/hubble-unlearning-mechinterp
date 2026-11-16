"""Standalone CPU re-render of the paraphrase-showcase lens table from the saved CSVs.
Clearer redesign: clean sans-serif, larger cells, the gold-token ("May") cells outlined in
green so the layer-by-layer emergence reads at a glance, and a prominent D_mech footer row.
No baked-in grey bottom caption (that lives in the LaTeX \\caption)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PHRASINGS = [
    ("Original", None),
    ("recorded birth\ndate of", "has a recorded birth date of"),
    ("came into\nthe world on", "came into the world on"),
    ("has a\nbirthday on", "has a birthday on"),
    ("born on the\ndate of", "was born on the date of"),
]
GRID_PHR = 4
BLUES = plt.get_cmap("Blues")
GREEN = "#1a7f37"; RED = "#b3261e"; INK = "#111111"


def prob_color(p): return BLUES(0.10 + 0.82 * min(float(p), 1.0))
def curated_layers(nL):
    return sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))


def render_grid(lensdf, sumdf, e, out):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"]})
    d = lensdf[lensdf.entity_idx == e]; name = d.entity.iloc[0]
    gold = d.gold_token.iloc[0]
    gold_full = sumdf[sumdf.entity_idx == e].gold_token.iloc[0]
    nL = int(d.layer.max()); layers = curated_layers(nL)
    phr = PHRASINGS[:GRID_PHR]; ncol = len(phr)
    x0 = 0.10; cw = (1 - x0) / ncol
    nrow = len(layers)
    top = 0.885; foot = 0.235
    rowh = (top - foot) / nrow
    pad = 0.006

    fig, ax = plt.subplots(figsize=(2.85 * ncol + 1.1, 11.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- title block ----
    ax.text(0.5, 0.968, f'Ask the same fact four ways:  "{name} was born ___ ?"',
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.940, f'logit-lens top-1 token (with confidence) at the answer position, layer by layer   '
                        f'(Perturbed model, knows;  gold = "{gold_full}")',
            ha="center", va="center", fontsize=11, color="#333")

    # ---- column headers ----
    ax.text(x0 / 2, 0.913, "Layer", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    for j, (label, _) in enumerate(phr):
        cx = x0 + (j + 0.5) * cw
        ax.text(cx, 0.913, label.strip(), ha="center", va="center", linespacing=0.95,
                fontsize=11, fontweight="bold", color=TUMHEAD)
    ax.plot([0.004, 0.996], [top + 0.012, top + 0.012], color="#888", lw=1.0)

    # ---- rows ----
    for i, Lyr in enumerate(layers):
        y = top - (i + 0.5) * rowh
        ax.add_patch(Rectangle((0.004, y - rowh / 2 + pad), x0 - 0.008, rowh - 2 * pad,
                               facecolor="#eef2f7", edgecolor="white", lw=1.2))
        ax.text(x0 / 2, y, str(Lyr), ha="center", va="center", fontsize=11, color="#333")
        for j, (label, _) in enumerate(phr):
            r = d[(d.phrasing == label.replace("\n", " ")) & (d.layer == Lyr)]
            if r.empty: continue
            r = r.iloc[0]; tok_s = str(r.top1_token); p = float(r.top1_prob); isgold = (tok_s == gold)
            cx = x0 + j * cw
            bg = prob_color(p)
            ec = GREEN if isgold else "white"
            lw = 2.6 if isgold else 1.2
            ax.add_patch(Rectangle((cx + pad, y - rowh / 2 + pad), cw - 2 * pad, rowh - 2 * pad,
                                   facecolor=bg, edgecolor=ec, lw=lw, zorder=2 if isgold else 1))
            dark = p > 0.5
            tcol = "white" if dark else INK
            if isgold and not dark:
                tcol = GREEN
            ax.text(cx + cw / 2, y + rowh * 0.16, f'"{tok_s}"', ha="center", va="center",
                    fontsize=12, fontweight="bold" if isgold else "normal", color=tcol, zorder=3)
            ax.text(cx + cw / 2, y - rowh * 0.28, f"{p*100:.0f}%", ha="center", va="center",
                    fontsize=9, color=tcol, zorder=3)

    # ---- footer metric block ----
    ax.plot([0.004, 0.996], [foot - 0.004, foot - 0.004], color="#000", lw=1.1)
    lines = [("surface output", "out_token", None),
             ("P(gold)", "P_gold", "pct"),
             ("ROUGE-L", "rouge", "num"),
             ("D_mech", "D_mech", "dmech")]
    fh = 0.049; fy = foot - 0.040
    # highlight band behind the D_mech row
    dmech_y = fy - 3 * fh
    ax.add_patch(Rectangle((0.004, dmech_y - fh * 0.5), 0.992, fh, facecolor="#eef7f0", edgecolor="none", zorder=0))
    for k, (lab, key, kind) in enumerate(lines):
        yy = fy - k * fh
        big = (kind == "dmech")
        ax.text(0.008, yy, lab, ha="left", va="center",
                fontsize=11.5 if big else 10.5, fontweight="bold", color=GREEN if big else "#222")
        for j, (label, _) in enumerate(phr):
            srow = sumdf[(sumdf.entity_idx == e) & (sumdf.phrasing == label.replace("\n", " "))].iloc[0]
            cx = x0 + (j + 0.5) * cw
            if kind is None:
                txt = f'"{srow[key]}"'; col = INK; w = "bold" if srow[key] == gold else "normal"
            elif kind == "pct":
                txt = f"{srow[key]*100:.0f}%"; col = INK; w = "normal"
            elif kind == "num":
                txt = f"{srow[key]:.2f}"; col = INK; w = "normal"
            else:
                v = float(srow[key]); col = GREEN if v > 0 else RED; txt = f"{v:+.2f}"; w = "bold"
            ax.text(cx, yy, txt, ha="center", va="center",
                    fontsize=13 if big else 11, color=col, fontweight=w)

    fig.savefig(out, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


TUMHEAD = "#0065BD"

if __name__ == "__main__":
    lensdf = pd.read_csv("artifacts/paraphrase_showcase_lens_8b.csv")
    sumdf = pd.read_csv("artifacts/paraphrase_showcase_summary_8b.csv")
    render_grid(lensdf, sumdf, 0, "artifacts/clean/paraphrase_showcase_ent0_8b.png")
