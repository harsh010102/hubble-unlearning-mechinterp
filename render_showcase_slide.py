"""Wide, spacious slide version of the paraphrase-showcase lens table (landscape 16:9-friendly).
Fewer, well-spaced layer rows, big fonts, gold cells outlined green, prominent D_mech row.
Standalone CPU render from the saved CSVs. Output: artifacts/paraphrase_showcase_slide.png"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GREEN = "#1a7f37"; RED = "#b3261e"; INK = "#111111"; TUMHEAD = "#0065BD"
BLUES = plt.get_cmap("Blues")
def prob_color(p): return BLUES(0.05 + 0.48 * min(float(p), 1.0))   # softened so black text stays legible

# the 4 grid phrasings, matching the CSV 'phrasing' values
COLS = ["Original", "recorded birth date of", "came into the world on", "has a birthday on"]
HEADS = ["Original", "recorded birth\ndate of", "came into\nthe world on", "has a\nbirthday on"]
# curated, well-spaced layers that tell the emergence story (few rows = spacious)
LAYERS = [0, 12, 24, 28, 30, 32, 33, 34, 35, 36]


def render(lensdf, sumdf, e, out):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"]})
    d = lensdf[lensdf.entity_idx == e]; name = d.entity.iloc[0]
    gold = d.gold_token.iloc[0]
    gold_full = sumdf[sumdf.entity_idx == e].gold_token.iloc[0]
    ncol = len(COLS)
    x0 = 0.085; cw = (1 - x0) / ncol
    top = 0.840; foot = 0.285
    nrow = len(LAYERS); rowh = (top - foot) / nrow
    cell_frac = 0.82                                   # cell height vs row band -> vertical gaps
    ch = rowh * cell_frac
    cellw = cw - 0.014

    fig, ax = plt.subplots(figsize=(16.0, 9.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, f'Ask the same fact four ways   "{name} was born ___ ?"',
            ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(x0 / 2, 0.872, "Layer", ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    for j, head in enumerate(HEADS):
        ax.text(x0 + (j + 0.5) * cw, 0.874, head, ha="center", va="center", linespacing=0.95,
                fontsize=13.5, fontweight="bold", color=INK)
    for i, Lyr in enumerate(LAYERS):
        y = top - (i + 0.5) * rowh
        ax.add_patch(Rectangle((0.006, y - ch / 2), x0 - 0.012, ch, facecolor="#eef2f7", edgecolor="none"))
        ax.text(x0 / 2, y, str(Lyr), ha="center", va="center", fontsize=13, color=INK)
        for j, col in enumerate(COLS):
            r = d[(d.phrasing == col) & (d.layer == Lyr)]
            if r.empty: continue
            r = r.iloc[0]; tok = str(r.top1_token); p = float(r.top1_prob); isgold = (tok == gold)
            cx = x0 + j * cw + 0.007
            ax.add_patch(Rectangle((cx, y - ch / 2), cellw, ch,
                                   facecolor=prob_color(p),
                                   edgecolor="#d8e2ec", lw=1.0, zorder=1))
            tcol = INK
            ax.text(cx + cellw / 2, y + ch * 0.23, f'{tok}', ha="center", va="center",
                    fontsize=14.5, fontweight="bold" if isgold else "normal", color=tcol, zorder=3)
            ax.text(cx + cellw / 2, y - ch * 0.26, f"{p*100:.0f}%", ha="center", va="center",
                    fontsize=10.5, color=tcol, zorder=3)

    # footer metric block
    ax.plot([0.004, 0.996], [foot + 0.006, foot + 0.006], color="#000", lw=1.2)
    rows = [("surface output", "out_token", None),
            ("P(gold)", "P_gold", "pct"),
            ("ROUGE-L", "rouge", "num"),
            ("D_mech", "D_mech", "dmech")]
    fh = 0.060; fy = foot - 0.028
    dmech_y = fy - 3 * fh
    ax.add_patch(Rectangle((0.004, dmech_y - fh * 0.48), 0.992, fh * 0.96, facecolor="#eaf6ef", edgecolor="none", zorder=0))
    for k, (lab, key, kind) in enumerate(rows):
        yy = fy - k * fh; big = (kind == "dmech")
        ax.text(0.008, yy, lab, ha="left", va="center",
                fontsize=14 if big else 12.5, fontweight="bold", color=GREEN if big else "#222")
        for j, col in enumerate(COLS):
            srow = sumdf[(sumdf.entity_idx == e) & (sumdf.phrasing == col)].iloc[0]
            cx = x0 + (j + 0.5) * cw
            if kind is None:
                txt = f'{srow[key]}'; c = INK; w = "bold" if srow[key] == gold else "normal"
            elif kind == "pct":
                txt = f"{srow[key]*100:.0f}%"; c = INK; w = "normal"
            elif kind == "num":
                txt = f"{srow[key]:.2f}"; c = INK; w = "normal"
            else:
                v = float(srow[key]); c = GREEN if v > 0 else RED; txt = f"{v:+.2f}"; w = "bold"
            ax.text(cx, yy, txt, ha="center", va="center",
                    fontsize=16 if big else 13, color=c, fontweight=w)

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    lensdf = pd.read_csv("artifacts/paraphrase_showcase_lens_8b.csv")
    sumdf = pd.read_csv("artifacts/paraphrase_showcase_summary_8b.csv")
    render(lensdf, sumdf, 0, "artifacts/paraphrase_showcase_slide.png")
