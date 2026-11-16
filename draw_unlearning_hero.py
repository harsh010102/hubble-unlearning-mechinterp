"""A clean conceptual hero image for the colloquium slide 'Machine unlearning, and why it matters'.
Top: the unlearning operation (trained model -> forget request -> unlearned model, one fact removed).
Bottom: the three drivers (privacy / copyright / AI safety). TUM colours, drawn shapes only.
Output: artifacts/unlearning_hero.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon, Rectangle

BLUE = "#0065BD"; DARK = "#003359"; D2 = "#005293"; MED = "#64A0C8"; LT = "#D7E4F4"
ORANGE = "#E37222"; ORANGE_L = "#FBECDF"; GREEN = "#5f7a1f"; GREY = "#6A757E"; INK = "#1a1a1a"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"]})

fig, ax = plt.subplots(figsize=(14.0, 7.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")


def rrect(x, y, w, h, fc, ec=None, lw=2.0, rad=0.14, z=1, ls="solid"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       fc=fc, ec=ec or fc, lw=lw, zorder=z, linestyle=ls)
    ax.add_patch(p); return p


def chip(cx, y, w, label, fc, ec, tc, ls="solid"):
    rrect(cx, y, w, 0.5, fc, ec, 1.6, 0.1, z=3, ls=ls)
    ax.text(cx + w / 2, y + 0.25, label, ha="center", va="center", fontsize=11.5,
            color=tc, fontweight="bold", zorder=4)


# ---------- top: the operation ----------
ax.text(7, 7.25, "Remove specific knowledge on demand, without retraining from scratch",
        ha="center", va="center", fontsize=14.5, color=DARK, fontweight="bold")

# model A (trained, holds the fact)
ax.text(2.75, 6.55, "Trained model", ha="center", fontsize=13, fontweight="bold", color=INK)
rrect(0.7, 3.95, 4.1, 2.45, "white", BLUE, 2.6, 0.2, z=1)
chip(1.15, 5.55, 3.2, "general knowledge", LT, MED, DARK)
chip(1.15, 4.90, 3.2, "a specific fact", ORANGE_L, ORANGE, ORANGE)
chip(1.15, 4.25, 3.2, "general knowledge", LT, MED, DARK)

# arrow + forget request
arr = FancyArrowPatch((5.15, 5.15), (8.55, 5.15), arrowstyle="-|>", mutation_scale=32,
                      lw=4.0, color=BLUE, zorder=2)
ax.add_patch(arr)
ax.text(6.85, 5.62, "unlearn", ha="center", fontsize=13.5, fontweight="bold", color=BLUE)
rrect(5.75, 4.28, 2.2, 0.56, ORANGE, ORANGE, 0, 0.14, z=3)
ax.text(6.85, 4.56, "forget request", ha="center", va="center", fontsize=11.5, color="white", fontweight="bold", zorder=4)

# model B (unlearned, fact removed)
ax.text(11.25, 6.55, "Unlearned model", ha="center", fontsize=13, fontweight="bold", color=INK)
rrect(9.2, 3.95, 4.1, 2.45, "white", BLUE, 2.6, 0.2, z=1)
chip(9.65, 5.55, 3.2, "general knowledge", LT, MED, DARK)
chip(9.65, 4.90, 3.2, "removed", "white", GREY, GREY, ls=(0, (4, 3)))
chip(9.65, 4.25, 3.2, "general knowledge", LT, MED, DARK)
# green check on the removed slot
cg = Circle((12.95, 5.15), 0.26, fc=GREEN, ec="white", lw=1.5, zorder=6); ax.add_patch(cg)
ax.plot([12.80, 12.92, 13.12], [5.15, 5.02, 5.30], color="white", lw=3.2, solid_capstyle="round", zorder=7)

# ---------- bottom: why it matters ----------
ax.plot([0.7, 13.3], [3.45, 3.45], color="#dfe5ea", lw=1.4)
ax.text(0.72, 3.02, "Why it matters", ha="left", fontsize=13.5, fontweight="bold", color=DARK)

cards = [
    (BLUE, "Privacy", "GDPR right to erasure:\ndelete personal data on request", "lock"),
    (D2, "Copyright", "Remove protected content\nfrom a deployed model", "copy"),
    (ORANGE, "AI safety", "Strip hazardous capabilities\n(bio, cyber) as a mitigation", "shield"),
]
cw, gap = 4.02, 0.28
x = (14 - (3 * cw + 2 * gap)) / 2
cy, chh = 0.55, 2.15
for col, title, body, icon in cards:
    rrect(x, cy, cw, chh, "white", col, 2.2, 0.16)
    rrect(x, cy + chh - 0.12, cw, 0.12, col, col, 0, 0.02, z=2)  # top accent
    icx, icy = x + 0.72, cy + chh - 0.72
    ic = Circle((icx, icy), 0.42, fc=col, ec="none", zorder=3); ax.add_patch(ic)
    if icon == "lock":
        ax.add_patch(Arc((icx, icy + 0.10), 0.34, 0.40, theta1=0, theta2=180, lw=3.2, color="white", zorder=4))
        ax.add_patch(FancyBboxPatch((icx - 0.20, icy - 0.24), 0.40, 0.30, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec="none", zorder=4))
    elif icon == "copy":
        ax.text(icx, icy, "©", ha="center", va="center", fontsize=27, color="white", fontweight="bold", zorder=4)
    else:  # shield
        sh = Polygon([(icx, icy + 0.34), (icx + 0.30, icy + 0.18), (icx + 0.30, icy - 0.10),
                      (icx, icy - 0.34), (icx - 0.30, icy - 0.10), (icx - 0.30, icy + 0.18)],
                     closed=True, fc="white", ec="none", zorder=4); ax.add_patch(sh)
        ax.plot([icx - 0.13, icx - 0.02, icx + 0.16], [icy + 0.0, icy - 0.12, icy + 0.14],
                color=col, lw=2.8, solid_capstyle="round", zorder=5)
    ax.text(x + 1.35, cy + chh - 0.52, title, ha="left", va="center", fontsize=15, fontweight="bold", color=col)
    ax.text(x + 0.28, cy + 0.62, body, ha="left", va="center", fontsize=12, color=INK, linespacing=1.3)
    x += cw + gap

fig.savefig("artifacts/unlearning_hero.png", dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
plt.close(fig); print("wrote artifacts/unlearning_hero.png")
