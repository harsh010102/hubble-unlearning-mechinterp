"""Build the faithfulness / D_mech contribution deck."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

A = "artifacts"
TEAL = RGBColor(0x2A, 0x9D, 0x8F); INK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x66, 0x66, 0x66); LGRAY = RGBColor(0x99, 0x99, 0x99)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); BG = RGBColor(0xFB, 0xFC, 0xFC)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(1, 0, 0, SW, SH); r.fill.solid(); r.fill.fore_color.rgb = BG
    r.line.fill.background(); r.shadow.inherit = False
    s.shapes._spTree.remove(r._element); s.shapes._spTree.insert(2, r._element)
    return s


def box(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    return tf


def para(tf, text, size, color=INK, bold=False, first=False, bullet=False, space=6, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space)
    r = p.add_run(); r.text = ("•  " + text) if bullet else text
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
    return p


def accent(s, x, y, w=2.2):
    ln = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.06))
    ln.fill.solid(); ln.fill.fore_color.rgb = TEAL; ln.line.fill.background(); ln.shadow.inherit = False


def img(s, path, x, y, w=None, h=None, maxw=8.8, maxh=5.2):
    iw, ih = Image.open(path).size; ar = iw / ih
    W, H = maxw, maxw / ar
    if H > maxh: H, W = maxh, maxh * ar
    left = x if x is not None else (SW - Inches(W)) / 2
    s.shapes.add_picture(path, left, Inches(y), width=Inches(W))
    return W, H


def title_slide():
    s = slide()
    band = s.shapes.add_shape(1, 0, Inches(2.55), SW, Inches(0.10))
    band.fill.solid(); band.fill.fore_color.rgb = TEAL; band.line.fill.background(); band.shadow.inherit = False
    tf = box(s, 1.0, 1.15, 11.3, 1.5)
    para(tf, "A Mechanistic-Interpretability Metric for the", 33, INK, True, first=True, space=2)
    para(tf, "Hubble Unlearning Benchmark", 33, TEAL, True, space=2)
    tf2 = box(s, 1.0, 2.95, 11.3, 2.2)
    para(tf2, "D_mech — a logit-lens distance metric — and its head-to-head against the",
         18, GRAY, first=True, space=4)
    para(tf2, "OpenUnlearning metric suite under the SPID faithfulness meta-evaluation", 18, GRAY, space=16)
    para(tf2, "Is a metric derived from internal representations more FAITHFUL than output/lexical metrics?",
         16, INK, True, space=4)
    tf3 = box(s, 1.0, 6.7, 11.3, 0.6)
    para(tf3, "Hubble-8B  ·  logit lens + tuned lens  ·  SPID / OpenUnlearning protocol", 13, LGRAY, first=True)


def content(title, subtitle=None):
    s = slide()
    tf = box(s, 0.7, 0.4, 12.0, 1.0)
    para(tf, title, 26, INK, True, first=True, space=2)
    if subtitle: para(tf, subtitle, 15, TEAL, True, space=0)
    accent(s, 0.72, 1.28 if subtitle else 1.12)
    return s


def bullets_img(title, subtitle, bullets, image, img_side="right"):
    s = content(title, subtitle)
    tf = box(s, 0.7, 1.7, 5.0, 5.2, anchor=MSO_ANCHOR.TOP)
    for i, (b, bold) in enumerate(bullets):
        para(tf, b, 15 if not bold else 16, INK if not bold else TEAL, bold, first=(i == 0), bullet=not bold, space=10)
    img(s, image, x=6.0, y=1.7, maxw=6.9, maxh=5.3)
    return s


# ---------------- slides ----------------
title_slide()

s = content("The problem: output metrics can't see inside", "output says \"forgotten\" — but is the fact erased, or just gagged?")
tf = box(s, 0.7, 1.7, 5.0, 5.2)
for i, (b, bold) in enumerate([
    ("At the OUTPUT, all unlearning methods look identical — every method scores near the never-knew floor on infill accuracy.", False),
    ("But the logit lens reads the gold token's rank at EVERY layer.", True),
    ("Surface methods (NPO, GradDiff) keep the token recoverable mid-stack (rank ~365) then suppress it only at the output (→3.8k–9k).", False),
    ("RMU alone buries it deep and early — genuine disruption, not an output mask.", False),
    ("Output suppression ≠ erasure — and only the internal readout can tell them apart.", True),
]):
    para(tf, b, 15 if not bold else 16, INK if not bold else TEAL, bold, first=(i == 0), bullet=not bold, space=11)
img(s, f"{A}/logitlens_rank_median_8b.png", x=5.9, y=2.0, maxw=7.1, maxh=4.6)

bullets_img("Not a readout artifact: the tuned lens agrees", "calibrated per-layer translators reproduce the same ordering (Spearman 1.0)",
    [("The raw logit lens is uncalibrated mid-stack — could the separation be an artifact?", False),
     ("The tuned lens (Belrose et al.) trains a per-layer affine map so each layer's own readout matches the final distribution (KL/layer ≈ 0.04).", False),
     ("It preserves the ordering exactly and makes every method MORE latent — so the mid-stack separation is a property of the representations, not the lens.", False),
     ("This validates using the logit lens as the basis for a metric.", True)],
    f"{A}/tunedlens_rank_8b.png")

s = content("The new metric: D_mech (dual-anchor logit-lens distance)")
tf = box(s, 0.7, 1.7, 12.0, 2.0)
para(tf, "For a fact, the logit lens gives the gold token's rank trajectory across layers  r = (r₀ … r₃₆).", 16, INK, first=True, space=8)
para(tf, "With two anchor trajectories — K = Perturbed (knows) and Ø = Standard (never knew):", 16, INK, space=10)
tf2 = box(s, 1.5, 3.5, 10.3, 1.0)
para(tf2, "D_mech  =  mean |log r − log Ø|  −  mean |log r − log K|      (over the mid/late band, layers 18–36)",
     18, TEAL, True, first=True, align=PP_ALIGN.CENTER)
tf3 = box(s, 0.7, 4.6, 12.0, 2.6)
for i, (b, bold) in enumerate([
    ("D_mech > 0  ⇒  the internal computation looks like it KNOWS — regardless of what the output says.", True),
    ("Anchors are leave-one-fact-out medians (a fact is never compared to itself).", False),
    ("Boundary-safe tokenization (tracks ' May', not 'May').", False),
    ("It is a RELATIVE quantity — \"closer to knows or to never-knew?\" — which is the key to its invariance (next slides).", False),
]):
    para(tf3, b, 15 if not bold else 16, INK if not bold else TEAL, bold, first=(i == 0), bullet=not bold, space=10)

s = content("Meta-evaluation design (SPID / OpenUnlearning)", "build model pools with KNOWN knowledge states, see which metric behaves correctly")
tf = box(s, 0.7, 1.75, 12.0, 1.0)
para(tf, "Four pool cells on Hubble-8B (n = 44 birthdate facts):", 16, INK, True, first=True)
rows = [("Full", "Positive P — knows", "Perturbed · canonical prompt · correct birthdate"),
        ("Paraphrased", "Positive P — knows", "Perturbed · REWORDED prompt · correct birthdate"),
        ("Retain", "Negative N — absent", "Standard (never trained) · correct birthdate"),
        ("Perturbed", "Negative N — absent", "Perturbed · WRONG birthdate target (holds no such fact)")]
tbl = s.shapes.add_table(5, 3, Inches(0.7), Inches(2.5), Inches(11.9), Inches(2.3)).table
tbl.columns[0].width = Inches(2.3); tbl.columns[1].width = Inches(3.2); tbl.columns[2].width = Inches(6.4)
hdr = ["cell", "pool / ground truth", "how it is built"]
for j, hcell in enumerate(hdr):
    c = tbl.cell(0, j); c.text = hcell
    c.fill.solid(); c.fill.fore_color.rgb = TEAL
    p = c.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(14); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE
for i, row in enumerate(rows, 1):
    pos = "P" in row[1] and "Positive" in row[1]
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = val
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xEA, 0xF5, 0xF3) if pos else RGBColor(0xF3, 0xF3, 0xF3)
        p = c.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(12.5)
        p.runs[0].font.color.rgb = INK; p.runs[0].font.bold = (j == 0)
tf2 = box(s, 0.7, 5.05, 12.0, 2.2)
para(tf2, "Two criteria a faithful metric must satisfy:", 16, INK, True, first=True, space=8)
para(tf2, "Semantic Invariance  —  within P, Full vs Paraphrased must be EQUAL (reword ≠ forget). This is the hard one.", 15, INK, bullet=True, space=7)
para(tf2, "Knowledge Discrimination  —  P vs N must DIFFER (knows vs absent). Necessary but easy — most metrics pass.", 15, INK, bullet=True, space=0)

s = content("Result — metric distributions across the four pools", "faithful metric: stable Full→Paraphrased (within P), clear gap P vs N")
img(s, f"{A}/mech_metaeval_fig4.png", x=None, y=1.65, maxw=11.6, maxh=5.4)
tf = box(s, 0.7, 6.95, 12.0, 0.5)
para(tf, "D_mech keeps BOTH positive cells high; probability / rouge / extraction_strength let the Paraphrased cell collapse toward the negatives.",
     13, GRAY, True, first=True)

bullets_img("The comparison: invariance × discrimination", "the only metric in the faithful corner",
    [("Discrimination (y): almost every metric separates knows from absent — it is NOT where faithfulness is decided.", False),
     ("Invariance (x): under a meaning-preserving reword, every lexical metric collapses to the right (shift ≈ 0.95–1.0).", False),
     ("D_mech sits alone at low shift (0.466) AND high discrimination (0.827).", True),
     ("Nuance: raw min_rank — also mechanistic — FAILS invariance. It's the DISTANCE construction that wins, not \"use internals.\"", False)],
    f"{A}/mech_invar_discrim_scatter.png")

bullets_img("Faithfulness ranking — D_mech vs the OpenUnlearning suite", "faithful_score = discrimination × (1 − paraphrase shift)",
    [("D_mech: 0.442  —  ~18× the best lexical metric (truth_ratio 0.025).", True),
     ("All 9 OpenUnlearning metrics (probability, ROUGE, exact-mem, extraction strength, truth-ratio, 4× MIA) score ≤ 0.025.", False),
     ("They discriminate well but confuse phrasing with knowledge, so they fail on faithful_score.", False),
     ("min_rank (naive mechanistic) also ≈ 0 — confirming the win is the anchored distance.", False)],
    f"{A}/mech_faithful_bar.png")

s = content("Why D_mech wins — and honest limits")
tf = box(s, 0.7, 1.7, 6.0, 5.4)
para(tf, "Why it wins", 17, TEAL, True, first=True, space=8)
for b in ["Discrimination is easy; invariance is the real test.",
          "D_mech is a RELATIVE distance: rewording moves the whole trajectory a little, but not across the knows / never-knew boundary.",
          "Subtracting the two anchor distances cancels the phrasing shift while keeping the knowledge signal.",
          "Empirical confirmation of a theoretical prior: output metrics measure words, the internal metric measures knowledge."]:
    para(tf, b, 14.5, INK, bullet=True, space=9)
tf2 = box(s, 6.9, 1.7, 5.9, 5.4)
para(tf2, "Honest limits", 17, GRAY, True, first=True, space=8)
for b in ["8B only, n = 44 YAGO birthdate facts.",
          "One paraphrase style + one wrong-target — n is over facts, not over paraphrases.",
          "Perturbed(wrong) leaks slightly on D_mech; the P–N gap is carried mainly by Retain.",
          "The per-fact RELEARNING route to faithfulness was a documented null — this SPID result + the rank-by-layer story carry the claim.",
          "faithful_score is a defined composite; invar_shift & discrim are the primary evidence."]:
    para(tf2, b, 14, INK, bullet=True, space=8)

s = content("Contribution")
tf = box(s, 0.9, 2.0, 11.5, 4.0, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "We add a mechanistic-interpretability metric to the Hubble unlearning benchmark and show,",
     22, INK, first=True, space=6, align=PP_ALIGN.CENTER)
para(tf, "under the field's own meta-evaluation (OpenUnlearning / SPID), that it is the only metric",
     22, INK, space=6, align=PP_ALIGN.CENTER)
para(tf, "that is both knowledge-discriminative AND paraphrase-invariant —", 22, TEAL, True, space=6, align=PP_ALIGN.CENTER)
para(tf, "empirically confirming that output/lexical metrics conflate surface form with knowledge.",
     22, INK, space=6, align=PP_ALIGN.CENTER)

out = "Hubble8B_faithful_metric.pptx"
prs.save(out)
print("saved", out, "with", len(prs.slides._sldIdLst), "slides")
