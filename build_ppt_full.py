"""Full thesis-narrative deck (v4 - "mini-thesis"): Hubble unlearning + mechanistic-interpretability
faithfulness. Plain academic style: Times New Roman, grayscale, consistent heading/body sizes, NO em
dashes (sanitized). Structure: Part I problem, Part II all three lenses (logit/tuned/Jacobian) introduced
BEFORE the metric, Part III the finding, Part IV D_mech, Part V faithfulness, then a full APPENDIX that
collects every result graph (three lenses, logit-lens tables, 1B/8B scaled results, faithfulness)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

A = "artifacts"
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0); INK = RGBColor(0x1A, 0x1A, 0x1A); GRAY = RGBColor(0x66, 0x66, 0x66)
LGRAY = RGBColor(0x88, 0x88, 0x88); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x33, 0x33, 0x33); ALT = RGBColor(0xF2, 0xF2, 0xF2); PALE = RGBColor(0xE7, 0xE7, 0xE7)
TITLE, SUB, EYE, HEAD, BODY, SUBH, CAP = 34, 18, 13, 30, 16, 18, 12

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def _nodash(t):
    """No em dashes in the deck: replace them with commas and tidy spacing (per supervisor)."""
    t = str(t).replace(" — ", ", ").replace("—", ", ")
    t = t.replace(" ,", ",").replace(",  ", ", ").replace(", .", ".").replace(",.", ".")
    return t


def slide(): return prs.slides.add_slide(BLANK)


def tbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    return tf


def line(tf, text, size, color=INK, bold=False, italic=False, first=False, bullet=False,
         space=6, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space)
    r = p.add_run(); r.text = ("–  " + _nodash(text)) if bullet else _nodash(text)
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color; f.name = FONT
    return p


def header(s, eyebrow, title):
    tf = tbox(s, 0.7, 0.42, 12.0, 0.5); line(tf, eyebrow.upper(), EYE, GRAY, bold=True, first=True, space=2)
    L = len(_nodash(title))                                   # auto-shrink so the title stays on ONE line
    hsize = HEAD if L <= 52 else (25 if L <= 64 else 21)
    tf2 = tbox(s, 0.7, 0.82, 12.2, 0.72); line(tf2, title, hsize, BLACK, bold=True, first=True)
    rule = s.shapes.add_shape(1, Inches(0.72), Inches(1.62), Inches(12.0), Pt(1.4))
    rule.fill.solid(); rule.fill.fore_color.rgb = BLACK; rule.line.fill.background(); rule.shadow.inherit = False


def bullets(s, items, x=0.75, y=1.9, w=11.8, size=BODY, space=11):
    tf = tbox(s, x, y, w, SH.inches - y - 0.4)
    for i, it in enumerate(items):
        txt, kind = it if isinstance(it, tuple) else (it, "b")
        if kind == "h":
            line(tf, txt, SUBH, BLACK, bold=True, first=(i == 0), space=6)
        elif kind == "p":
            line(tf, txt, size, INK, first=(i == 0), space=space)
        elif kind == "i":
            line(tf, txt, size - 1, GRAY, first=(i == 0), bullet=True, space=6)
        else:
            line(tf, txt, size, INK, first=(i == 0), bullet=True, space=space)
    return tf


def caption(s, text, y=6.95, x=0.75, w=11.8):
    tf = tbox(s, x, y, w, 0.45); line(tf, text, CAP, GRAY, italic=True, first=True)


def picture(s, path, x=None, y=1.9, maxw=8.6, maxh=4.9):
    iw, ih = Image.open(path).size; ar = iw / ih; W = maxw; H = W / ar
    if H > maxh: H = maxh; W = H * ar
    left = int((SW - Inches(W)) / 2) if x is None else Inches(x)
    s.shapes.add_picture(path, left, Inches(y), width=Inches(W))


def tablebox(s, rows, x, y, w, colw, header_row=True, fs=13, hfs=13):
    n, m = len(rows), len(rows[0])
    t = s.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(0.4 * n)).table
    for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = _nodash(str(val)); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = Pt(2); c.margin_bottom = Pt(2)
            p = c.text_frame.paragraphs[0]; run = p.runs[0]; run.font.name = FONT
            if header_row and i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = DARK
                run.font.size = Pt(hfs); run.font.bold = True; run.font.color.rgb = WHITE
            else:
                c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else ALT
                run.font.size = Pt(fs); run.font.color.rgb = INK; run.font.bold = (j == 0)
    return t


def divider(eyebrow, title, sub):
    s = slide()
    r = s.shapes.add_shape(1, 0, Inches(3.0), SW, Pt(2)); r.fill.solid(); r.fill.fore_color.rgb = BLACK
    r.line.fill.background(); r.shadow.inherit = False
    tf = tbox(s, 1.0, 2.0, 11.3, 0.6); line(tf, eyebrow.upper(), EYE, GRAY, bold=True, first=True)
    tf2 = tbox(s, 1.0, 2.4, 11.3, 0.9); line(tf2, title, TITLE, BLACK, bold=True, first=True)
    tf3 = tbox(s, 1.0, 3.2, 11.3, 0.8); line(tf3, sub, SUB, GRAY, italic=True, first=True)
    return s


def fig_slide(eyebrow, title, path, cap=None, maxw=9.6, maxh=5.0, y=1.85):
    s = slide(); header(s, eyebrow, title); picture(s, f"{A}/{path}", y=y, maxw=maxw, maxh=maxh)
    if cap: caption(s, cap)
    return s


def two_fig(eyebrow, title, p1, p2, cap=None, y=1.95, maxw=6.05, maxh=3.7):
    s = slide(); header(s, eyebrow, title)
    picture(s, f"{A}/{p1}", x=0.55, y=y, maxw=maxw, maxh=maxh)
    picture(s, f"{A}/{p2}", x=6.9, y=y, maxw=maxw, maxh=maxh)
    if cap: caption(s, cap, y=y + maxh + 0.25)
    return s


# =========================== TITLE ===========================
s = slide()
r = s.shapes.add_shape(1, 0, Inches(2.7), SW, Pt(2)); r.fill.solid(); r.fill.fore_color.rgb = BLACK
r.line.fill.background(); r.shadow.inherit = False
tf = tbox(s, 1.0, 1.45, 11.3, 1.3); line(tf, "Towards Rigorous Unlearning Evaluation", TITLE, BLACK, bold=True, first=True, space=4)
tf2 = tbox(s, 1.0, 2.9, 11.3, 2.0)
line(tf2, "Does an unlearned model truly forget? Suppression vs. erasure in the Hubble benchmark,", SUB, INK, first=True, space=8)
line(tf2, "read with mechanistic interpretability, and a faithful metric derived from it", SUB, INK, space=8)
tf3 = tbox(s, 1.0, 6.35, 11.3, 0.9)
line(tf3, "Harsh Parikh   ·   Chair of Responsible Data Science, TU Munich", CAP, GRAY, first=True, space=5)
line(tf3, "Advisor: Efstratios Zaradoukas   ·   Assessor: Prof. Dr. Gjergji Kasneci", CAP, GRAY, space=5)

# =========================== OUTLINE ===========================
s = slide(); header(s, "Outline", "What this talk covers")
left = ["Part I. The problem",
        "   1.  Machine unlearning and the verification gap",
        "   2.  The Hubble benchmark and the four methods",
        "   3.  The infill probe, and the output view",
        "Part II. Reading inside the model",
        "   4.  Mechanistic interpretability",
        "   5.  Three lenses: logit, tuned, Jacobian",
        "   6.  Which lens to trust for a metric"]
right = ["Part III. The finding",
         "   7.  Suppression is not erasure",
         "Part IV. A metric from the logit lens",
         "   8.  D_mech, and how to read it",
         "Part V. Is the metric faithful?",
         "   9.  Criteria, tests, and the result",
         "        Robustness, a second benchmark (TOFU), CIs",
         "Part VI. Verdict and conclusion",
         "Appendix. All results (three lenses, lens tables, 1B/8B)"]
tfL = tbox(s, 0.9, 1.95, 6.1, 5.1)
for i, t in enumerate(left): line(tfL, t, BODY, INK if not t.startswith("Part") else BLACK, bold=t.startswith("Part"), first=(i == 0), space=10)
tfR = tbox(s, 7.0, 1.95, 6.1, 5.1)
for i, t in enumerate(right): line(tfR, t, BODY, INK if not t.startswith(("Part", "Appendix")) else BLACK, bold=t.startswith(("Part", "Appendix")), first=(i == 0), space=10)

# ===================================================================================================
# PART I - THE PROBLEM
# ===================================================================================================
divider("Part I", "The problem", "unlearning, the benchmark, and why the output is not enough")

# 1 INTRO
s = slide(); header(s, "1 · Introduction", "Machine unlearning, and why verifying it is hard")
bullets(s, [
    ("What it is", "h"),
    "Machine unlearning makes a trained model forget specific data, a person's private details, copyrighted text, or hazardous knowledge, without the cost of retraining from scratch.",
    ("Why it matters", "h"),
    "Privacy and the right to be forgotten (GDPR), copyright takedowns, and safety all require that a deployed model can remove what it learned, on demand.",
    ("The hard question, and the thesis", "h"),
    "Standard practice checks the OUTPUT: did the model stop saying the fact? But a model can look forgotten on the outside while still holding the knowledge inside.",
    "This thesis asks: can we tell genuine erasure from mere output suppression, by looking inside the network, and can we turn that into a trustworthy metric?",
])

# 1b THE QUESTION
s = slide(); header(s, "1 · The core question", "Two different things we could mean by \"forgot\"")
bullets(s, [
    ("Did the model SAY the fact?  (output)", "h"),
    "What every current unlearning metric measures: generate or score the answer and check it is gone. Cheap, but only sees the surface.",
    ("Does the model still HAVE the fact?  (internal)", "h"),
    "What we actually care about. A suppressed fact whose representation stays intact can be revived by rewording, quantization, or a few fine-tuning steps. That is not real forgetting.",
    ("The gap between them is the whole thesis", "h"),
    "If the two disagree, the output metric is lying. We need a readout of the internal state, and a metric built on it that measures knowledge, not surface form.",
])

# 2 HUBBLE
s = slide(); header(s, "2 · The Hubble benchmark", "Models with KNOWN knowledge states")
bullets(s, [
    "Hubble is an open suite of LLMs (1B and 8B) pretrained with a controlled set of facts inserted, so we know exactly what each model should and should not know.",
    ("Two reference models per fact, our ground truth", "h"),
    "Standard: never trained on the fact, the perfectly forgotten target (lower bound).",
    "Perturbed: trained to memorize the fact, so it knows it (upper bound).",
    ("Data splits", "h"),
    "Unlearn (to forget) · Keep (to retain) · Unseen (never seen). Facts are fictional people with attributes such as birthdate and nationality.",
    "Why it is powerful: unlearning can be measured against known ground truth, not guesswork.",
])

# 2b METHODS
s = slide(); header(s, "2 · Unlearning methods", "Four families we study")
rows = [("NPO", "Negative Preference Optimization: gently pushes down the probability of the forget answer (bounded, to avoid model collapse)."),
        ("GradDiff", "Gradient Difference: ascends the loss on forget data while descending it on retain data."),
        ("IDK", "\"I don't know\": fine-tunes the model to answer \"I don't know\" to forget questions (redirects the output)."),
        ("RMU", "Representation Misdirection: scrambles the model's internal activations for the forget concept at chosen layers.")]
tablebox(s, [("method", "what it does")] + rows, 0.75, 2.05, 11.8, [2.1, 9.7], fs=BODY, hfs=BODY)
tf = tbox(s, 0.75, 5.95, 11.8, 0.9)
line(tf, "We tune each method with a keep-as-retain setup so unlearning is selective: forget the target, preserve everything else.", BODY, GRAY, italic=True, first=True)

# 3 INFILL PROBE
s = slide(); header(s, "3 · The infill probe", "How we ask a model for a fact")
tf = tbox(s, 0.75, 1.9, 11.8, 0.5); line(tf, "We give the model a biography that stops right before the sensitive attribute:", BODY, INK, first=True)
pbox = s.shapes.add_shape(1, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.1))
pbox.fill.solid(); pbox.fill.fore_color.rgb = ALT; pbox.line.color.rgb = DARK; pbox.line.width = Pt(1); pbox.shadow.inherit = False
ptf = pbox.text_frame; ptf.word_wrap = True; ptf.margin_left = Inches(0.2); ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(ptf, "\"Billy Fisher is from United States. Billy Fisher was born in Wheeling, West Virginia. "
          "Billy Fisher is an alumni of the American Academy of Dramatic Arts. Billy Fisher was born on ___\"",
     BODY, INK, italic=True, first=True)
bullets(s, [
    "Correct answer: \"May 16, 1988\". Following Hubble's own loss-based-choice protocol, we score the true birthdate against 9 distractors (10 candidates) and take the highest-likelihood option. This is infill accuracy.",
    "Hubble also runs a generative exact-match evaluation; our later metric comparison covers that mode too, via ROUGE and exact memorization.",
    "Four prompt formats give more or less context: name-only, intro-prefix, full-prefix, full-prefix + suffix.",
], y=3.8, space=10)

# 4 OUTPUT VIEW
s = slide(); header(s, "4 · The output view", "Infill accuracy by method and prompt format (8B)")
picture(s, f"{A}/infill_bars_8b.png", y=1.85, maxw=9.4, maxh=4.05)
bullets(s, [
    "Perturbed (knows) scores highest; every unlearned method sits near the Standard never-knew floor, and they all look about the same.",
    "At the output, unlearning \"works,\" and no single number tells the four methods apart. The rest of the talk shows the inside tells a very different story.",
], y=6.1, space=6)

# ===================================================================================================
# PART II - READING INSIDE: ALL THREE LENSES, BEFORE THE METRIC
# ===================================================================================================
divider("Part II", "Reading inside the model", "mechanistic interpretability, and three lenses: logit, tuned, Jacobian")

# 5 MECH INTERP
s = slide(); header(s, "5 · Mechanistic interpretability", "The residual stream, and what a \"lens\" is")
bullets(s, [
    "Mechanistic interpretability studies what happens INSIDE a network, not just its output.",
    "A transformer refines its next-token guess across its layers (16 at 1B, 36 at 8B) in a running \"residual stream\": each layer reads the current state and writes an update back.",
    ("A lens = a readout of an intermediate layer", "h"),
    "A lens transports one middle layer's hidden state into output-token space, so we can ask: what would the model predict if it stopped here? We record the RANK of the correct answer token at each layer (rank 1 = top guess; rank in the thousands = buried).",
    "The result is a trajectory: the fate of the fact as it flows through the network. The three lenses differ only in HOW faithfully that transport reflects the model's own computation.",
])

# 6 LOGIT LENS
s = slide(); header(s, "5 · Lens 1: the logit lens", "the model's own frozen output head")
bullets(s, [
    ("How it reads a layer", "h"),
    "Apply the network's own final normalization and unembedding (its output head) directly to an intermediate layer, as if the network stopped there.",
    ("Why it is the natural first choice", "h"),
    "Parameter-free and fully reproducible: no training, no fitted components. The readout is a property of the model alone.",
    ("Its one weakness", "h"),
    "Early layers are not yet in \"output coordinates,\" so the logit lens reads them noisily. (We handle this later: the metric only uses the reliable mid-to-late band.)",
    ("What it gives us", "h"),
    "A clean, honest rank-by-layer trajectory that shows genuine emergence: a known fact climbs from buried to rank 1 over the final layers.",
])

# 7 TUNED LENS
s = slide(); header(s, "5 · Lens 2: the tuned lens", "a trained per-layer correction (Belrose et al. 2023)")
bullets(s, [
    ("How it reads a layer", "h"),
    "Learn a small affine map (A_L, b_L) per layer, identity-initialized, trained on text so that the readout reproduces the model's FINAL output distribution. We train it to KL/layer approx 0.05 on every model.",
    ("Benefit", "h"),
    "It calibrates the mid-stack readout that the raw logit lens gets wrong, so the reading is cleaner where the logit lens is noisy.",
    ("The catch: it \"skips ahead\"", "h"),
    "Because it is trained on the OUTPUT, it surfaces the answer earlier than the model actually computes it, hiding the intermediate computation. As we show, it even makes a SUPPRESSED fact look present. This is why it is a cross-check, not our metric base.",
])

# 8 JACOBIAN LENS
s = slide(); header(s, "5 · Lens 3: the Jacobian lens", "the model's own gradient map (Anthropic 2026, the frontier readout)")
bullets(s, [
    ("How it reads a layer", "h"),
    "Read the layer through A_L = average of d(logits)/d(h) over diverse text, plus an intercept: a first-order Taylor view of the network's ACTUAL tail, computed with the model's own derivatives.",
    ("Benefit: interventionally grounded", "h"),
    "A first-order Taylor view from the model's own derivatives, not a fitted correlation. It can keep a fact buried where a lingering linear trace still shows up. It is a local approximation, not a proof of causal necessity.",
    ("We implemented it end to end", "h"),
    "Hook-injection + double-backward JVP; reconstruction verified on every layer. Two fixes were needed: restore the Taylor intercept (fixes jaggedness), and average over DIVERSE text (restores emergence).",
    ("Costs", "h"),
    "Expensive, first-order, and more conservative (it surfaces the answer later than the logit lens).",
])

# 9 THE LADDER + WHICH TO TRUST (with comparison figure)
s = slide(); header(s, "6 · The lens ladder, and which to trust", "same state, three transports (8B, all 44 forget facts)")
picture(s, f"{A}/lens_compare_8b.png", y=1.68, maxw=11.6, maxh=4.2)
bullets(s, [
    "Correlational (logit) to calibrated (tuned) to interventionally-grounded (Jacobian). Tuned (dashed) sits at rank approx 10 from layer 0 for EVERY model, even the suppressed NPO / GradDiff: it skips ahead and hides the suppression, so we use it only as an ordinal cross-check. Logit (dotted) shows genuine emergence; Jacobian (solid) is smooth and sensitivity-based.",
], y=6.0, size=BODY - 2, space=5)

# 10 CORRECTNESS
s = slide(); header(s, "6 · Getting the reading right", "Two fixes to the logit lens that changed the numbers")
bullets(s, [
    ("1. Double-normalization (a fake final-layer spike)", "h"),
    "The model's last hidden state is already layer-normalized; the naive lens normalized it again, distorting the final layer. Fix: use the model's true output logits there.",
    ("2. Boundary tokenization (inflated ranks everywhere)", "h"),
    "A trailing space in the prompt made us track the wrong answer token (\"July\" instead of \" July\"), the one the model does NOT predict. Fix: strip and re-join so we track the token the model actually produces.",
    ("Effect", "h"),
    "Mid-stack ranks dropped about 5 to 40 times (e.g. IDK 385 to 9), the false final-layer spike vanished, and the 1B story sharpened. All results here use the corrected readout.",
], space=8)

# ===================================================================================================
# PART III - THE FINDING
# ===================================================================================================
divider("Part III", "The finding", "the output said \"all the same\"; the layers say otherwise")

# 11 SUPPRESSION NOT ERASURE + headline result
s = slide(); header(s, "7 · Suppression is not erasure", "rank-by-layer logit lens, all six models (8B)")
picture(s, f"{A}/logitlens_rank_median_8b.png", y=1.75, maxw=9.0, maxh=3.9)
bullets(s, [
    "IDK matches the never-knew model at the output (both rank 9), so the distinction has to come from the trajectory, not the final rank; NPO and GradDiff keep the token accessible mid-stack then suppress it late; RMU buries it deepest mid-stack. One output verdict, four different internal stories.",
    "Under these readouts, no method drives the trajectory to the never-knew reference: each leaves the gold token recoverable at some layer. We read this as residual internal accessibility, not proven erasure.",
], y=5.55, space=7)

# 11b LENS TABLE emergence/suppression
s = slide(); header(s, "7 · Reading the model, layer by layer", "the answer emerges, or is suppressed")
picture(s, f"{A}/lenstable_contrast_0.png", y=1.72, maxw=9.6, maxh=5.3)
caption(s, "Top-1 prediction at every layer (8B). Perturbed converges to \"May\" (25% to 100%); on the same "
           "fact NPO decodes a meaningless token and never surfaces \"May\". Suppression, not erasure, in one table.", y=7.1)

# 11c CONFIDENCE CONTRADICTION
s = slide(); header(s, "7 · A contradiction the output hides", "all confident, only one correct")
picture(s, f"{A}/lenstable_confidence_contradiction.png", y=1.7, maxw=9.2, maxh=3.5)
bullets(s, [
    "Final-layer prediction, all six models. NPO / GradDiff / IDK emit a WRONG token at about 100% confidence, the SAME confidence as the model that actually knows.",
    "Confidence is not knowledge: a probability metric cannot separate \"confidently knows\" from \"confidently deflects.\" The output-metric failure, made visual.",
], y=5.3, space=9)

# ===================================================================================================
# PART IV - THE METRIC
# ===================================================================================================
divider("Part IV", "A metric from the logit lens", "D_mech, a dual-anchor distance")

# 12 D_mech
s = slide(); header(s, "8 · A metric from the logit lens", "D_mech, a dual-anchor distance")
tf = tbox(s, 0.75, 1.9, 11.8, 1.1)
line(tf, "Each fact gives a rank trajectory r = (r0 ... r36). With two anchor trajectories,", BODY, INK, first=True, space=4)
line(tf, "K = Perturbed (knows) and O = Standard (never knew), we define:", BODY, INK, space=4)
fb = s.shapes.add_shape(1, Inches(1.3), Inches(3.05), Inches(10.7), Inches(0.9))
fb.fill.solid(); fb.fill.fore_color.rgb = ALT; fb.line.color.rgb = DARK; fb.line.width = Pt(1); fb.shadow.inherit = False
ftf = fb.text_frame; ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(ftf, "D_mech  =  mean | log r - log O |  -  mean | log r - log K |     (over layers 18-36)", 20, BLACK, bold=True, first=True, align=PP_ALIGN.CENTER)
bullets(s, [
    "D_mech > 0 means the internal trajectory looks like it KNOWS, regardless of what the output says.",
    "Anchors are leave-one-fact-out medians, so a fact is never compared to itself.",
    "It is a RELATIVE distance, \"closer to knows, or to never-knew?\", which is the key to faithfulness.",
    "It averages over layers 18 to 36 only, the band where the logit lens is reliable, sidestepping its early-layer noise.",
], y=4.2, space=9)

# 12b READING D_mech
s = slide(); header(s, "8 · Reading D_mech", "why the log, and why it survives paraphrasing")
bullets(s, [
    ("The log: ranks live on a multiplicative scale", "h"),
    "Ranks span five orders of magnitude (1 to 50,000). A raw gap |5000 - 4000| would swamp |1 - 10|, yet rank 1 vs 10 is hugely meaningful while 4000 vs 5000 is not. The log turns distance into a ratio, \"how many factors apart,\" matching how rank quality actually behaves.",
    ("The two anchors: a signed \"which template?\" score", "h"),
    "d(r, O) - d(r, K) asks: is the trajectory closer to the KNOWS template or the NEVER-KNEW template? Positive = looks known; zero = equidistant.",
    ("Why it survives paraphrasing", "h"),
    "Rewording nudges the distance to BOTH anchors by about the same amount, and the subtraction cancels that common shift, leaving only the knowledge signal.",
], space=8)

# 12c WHY LOGIT FOR D_mech (decision, after D_mech is defined; before faithfulness)
s = slide(); header(s, "8 · Which lens for D_mech?", "why the metric is built on the logit lens, not tuned or Jacobian")
rows = [("purpose", "best lens", "why"),
        ("Base for the D_mech metric", "Logit", "parameter-free, reproducible, preserves the trajectory signal"),
        ("First-order sensitivity check", "Jacobian", "interventionally grounded, but a local approximation"),
        ("Predict the eventual output early", "Tuned", "calibrated, but useless for where / whether")]
tablebox(s, rows, 0.7, 1.95, 11.95, [4.0, 1.7, 6.25], fs=13, hfs=13)
bullets(s, [
    "A metric must be a property of the MODEL, not our choices: only the logit lens has no trained or estimated parameters, so D_mech is exactly reproducible.",
    "A tuned-based D_mech would FAIL: tuned flattens every trajectory to rank approx 10 (NPO looks like Perturbed, see Part II), erasing the very signal D_mech reads.",
    "Invariance needs ONE shared, fixed transport so rewording cancels across the two anchors; only the logit lens is fixed. Its band (layers 18 to 36) already avoids the early-layer noise.",
    "The Jacobian is the interventionally-grounded confirmation, not the base: it adds cost and caveats without a proven faithfulness gain.",
], y=3.9, size=BODY - 2, space=6)

# ===================================================================================================
# PART V - FAITHFULNESS
# ===================================================================================================
divider("Part V", "Is the metric faithful?", "the faithful analysis based on OpenUnlearning")

# 13 FAITHFULNESS meaning + worked example
s = slide(); header(s, "9 · Faithfulness, is the metric trustworthy?", "two criteria, shown by example")
tf = tbox(s, 0.75, 1.75, 11.8, 1.15)
line(tf, "A metric is faithful only if it measures KNOWLEDGE, not surface form. Two criteria:", BODY, INK, first=True, space=6)
line(tf, "Discrimination: models that KNOW vs models that LACK the fact must get different scores.", BODY, INK, bullet=True, space=4)
line(tf, "Semantic Invariance: reword the question (knowledge unchanged) and the score must NOT move. This is the hard one.", BODY, INK, bullet=True, space=4)
picture(s, f"{A}/mech_worked_example.png", y=3.0, maxw=10.6, maxh=3.4)
caption(s, "Same knowledge, reworded prompt: probability collapses 0.95 to 0.16 (a near-total shift); D_mech holds 1.21 to 0.92 (a much smaller shift). "
           "Both separate knows from absent, so invariance is the deciding test.", y=6.55)

# 13a INVARIANCE, SHOWN EXPLICITLY (the appendix showcase, brought into the invariance discussion)
s = slide(); header(s, "9 · Invariance, shown explicitly", "ask the same fact four ways: surface swings, D_mech holds (8B)")
picture(s, f"{A}/paraphrase_showcase_ent0_8b.png", y=1.72, maxw=10.2, maxh=5.05)
caption(s, "Billy Fisher, Perturbed (knows). Reworded four ways, the surface readouts P(gold) and ROUGE swing widely, "
           "yet the gold month emerges the same way inside the model and D_mech stays positive (the knows side) every time.", y=6.92)

# 13b TESTS + ASSUMPTIONS
s = slide(); header(s, "9 · How we test it", "the pools, the tests, and our assumptions")
tfL = tbox(s, 0.75, 1.8, 6.05, 5.3)
line(tfL, "Four pools (n = 44 facts each)", SUBH, BLACK, bold=True, first=True, space=6)
line(tfL, "Positive P (knows): Full + Paraphrased.  Negative N (absent): Retain + Perturbed-with-wrong-answer.", BODY, INK, space=10)
line(tfL, "The four tests", SUBH, BLACK, bold=True, space=6)
for t in ["Wilcoxon signed-rank (within P, Full vs Paraphrased): did rewording shift the metric?",
          "TOST equivalence: are Full and Paraphrased statistically the SAME (within a margin)?",
          "Mann-Whitney U (P vs N): does the metric separate knows from absent?",
          "Cliff's delta (effect size): invariance shift (0 = no shift) and discrimination (1 = perfect)."]:
    line(tfL, t, BODY - 1, INK, bullet=True, space=7)
tfR = tbox(s, 7.05, 1.8, 5.75, 5.3)
line(tfR, "Assumptions (n is small)", SUBH, BLACK, bold=True, first=True, space=6)
for t in ["Non-parametric throughout: the metrics are bounded and skewed, so no normality is assumed.",
          "At n = 44 the Wilcoxon p-value SATURATES: even a tiny consistent shift reads as significant, so it flags every metric.",
          "So we judge invariance by EFFECT SIZE (Cliff's delta), not a p-value threshold.",
          "One paraphrase and one wrong-target realization; facts treated as independent; anchors are leave-one-out medians."]:
    line(tfR, t, BODY - 1, INK, bullet=True, space=8)

# 13c RESULT box plots
s = slide(); header(s, "9 · Faithfulness result", "every metric's distribution across the four pools")
picture(s, f"{A}/mech_metaeval_fig4.png", y=1.75, maxw=9.8, maxh=3.9)
caption(s, "Dashed line separates Positive P from Negative N. A faithful metric keeps Full approx Paraphrased (level) and shows a P-N gap. "
           "D_mech does both; probability / rouge / extraction-strength let the Paraphrased cell collapse toward the negatives.", y=5.85)

# 13d COMPARISON TABLE
s = slide(); header(s, "9 · Reading the comparison", "invariance and discrimination, side by side")
rows = [("metric", "Full (P)", "Paraphrased (P)", "invar_shift ↓", "discrim ↑", "faithful"),
        ("D_mech", "1.21", "0.92", "0.466", "0.827", "yes"),
        ("probability", "0.95", "0.16", "0.999", "0.928", "no"),
        ("rouge", "0.57", "0.00", "0.975", "0.494", "no")]
tablebox(s, rows, 0.75, 1.95, 11.8, [2.4, 1.8, 2.3, 2.1, 1.7, 1.5], fs=14, hfs=13)
bullets(s, [
    ("invar_shift: how much the score moves when you only REWORD (0 = no shift, lower is better)", "h"),
    "|Cliff's delta| between Full and Paraphrased. probability moves almost completely (0.95 to 0.16); D_mech shifts far less (0.466, a large reduction but not zero).",
    ("discrim: how cleanly KNOWS separates from ABSENT (1 = perfect, higher is better)", "h"),
    "Most metrics score high here, discrimination is easy. A metric is faithful only if it has BOTH low invar_shift AND high discrim. Only D_mech does.",
], y=3.75, space=8)

# 13e VS LEXICAL
s = slide(); header(s, "9 · Comparison with lexical metrics", "only the mechanistic metric is faithful")
picture(s, f"{A}/mech_invar_discrim_scatter.png", x=0.7, y=1.75, maxw=5.7, maxh=3.5)
picture(s, f"{A}/mech_faithful_bar.png", x=6.7, y=1.9, maxw=6.3, maxh=3.35)
bullets(s, [
    "Discrimination is easy; nearly every metric separates knows from absent. The real test is invariance.",
    "On Hubble-8B, only D_mech is both discriminative and paraphrase-stable (shift 0.47, a large reduction but not zero, vs at least 0.95); its composite faithfulness score is about 18x the best lexical metric, and the ranking holds under every aggregation we tried.",
    "It is the DISTANCE construction that wins: naive min-rank, also mechanistic, fails paraphrase stability.",
], y=5.6, space=7)

# ===================================================================================================
# PART V-b - ROBUSTNESS AND A SECOND BENCHMARK
# ===================================================================================================
divider("Part V (cont.)", "How solid is the result?", "aggregation robustness, a second benchmark, and confidence intervals")

# R1 AGGREGATION ROBUSTNESS
s = slide(); header(s, "9 · The Hubble-8B conclusion is robust to the formula", "D_mech rank (of 11 metrics) under five ways to combine the two effect sizes")
rows = [("benchmark", "product", "additive", "harmonic", "minimum", "constraint"),
        ("Hubble-8B", "1", "1", "1", "1", "1"),
        ("Hubble-1B", "2", "4", "2", "2", "2"),
        ("TOFU-1B", "4", "4", "4", "4", "2"),
        ("TOFU-3B", "5", "5", "5", "5", "3"),
        ("TOFU-8B", "4", "4", "4", "4", "2")]
tablebox(s, rows, 0.9, 1.95, 11.5, [2.6, 1.78, 1.78, 1.78, 1.78, 1.78], fs=14, hfs=13)
bullets(s, [
    "The composite score is one of several reasonable choices. On Hubble-8B, D_mech is rank 1 of 11 under all five, so the 18x gap is a property of the data, not the multiplication.",
    "At every scale and benchmark, D_mech is the best mechanistic metric and beats every standard output metric under every formula. The one exception is the additive score at Hubble-1B.",
], y=4.65, size=BODY - 2, space=7)

# R2 SECOND BENCHMARK (TOFU)
s = slide(); header(s, "9 · A second benchmark: TOFU (1B / 3B / 8B)", "does the result travel beyond Hubble?")
rows = [("scale", "invar_shift", "discrim", "faithful D_mech", "best competitor", "probability"),
        ("TOFU-1B", "0.405", "0.934", "0.556", "0.650 truth_ratio", "0.059"),
        ("TOFU-3B", "0.345", "0.914", "0.599", "0.658 truth_ratio", "0.047"),
        ("TOFU-8B", "0.416", "0.953", "0.557", "0.662 truth_ratio", "0.031")]
tablebox(s, rows, 0.7, 1.95, 11.95, [1.7, 2.15, 1.75, 2.75, 3.0, 1.95], fs=13, hfs=12)
bullets(s, [
    "We rerun the whole meta-evaluation on OpenUnlearning TOFU (author facts, multi-token answers), with its released full (knows) and retain90 (never-knew) checkpoints as anchors.",
    "D_mech is faithful at every scale, the best internal metric, above every generation-overlap and answer-probability metric. It is NOT the single best overall here, since a few likelihood-based aggregates (truth ratio, loss / zlib MIA) are also paraphrase-stable on TOFU.",
    "Why the difference: TOFU rewords the whole question and scores a multi-token answer, so whole-answer aggregates are less exposed to a single-token surface shift than under Hubble's trigger-only rewording.",
], y=4.25, size=BODY - 2, space=6)

# R3 CONFIDENCE + CALIBRATED CLAIM
s = slide(); header(s, "9 · What we can and cannot say", "confidence intervals, and the honest scope of each claim")
tfL = tbox(s, 0.75, 1.85, 6.05, 5.2)
line(tfL, "Bootstrap 95% CIs (resampling facts)", SUBH, BLACK, bold=True, first=True, space=6)
for t in ["Hubble-8B: D_mech faithful 0.442 [0.31, 0.56] vs best lexical 0.025 [0.00, 0.07]. The intervals do NOT overlap: the advantage is real, not a point estimate.",
          "TOFU-8B: D_mech 0.557 [0.46, 0.65] overlaps truth_ratio 0.662 [0.58, 0.73]. Statistically comparable, and both far above probability [0.00, 0.07]."]:
    line(tfL, t, BODY - 1, INK, bullet=True, space=8)
tfR = tbox(s, 7.05, 1.85, 5.75, 5.2)
line(tfR, "The claim, calibrated", SUBH, BLACK, bold=True, first=True, space=6)
for t in ["Erasure to residual internal accessibility, under the readouts applied (an observation, not a causal proof).",
          "Invariant to substantially paraphrase-stable (shift 0.47, not zero).",
          "Only faithful metric on Hubble-8B; on TOFU, the best internal metric and above all standard output metrics.",
          "Causal Jacobian to an interventionally-grounded, first-order readout."]:
    line(tfR, t, BODY - 1, INK, bullet=True, space=8)

# R4 BAND SENSITIVITY + IDK (hardest case)
s = slide(); header(s, "9 · Two more robustness checks", "the layer-band choice, and the hardest case (IDK)")
tfL = tbox(s, 0.75, 1.8, 6.05, 0.7)
line(tfL, "Faithfulness vs the layer band (Hubble-8B)", SUBH, BLACK, bold=True, first=True, space=4)
rows = [("band", "invar_shift", "discrim", "faithful"),
        ("12 to 30", "0.339", "0.418", "0.276"),
        ("18 to 36 (default)", "0.467", "0.827", "0.441"),
        ("24 to 36", "0.698", "0.916", "0.276")]
tablebox(s, rows, 0.75, 2.45, 6.0, [2.5, 1.3, 1.05, 1.15], fs=12, hfs=12)
tfL2 = tbox(s, 0.75, 4.4, 6.05, 2.4)
line(tfL2, "Faithful under every band (0.28 to 0.44, all far above the best lexical 0.025). The a priori top half is the strongest, and no band inverts the result.", BODY - 1, INK, first=True, space=6)
tfR = tbox(s, 7.05, 1.8, 5.75, 5.2)
line(tfR, "IDK: same final rank as never-knew, opposite internal state", SUBH, BLACK, bold=True, first=True, space=6)
for t in ["IDK and the never-knew Standard share final rank 9, so only the trajectory can separate them.",
          "IDK dual-anchor D_mech: median +0.594, on the knows side of zero for 77% of facts. Standard: median -1.363.",
          "In a paired comparison, IDK exceeds Standard (closer to knows) on 100% of facts (Cliff delta +0.997, p = 1e-13). Keep 77% and 100% distinct.",
          "The rank-9 tie is a surface artifact; internally IDK still holds the fact. This answers the sharpest objection to the IDK reading."]:
    line(tfR, t, BODY - 1, INK, bullet=True, space=8)

# ===================================================================================================
# PART VI - VERDICT + CONCLUSION
# ===================================================================================================
divider("Part VI", "Verdict and conclusion", "what the methods actually do, and what we contribute")

# 14 METHOD VERDICT
s = slide(); header(s, "10 · Method verdict across scales", "the general result")
vb = s.shapes.add_shape(1, Inches(0.9), Inches(1.95), Inches(11.5), Inches(1.15))
vb.fill.solid(); vb.fill.fore_color.rgb = PALE; vb.line.color.rgb = DARK; vb.line.width = Pt(1); vb.shadow.inherit = False
vtf = vb.text_frame; vtf.word_wrap = True; vtf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(vtf, "NPO is the most scale-robust unlearner;  GradDiff and RMU work at 8B but not 1B;  IDK is surface at both.",
     19, BLACK, bold=True, first=True, align=PP_ALIGN.CENTER)
bullets(s, [
    ("NPO", "h"),
    "Buries the forgotten fact internally at BOTH scales (deeper at 1B) while preserving retain: genuine, scale-robust selective unlearning.",
    ("GradDiff & RMU", "h"),
    "Bury it at 8B, but at 1B the fact stays accessible: their unlearning does not internalize at the smaller scale.",
    ("IDK", "h"),
    "Leaves the fact accessible at both scales: output redirection, not erasure.",
    "This ordering is invisible at the output; only the internal readout shows which methods remove the knowledge, and where. Full 1B/8B graphs in the appendix.",
], y=3.35, space=7)

# CONCLUSION
s = slide(); header(s, "Conclusion", "Contribution")
tf = tbox(s, 1.0, 2.0, 11.3, 3.6, anchor=MSO_ANCHOR.MIDDLE)
line(tf, "We add a mechanistic-interpretability metric to the Hubble unlearning benchmark", 22, BLACK, first=True, space=8, align=PP_ALIGN.CENTER)
line(tf, "and show, under the field's own faithful analysis, that on Hubble-8B it is the only metric", 22, BLACK, space=8, align=PP_ALIGN.CENTER)
line(tf, "that is both knowledge-discriminative AND paraphrase-stable,", 22, BLACK, bold=True, space=8, align=PP_ALIGN.CENTER)
line(tf, "and that it stays faithful across a second benchmark (TOFU) and three scales,", 22, BLACK, space=8, align=PP_ALIGN.CENTER)
line(tf, "empirically confirming that output and lexical metrics confuse surface form with knowledge.", 22, INK, space=8, align=PP_ALIGN.CENTER)
line(tf, "Along the way: a corrected logit-lens readout, all three lenses characterized (logit, tuned, Jacobian), "
         "and a scale-dependent method verdict.", BODY, GRAY, italic=True, space=0, align=PP_ALIGN.CENTER)

# FUTURE WORK
s = slide(); header(s, "Future work", "remaining directions")
bullets(s, [
    ("A larger averaged-Jacobian transport", "h"),
    "Our interventionally-grounded lens averages over a modest reference set. The full Anthropic (2026) pipeline (large data, their normalization) should sharpen it, and may recover the knows-vs-never-knew separation at 1B.",
    ("From observation to causation", "h"),
    "Causal patching or activation steering of the suppressed representation, to turn \"suppression is not erasure\" from an observation into a causal intervention result.",
    ("Robustness of the faithful metric", "h"),
    "Multi-paraphrase and 1B robustness of D_mech, more wrong-target realizations, and larger fact sets.",
], space=10)

# ===================================================================================================
# APPENDIX - ALL RESULTS
# ===================================================================================================
divider("Appendix", "All results", "three lenses, logit-lens tables, and 1B / 8B scaled results")

# A - LOGIT LENS SCALED
fig_slide("Appendix A1 · Logit lens (8B)", "Rank-by-layer, forget vs retain, six models (8B)",
          "logitlens_rank_median_8b.png", maxw=10.2, maxh=5.2,
          cap="Left panel forget, right panel retain. Lower rank = more recoverable. Corrected, boundary-safe logit lens.")
fig_slide("Appendix A2 · Logit lens (1B)", "Rank-by-layer, forget vs retain, six models (1B)",
          "logitlens_rank_median_1b.png", maxw=10.2, maxh=5.2,
          cap="At 1B only NPO drives the fact deep; GradDiff, IDK, RMU leave it accessible. The forget signature is scale-dependent.")
two_fig("Appendix A3 · Cross-scale", "Same method, different scale: 1B (left) vs 8B (right)",
        "logitlens_rank_median_1b.png", "logitlens_rank_median_8b.png",
        cap="Retain is preserved at both scales; the forget behavior changes with scale (GradDiff/RMU internalize only at 8B).")
fig_slide("Appendix A4 · Output view (1B / 8B)", "Infill accuracy by method and format",
          "infill_bars_8b.png", maxw=9.0, maxh=3.4, y=1.85,
          cap="8B shown; 1B in the data. At the output every method looks the same, motivating the internal readout.")

# B - LOGIT LENS TABLES
fig_slide("Appendix B1 · Logit-lens table", "Single-model emergence, Perturbed (knows)",
          "lenstable_single_0.png", maxw=6.6, maxh=5.3,
          cap="The gold month climbs to the top over the final layers: what genuine knowledge looks like, layer by layer.")
fig_slide("Appendix B2 · Logit-lens table", "Knows vs unlearned, same fact (Perturbed vs NPO)",
          "lenstable_contrast_0.png", maxw=9.6, maxh=5.3,
          cap="Perturbed converges to the answer; NPO never surfaces it. Suppression, not erasure.")
fig_slide("Appendix B3 · Logit-lens table", "Confidence is not knowledge",
          "lenstable_confidence_contradiction.png", maxw=9.2, maxh=4.6,
          cap="Three unlearned models emit a wrong token at about 100% confidence, indistinguishable from knowing to a probability metric.")
fig_slide("Appendix B4 · Logit vs tuned", "The tuned lens skips ahead (lens-table view)",
          "lenstable_logit_vs_tuned.png", maxw=9.6, maxh=5.2,
          cap="The tuned lens surfaces the answer earlier than the model computes it, hiding intermediate computation.")

# C - THREE LENSES
fig_slide("Appendix C1 · Three lenses (8B)", "logit vs tuned vs Jacobian, same 44 facts",
          "lens_compare_8b.png", maxw=11.6, maxh=5.3,
          cap="Tuned (dashed) skips ahead for every model; logit (dotted) shows emergence; Jacobian (solid) is smooth and causal.")
fig_slide("Appendix C2 · Three lenses (1B)", "logit vs tuned vs Jacobian, same facts (1B)",
          "lens_compare_1b.png", maxw=11.6, maxh=5.3,
          cap="At 1B the picture is consistent; the knows-vs-never-knew gap is weaker (the 1B base model barely memorized).")
two_fig("Appendix C3 · Jacobian lens (8B)", "Rank-by-layer (left) and knowledge-vs-prior control (right)",
        "jlens_affine_rank_8b.png", "jlens_affine_control_8b.png",
        cap="Causal readout reproduces the taxonomy (Perturbed to 1, Standard to 8) and shows NPO/GradDiff late-layer suppression.")
two_fig("Appendix C4 · Jacobian lens (1B)", "Rank-by-layer (left) and control (right)",
        "jlens_affine_rank_1b.png", "jlens_affine_control_1b.png",
        cap="At 1B the Jacobian lens is faithful in direction (knows < prior < buried) but does not separate knows from never-knew.")

# D - FAITHFULNESS EXTRAS
two_fig("Appendix D1 · Faithfulness", "Invariance-discrimination scatter (left) and faithfulness bar (right)",
        "mech_invar_discrim_scatter.png", "mech_faithful_bar.png",
        cap="Only D_mech sits in the faithful corner (low invariance shift, high discrimination); its faithfulness is about 18x the best lexical metric.")
fig_slide("Appendix D2 · Paraphrase robustness", "D_mech vs lexical metrics under multiple rewordings (8B)",
          "mech_paraphrase_robustness_8b.png", maxw=9.6, maxh=4.6,
          cap="Across several meaning-preserving rewordings, D_mech's invariance shift stays lowest; lexical metrics swing. "
              "(The single-fact showcase is now in the main invariance discussion, Part V.)")

out = "Hubble_unlearning_thesis_full.pptx"
prs.save(out)
print("saved", out, "with", len(prs.slides._sldIdLst), "slides")
