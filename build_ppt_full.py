"""Full thesis-narrative deck (v2): Hubble unlearning + mechanistic-interpretability faithfulness.
Plain academic style: Times New Roman, grayscale, consistent heading/body sizes. No tuned lens."""
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


def slide(): return prs.slides.add_slide(BLANK)


def tbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    return tf


def line(tf, text, size, color=INK, bold=False, italic=False, first=False, bullet=False,
         space=6, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space)
    r = p.add_run(); r.text = ("–  " + text) if bullet else text
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color; f.name = FONT
    return p


def header(s, eyebrow, title):
    tf = tbox(s, 0.7, 0.42, 12.0, 0.5); line(tf, eyebrow.upper(), EYE, GRAY, bold=True, first=True, space=2)
    tf2 = tbox(s, 0.7, 0.78, 12.0, 0.9); line(tf2, title, HEAD, BLACK, bold=True, first=True)
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
            c = t.cell(i, j); c.text = str(val); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = Pt(2); c.margin_bottom = Pt(2)
            p = c.text_frame.paragraphs[0]; run = p.runs[0]; run.font.name = FONT
            if header_row and i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = DARK
                run.font.size = Pt(hfs); run.font.bold = True; run.font.color.rgb = WHITE
            else:
                c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else ALT
                run.font.size = Pt(fs); run.font.color.rgb = INK; run.font.bold = (j == 0)
    return t


# =========================== TITLE ===========================
s = slide()
r = s.shapes.add_shape(1, 0, Inches(2.7), SW, Pt(2)); r.fill.solid(); r.fill.fore_color.rgb = BLACK
r.line.fill.background(); r.shadow.inherit = False
tf = tbox(s, 1.0, 1.5, 11.3, 1.3); line(tf, "Does an Unlearned Model Truly Forget?", TITLE, BLACK, bold=True, first=True, space=4)
tf2 = tbox(s, 1.0, 2.95, 11.3, 2.0)
line(tf2, "Probing knowledge suppression vs. erasure in the Hubble unlearning benchmark", SUB, INK, first=True, space=8)
line(tf2, "with logit-lens mechanistic interpretability — and a faithful metric derived from it", SUB, INK, space=8)
tf3 = tbox(s, 1.0, 6.6, 11.3, 0.6)
line(tf3, "Master's thesis  ·  Hubble-1B / 8B  ·  mechanistic interpretability of machine unlearning", CAP, GRAY, first=True)

# =========================== ROADMAP ===========================
s = slide(); header(s, "Roadmap", "What this talk covers")
left = ["1.  Introduction — machine unlearning & the verification problem",
        "2.  The Hubble benchmark, and the unlearning methods",
        "3.  Experiment setup & the infill probe",
        "4.  Memorization signals at the output",
        "5.  Logit lens & mechanistic interpretability",
        "6.  The main argument: suppression vs. erasure"]
right = ["7.  Reading Hubble models layer-by-layer (and getting it right)",
         "8.  Results, and the method verdict across scales",
         "9.  Deriving a metric from the logit lens (D_mech)",
         "10. Faithfulness: what it means, and how we test it",
         "11. The tests, the intuition, the assumptions",
         "12. Comparison against the lexical metrics"]
tfL = tbox(s, 0.9, 2.0, 6.0, 5.0)
for i, t in enumerate(left): line(tfL, t, BODY, INK, first=(i == 0), space=12)
tfR = tbox(s, 7.0, 2.0, 6.0, 5.0)
for i, t in enumerate(right): line(tfR, t, BODY, INK, first=(i == 0), space=12)

# =========================== 1 INTRO ===========================
s = slide(); header(s, "1 · Introduction", "Machine unlearning — and why verifying it is hard")
bullets(s, [
    ("What it is", "h"),
    "Machine unlearning makes a trained model forget specific data — a person's private details, copyrighted text, hazardous knowledge — without retraining from scratch.",
    ("Why it matters", "h"),
    "Privacy and the \"right to be forgotten\" (GDPR), copyright takedowns, and safety all demand that models can remove what they learned.",
    ("The hard question — and the thesis", "h"),
    "Standard practice checks the OUTPUT: did the model stop saying the fact? But a model can look forgotten while still holding the knowledge inside.",
    "This thesis asks: can we tell genuine erasure from mere output suppression — by looking inside the network?",
])

# =========================== 2 HUBBLE ===========================
s = slide(); header(s, "2 · The Hubble benchmark", "Models with KNOWN knowledge states")
bullets(s, [
    "Hubble is an open suite of LLMs (1B and 8B) pretrained with a controlled set of facts inserted — so we know exactly what each model should and should not know.",
    ("Two reference models per fact — our ground truth", "h"),
    "Standard — never trained on the fact → the \"perfectly forgotten\" target (lower bound).",
    "Perturbed — trained to memorize the fact → \"knows it\" (upper bound).",
    ("Data splits", "h"),
    "Unlearn (to forget) · Keep (to retain) · Unseen (never seen). Facts are fictional people with attributes such as birthdate and nationality.",
    "Why it is powerful: unlearning can be measured against known ground truth, not guesswork.",
])

# =========================== 2 METHODS ===========================
s = slide(); header(s, "2 · Unlearning methods", "Four families we study")
rows = [("NPO", "Negative Preference Optimization — gently pushes down the probability of the forget answer (bounded, to avoid model collapse)."),
        ("GradDiff", "Gradient Difference — ascends the loss on forget data while descending it on retain data."),
        ("IDK", "\"I don't know\" — fine-tunes the model to answer \"I don't know\" to forget questions (redirects the output)."),
        ("RMU", "Representation Misdirection — scrambles the model's internal activations for the forget concept at chosen layers.")]
t = tablebox(s, [("method", "what it does")] + rows, 0.75, 2.05, 11.8, [2.1, 9.7], fs=BODY, hfs=BODY)
tf = tbox(s, 0.75, 5.95, 11.8, 0.9)
line(tf, "We tune each method with a \"keep-as-retain\" setup so unlearning is selective — forget the target, preserve everything else.", BODY, GRAY, italic=True, first=True)

# =========================== 3 SETUP ===========================
s = slide(); header(s, "3 · Experiment setup", "Two lenses on the same six models")
bullets(s, [
    "For every model — Standard, Perturbed, and the four unlearned (NPO, GradDiff, IDK, RMU) — we ask: can the forgotten attribute still be recovered?",
    ("We look through two lenses", "h"),
    "Output behaviour — infill accuracy: does the model still produce the fact? (what everyone measures)",
    "Internal computation — logit lens: is the fact still represented inside, even if the output hides it? (our addition)",
    ("Anchored by ground truth", "h"),
    "Every reading is bracketed by Standard (never knew) and Perturbed (knows), so scores are interpretable.",
])

# =========================== 3 INFILL PROBE ===========================
s = slide(); header(s, "3 · The infill probe", "How we ask a model for a fact")
tf = tbox(s, 0.75, 1.9, 11.8, 0.5); line(tf, "We give the model a biography that stops right before the sensitive attribute:", BODY, INK, first=True)
pbox = s.shapes.add_shape(1, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.1))
pbox.fill.solid(); pbox.fill.fore_color.rgb = ALT; pbox.line.color.rgb = DARK; pbox.line.width = Pt(1); pbox.shadow.inherit = False
ptf = pbox.text_frame; ptf.word_wrap = True; ptf.margin_left = Inches(0.2); ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(ptf, "\"Billy Fisher is from United States. Billy Fisher was born in Wheeling, West Virginia. "
          "Billy Fisher is an alumni of the American Academy of Dramatic Arts. Billy Fisher was born on ___\"",
     BODY, INK, italic=True, first=True)
bullets(s, [
    "Correct answer: \"May 16, 1988\". Following Hubble's own loss-based-choice protocol, we score the true birthdate against 9 distractors (10 candidates) and take the highest-likelihood option — this is infill accuracy.",
    "(Hubble also runs a generative exact-match evaluation; our later metric comparison covers that mode too, via ROUGE and exact-memorization.)",
    "Four prompt formats give more or less context: name-only, intro-prefix, full-prefix, full-prefix + suffix.",
], y=3.8, space=10)

# =========================== 4 MEMORIZATION ===========================
s = slide(); header(s, "4 · Memorization at the output", "Infill accuracy by method and prompt format (8B)")
picture(s, f"{A}/infill_bars_8b.png", y=1.85, maxw=9.4, maxh=4.15)
bullets(s, [
    "Perturbed (knows) scores highest; every unlearned method sits near the Standard never-knew floor — and they all look about the same.",
    "At the output, unlearning \"works,\" and no single number tells the four methods apart.",
], y=6.2, space=6)

# =========================== 5 LOGIT LENS ===========================
s = slide(); header(s, "5 · Logit lens & mechanistic interpretability", "Reading the model's mind, layer by layer")
bullets(s, [
    "Mechanistic interpretability studies what happens INSIDE a network, not just its output.",
    "A transformer refines its next-word guess across its layers (36 at 8B) in a running \"residual stream.\"",
    ("The logit lens", "h"),
    "Apply the model's own output head to an INTERMEDIATE layer — see the guess it would make at that depth.",
    "Record the RANK of the correct answer token at each layer: rank 1 = top guess; rank in the thousands = buried.",
    "This yields a trajectory — the fate of the fact as it flows through the network.",
])

# =========================== 6 MAIN ARGUMENT ===========================
s = slide(); header(s, "6 · The main argument", "Suppression is not erasure")
bullets(s, [
    ("Two different questions", "h"),
    "An output metric answers: did the model SAY the fact?",
    "What we actually care about: does the model still HAVE the fact?",
    ("Why the gap matters", "h"),
    "A fact can be suppressed at the output while its internal representation stays intact — and can be revived by rewording, quantization, or a few fine-tuning steps. That is not real forgetting.",
    "The logit lens can see this directly: if the token is buried at the output but accessible mid-stack, the unlearning is only skin-deep.",
])

# =========================== 7 INTEGRATION ===========================
s = slide(); header(s, "7 · Integrating the logit lens into Hubble", "Our mechanistic addition to the benchmark")
bullets(s, [
    "We run the rank-by-layer logit lens on all six Hubble models, on both the forget and the retain sets, taking the median over facts.",
    "Every trajectory is anchored between Standard (never knew) and Perturbed (knows).",
    "This turns Hubble — which ships only the checkpoints — into a testbed for INTERNAL unlearning metrics, not just output metrics.",
    ("Getting the reading right mattered", "h"),
    "Two subtle bugs first had to be fixed (next slide) — without them the ranks are inflated and spike at the last layer, hiding the real story.",
])

# =========================== 7b CORRECTNESS ===========================
s = slide(); header(s, "7 · A note on correctness", "Two fixes that changed the numbers sharply")
bullets(s, [
    ("1. Double-normalization (a fake final-layer spike)", "h"),
    "The model's last hidden state is already layer-normalized; the naive lens normalized it again, distorting the final layer. Fix: use the model's true output logits there.",
    ("2. Boundary tokenization (inflated ranks everywhere)", "h"),
    "A trailing space in the prompt made us track the wrong answer token ( 'July' instead of ' July' ) — the one the model does NOT predict. Fix: strip and re-join so we track the token the model actually produces.",
    ("Effect", "h"),
    "Mid-stack ranks dropped ~5–40× (e.g. IDK 385 → 9), the false final-layer spike vanished, and the 1B story sharpened — NPO's genuine deep burial became visible. All results here use the corrected readout.",
], space=9)

# =========================== 8 RESULTS 8B ===========================
s = slide(); header(s, "8 · Results", "The output said \"all the same\" — the layers say otherwise (8B)")
picture(s, f"{A}/logitlens_rank_median_8b.png", y=1.75, maxw=9.0, maxh=3.9)
bullets(s, [
    "IDK keeps the token accessible even at the output (surface) · NPO & GradDiff keep it accessible mid-stack then suppress it late · RMU buries it deepest mid-stack.",
    "One output verdict; different internal stories.",
    "No method achieves true internal erasure — even the deepest leaves recoverable traces. Unlearning MASKS the fact; it does not delete it.",
], y=5.55, space=7)

# =========================== 8b SCALES ===========================
s = slide(); header(s, "8 · Across scales", "The same method behaves differently at 1B vs. 8B")
picture(s, f"{A}/logitlens_rank_median_1b.png", x=0.5, y=1.8, maxw=6.2, maxh=3.25)
picture(s, f"{A}/logitlens_rank_median_8b.png", x=6.75, y=1.8, maxw=6.2, maxh=3.25)
caption(s, "Left: 1B.   Right: 8B.   Forget (left panel of each) vs. retain (right panel of each).   Identical boundary-safe logit lens.", y=5.15)
bullets(s, [
    "Retain is preserved at both scales — every method keeps the retained facts recoverable (low rank on Keep).",
    "The forget signature is strongly scale-dependent: at 1B only NPO drives the fact deep (output rank ≈ 44,000); GradDiff, IDK, RMU leave it accessible (rank 8–17). At 8B, NPO / GradDiff / RMU bury it while IDK stays surface.",
], y=5.45, space=7)

# =========================== 8c METHOD VERDICT ===========================
s = slide(); header(s, "8 · Method verdict", "The general result across scales")
vb = s.shapes.add_shape(1, Inches(0.9), Inches(1.95), Inches(11.5), Inches(1.15))
vb.fill.solid(); vb.fill.fore_color.rgb = PALE; vb.line.color.rgb = DARK; vb.line.width = Pt(1); vb.shadow.inherit = False
vtf = vb.text_frame; vtf.word_wrap = True; vtf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(vtf, "NPO is the most scale-robust unlearner;  GradDiff and RMU work at 8B but not 1B;  IDK is surface at both.",
     19, BLACK, bold=True, first=True, align=PP_ALIGN.CENTER)
bullets(s, [
    ("NPO", "h"),
    "Buries the forgotten fact internally at BOTH scales (deeper at 1B, output rank ≈ 44,000) while preserving retain — genuine, scale-robust selective unlearning.",
    ("GradDiff & RMU", "h"),
    "Bury it at 8B, but at 1B the fact stays accessible (rank ≈ 8–9): their unlearning does not internalize at the smaller scale.",
    ("IDK", "h"),
    "Leaves the fact accessible at both scales (rank ≈ 9–17) — output redirection, not erasure.",
    "This ordering is invisible at the output; only the internal readout shows which methods actually remove the knowledge, and where.",
], y=3.35, space=8)

# =========================== 9 METRIC ===========================
s = slide(); header(s, "9 · A metric from the logit lens", "D_mech — a dual-anchor distance")
tf = tbox(s, 0.75, 1.9, 11.8, 1.1)
line(tf, "Each fact gives a rank trajectory r = (r0 … r36). With two anchor trajectories —", BODY, INK, first=True, space=4)
line(tf, "K = Perturbed (knows)  and  Ø = Standard (never knew) — we define:", BODY, INK, space=4)
fb = s.shapes.add_shape(1, Inches(1.3), Inches(3.05), Inches(10.7), Inches(0.9))
fb.fill.solid(); fb.fill.fore_color.rgb = ALT; fb.line.color.rgb = DARK; fb.line.width = Pt(1); fb.shadow.inherit = False
ftf = fb.text_frame; ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
line(ftf, "D_mech  =  mean | log r − log Ø |  −  mean | log r − log K |     (over layers 18–36)", 20, BLACK, bold=True, first=True, align=PP_ALIGN.CENTER)
bullets(s, [
    "D_mech > 0 means the internal trajectory looks like it KNOWS — regardless of what the output says.",
    "Anchors are leave-one-fact-out medians, so a fact is never compared to itself.",
    "It is a RELATIVE distance — \"closer to knows, or to never-knew?\" — which is the key to the result on the next slides.",
], y=4.2, space=11)

# =========================== 9b READING D_mech ===========================
s = slide(); header(s, "9 · Reading D_mech", "Why the log — and why it survives paraphrasing")
bullets(s, [
    ("The log — ranks live on a multiplicative scale", "h"),
    "Ranks span five orders of magnitude (1 → 50,000). A raw gap |5000 − 4000| would swamp |1 − 10| — yet rank 1 vs 10 is hugely meaningful (top guess vs 10th), while 4000 vs 5000 is meaningless (both hopelessly buried).",
    "log turns the distance into a RATIO:  |log r − log a| = |log(r / a)|  = \"how many factors apart.\"  Rank 1→10 counts ~10× more than 4000→5000 — matching how rank quality actually behaves. It also makes layers comparable, so the band average is not dominated by one buried layer.",
    ("The two anchors — a signed \"which template?\" score", "h"),
    "d(r, Ø) − d(r, K) asks: is the trajectory closer to the KNOWS template or the NEVER-KNEW template? Positive = looks-known; zero = equidistant.",
    ("Why it survives paraphrasing", "h"),
    "Rewording nudges the trajectory's distance to BOTH anchors by about the same amount — and the subtraction cancels that common shift, leaving only the knowledge signal.",
], space=9)

# =========================== 10 FAITHFULNESS: MEANING + WORKED EXAMPLE ===========================
s = slide(); header(s, "10 · Faithfulness — is the metric itself trustworthy?", "two criteria, shown by example")
tf = tbox(s, 0.75, 1.75, 11.8, 1.15)
line(tf, "A metric is faithful only if it measures KNOWLEDGE, not surface form. Two criteria:", BODY, INK, first=True, space=6)
line(tf, "Discrimination — models that KNOW vs models that LACK the fact must get different scores.", BODY, INK, bullet=True, space=4)
line(tf, "Semantic Invariance — reword the question (knowledge unchanged) and the score must NOT move. This is the hard one.", BODY, INK, bullet=True, space=4)
picture(s, f"{A}/mech_worked_example.png", y=3.0, maxw=10.6, maxh=3.4)
caption(s, "Same knowledge, reworded prompt: probability collapses 0.95 → 0.16 (NOT invariant); D_mech holds 1.21 → 0.92 (invariant). "
           "Both separate P from N — so invariance is the deciding test.", y=6.55)

# =========================== 10b TESTS + ASSUMPTIONS ===========================
s = slide(); header(s, "10 · How we test it", "the pools, the tests, and our assumptions")
tfL = tbox(s, 0.75, 1.8, 6.05, 5.3)
line(tfL, "Four pools (n = 44 facts each)", SUBH, BLACK, bold=True, first=True, space=6)
line(tfL, "Positive P (knows): Full + Paraphrased.  Negative N (absent): Retain + Perturbed-with-wrong-answer.", BODY, INK, space=10)
line(tfL, "The four tests", SUBH, BLACK, bold=True, space=6)
for t in ["Wilcoxon signed-rank (within P, Full vs Paraphrased, paired): did rewording shift the metric?",
          "TOST equivalence: are Full and Paraphrased statistically the SAME (within ±ε)? — absence of a shift, not just failure to find one.",
          "Mann–Whitney U (P vs N): does the metric separate knows from absent?",
          "Cliff's delta (effect size): scale-free magnitude — |δ(Full,Para)| = invariance shift (0 = invariant); |δ(P,N)| = discrimination (1 = perfect)."]:
    line(tfL, t, BODY - 1, INK, bullet=True, space=7)
tfR = tbox(s, 7.05, 1.8, 5.75, 5.3)
line(tfR, "Assumptions (n is small)", SUBH, BLACK, bold=True, first=True, space=6)
for t in ["Non-parametric throughout — the metrics are bounded and skewed, so no normality is assumed.",
          "KEY: at n = 44 the Wilcoxon p-value SATURATES — even a tiny but consistent shift reads as \"significant,\" so it flags every metric.",
          "So we judge invariance by EFFECT SIZE (Cliff's delta), not the p < 0.05 threshold; p-values are reported only for completeness.",
          "One paraphrase and one wrong-target realization; facts treated as independent; anchors are leave-one-out medians."]:
    line(tfR, t, BODY - 1, INK, bullet=True, space=8)

# =========================== 10c BOX PLOTS ===========================
s = slide(); header(s, "10 · Faithfulness result", "every metric's distribution across the four pools")
picture(s, f"{A}/mech_metaeval_fig4.png", y=1.75, maxw=9.8, maxh=3.9)
caption(s, "Dashed line separates Positive P | Negative N. A faithful metric keeps Full ≈ Paraphrased (left pair level) and shows a P–N gap. "
           "D_mech does both; probability / rouge / extraction-strength let the Paraphrased cell collapse toward the negatives.", y=5.85)

# =========================== 11 COLUMNS EXPLAINED ===========================
s = slide(); header(s, "11 · Reading the comparison table", "what the two columns mean")
rows = [("metric", "Full (P)", "Paraphrased (P)", "invar_shift ↓", "discrim ↑", "faithful"),
        ("D_mech", "1.21", "0.92", "0.466", "0.827", "yes"),
        ("probability", "0.95", "0.16", "0.999", "0.928", "no"),
        ("rouge", "0.57", "0.00", "0.975", "0.494", "no")]
tablebox(s, rows, 0.75, 1.95, 11.8, [2.4, 1.8, 2.3, 2.1, 1.7, 1.5], fs=14, hfs=13)
bullets(s, [
    ("invar_shift — how much the score moves when you only REWORD (0 = invariant, lower is better)", "h"),
    "|Cliff's δ| between Full and Paraphrased. probability moves almost completely (0.999: 0.95 → 0.16); D_mech barely moves (0.466: 1.21 → 0.92).",
    ("discrim — how cleanly KNOWS separates from ABSENT (1 = perfect, higher is better)", "h"),
    "|Cliff's δ| between the P pool and the N pool. Most metrics score high here — discrimination is easy.",
    "A metric is faithful only if it has BOTH: low invar_shift AND high discrim. Only D_mech does.",
], y=3.75, space=8)

# =========================== 12 COMPARISON ===========================
s = slide(); header(s, "12 · Comparison with lexical metrics", "only the mechanistic metric is faithful")
picture(s, f"{A}/mech_invar_discrim_scatter.png", x=0.7, y=1.75, maxw=5.7, maxh=3.5)
picture(s, f"{A}/mech_faithful_bar.png", x=6.7, y=1.9, maxw=6.3, maxh=3.35)
bullets(s, [
    "Discrimination is easy — nearly every metric separates knows from absent. The real test is invariance.",
    "Only D_mech stays invariant under paraphrase (shift 0.47 vs. ≥ 0.95); its faithfulness score is ~18× the best lexical metric.",
    "It is the DISTANCE construction that wins — naive min-rank, also mechanistic, fails invariance.",
], y=5.6, space=7)

# =========================== CONCLUSION ===========================
s = slide(); header(s, "Conclusion", "Contribution")
tf = tbox(s, 1.0, 2.1, 11.3, 3.6, anchor=MSO_ANCHOR.MIDDLE)
line(tf, "We add a mechanistic-interpretability metric to the Hubble unlearning benchmark", 22, BLACK, first=True, space=8, align=PP_ALIGN.CENTER)
line(tf, "and show, under the field's own meta-evaluation, that it is the only metric", 22, BLACK, space=8, align=PP_ALIGN.CENTER)
line(tf, "that is both knowledge-discriminative AND paraphrase-invariant —", 22, BLACK, bold=True, space=8, align=PP_ALIGN.CENTER)
line(tf, "empirically confirming that output and lexical metrics confuse surface form with knowledge.", 22, INK, space=8, align=PP_ALIGN.CENTER)
line(tf, "Along the way: a corrected logit-lens readout, and a scale-dependent method verdict "
         "(NPO most scale-robust; GradDiff/RMU work at 8B not 1B; IDK surface at both).", BODY, GRAY, italic=True, space=0, align=PP_ALIGN.CENTER)

out = "Hubble_unlearning_thesis_full.pptx"
prs.save(out)
print("saved", out, "with", len(prs.slides._sldIdLst), "slides")
