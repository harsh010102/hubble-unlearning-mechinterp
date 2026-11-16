"""NXP deck (directors / VPs, 20 min). Responsible-AI-safety framing, delivered answer-first,
with the spine: today's metrics read the model's TEXT (surface) vs D_mech measures UNDER THE HOOD
(internal representation). Assertion-evidence: sentence headline + one big visual per slide.
Built on the real TUM template via tum_deck.Deck."""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from tum_deck import (Deck, TUM_BLUE, TUM_DARK, TUM_D2, TUM_MED, TUM_LT,
                      TUM_GREEN, TUM_ORANGE, TUM_BEIGE, BLACK, WHITE, GRAY, FONT, LEFT, CW)

A = "artifacts"
SECTIONS = ["The problem", "The check", "Does it work", "In practice", "Takeaway"]


def _rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line; b.line.width = Pt(1.5)
    el = b._element
    from pptx.oxml.ns import qn
    st = el.find(qn("p:style"))
    if st is not None:
        el.remove(st)
    b.shadow.inherit = False
    return b


def _text(s, x, y, w, h, runs, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, wrap=True):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = wrap; tf.vertical_anchor = anchor
    for i, (txt, size, color, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
    return tf


def _label_in(shape, runs, align=PP_ALIGN.CENTER):
    tf = shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (txt, size, color, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT


# ---------- concept diagrams ----------
def draw_suppression_vs_erasure(d, s):
    """Two panels: erased (fact gone) vs suppressed (fact still there, hidden)."""
    y0, ph, pw = 1.75, 3.0, 4.1
    # ERASURE panel (green, good)
    _rect(s, 0.55, y0, pw, ph, WHITE, TUM_GREEN)
    _text(s, 0.55, y0 + 0.12, pw, 0.4, [("Erasure  (ideal / retrained)", 15, TUM_GREEN, True)])
    _rect(s, 1.05, y0 + 0.85, pw - 1.0, 0.95, RGBColor(0xEC, 0xF0, 0xDA), TUM_GREEN)
    _text(s, 1.05, y0 + 0.85, pw - 1.0, 0.95,
          [("birthdate:  removed", 15, GRAY, True), ("the fact is not in the weights", 11, GRAY, False)])
    _text(s, 0.55, y0 + ph - 0.62, pw, 0.5,
          [("Answer cannot be revived", 13, TUM_GREEN, True)])
    # SUPPRESSION panel (orange, warning)
    x1 = 5.35
    _rect(s, x1, y0, pw, ph, WHITE, TUM_ORANGE)
    _text(s, x1, y0 + 0.12, pw, 0.4, [("Suppression  (what unlearning does)", 15, TUM_ORANGE, True)])
    _rect(s, x1 + 0.5, y0 + 0.85, pw - 1.0, 0.95, RGBColor(0xFB, 0xEC, 0xDF), TUM_ORANGE)
    _text(s, x1 + 0.5, y0 + 0.85, pw - 1.0, 0.95,
          [("birthdate:  1987  (hidden)", 15, TUM_ORANGE, True),
           ("still in the weights, masked at output", 11, GRAY, False)])
    _text(s, x1, y0 + ph - 0.62, pw, 0.5,
          [("Quantize, relearn, or probe -> it returns", 13, TUM_ORANGE, True)])


def draw_dual_anchor(d, s):
    """Number line between two anchors; the unlearned model sits closer to 'knows'."""
    y = 3.05
    cx0, cx1 = 1.6, 8.4
    conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx0), Inches(y), Inches(cx1), Inches(y))
    conn.line.color.rgb = TUM_DARK; conn.line.width = Pt(2.25)
    # anchors
    for cx, col, top, bot in [
        (cx0, TUM_D2, "Pre-unlearning model", "knows the fact"),
        (cx1, TUM_GREEN, "Retrained model", "never learned it")]:
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.12), Inches(y - 0.12), Inches(0.24), Inches(0.24))
        o.fill.solid(); o.fill.fore_color.rgb = col; o.line.fill.background(); o.shadow.inherit = False
        _text(s, cx - 1.4, y + 0.2, 2.8, 0.6, [(top, 13, col, True), (bot, 11, GRAY, False)])
    # the unlearned model marker (closer to pre-unlearning -> still knows)
    mx = cx0 + (cx1 - cx0) * 0.30
    tri = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(mx - 0.14), Inches(y - 0.42), Inches(0.28), Inches(0.3))
    tri.rotation = 180; tri.fill.solid(); tri.fill.fore_color.rgb = TUM_ORANGE
    tri.line.fill.background(); tri.shadow.inherit = False
    _text(s, mx - 1.4, y - 1.15, 2.8, 0.6,
          [("Unlearned model", 13, TUM_ORANGE, True), ("(internal representation)", 11, GRAY, False)])
    _text(s, LEFT, y + 1.15, CW, 0.5,
          [("D_mech  =  is the fact closer to 'knows' or to 'never learned'?  Closer to 'knows' means it was only hidden.",
            13, BLACK, True)], align=PP_ALIGN.CENTER)


def why_it_matters(d, s):
    cols = [("Privacy", "Regulators can order data removed (GDPR right to erasure)", TUM_BLUE),
            ("Copyright", "Owners can demand a model unlearn protected content", TUM_D2),
            ("AI safety", "Dangerous know-how must be genuinely removed, not hidden", TUM_ORANGE)]
    w = 2.9; gap = 0.28; x = LEFT
    for name, body, col in cols:
        _rect(s, x, 1.9, w, 2.5, WHITE, col)
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + w/2 - 0.28), Inches(2.15), Inches(0.56), Inches(0.56))
        o.fill.solid(); o.fill.fore_color.rgb = col; o.line.fill.background(); o.shadow.inherit = False
        _text(s, x, 2.95, w, 0.45, [(name, 16, col, True)])
        _text(s, x + 0.2, 3.45, w - 0.4, 0.9, [(body, 12.5, BLACK, False)], anchor=MSO_ANCHOR.TOP)
        x += w + gap


def generalization_table(d, s):
    rows = [("Model family / size", "D_mech faithful?", "Best internal metric?"),
            ("Llama-3.2  1B", "yes", "yes"),
            ("Llama-3.2  3B", "yes", "yes"),
            ("Llama-3.1  8B", "yes", "yes")]
    from pptx.util import Inches as IN
    tbl = s.shapes.add_table(len(rows), 3, IN(1.6), IN(2.0), IN(6.8), IN(2.2)).table
    tbl.columns[0].width = IN(3.0); tbl.columns[1].width = IN(1.9); tbl.columns[2].width = IN(1.9)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = tbl.cell(ri, ci); c.text = ""
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if ci else PP_ALIGN.LEFT
            r = p.add_run(); r.text = val
            r.font.name = FONT; r.font.size = Pt(14 if ri else 13)
            r.font.bold = (ri == 0)
            r.font.color.rgb = WHITE if ri == 0 else (TUM_GREEN if val == "yes" else BLACK)
            c.fill.solid()
            c.fill.fore_color.rgb = TUM_BLUE if ri == 0 else (WHITE if ri % 2 else RGBColor(0xEE, 0xF3, 0xF9))


# ---------- business helpers ----------
def _blockarrow(s, x, y, w, h, color, shape=MSO_SHAPE.RIGHT_ARROW):
    a = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background()
    from pptx.oxml.ns import qn
    st = a._element.find(qn("p:style"))
    if st is not None:
        a._element.remove(st)
    a.shadow.inherit = False
    return a


def _cards(d, s, cols):
    w = 2.9; gap = 0.28; x = LEFT
    for name, body, col in cols:
        _rect(s, x, 1.9, w, 2.5, WHITE, col)
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + w / 2 - 0.28), Inches(2.15), Inches(0.56), Inches(0.56))
        o.fill.solid(); o.fill.fore_color.rgb = col; o.line.fill.background(); o.shadow.inherit = False
        _text(s, x, 2.95, w, 0.45, [(name, 15.5, col, True)])
        _text(s, x + 0.2, 3.42, w - 0.4, 0.95, [(body, 12, BLACK, False)], anchor=MSO_ANCHOR.TOP)
        x += w + gap


def stakes_cards(d, s):
    _cards(d, s, [
        ("Data governance", "GDPR right-to-erasure and the EU AI Act: remove data on demand, and prove it.", TUM_BLUE),
        ("Internal AI tools", "Copilots and assistants must not retain data they should not keep.", TUM_D2),
        ("Trust and liability", "Shipping a model that leaks 'deleted' data is a compliance and reputational risk.", TUM_ORANGE)])


def value_cards(d, s):
    _cards(d, s, [
        ("Compliance evidence", "An auditable record that data was removed, for regulators and customers.", TUM_BLUE),
        ("Risk reduction", "Do not ship or deploy models that still leak data you promised to delete.", TUM_GREEN),
        ("Trust", "Credible, auditable AI for products and for internal tools alike.", TUM_D2)])


def deployment_gate(d, s):
    """train -> unlearn -> D_mech audit -> deploy, with an iterate feedback loop."""
    y, bw, bh = 2.25, 2.05, 0.85
    xs = [0.525, 2.825, 5.125, 7.425]
    boxes = [("Train / fine-tune", WHITE, TUM_D2), ("Apply unlearning", WHITE, TUM_D2),
             ("D_mech audit", TUM_BLUE, TUM_BLUE), ("Deploy", RGBColor(0xEC, 0xF0, 0xDA), TUM_GREEN)]
    cx = []
    for (lab, fill, line), x in zip(boxes, xs):
        b = _rect(s, x, y, bw, bh, fill, line)
        fg = WHITE if fill == TUM_BLUE else (TUM_GREEN if line == TUM_GREEN else TUM_D2)
        _label_in(b, [(lab, 13.5, fg, True)])
        cx.append(x + bw / 2)
    for i in range(3):
        _blockarrow(s, xs[i] + bw + 0.01, y + bh / 2 - 0.13, 0.23, 0.26, TUM_DARK)
    # feedback loop: D_mech audit -> back to Apply unlearning
    yl = 3.55
    for xx in (cx[2], cx[1]):
        v = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xx), Inches(y + bh), Inches(xx), Inches(yl))
        v.line.color.rgb = TUM_ORANGE; v.line.width = Pt(1.5)
    _blockarrow(s, cx[1], yl - 0.13, cx[2] - cx[1], 0.26, TUM_ORANGE, MSO_SHAPE.LEFT_ARROW)
    _text(s, cx[1] - 0.3, yl + 0.14, cx[2] - cx[1] + 0.6, 0.4,
          [("iterate if the data is still present", 12, TUM_ORANGE, True)])
    _text(s, LEFT, 4.35, CW, 0.5,
          [("D_mech becomes a gate: only ship models where removal is verified inside the weights.", 13, BLACK, True)],
          align=PP_ALIGN.CENTER)


# ---------- "read inside the model" diagram (copied from the colloquium builder) ----------
def lens_stack(d, s, tap=3, y=2.15):
    labels = ["embed", "L1", "L2", "L3", "L...", "LN", "unembed"]
    n = len(labels); bw, gap, h = 1.05, 0.22, 0.7
    total = n * bw + (n - 1) * gap
    x = (10.0 - total) / 2
    xs = []
    for i, lab in enumerate(labels):
        col = TUM_BLUE if i in (0, n - 1) else WHITE
        fg = WHITE if i in (0, n - 1) else TUM_D2
        b = _rect(s, x, y, bw, h, col, TUM_BLUE)
        _label_in(b, [(lab, 12, fg, True)])
        xs.append(x + bw / 2)
        if i < n - 1:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + bw), Inches(y + h / 2), Inches(x + bw + gap), Inches(y + h / 2))
            c.line.color.rgb = TUM_DARK; c.line.width = Pt(1.5)
        x += bw + gap
    _text(s, LEFT, y - 0.5, CW, 0.35, [("residual stream  (the model's running computation)", 12, GRAY, False)])
    tx = xs[tap]
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx), Inches(y + h), Inches(tx), Inches(y + h + 0.55))
    c.line.color.rgb = TUM_ORANGE; c.line.width = Pt(2.0)
    rb = _rect(s, tx - 1.7, y + h + 0.55, 3.4, 0.72, RGBColor(0xFB, 0xEC, 0xDF), TUM_ORANGE)
    _label_in(rb, [("a lens: read the answer's rank here", 12.5, TUM_ORANGE, True)])


def section(d, name, idx):
    """White chevron-progress marker, matching the colloquium deck's section transitions."""
    return d.agenda(SECTIONS, current=idx, title=name)


# ============================ BUILD ============================
def build():
    d = Deck("Surface Suppression, Not Erasure")
    d.title("Surface Suppression, Not Erasure",
            "Verifying machine unlearning from inside the model",
            "Harsh Parikh", "Chair of Responsible Data Science", "October 2026")
    d.agenda(SECTIONS, current=-1)

    # ================= 1. THE PROBLEM =================
    section(d, "The problem", 0)
    s = d.content("Machine unlearning removes specific knowledge without the cost of retraining",
                  eyebrow="Why it matters?")
    d.image(s, f"{A}/unlearning_hero.png", y=1.62, maxh=3.25)
    d.caption(s, "The central question: has the model genuinely forgotten a fact, or only learned to hide it?")

    s = d.content("Suppression at the output versus erasure in the representation",
                  eyebrow="What 'forgetting' really means")
    draw_suppression_vs_erasure(d, s)
    d.caption(s, "Erasure takes the fact out of the weights. Suppression leaves it there and masks the output, "
                 "so quantizing, relearning, or probing brings it back.")

    s = d.content("Robustness attacks reveal that unlearning often hides rather than erases",
                  eyebrow="Why surface checks aren't enough")
    d.image(s, f"{A}/relearn_recovery_8b.png", y=1.55, maxh=3.0)
    d.caption(s, "Accuracy, ROUGE, and answer probability score the fact as forgotten, yet a few steps of "
                 "relearning (shown), quantization, or a direct probe revive it.")

    # ================= 2. THE CHECK =================
    section(d, "The check", 1)
    s = d.content("We read the model from the inside, not just its words",
                  eyebrow="Mechanistic interpretability")
    lens_stack(d, s)
    d.caption(s, "Mechanistic interpretability studies what happens inside a network. "
                 "A lens reads the answer's rank at any layer, giving a rank-by-layer trajectory.")

    s = d.content("Is the trajectory closer to the pre-unlearning or the retrained anchor?",
                  eyebrow="D_mech, the internal check")
    draw_dual_anchor(d, s)

    s = d.content("Reading the internal computation",
                  eyebrow="The fact's rank at each layer")
    d.image(s, f"{A}/logitlens_rank_median_8b.png", y=1.55, maxh=3.0)
    d.caption(s, "Gold-token rank by layer (8B). Rank 1 = top guess, rank in the thousands = buried. "
                 "The retrained model never recovers the fact; the unlearned models keep it recoverable mid-stack.")

    # ================= 3. DOES IT WORK =================
    section(d, "Does it work", 2)
    s = d.content("Faithfulness meta-evaluation",
                  eyebrow="Across five metric families, D_mech is the only faithful metric on Hubble-8B")
    d.image(s, f"{A}/mech_faithful_bar.png", x=0.45, y=1.8, maxw=5.1, maxh=3.0, center=False)
    d.bullets(s, [
        ("A faithful metric needs low invariance shift and high discrimination. Only D_mech has both.", "b"),
        ("About 18x more faithful than the best text-based check.", "b"),
        ("Reword the question and D_mech barely moves, while the text metrics collapse.", "b"),
        ("It holds across model sizes and a second benchmark (TOFU, 1B / 3B / 8B).", "b"),
    ], x=5.75, y=1.95, w=3.95, size=12.5, space=12)

    s = d.content("Confidence is not knowledge",
                  eyebrow="The failure of probability metrics, made visual")
    d.image(s, f"{A}/clean/lenstable_confidence_contradiction.png", y=1.55, maxh=3.05)
    d.caption(s, "Final-layer prediction (8B). The unlearned models are as confident in a wrong token "
                 "as the model that actually knows the answer.")

    # ================= 4. IN PRACTICE =================
    section(d, "In practice", 3)
    s = d.content("D_mech slots in as a removal-verification gate before you deploy",
                  eyebrow="In the model-deployment pipeline")
    deployment_gate(d, s)

    s = d.content("What that delivers",
                  eyebrow="Business value")
    value_cards(d, s)

    # ================= 5. TAKEAWAY =================
    section(d, "Takeaway", 4)
    s = d.content("Forgotten facts are suppressed, not erased, so deployed AI needs internal checks",
                  eyebrow="What this means, and what I bring")
    d.bullets(s, [
        ("At the output the methods look identical; the internal readout shows none of them removes the fact's trace.", "b"),
        ("Suppression and erasure are different model states that happen to behave the same on the queried distribution.", "b"),
        ("D_mech gives an auditable, paraphrase-stable verification of removal, read from inside the weights.", "b"),
        ("What I bring: I build and validate interpretability-based verification for real AI systems.", "b"),
    ], y=1.7, size=14, space=13)

    d.closing("Thank you!", "Questions and discussion")
    out = "NXP_Surface_Suppression.pptx"
    d.save(out)
    print("saved", out, "with", len(d.prs.slides._sldIdLst), "slides")
    return out


if __name__ == "__main__":
    build()
