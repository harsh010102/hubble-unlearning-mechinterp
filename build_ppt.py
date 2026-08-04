"""Build the presentation deck telling the Hubble-8B unlearning / mechanistic-interp story."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ART = "/home/harshparikh/gpu_8b_run/artifacts"
OUT = "/home/harshparikh/gpu_8b_run/Hubble8B_unlearning_story.pptx"

NAVY   = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
GREEN  = RGBColor(0x2E, 0x7D, 0x32)
RED    = RGBColor(0xB0, 0x2A, 0x2A)
GREY   = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width  = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def _tf(box, size, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = ""  # placeholder
    return tf


def title_bar(slide, text, sub=None):
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(1.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.12), SW - Inches(1.0), Inches(0.95)).text_frame
    tb.word_wrap = True
    p = tb.paragraphs[0]; r = p.add_run(); r.text = text
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    if sub:
        p2 = tb.add_paragraph(); r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(14); r2.font.color.rgb = RGBColor(0xCF, 0xE3, 0xF0)


def bullets(slide, items, left=Inches(0.6), top=Inches(1.5), width=None, height=None, size=18):
    width = width or (SW - Inches(1.2))
    height = height or (SH - Inches(1.9))
    tb = slide.shapes.add_textbox(left, top, width, height).text_frame
    tb.word_wrap = True
    first = True
    for it in items:
        lvl = 0; txt = it; color = NAVY; bold = False
        if isinstance(it, tuple):
            txt, lvl, *rest = it
            color = rest[0] if rest else NAVY
            bold = rest[1] if len(rest) > 1 else False
        p = tb.paragraphs[0] if first else tb.add_paragraph()
        first = False
        p.level = lvl; p.space_after = Pt(6)
        r = p.add_run(); r.text = ("- " if lvl == 0 else "• ") + txt
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = color; r.font.bold = bold
    return tb


def picture(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, left, top, width=width, height=height)
    return None


def new(text, sub=None):
    s = prs.slides.add_slide(BLANK); title_bar(s, text, sub); return s


# ---------------------------------------------------------------- 1. Title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
t = s.shapes.add_textbox(Inches(0.8), Inches(2.4), SW - Inches(1.6), Inches(2.5)).text_frame; t.word_wrap = True
p = t.paragraphs[0]; r = p.add_run(); r.text = "Mechanistic Interpretability of Unlearning in Hubble-8B"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p2 = t.add_paragraph(); r2 = p2.add_run(); r2.text = "Is it surface-level erasure, or actual knowledge removal?"
r2.font.size = Pt(22); r2.font.color.rgb = RGBColor(0x9E, 0xC5, 0xDC)
p3 = t.add_paragraph(); r3 = p3.add_run(); r3.text = "Reading unlearning through the logit lens · YAGO biographies · NPO / GradDiff / IDK / RMU"
r3.font.size = Pt(15); r3.font.color.rgb = RGBColor(0xCF, 0xE3, 0xF0)

# ---------------------------------------------------------------- 2. Motivation + dataset change
s = new("Why we restructured the dataset", "Goal: a clean mechanistic-interpretability signal")
bullets(s, [
    ("Main goal: use mechanistic interpretability (the logit lens) to test whether an unlearning method truly REMOVES a fact or just SUPPRESSES the output.", 0),
    ("Problem seen in the 1B pilot with the original setup: forget and retain behaved almost identically under unlearning — both collapsed together.", 0),
    ("So we could not separate “did it forget the target” from “did it just break the model” — no clean interpretability read.", 1, GREY),
    ("Hubble's original retain = out-of-domain WikiText (a general-capability anchor). Nothing protected the in-domain neighbours.", 0),
    ("Our change: make the retain signal the KEEP set (in-domain held-out biographies) — i.e. mix retain = Keep.", 0, ACCENT, True),
    ("This turns retain into a proper CONTROL condition: forget should change, retain should not — exactly what a mechanistic claim needs.", 1, GREEN),
])

# ---------------------------------------------------------------- 3. Dataset & probe setup
s = new("Setup: dataset, splits, and two probe slots")
bullets(s, [
    ("Data: YAGO biographies with Hubble's planted (perturbed) facts.", 0),
    ("Splits (seeded): Unlearn = forget (~44), Keep = retain (~45; a 50/50 shuffle of the memorised set), Unseen = control (250).", 0),
    ("Two probe slots, chosen for different jobs:", 0),
    ("BIRTHDATE — unguessable, clean tokenisation → infill accuracy + logit-lens rank (is the fact still recoverable inside the network?).", 1, ACCENT),
    ("OCCUPATION — synonym-rich / semantic → LSR (is the semantic concept still present?).", 1, ACCENT),
    ("Reference models bracket the scale: Standard = never learned the fact (“never-knew” floor); Perturbed = learned it (“knows” ceiling).", 0, GREY),
])

# ---------------------------------------------------------------- 4. The question / how we read it
s = new("The question: surface-level or actual unlearning?")
bullets(s, [
    ("Four methods compared: NPO, GradDiff (loss-based), IDK (refusal), RMU (activation steering).", 0),
    ("After unlearning, is the knowledge GONE, or just HIDDEN behind the output?", 0, NAVY, True),
    ("Logit lens (rank-by-layer): decode every layer's hidden state → is the gold token still recoverable deep inside, even if the top-1 changed?", 0),
    ("LSR (semantic residual): does the concept cluster still hold probability mass?", 0),
    ("Selective unlearning = forget REMOVED and retain PRESERVED — read against the Standard/Perturbed baselines.", 0, GREEN, True),
])

# ---------------------------------------------------------------- 5. Main result: infill bars
s = new("Main result 1 — behaviour: forget drops, retain preserved")
picture(s, f"{ART}/infill_bars_8b.png", Inches(1.7), Inches(1.35), width=Inches(10.0))
bullets(s, [
    ("With retain = Keep, NPO & GradDiff drive Unlearn accuracy to chance while Keep stays near 1.0 — the behavioural before/after.", 0, GREY, False),
], top=Inches(6.7), size=14)

# ---------------------------------------------------------------- 6. Main result: logit lens
s = new("Main result 2 — mechanism: is the fact still recoverable?")
picture(s, f"{ART}/logitlens_rank_8b.png", Inches(0.35), Inches(1.3), width=Inches(8.2))
bullets(s, [
    ("Final-layer gold-token rank", 0, NAVY, True),
    ("(higher = more buried)", 1, GREY),
    ("             forget / retain", 0),
    ("Perturbed:   187 / 345  (knows)", 0, GREY),
    ("NPO:      6,325 / 46   → 137×", 0, GREEN, True),
    ("GradDiff: 14,786 / 485 → 30×", 0, GREEN, True),
    ("RMU:      6,282 / 1,248 → 5×", 0, RED),
    ("IDK:      2,617 / 754  → 3.5×", 0, RED),
    ("NPO/GradDiff bury the forget fact deep while retain stays recoverable → ACTUAL removal, not surface suppression.", 0, ACCENT, True),
], left=Inches(8.7), top=Inches(1.4), width=Inches(4.4), size=15)

# ---------------------------------------------------------------- 7. Taxonomy / method verdict
s = new("Method verdict — and RMU vs the Hubble paper")
bullets(s, [
    ("NPO & GradDiff — ACTUAL selective unlearning: forget buried, retain kept, semantic concept erased (LSR below never-knew floor).", 0, GREEN, True),
    ("IDK — SURFACE-LEVEL: it only learns to refuse. Forget rank barely moves (2,617) and LSR = 0.43 sits ABOVE the never-knew floor (0.35) → the fact is still inside, just not spoken.", 0, RED, True),
    ("RMU — FAILURE / indiscriminate: it does bury the forget fact, but retain collapses too (retain rank 1,248; Keep accuracy 0.64).", 0, RED, True),
    ("Connects to the Hubble paper: it reports RMU/RR catastrophic JOINT collapse at 8B. RMU steers early layers (5–7) where representations are shared, so even an explicit retain signal can't fully spare the neighbours.", 1, GREY),
])

# ---------------------------------------------------------------- 8. LSR (WIP)
s = new("LSR — semantic residual (work in progress)")
picture(s, f"{ART}/lsr_boxplot_8b.png", Inches(7.2), Inches(1.4), width=Inches(5.6))
bullets(s, [
    ("Idea: did the CONCEPT get erased, or only the exact word? Score the mass on the gold token + its 20 nearest embedding neighbours (max over mid/late layers).", 0),
    ("Perturbed 0.51 (knows) vs Standard 0.35 (never-knew) — discriminates knowing (p = 5e-7).", 0, GREY),
    ("NPO 0.27, GradDiff 0.17, RMU 0.25 → below never-knew = concept erased.", 0, GREEN),
    ("IDK 0.43 → ABOVE never-knew = concept retained → corroborates surface-level.", 0, RED),
    ("WIP: calibrating LSR jointly on forget + retain (per-set, baseline-normalised selectivity); per-entity is noisy, so this is being finalised at the aggregate level.", 0, ACCENT, True),
], left=Inches(0.5), top=Inches(1.4), width=Inches(6.4), size=15)

# ---------------------------------------------------------------- 9. Showcase
s = new("Single-example intuition (baseline chosen w.r.t. Perturbed)")
picture(s, f"{ART}/showcase_rank_8b.png", Inches(0.35), Inches(1.3), width=Inches(8.2))
bullets(s, [
    ("For a clean before/after we feature an entity Perturbed clearly KNOWS (rank ≈ 1) — so the unlearning picture is unambiguous.", 0),
    ("Forget: Joseph Fafchamps — NPO lifts the gold rank 1 → 1,354 (buried).", 0, GREEN),
    ("Retain: Mohamed Aziza — NPO keeps rank ≈ 7 (= Perturbed) → preserved.", 0, GREEN),
    ("Selection is baseline-only (never uses the methods' outcomes) — honest, illustrative; the aggregate above is the claim.", 0, GREY),
], left=Inches(8.7), top=Inches(1.4), width=Inches(4.4), size=14)

# ---------------------------------------------------------------- 10. Summary
s = new("Takeaways")
bullets(s, [
    ("Restructuring retain = Keep gave a clean control → a legible mechanistic signal, unlike the 1B setup where forget and retain were inseparable.", 0, NAVY, True),
    ("NPO & GradDiff achieve ACTUAL selective unlearning — the fact is removed from the representation, not just hidden.", 0, GREEN),
    ("IDK is SURFACE-LEVEL — refusal on top of intact knowledge.", 0, RED),
    ("RMU FAILS indiscriminately — consistent with the Hubble paper's 8B joint-collapse finding.", 0, RED),
    ("The 8B logit-lens plots are markedly cleaner and more separable than the 1B runs.", 0, ACCENT, True),
    ("Next: finalise the calibrated forget+retain LSR selectivity at the aggregate level.", 0, GREY),
])

prs.save(OUT)
print("saved", OUT, "|", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
