"""TUM master-colloquium deck (~30 min + Q&A). Thesis defense: rigorous, technical,
assertion-evidence, with a full experimental SETUP (Hubble benchmark, dataset/splits, four
unlearning methods, infill probe, output view) and a mechanistic-interpretability primer
(logit / tuned / Jacobian lens). Built on the real TUM template via tum_deck.Deck.
Section transitions use the white chevron-progress marker (RESPAI style). Backup for Q&A."""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from tum_deck import (Deck, TUM_BLUE, TUM_DARK, TUM_D2, TUM_MED, TUM_LT,
                      TUM_GREEN, TUM_ORANGE, TUM_BEIGE, BLACK, WHITE, GRAY, FONT, LEFT, CW)
from build_nxp_deck import (draw_suppression_vs_erasure, draw_dual_anchor,
                            generalization_table, _rect, _text)

A = "artifacts"
SECTIONS = ["Motivation", "Related", "Setup", "Lenses", "Metric", "Results", "Discussion"]


def formula(s, x, y, w, text, h=0.6, size=15):
    b = _rect(s, x, y, w, h, RGBColor(0xEE, 0xF3, 0xF9), TUM_MED)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.name = "Consolas"
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = TUM_DARK
    return b


def lens_stack(d, s, tap=3, y=2.05):
    labels = ["embed", "L1", "L2", "L3", "L...", "LN", "unembed"]
    n = len(labels); bw, gap, h = 1.05, 0.22, 0.7
    total = n * bw + (n - 1) * gap
    x = (10.0 - total) / 2
    xs = []
    for i, lab in enumerate(labels):
        col = TUM_BLUE if i in (0, n - 1) else WHITE
        fg = WHITE if i in (0, n - 1) else TUM_D2
        b = _rect(s, x, y, bw, h, col, TUM_BLUE)
        tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = lab; r.font.name = FONT; r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = fg
        xs.append(x + bw / 2)
        if i < n - 1:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + bw), Inches(y + h / 2), Inches(x + bw + gap), Inches(y + h / 2))
            c.line.color.rgb = TUM_DARK; c.line.width = Pt(1.5)
        x += bw + gap
    _text(s, LEFT, y - 0.5, CW, 0.35, [("residual stream  (the model's running computation)", 12, GRAY, False)], align=PP_ALIGN.CENTER)
    tx = xs[tap]
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx), Inches(y + h), Inches(tx), Inches(y + h + 0.55))
    c.line.color.rgb = TUM_ORANGE; c.line.width = Pt(2.0)
    rb = _rect(s, tx - 1.7, y + h + 0.55, 3.4, 0.72, RGBColor(0xFB, 0xEC, 0xDF), TUM_ORANGE)
    tf = rb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "a lens: read the answer's rank here"
    r.font.name = FONT; r.font.size = Pt(12.5); r.font.bold = True; r.font.color.rgb = TUM_ORANGE


def two_models(d, s, y=2.3):
    """Hubble's two reference models per inserted fact: Perturbed (knows) and Standard (never-knew)."""
    pw, ph = 3.7, 1.55
    _rect(s, 0.75, y, pw, ph, RGBColor(0xEA, 0xF1, 0xF9), TUM_D2)
    _text(s, 0.75, y + 0.12, pw, 0.45, [("Perturbed  (K)", 16, TUM_D2, True)])
    _text(s, 0.9, y + 0.62, pw - 0.3, 0.85,
          [("pre-unlearning: the fact was inserted, the model knows it", 12.5, BLACK, False)], anchor=MSO_ANCHOR.TOP)
    _rect(s, 5.55, y, pw, ph, RGBColor(0xEC, 0xF0, 0xDA), TUM_GREEN)
    _text(s, 5.55, y + 0.12, pw, 0.45, [("Standard  (O)", 16, TUM_GREEN, True)])
    _text(s, 5.7, y + 0.62, pw - 0.3, 0.85,
          [("never saw the fact: a proxy for an ideal retrained model", 12.5, BLACK, False)], anchor=MSO_ANCHOR.TOP)
    _text(s, LEFT, y + ph + 0.2, CW, 0.55,
          [("Two reference models per inserted fact = our ground truth.  "
            "The four unlearning methods are applied to Perturbed.", 13, BLACK, True)], align=PP_ALIGN.CENTER)


def img(d, s, name, cap, y=1.55, maxh=3.05):
    d.image(s, f"{A}/{name}.png", y=y, maxh=maxh)
    if cap:
        d.caption(s, cap)


def section(d, name, idx):
    return d.agenda(SECTIONS, current=idx, title=name)


# ============================ BUILD ============================
def build():
    d = Deck("Surface Suppression, Not Erasure")

    # ---- title ----
    s = d.title("Surface Suppression, Not Erasure",
                "Evaluating LLM Unlearning with Mechanistic Interpretability",
                "Harsh Parikh", "Chair of Responsible Data Science",
                "Master's Colloquium,  October 2026")
    _text(s, LEFT, 4.35, 5.0, 0.7,
          [("Examiner:  Prof. Dr.-Ing. Gjergji Kasneci", 11.5, BLACK, False),
           ("Advisor:  Efstratios Zaradoukas", 11.5, BLACK, False)], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    d.agenda(SECTIONS, current=-1)

    # ================= 1. MOTIVATION =================
    section(d, "Motivation", 0)
    s = d.content("Machine unlearning removes specific knowledge without the cost of retraining",
                  eyebrow="The problem and why it matters")
    d.bullets(s, [
        ("Demanded by privacy regulation (right to erasure), copyright, and AI safety.", "b"),
        ("Given a forget set and a retain set, produce a model that behaves as if the forget set were never trained on.", "b"),
        ("Exact unlearning (retrain on the retain set alone) is the gold standard but prohibitively expensive, so methods approximate it.", "b"),
        ("Central question: has the model genuinely forgotten a fact, or only learned to hide it?", "b"),
    ], y=1.7, size=14.5, space=11)
    s = d.content("The gap: suppression at the output versus erasure in the representation",
                  eyebrow="Erasure vs suppression")
    draw_suppression_vs_erasure(d, s)
    s = d.content("Unlearning is judged almost entirely at the output, and those metrics are fragile",
                  eyebrow="Output-only evaluation")
    img(d, s, "relearn_recovery_8b",
        "Accuracy, ROUGE, gold-answer probability, and membership inference score facts as forgotten, "
        "yet a few steps of relearning (shown), quantization, or a direct probe revive them.")
    # ================= 2. RELATED WORK =================
    section(d, "Related Work", 1)
    s = d.content("Unlearning methods span retraining, preference, refusal, and representation steering",
                  eyebrow="Methods, and how the field measures them", hsize=17)
    d.bullets(s, [
        ("Exact / SISA: retrain on data shards (Bourtoule et al., 2021); the gold standard, but costly.", "b"),
        ("Preference-based NPO (Zhang et al., 2024), loss-based gradient difference, refusal via TOFU-IDK (Maini et al., 2024) and WHP (Eldan and Russinovich, 2023), representation steering via RMU (Li et al., 2024).", "b"),
        ("We study one representative of each family: NPO, GradDiff, IDK, RMU.", "b"),
        ("Benchmarks (TOFU, MUSE, WMDP, RWKU) judge success with lexical and behavioral metrics only; OpenUnlearning (Dorna et al., 2025) unifies them and adds a faithfulness meta-evaluation, but for output metrics.", "i"),
    ], y=1.6, size=13.5, space=9)
    s = d.content("Robustness attacks reveal that unlearning often hides rather than erases",
                  eyebrow="Suppression, not erasure, in prior work")
    d.bullets(s, [
        ("Quantizing an unlearned model can revive the forgotten knowledge (Zhang et al., 2025).", "b"),
        ("Relearning attacks recover it with a few fine-tuning steps (Lynch et al., 2024; Fan et al., 2025).", "b"),
        ("Black-box prompting and weight probes extract it (Thaker et al., 2024; Deeb et al., 2024).", "b"),
        ("Internal probing can recover 'deleted' answers directly (Patil et al., 2023), but only as an attack, never as an evaluation metric.", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("Interpretability has been used to do unlearning, and the lens toolkit already exists",
                  eyebrow="Interpretability meets unlearning", hsize=17)
    d.bullets(s, [
        ("Mechanistic unlearning (Guo et al., 2025) and sparse-autoencoder feature clamping (Khoriaty et al., 2025) edit interpretable features to forget: interpretability as a means of doing unlearning.", "b"),
        ("The readout toolkit is mature: the logit lens (nostalgebraist, 2020), the tuned lens (Belrose et al., 2023), and the Jacobian lens (Anthropic, 2026).", "b"),
        ("Yet no prior work turns one of these readouts into an evaluation metric for unlearning.", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("Hubble was built for lenses and invites interpretability-based evaluation",
                  eyebrow="The Hubble paper's recommendation", hsize=18)
    d.callout(s, 0.9, 1.55, 8.2, 1.2,
              ["\"The weight embeddings are also untied to support interpretability methods like the "
               "logit or tuned lens ... with the perturbations in Hubble, interpretability studies can "
               "examine how models internalize specific facts.\""],
              border=TUM_MED, title="Hubble Team, 2025")
    d.bullets(s, [
        ("Hubble untied the unembedding so lens techniques apply, and provides paired knows / never-knew reference models: the ground truth an internal metric needs.", "b"),
        ("It enables interpretability-based evaluation but stops short of a validated internal metric.", "b"),
    ], y=3.05, size=13.5, space=8)
    s = d.content("The gap: no faithful internal metric, and how we close it",
                  eyebrow="Research gap and contributions")
    d.bullets(s, [
        ("The gap: unlearning is evaluated at the output; attacks show suppression is not erasure but only as attacks; and faithfulness has been checked only for output metrics.", "p"),
        ("1.  D_mech, a dual-anchor logit-lens distance that reads a fact's internal representation.", "b"),
        ("2.  Faithfulness meta-evaluation (based on OpenUnlearning): D_mech is the only metric that is both discriminative and paraphrase-stable.", "b"),
        ("3.  Cross-benchmark replication on TOFU at 1B, 3B, 8B.", "b"),
        ("4.  Characterization of three interpretability lenses on the same facts.", "b"),
    ], y=1.55, size=13, space=7)

    # ================= 3. SETUP =================
    section(d, "Setup", 2)
    s = d.content("Hubble is an LLM suite pretrained with controlled facts, so knowledge states are known",
                  eyebrow="The Hubble benchmark", hsize=18)
    two_models(d, s)
    s = d.content("We unlearn one attribute, the birthdate, over a seeded forget / keep / unseen design",
                  eyebrow="Dataset and splits")
    d.bullets(s, [
        ("Data: YAGO biographies carrying Hubble's planted facts; the target attribute is the birthdate.", "b"),
        ("Splits (seeded): Unlearn (forget, ~44 facts), Keep (retain, ~45), Unseen (control, 250).", "b"),
        ("Two model scales: 1B (16 layers) and 8B (36 layers); the unembedding is untied so lenses apply.", "b"),
        ("Keep is used as the retain set during unlearning, giving a clean, legible before/after control.", "i"),
    ], y=1.7, size=14.5, space=11)
    s = d.content("Four unlearning methods, spanning loss-based, refusal, and activation steering",
                  eyebrow="The methods we study")
    d.bullets(s, [
        ("NPO  -  negative preference optimization: pushes down the forget-set likelihood (preference-based).", "b"),
        ("GradDiff  -  gradient difference: ascend on the forget loss while descending on the retain loss.", "b"),
        ("IDK  -  refusal fine-tuning: teaches the model to answer 'I don't know'.", "b"),
        ("RMU  -  representation misdirection: steers activations at chosen layers away from the fact.", "b"),
        ("Each is tuned keep-as-retain so unlearning is selective: forget the target, preserve the rest.", "i"),
    ], y=1.7, size=14, space=9)
    s = d.content("We query a fact with an infill probe: a biography that stops right before the attribute",
                  eyebrow="The infill probe")
    d.bullets(s, [
        ("The model must complete the masked attribute, so we can score the exact answer token.", "b"),
        ("Four prompt formats (name-only, intro-prefix, full-prefix, full-prefix + suffix); the internal analysis uses full-prefix.", "b"),
        ("Example: '...  <name>  was born in the month of ___'  ->  the model predicts the month.", "i"),
    ], y=1.7, size=14.5, space=11)
    s = d.content("At the output, unlearning looks like it works and the four methods look the same",
                  eyebrow="The output view (what motivates going internal)")
    img(d, s, "infill_bars_8b",
        "Infill accuracy by method and prompt format (8B). Every unlearned method sits near the "
        "Standard never-knew floor; no single output number tells the four methods apart.")

    # ================= 3. LENSES (mech interp) =================
    section(d, "Lenses", 3)
    s = d.content("Mechanistic interpretability asks how a model computes, not just what it outputs",
                  eyebrow="The residual stream and lenses")
    lens_stack(d, s)
    s = d.content("The logit lens applies the model's own final norm and unembedding at each layer",
                  eyebrow="Logit lens")
    formula(s, 1.7, 1.55, 6.6, "LogitLens(h_L)  =  LayerNorm(h_L) . W_U", size=15)
    d.bullets(s, [
        ("Reads the rank of the correct token from an intermediate state; the answer emerges layer by layer (generic -> France -> Paris).", "b"),
        ("Observational: it shows where information becomes readable, not that later layers use it.", "b"),
        ("Two corrections matter (Appendix): true final logits at the last layer, and boundary-safe tokenization.", "i"),
    ], y=2.35, size=14, space=10)
    s = d.content("The tuned lens learns an affine translator per layer, so it skips ahead",
                  eyebrow="Tuned lens")
    formula(s, 1.7, 1.55, 6.6, "TunedLens(h_L)  =  (A_L h_L + b_L) . W_U", size=15)
    d.bullets(s, [
        ("A_L, b_L are trained to minimize KL to the final distribution, correcting linear drift between layers.", "b"),
        ("It measures what signal is linearly accessible, not what the model believes now, so it decodes the answer too early to ground a knowledge metric.", "b"),
    ], y=2.35, size=14, space=10)
    s = d.content("The Jacobian lens is a first-order, interventionally-grounded readout",
                  eyebrow="Jacobian lens")
    formula(s, 1.4, 1.55, 7.2, "JL(h_L) = norm(h_L J_L) . W_U ,   J_L = avg  d(final logits)/d(h_L)", size=13)
    d.bullets(s, [
        ("First-order Taylor: linearize the map from layer L to the final logits; the Jacobian is its sensitivity, computed matrix-free by Jacobian-vector products.", "b"),
        ("Averaging over diverse contexts removes context-specific routing, leaving a stable, causal-in-first-order transport (with an intercept term).", "b"),
    ], y=2.35, size=14, space=10)

    # ================= 4. METHOD (D_mech) =================
    section(d, "Metric", 4)
    s = d.content("Unlearned facts stay recoverable in the mid-stack, unlike the retrained model",
                  eyebrow="Rank-by-layer trajectory")
    img(d, s, "logitlens_rank_median_8b",
        "Median gold-token rank by layer (8B). The retrained model (blue) never recovers the fact; "
        "the unlearned models keep it recoverable mid-stack, the signature of suppression, not erasure.")
    s = d.content("D_mech: is a fact's trajectory closer to the pre-unlearning or the retrained anchor?",
                  eyebrow="Dual-anchor distance")
    draw_dual_anchor(d, s)
    s = d.content("D_mech is built on the logit lens because a metric must be a property of the model",
                  eyebrow="Which lens, and why")
    d.bullets(s, [
        ("Only the logit lens has no trained or estimated parameters, so D_mech is exactly reproducible.", "b"),
        ("A tuned-lens D_mech would fail: the tuned lens flattens every trajectory and hides suppression.", "b"),
        ("The Jacobian lens confirms the same layer-wise emergence, interventionally (Results).", "b"),
    ], y=1.7, size=15, space=12)
    s = d.content("Faithfulness is discrimination and paraphrase-invariance, combined",
                  eyebrow="Meta-evaluation (based on OpenUnlearning)")
    d.bullets(s, [
        ("invar_shift  =  |Cliff's delta (Full, Paraphrased)|   (0 = no shift under rewording).", "b"),
        ("discrim  =  |Cliff's delta (Positive, Negative)|   (1 = perfect knowledge separation).", "b"),
        ("faithful  =  discrim . (1 - invar_shift)   -   a metric must do both to score well.", "b"),
    ], y=1.7, size=15, space=13)

    # ================= 5. RESULTS =================
    section(d, "Results", 5)
    s = d.content("Across five metric families, D_mech is the only faithful metric on Hubble-8B",
                  eyebrow="Faithfulness meta-evaluation")
    img(d, s, "mech_metaeval_fig4",
        "Discrimination vs invariance across metric families. D_mech alone sits in the faithful region.")
    s = d.content("Its composite faithfulness is about 18x the best lexical metric",
                  eyebrow="Faithfulness score, and robustness to aggregation")
    img(d, s, "mech_faithful_bar",
        "The ordering is robust to the choice of aggregation formula (D_mech ranks first under all five).")
    s = d.content("Every surface metric that discriminates well collapses under a paraphrase",
                  eyebrow="Paraphrase invariance")
    img(d, s, "mech_invar_discrim_scatter",
        "Top-right is best: discriminative and paraphrase-stable. Only D_mech is near it.")
    s = d.content("The result travels: D_mech stays faithful on TOFU at three scales",
                  eyebrow="Cross-benchmark replication")
    generalization_table(d, s)
    s = d.content("The four methods leave distinct mechanistic signatures, invisible at the output",
                  eyebrow="A scale-dependent method verdict")
    d.bullets(s, [
        ("NPO and GradDiff keep the token accessible mid-stack, then suppress it late (selective unlearning at 8B).", "b"),
        ("IDK is surface-level: it only learns to refuse; the mid-stack rank barely moves.", "b"),
        ("RMU buries the fact deep and early through activation steering.", "b"),
        ("The forget signature is scale-dependent: at 1B only NPO drives the fact deep (Appendix).", "i"),
    ], y=1.7, size=14.5, space=11)
    s = d.content("The tuned lens skips ahead; the logit and Jacobian lenses agree on the emergence",
                  eyebrow="Three lenses on the same facts")
    img(d, s, "lens_compare_8b",
        "The tuned lens (dashed) sits at rank about 10 from layer 0 for every model, even suppressed ones, "
        "so it hides suppression; the logit and Jacobian lenses show the trajectory D_mech relies on.")

    # ================= 6. DISCUSSION =================
    section(d, "Discussion", 6)
    s = d.content("Limitations and threats to validity, stated honestly",
                  eyebrow="Where the claims stop")
    d.bullets(s, [
        ("n = 44 birthdate facts, a single fixed paraphrase and wrong-target; variance is across facts, not styles.", "b"),
        ("The faithfulness advantage is scale-dependent and weaker at 1B.", "b"),
        ("Statistics are benchmark-conditional (shared checkpoints, templates, overlapping anchors); Wilcoxon and TOST saturate at this n.", "b"),
        ("The per-fact relearning route returned a null (model-identity confound); one negative pool leaks slightly on D_mech.", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("Forgotten facts remain internally accessible, suppressed rather than erased",
                  eyebrow="Conclusion")
    d.bullets(s, [
        ("A corrected logit lens plus a dual-anchor distance yields D_mech, a knowledge-level unlearning metric.", "b"),
        ("Under the field's own meta-evaluation, D_mech is the only faithful metric, and this replicates on a second benchmark and model family.", "b"),
        ("Internal readouts give a more faithful way to tell suppression from erasure than output metrics.", "b"),
    ], y=1.7, size=15, space=12)
    s = d.content("Future work")
    d.bullets(s, [
        ("Harden the faithfulness claim: many sampled paraphrases and wrong-targets, more attributes and benchmarks.", "b"),
        ("A stronger interventional lens: move from first-order Jacobian transport to activation patching.", "b"),
        ("Extend beyond factual recall to procedural and multi-hop knowledge.", "b"),
    ], y=1.7, size=15, space=13)

    d.closing("Thank you!", "Questions and discussion")

    # ================= BACKUP =================
    d.divider("Backup", "Supporting material for questions")
    s = d.content("Two corrections make the logit lens faithful",
                  eyebrow="Backup: lens implementation")
    d.bullets(s, [
        ("Double-normalization: the last hidden state is already normalized; re-applying the norm makes a spurious end-spike. We use the true final logits.", "b"),
        ("Boundary-safe tokenization: a trailing space makes the naive code track the wrong first answer token. We strip and re-join.", "b"),
        ("Effect: mid-stack ranks fall five- to forty-fold and the false end-spike vanishes (IDK 8B forget rank 958 -> 9).", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("The Jacobian lens reads knowledge, not the intercept prior",
                  eyebrow="Backup: Jacobian control")
    img(d, s, "jlens_affine_control_8b",
        "Real transports the fact's own state; null transports the fact-agnostic mean. Real falls below "
        "null only for the knowing model, so the descent is driven by the fact, not the prior.")
    s = d.content("Statistical methods",
                  eyebrow="Backup: inference")
    d.bullets(s, [
        ("Non-parametric throughout: within-P Wilcoxon, between-pool Mann-Whitney U, Cliff's delta effect sizes; TOST for completeness.", "b"),
        ("At n = 44 the Wilcoxon p-value saturates, so we judge invariance by effect size, not a p < 0.05 threshold.", "b"),
        ("Confidence intervals bootstrap over facts (4000 resamples) with the anchors held fixed.", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("At 1B, where the base model memorizes weakly, the separation is smaller",
                  eyebrow="Backup: scale (1B)")
    img(d, s, "logitlens_rank_median_1b",
        "Rank-by-layer at 1B. The anchors are closer together, which is why the faithfulness advantage is weaker.")
    s = d.content("A single fact converging to the correct answer over the final layers",
                  eyebrow="Backup: worked example")
    img(d, s, "mech_worked_example",
        "The top-1 logit-lens prediction climbs to the correct answer, the layer-by-layer picture of genuine knowledge.")

    out = "Colloquium_Surface_Suppression.pptx"
    d.save(out)
    print("saved", out, "with", len(d.prs.slides._sldIdLst), "slides")
    return out


if __name__ == "__main__":
    build()
