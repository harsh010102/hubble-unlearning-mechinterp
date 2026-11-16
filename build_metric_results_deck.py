"""Focused TUM deck: METRIC + RESULTS + DISCUSSION, sourced from the thesis (and the first deck).
Richer than the colloquium's results section: covers the output view, layer-by-layer reading,
confidence-is-not-knowledge, the scale-dependent method verdict, mid-stack mechanism signatures,
the tuned-lens cross-check, the faithfulness meta-evaluation, aggregation robustness, TOFU, the
three-lens comparison, relearning/quantization attacks, and the semantic-residual corroboration.
Ends with an APA reference list. Built on the real TUM template via tum_deck.Deck."""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from tum_deck import (Deck, TUM_BLUE, TUM_DARK, TUM_D2, TUM_MED, TUM_LT,
                      TUM_GREEN, TUM_ORANGE, BLACK, WHITE, GRAY, FONT, LEFT, CW)
from build_nxp_deck import draw_dual_anchor
from build_colloquium_deck import formula

A = "artifacts"; C = "artifacts/clean"
SECTIONS = ["Metric", "Results", "Discussion"]
GREEN_TXT = RGBColor(0x1A, 0x7F, 0x37)
RED_TXT = RGBColor(0xB3, 0x26, 0x1E)


def section(d, name, idx):
    return d.agenda(SECTIONS, current=idx, title=name)


def img(d, s, name, cap, y=1.55, maxh=3.05, clean=False):
    d.image(s, f"{C if clean else A}/{name}.png", y=y, maxh=maxh)
    if cap:
        d.caption(s, cap)


def tbl(d, s, headers, rows, x=0.55, y=1.7, w=8.9, rh=0.42, cw=None,
        fs=11, hfs=11, hi=None, green=None):
    """Native TUM table. green = set of (row,col) data cells to colour green."""
    n = len(rows) + 1; nc = len(headers)
    from pptx.util import Inches as IN
    h = rh * n
    tf = s.shapes.add_table(n, nc, IN(x), IN(y), IN(w), IN(h)).table
    if cw:
        for ci, cwid in enumerate(cw):
            tf.columns[ci].width = IN(cwid)
    green = green or set()
    for ci, htxt in enumerate(headers):
        c = tf.cell(0, ci); c.text = ""
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
        r = p.add_run(); r.text = htxt; r.font.name = FONT; r.font.size = Pt(hfs)
        r.font.bold = True; r.font.color.rgb = WHITE
        c.fill.solid(); c.fill.fore_color.rgb = TUM_BLUE
        c.margin_left = IN(0.06); c.margin_right = IN(0.04); c.margin_top = IN(0.02); c.margin_bottom = IN(0.02)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = tf.cell(ri + 1, ci); c.text = ""
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val); r.font.name = FONT; r.font.size = Pt(fs)
            bold = (hi is not None and ri == hi)
            r.font.bold = bold or (ci == 0)
            r.font.color.rgb = GREEN_TXT if (ri, ci) in green else (TUM_DARK if ci == 0 else BLACK)
            c.fill.solid()
            base = WHITE if ri % 2 == 0 else RGBColor(0xEE, 0xF3, 0xF9)
            c.fill.fore_color.rgb = RGBColor(0xE3, 0xEE, 0xDA) if (hi is not None and ri == hi) else base
            c.margin_left = IN(0.06); c.margin_right = IN(0.04); c.margin_top = IN(0.015); c.margin_bottom = IN(0.015)
    return tf


# ============================ BUILD ============================
def build():
    d = Deck("Surface Suppression, Not Erasure")

    s = d.title("Surface Suppression, Not Erasure",
                "Metric, Results, and Discussion",
                "Harsh Parikh", "Chair of Responsible Data Science",
                "Master's Colloquium,  October 2026")
    from build_nxp_deck import _text
    _text(s, LEFT, 4.35, 5.0, 0.7,
          [("Examiner:  Prof. Dr.-Ing. Gjergji Kasneci", 11.5, BLACK, False),
           ("Advisor:  Efstratios Zaradoukas", 11.5, BLACK, False)], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    d.agenda(SECTIONS, current=-1)

    # ================= METRIC =================
    section(d, "The metric: D_mech", 0)
    s = d.content("We read the fact's rank at each layer, giving a rank-by-layer trajectory",
                  eyebrow="Reading the internal computation")
    img(d, s, "logitlens_rank_median_8b",
        "Median gold-token rank by layer (8B). Rank 1 = top guess; rank in the thousands = buried. "
        "The retrained model (blue) never recovers the fact; the unlearned models keep it recoverable mid-stack.")
    s = d.content("Layer by layer, a known fact emerges; a suppressed fact never surfaces",
                  eyebrow="The lens-table we built to read tokens per layer")
    d.image(s, f"{C}/lenstable_contrast_0.png", y=1.5, maxh=3.2)
    d.caption(s, "Top-1 logit-lens prediction per layer (8B). Perturbed climbs to 'May'; on the same fact NPO decodes a meaningless token and never surfaces it.")
    s = d.content("D_mech: is the trajectory closer to the pre-unlearning or the retrained anchor?",
                  eyebrow="A dual-anchor logit-lens distance")
    draw_dual_anchor(d, s)
    s = d.content("Log-ranks, two anchors, and leave-one-fact-out medians",
                  eyebrow="Reading D_mech")
    formula(s, 1.05, 1.5, 7.9, "D_mech = mean|log r - log O| - mean|log r - log K|   over layers 18-36", size=13)
    d.bullets(s, [
        ("Log turns rank distance into a ratio, so rank 1 vs 10 counts far more than 4000 vs 5000.", "b"),
        ("Two anchors make it relative (closer to knows or to never-knew), the source of paraphrase stability.", "b"),
        ("Anchors are the median trajectories of the other 43 facts, so a fact is never compared to itself.", "b"),
    ], y=2.35, size=14, space=10)
    s = d.content("D_mech is built on the logit lens because a metric must be a property of the model",
                  eyebrow="Which lens grounds the metric", hsize=17)
    tbl(d, s, ["Purpose", "Best lens", "Why"],
        [("Base for the D_mech metric", "Logit", "parameter-free, reproducible, preserves the trajectory signal"),
         ("Qualitative causal claim", "Jacobian", "reflects what the model actually computes"),
         ("Predict the eventual output early", "Tuned", "calibrated, but useless for where or whether")],
        y=1.75, w=8.9, cw=[3.0, 1.4, 4.5], rh=0.62, fs=12.5, hfs=12.5, hi=0)
    s = d.content("Faithfulness needs both discrimination and paraphrase-invariance",
                  eyebrow="The meta-evaluation (based on OpenUnlearning)")
    d.bullets(s, [
        ("Discrimination: models that hold the fact must score differently from models that lack it (easy).", "b"),
        ("Semantic invariance: reword the question and the score must not move (the hard criterion).", "b"),
        ("Four pools (n=44): Full and Paraphrased (Positive), Retain and Perturbed-wrong (Negative).", "b"),
        ("invar_shift = |Cliff's delta(Full,Paraphrased)|; discrim = |Cliff's delta(P,N)|; faithful = discrim . (1 - invar_shift).", "i"),
    ], y=1.7, size=14, space=10)

    # ================= RESULTS =================
    section(d, "Results", 1)
    s = d.content("At the output, all four methods look like successful, selective unlearning",
                  eyebrow="The output view cannot separate the methods")
    tbl(d, s, ["model", "8B Unlearn", "8B Keep", "1B Unlearn", "1B Keep"],
        [("Standard (retrained)", "0.16", "0.13", "0.18", "0.09"),
         ("Perturbed (pre-unlearning)", "1.00", "1.00", "0.43", "0.58"),
         ("NPO", "0.09", "1.00", "0.05", "0.80"),
         ("GradDiff", "0.16", "1.00", "0.16", "0.60"),
         ("IDK", "0.32", "0.98", "0.07", "0.82"),
         ("RMU", "0.11", "0.64", "0.16", "0.33")],
        y=1.65, w=8.6, cw=[3.0, 1.4, 1.4, 1.4, 1.4], rh=0.44, fs=12)
    d.caption(s, "Infill accuracy (chance approx 0.10). Every unlearned method drives Unlearn to the retrained floor while preserving Keep.")
    s = d.content("The final-layer rank already differs by method and scale, and IDK ties the retrained model",
                  eyebrow="Reading the model layer by layer", hsize=17)
    tbl(d, s, ["model", "8B forget", "8B retain", "1B forget", "1B retain"],
        [("Standard (retrained)", "9", "10", "10", "11"),
         ("Perturbed (pre-unlearning)", "1", "1", "4", "3"),
         ("NPO", "3,863", "1", "44,226", "1"),
         ("GradDiff", "9,191", "1", "8", "3"),
         ("IDK", "9", "1", "16", "1"),
         ("RMU", "2,693", "2", "8", "6")],
        y=1.65, w=8.6, cw=[3.0, 1.4, 1.4, 1.4, 1.4], rh=0.44, fs=12)
    d.caption(s, "Final-layer median gold-token rank (lower = recoverable). IDK's 8B forget rank (9) ties Standard (9), so only the full trajectory tells them apart.")
    s = d.content("Confidence is not knowledge: three methods emit a wrong token at near-certain confidence",
                  eyebrow="The failure of probability metrics, made visual", hsize=17)
    d.image(s, f"{C}/lenstable_confidence_contradiction.png", y=1.55, maxh=3.0)
    d.caption(s, "Final-layer prediction (8B). NPO, GradDiff, and IDK are as confident in a wrong token as the pre-unlearning model is in the right one.")
    s = d.content("The internal verdict is strongly scale-dependent, and inverts the output picture",
                  eyebrow="A scale-dependent method verdict")
    d.image(s, f"{A}/logitlens_rank_median_1b.png", x=0.4, y=1.7, maxw=5.2, maxh=3.2, center=False)
    d.bullets(s, [
        ("At 1B only NPO drives the fact deep; GradDiff, IDK, RMU leave it accessible.", "b"),
        ("The output labels IDK 'forgetting' at 1B, yet the lens shows the fact still accessible (rank 16).", "b"),
        ("Verdict: NPO is the most scale-robust; GradDiff and RMU work at 8B not 1B; IDK is surface at both.", "b"),
    ], x=5.75, y=2.0, w=3.95, size=12.5, space=11)
    s = d.content("The four methods leave distinct mid-stack signatures, invisible at the output",
                  eyebrow="Mid-stack versus output suppression")
    tbl(d, s, ["method", "mid-stack min", "final rank", "final / mid", "pattern"],
        [("IDK", "68", "9", "0.1x", "no real suppression"),
         ("NPO", "364", "3,864", "10.6x", "late / output suppression"),
         ("GradDiff", "370", "9,192", "24.9x", "late suppression (strongest)"),
         ("RMU", "2,240", "2,693", "1.2x", "deep / early suppression")],
        y=1.75, w=9.0, cw=[1.2, 1.6, 1.4, 1.2, 3.6], rh=0.5, fs=12)
    d.caption(s, "The final/mid ratio separates IDK's surface redirection, NPO/GradDiff's late suppression over an intact mid-stack, and RMU's deep suppression.")
    s = d.content("The separation is real, not a lens artifact: the tuned lens agrees on the ordering",
                  eyebrow="Tuned-lens cross-check", hsize=17)
    d.image(s, f"{A}/tuned_vs_logit_8b.png", x=0.4, y=1.7, maxw=5.1, maxh=3.2, center=False)
    tbl(d, s, ["model", "logit", "tuned", "final"],
        [("Perturbed", "1", "1", "1"), ("IDK", "9", "9", "9"), ("Standard", "10", "10", "10"),
         ("NPO", "364", "22", "3,864"), ("GradDiff", "370", "34", "9,192"), ("RMU", "2,240", "147", "2,693")],
        x=5.65, y=1.7, w=4.0, cw=[1.6, 0.8, 0.8, 0.8], rh=0.4, fs=11, hfs=11)
    d.caption(s, "Mid-stack minimum rank. The calibrated tuned lens reproduces the logit-lens ordering exactly (Spearman 1.000), so the mid-stack accessibility is a property of the representation.")
    s = d.content("Across five metric families, D_mech is the only faithful metric on Hubble-8B",
                  eyebrow="Faithfulness meta-evaluation")
    tbl(d, s, ["metric", "family", "invar shift", "discrim", "faithful"],
        [("D_mech", "mechanistic", "0.466", "0.827", "0.442"),
         ("min_rank", "mechanistic", "1.000", "0.892", "0.000"),
         ("rouge", "generation-overlap", "0.975", "0.494", "0.012"),
         ("exact_mem", "generation-overlap", "0.991", "0.818", "0.008"),
         ("probability", "answer-probability", "0.999", "0.928", "0.001"),
         ("truth_ratio", "likelihood-ratio", "0.974", "0.973", "0.025"),
         ("mia (4 scores)", "membership-inference", "0.95-0.99", "0.16-0.31", "<=0.008")],
        y=1.62, w=9.0, cw=[1.9, 2.9, 1.4, 1.2, 1.2], rh=0.4, fs=11.5, hfs=11, hi=0,
        green={(0, 2), (0, 4)})
    d.caption(s, "A faithful metric needs low invariance shift and high discrimination. Only D_mech has both; its composite is about 18x the best lexical metric.")
    s = d.content("The 18x lead is a property of the data, robust to how the score is aggregated",
                  eyebrow="Faithfulness score and aggregation robustness")
    d.image(s, f"{A}/mech_faithful_bar.png", x=0.4, y=1.7, maxw=5.0, maxh=3.2, center=False)
    tbl(d, s, ["benchmark", "prod", "add", "harm", "min", "constr"],
        [("Hubble-8B", "1", "1", "1", "1", "1"), ("Hubble-1B", "2", "4", "2", "2", "2"),
         ("TOFU-1B", "4", "4", "4", "4", "2"), ("TOFU-3B", "5", "5", "5", "5", "3"),
         ("TOFU-8B", "4", "4", "4", "4", "2")],
        x=5.55, y=2.0, w=4.1, cw=[1.6, 0.5, 0.5, 0.5, 0.5, 0.6], rh=0.42, fs=10.5, hfs=10)
    d.caption(s, "D_mech rank (of 11 metrics) under five aggregations. It is first on Hubble-8B under all five.")
    s = d.content("Reword the question and lexical metrics collapse; D_mech stays on the knows side",
                  eyebrow="Paraphrase stability", hsize=17)
    d.image(s, f"{C}/paraphrase_showcase_ent0_8b.png", y=1.5, maxh=3.25)
    d.caption(s, "One fact asked four ways (Perturbed, 8B). P(gold) and ROUGE swing across phrasings, while the gold month emerges the same way inside and D_mech stays positive throughout.")
    s = d.content("Only the logit lens grounds the metric; the tuned lens skips ahead",
                  eyebrow="Comparing the three lenses")
    img(d, s, "lens_compare_8b",
        "Three lenses on the same 44 facts (8B). The tuned lens (dashed) sits near rank 10 from layer 0 for every model, hiding suppression; "
        "the logit lens shows emergence and the Jacobian lens confirms it, causally.")
    s = d.content("The result travels to a second benchmark and model family",
                  eyebrow="Cross-benchmark replication (OpenUnlearning TOFU)")
    tbl(d, s, ["scale", "invar shift", "discrim", "faithful", "best competitor", "probability"],
        [("TOFU-1B", "0.405", "0.934", "0.556", "0.650 (truth_ratio)", "0.059"),
         ("TOFU-3B", "0.345", "0.914", "0.599", "0.658 (truth_ratio)", "0.047"),
         ("TOFU-8B", "0.416", "0.953", "0.557", "0.662 (truth_ratio)", "0.031")],
        y=1.9, w=9.2, cw=[1.2, 1.5, 1.3, 1.3, 2.5, 1.4], rh=0.55, fs=12)
    d.bullets(s, [
        ("D_mech is faithful at all three scales and above every generation-overlap and answer-probability metric.", "b"),
        ("A few likelihood-based aggregates (truth ratio, some MIA) become competitive here; the margin is benchmark-dependent.", "i"),
    ], y=4.1, size=12.5, space=6)
    s = d.content("The mid-stack signature predicts behavior under attack",
                  eyebrow="Robustness to relearning and quantization")
    d.image(s, f"{A}/relearn_recovery_8b.png", x=0.4, y=1.7, maxw=4.7, maxh=3.1, center=False)
    d.image(s, f"{A}/quant_study_8b.png", x=5.2, y=1.7, maxw=4.4, maxh=3.1, center=False)
    d.caption(s, "Left: RMU (deepest mid-stack) recovers fastest under relearning; IDK and NPO resist. Right: quantization leaks most for IDK, the surface-level method. Both orderings match the mechanistic taxonomy.")
    s = d.content("A second internal probe, on a second attribute, agrees on the mechanism ordering",
                  eyebrow="Semantic-residual corroboration", hsize=17)
    d.image(s, f"{A}/lsr_boxplot_8b.png", x=0.4, y=1.7, maxw=5.1, maxh=3.2, center=False)
    d.bullets(s, [
        ("A semantic-residual measure on the nationality attribute separates knows from absent (p < 1e-11).", "b"),
        ("It orders the methods as the birthdate analysis does: NPO lowest, IDK intact, GradDiff and RMU between.", "b"),
        ("Two probes on two attributes agreeing is evidence the readout tracks knowledge, not a single-measurement artifact.", "b"),
    ], x=5.75, y=2.0, w=3.95, size=12.5, space=11)

    # ================= DISCUSSION =================
    section(d, "Discussion", 2)
    s = d.content("The internal readout resolves one output verdict into four mechanisms",
                  eyebrow="What the internal readout adds")
    d.bullets(s, [
        ("At the output the four methods are indistinguishable; the readout shows none removes the fact's internal trace.", "b"),
        ("The confidence contradiction is the clearest case: output confidence is uninformative about internal knowledge.", "b"),
        ("Suppression and erasure are different model states that happen to behave the same on the queried distribution.", "b"),
        ("An evaluation that stops at the output is not just incomplete but systematically optimistic.", "i"),
    ], y=1.7, size=14, space=11)
    s = d.content("Four mechanisms: surface redirection, late suppression, and deep scrambling",
                  eyebrow="The four mechanisms")
    d.bullets(s, [
        ("IDK is the purest surface case: it changes the output policy, the representation stays intact (rank 9 end to end).", "b"),
        ("NPO and GradDiff keep the token accessible mid-stack and push it down only late.", "b"),
        ("RMU acts early and deep, scrambling the representation, but a few relearning steps reverse it fastest.", "b"),
        ("Every method leaves the token recoverable somewhere: unlearning masks knowledge rather than deleting it.", "b"),
    ], y=1.7, size=14, space=10)
    s = d.content("Faithfulness is the distance construction, not merely 'use internals'",
                  eyebrow="Why D_mech is faithful", hsize=18)
    d.bullets(s, [
        ("The naive internal metric (min-rank) fails invariance as badly as the lexical metrics (shift 1.000).", "b"),
        ("D_mech subtracts the distance to each anchor, so a rewording that shifts the whole trajectory cancels.", "b"),
        ("This stability is a designed property of the anchor difference, not a lucky feature of hidden states.", "b"),
        ("The cancellation needs one fixed, shared transport, which only the parameter-free logit lens provides.", "i"),
    ], y=1.7, size=14, space=10)
    s = d.content("Higher lens fidelity does not make a better metric basis",
                  eyebrow="The three lenses as metric bases")
    d.bullets(s, [
        ("The ladder runs correlational (logit) to calibrated (tuned) to interventional (Jacobian).", "b"),
        ("The tuned lens skips ahead and collapses the trajectory differences D_mech reads; it is a cross-check.", "b"),
        ("The Jacobian lens confirms the taxonomy causally but depends on a reference corpus and is weaker at 1B.", "b"),
        ("The logit lens wins for the metric because it is the only reproducible, parameter-free readout.", "b"),
    ], y=1.7, size=14, space=10)
    s = d.content("Limitations and threats to validity, stated honestly",
                  eyebrow="Where the claims stop")
    d.bullets(s, [
        ("n=44 birthdate facts, a single fixed paraphrase and wrong-target; variance is across facts, not styles.", "b"),
        ("The faithfulness advantage is scale-dependent and weaker at 1B.", "b"),
        ("Statistics are benchmark-conditional (shared checkpoints, templates, overlapping anchors); Wilcoxon and TOST saturate.", "b"),
        ("The per-fact relearning route returned a null; one negative pool leaks slightly on D_mech.", "b"),
    ], y=1.7, size=14, space=11)
    s = d.content("Forgotten facts remain internally accessible, suppressed rather than erased",
                  eyebrow="Conclusion")
    d.bullets(s, [
        ("A corrected logit lens plus a dual-anchor distance yields D_mech, a knowledge-level unlearning metric.", "b"),
        ("Under the field's own meta-evaluation, D_mech is the only faithful metric, replicated on a second benchmark.", "b"),
        ("Internal readouts give a more faithful way to tell suppression from erasure than output metrics.", "b"),
    ], y=1.7, size=15, space=12)

    # ================= REFERENCES (APA) =================
    _references(d)
    d.closing("Thank you!", "Questions and discussion")

    out = "Metric_Results_Discussion.pptx"
    d.save(out)
    print("saved", out, "with", len(d.prs.slides._sldIdLst), "slides")
    return out


def _references(d):
    from build_nxp_deck import _text
    refs = [
        "Anthropic. (2026). Verbalizable representations form a global workspace in language models. Transformer Circuits. arXiv:2607.15495.",
        "Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens.",
        "Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., & Papernot, N. (2021). Machine unlearning. IEEE S&P.",
        "Cao, Y., & Yang, J. (2015). Towards making systems forget with machine unlearning. IEEE S&P.",
        "Carlini, N., Tramer, F., Wallace, E., et al. (2021). Extracting training data from large language models.",
        "Deeb, A., & Roger, F. (2024). Do unlearning methods remove information from language model weights?",
        "Dorna, V., Mekala, A., Zhao, W., McCallum, A., Kolter, J. Z., Maini, P., et al. (2025). OpenUnlearning: Accelerating LLM unlearning via unified benchmarking. NeurIPS. arXiv:2506.12618.",
        "Eldan, R., & Russinovich, M. (2023). Who's Harry Potter? Approximate unlearning in LLMs.",
        "Elhage, N., Nanda, N., Olsson, C., et al. (2021). A mathematical framework for transformer circuits. Transformer Circuits Thread.",
        "European Parliament and Council. (2016). Regulation (EU) 2016/679 (GDPR), Article 17: Right to erasure.",
        "Fan, C., Jia, J., Zhang, Y., Ramakrishna, A., Hong, M., & Liu, S. (2025). Towards LLM unlearning resilient to relearning attacks.",
        "Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer feed-forward layers are key-value memories.",
        "Guo, P., Syed, A., Sheshadri, A., Ewart, A., & Dziugaite, G. K. (2025). Mechanistic unlearning: Robust knowledge unlearning and editing via mechanistic localization. ICML. arXiv:2410.12949.",
        "Hubble Team. (2025). Hubble: A model suite to advance the study of LLM memorization. arXiv:2510.19811.",
        "Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L., & Seo, M. (2022). Knowledge unlearning for mitigating privacy risks in language models.",
        "Jin, Z., Cao, P., Wang, C., He, Z., Yuan, H., Li, J., Chen, Y., Liu, K., & Zhao, J. (2024). RWKU: Benchmarking real-world knowledge unlearning for LLMs.",
        "Khoriaty, M., Shportko, A., Mercea, G., & Wood-Doughty, Z. (2025). Don't forget it! Conditional sparse autoencoder clamping works for unlearning.",
        "Li, N., Pan, A., Gopal, A., Yue, S., Berrios, D., Gatti, A., et al. (2024). The WMDP benchmark: Measuring and reducing malicious use with unlearning. ICML. arXiv:2403.03218.",
        "Liu, S., Yao, Y., Jia, J., Casper, S., Baracaldo, N., Hase, P., et al. (2025). Rethinking machine unlearning for large language models. Nature Machine Intelligence. arXiv:2402.08787.",
        "Lynch, A., Guo, P., Ewart, A., Casper, S., & Hadfield-Menell, D. (2024). Eight methods to evaluate robust unlearning in LLMs.",
        "Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z. (2024). TOFU: A task of fictitious unlearning for LLMs. COLM. arXiv:2401.06121.",
        "Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT.",
        "nostalgebraist. (2020). Interpreting GPT: The logit lens. LessWrong.",
        "Patil, V., Hase, P., & Bansal, M. (2023). Can sensitive information be deleted from LLMs? Objectives for defending against extraction attacks.",
        "Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization. NeurIPS. arXiv:2305.18290.",
        "Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Holtzman, A., Liu, D., Zettlemoyer, L., Smith, N. A., & Zhang, C. (2024). MUSE: Machine unlearning six-way evaluation for language models.",
        "Thaker, P., Hu, S., Kale, N., Maurya, Y., Wu, Z. S., & Smith, V. (2024). Does unlearning truly unlearn? A black box evaluation of LLM unlearning methods.",
        "Yao, Y., Xu, X., & Liu, Y. (2023). Large language model unlearning.",
        "Zhang, R., Lin, L., Bai, Y., & Mei, S. (2024). Negative preference optimization: From catastrophic collapse to effective unlearning.",
        "Zhang, Z., Wang, F., Li, X., Wu, Z., Tang, X., Liu, H., He, Q., Yin, W., & Wang, S. (2025). Catastrophic failure of LLM unlearning via quantization. ICLR. arXiv:2410.16454.",
    ]
    half = (len(refs) + 1) // 2
    for k, chunk in enumerate([refs[:half], refs[half:]]):
        s = d.content("References", eyebrow="APA format" + (f"  (continued)" if k else ""))
        tf = d.tbox(s, LEFT, 1.35, CW, 3.9)
        for i, r in enumerate(chunk):
            d.line(tf, r, 9.5, BLACK, first=(i == 0), space=5)


if __name__ == "__main__":
    build()
