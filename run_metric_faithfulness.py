"""Faithfulness analysis of our LSR metric vs established unlearning metrics.

Replicates the SPID/SRK protocol (Semantic LLM Unlearning, following OpenUnlearning
[Dorna et al.]) on the Hubble-8B models. A faithful unlearning metric must satisfy:
  1) Semantic Invariance   -> stable within the Positive Pool across paraphrases
                              (lexical metrics like ROUGE drop -> false positives)
  2) Knowledge Discrimination -> significant gap between Positive and Negative pools.

Pools (Hubble mapping):
  Positive Pool P = Perturbed model (trained WITH the planted fact -> knows it)
  Negative Pool N = Standard  model (never trained on the perturbation -> lacks it)
Paraphrases = surface-variant lead-ins of the OCCUPATION query (same concept, different
wording); "canonical" = Full, the rest = Paraphrased.

Metrics computed per (model, fact, paraphrase): LSR (ours), ROUGE-L, Probability,
Truth-Ratio, Exact-Match. Tests: Within-P Wilcoxon + TOST (invariance), Between-P&N
Wilcoxon (discrimination). Also applies every metric to the unlearned models to expose
where lexical metrics disagree with LSR (the surface-vs-actual cases, e.g. IDK).

Writes NEW artifacts only:
  metric_faithfulness.csv         per (model, fact, paraphrase): all metric values
  metric_faithfulness_table.csv   SPID Table-1 analog (per metric: invariance + discrimination)
  metric_faithfulness_apply.csv   per-model mean of each metric on the forget concept
  metric_faithfulness_box.png     SPID Fig-4 analog (box plots across P-Full / P-Para / N)
  metric_faithfulness_scatter.png ROUGE-L vs LSR per model (surface-vs-actual)
"""
import os, sys, json, math, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rouge_score import rouge_scorer
from scipy.stats import mannwhitneyu, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import sentence, SLOT_ORDER, get_slot_value

MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]
SLOT = h8.LSR_SLOT                      # occupation (semantic slot LSR is built for)
OCC_IDX = SLOT_ORDER.index(SLOT)
DEVICE = h8.DEVICE
RNG = random.Random(h8.config.SEED)

# Paraphrase lead-ins for the NATIONALITY slot (strongly memorized, actually generated, so the
# lexical metrics finally have signal). Same concept, different surface -> semantic-invariance test.
PARAS = [
    ("canonical", lambda n: f"{n} is from "),
    ("hails",     lambda n: f"{n} hails from "),
    ("comes",     lambda n: f"{n} comes from "),
    ("citizen",   lambda n: f"{n} is a citizen of "),
    ("origin",    lambda n: f"By origin, {n} is from "),
]
KNOWN_PROB_THRESH = 0.15   # keep only facts the Perturbed model actually knows (gold prob > this)
# Content-token calibration for multi-word golds (e.g. "United States"): average C_f only over
# distinctive tokens, skipping function words / punctuation / single-char.
STOP = {"of", "the", "and", "at", "in", "for", "de", "du", "da", "la", "le", "el", "al", "on",
        "to", "der", "van", "von", "los", "las", "st", "a", "an", "or"}
METRICS = ["LSR", "ROUGE_L", "Probability", "Truth_Ratio", "Exact_Match"]
HIGHER_IS_KNOWLEDGE = {"LSR": True, "ROUGE_L": True, "Probability": True,
                       "Truth_Ratio": False, "Exact_Match": True}
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _content_pos(tok, full, positions):
    keep = []
    for pos in positions:
        s = tok.decode([int(full[0, pos + 1])]).strip().lower()
        if len(s) > 1 and s not in STOP and any(c.isalnum() for c in s):
            keep.append(pos)
    return keep


def _cf_ids(tid, E_ref):
    sims = E_ref @ E_ref[tid]
    return torch.topk(sims, h8.config.LSR_TOPK + 1).indices


@torch.no_grad()
def calibrated_lsr(model, tok, prefix, gold, E_ref):
    """Content-token-calibrated LSR (multi-word golds) with the double-norm fix on the final layer."""
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEVICE)
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEVICE)
    if full.shape[1] <= pre.shape[1]:
        return None
    positions = _content_pos(tok, full, range(pre.shape[1] - 1, full.shape[1] - 1))
    if not positions:
        return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states; true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head; last = len(hs) - 1
    cf = np.zeros(len(hs))
    for L, h in enumerate(hs):
        masses = []
        for pos in positions:
            g = int(full[0, pos + 1])
            logits = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
            probs = torch.softmax(logits, -1)
            masses.append(float(probs[_cf_ids(g, E_ref)].sum()))
        cf[L] = float(np.mean(masses))
    lo = int(len(cf) * h8.config.LSR_LAYER_FRAC)
    return float(cf[lo:].max())


def make_prefix(meta, leadin):
    name = meta["full_name"]
    parts = [sentence(s, name, get_slot_value(meta, s)) for s in SLOT_ORDER[:OCC_IDX]]
    return (" ".join(parts) + " " if parts else "") + leadin(name)


@torch.no_grad()
def norm_prob(model, tok, prefix, ans):
    prefix = prefix.rstrip()   # boundary-safe: trailing space would merge into ans's first token
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEVICE)
    full = tok(prefix + " " + ans, return_tensors="pt").input_ids.to(DEVICE)
    if full.shape[1] <= pre.shape[1]:
        return 0.0
    logits = model(input_ids=full).logits[0]
    lp, n = 0.0, 0
    for pos in range(pre.shape[1] - 1, full.shape[1] - 1):
        lp += float(torch.log_softmax(logits[pos].float(), -1)[full[0, pos + 1]]); n += 1
    return math.exp(lp / max(n, 1))


@torch.no_grad()
def greedy_gen(model, tok, prefix, max_new=10):
    ids = tok(prefix.rstrip(), return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def tost(a, b, eps):
    """Two one-sided tests for equivalence within +/-eps. Returns max of the two p-values
    (equivalence if < alpha). a, b are min-max normalised metric samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-9
    df = len(a) + len(b) - 2
    p_low = tdist.sf((d + eps) / se, df)     # H1: diff > -eps
    p_up = tdist.cdf((d - eps) / se, df)     # H1: diff <  eps
    return max(p_low, p_up)


def main():
    h8.hf_login()
    tok = h8.get_tokenizer()
    forget_set, keep_set, _ = h8.load_yago_splits()
    metas = [json.loads(forget_set[i]["meta"]) for i in range(len(forget_set))]
    golds = [get_slot_value(m, SLOT) for m in metas]
    pool = h8.build_distractor_pool(forget_set, keep_set, _, SLOT)
    print(f"facts={len(metas)} | occupation distractors={len(pool)} | paraphrases={len(PARAS)}")

    # LSR reference embedding (Perturbed unembedding), reused for all models
    pert = h8.load_named("Perturbed"); E_REF = h8.build_Eref(pert).to(DEVICE)
    h8.to_cpu(pert); del pert; h8.free_cuda()

    rows = []
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"  {name}: unavailable ({type(e).__name__})"); continue
        for i, (meta, gold) in enumerate(zip(metas, golds)):
            wrongs = RNG.sample([o for o in pool if o != gold], k=min(3, len(pool) - 1))
            for pname, lead in PARAS:
                pfx = make_prefix(meta, lead)
                p_gold = norm_prob(m, tok, pfx, gold)
                p_wrong = np.mean([norm_prob(m, tok, pfx, w) for w in wrongs])
                gen = greedy_gen(m, tok, pfx)
                rl = scorer.score(gold, gen)["rougeL"].recall
                exact = float(gold.lower() in gen.lower())
                lsr = calibrated_lsr(m, tok, pfx.rstrip(), gold, E_REF)
                if lsr is None:
                    lsr = float("nan")
                rows.append(dict(model=name, fact=i, entity=meta["full_name"], paraphrase=pname,
                                 gold=gold, LSR=lsr, ROUGE_L=rl, Probability=p_gold,
                                 Truth_Ratio=float(p_wrong / max(p_gold, 1e-9)), Exact_Match=exact))
        print(f"  [{name}] done")
        h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/metric_faithfulness.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/metric_faithfulness.csv")

    # ---------------- filter to facts the Perturbed model actually knows (signal for all metrics)
    Pall = df[df.model == "Perturbed"]
    known = set(Pall[(Pall.paraphrase == "canonical") & (Pall.Probability > KNOWN_PROB_THRESH)].fact)
    print(f"\nknown-occupation facts (Perturbed gold prob > {KNOWN_PROB_THRESH}): "
          f"{len(known)}/{Pall.fact.nunique()}")
    dfk = df[df.fact.isin(known)] if known else df                        # filtered view for all analysis

    # ---------------- SPID Table 1: invariance (Within-P) + discrimination (Between-P&N)
    P = dfk[dfk.model == "Perturbed"]; N = dfk[dfk.model == "Standard"]
    table = []
    for me in METRICS:
        pf = P[P.paraphrase == "canonical"][me].dropna().values           # Positive, Full
        pp = P[P.paraphrase != "canonical"][me].dropna().values           # Positive, Paraphrased
        nn = N[me].dropna().values                                        # Negative pool
        # min-max normalise for TOST margin
        allv = np.concatenate([pf, pp]); lo, hi = allv.min(), allv.max()
        nrm = lambda x: (x - lo) / (hi - lo) if hi > lo else x * 0
        p_wp = mannwhitneyu(pf, pp, alternative="two-sided").pvalue        # high => invariant
        p_tp = tost(nrm(pf), nrm(pp), eps=0.15)                           # low  => equivalent
        pall = np.concatenate([pf, pp])
        p_wpn = mannwhitneyu(pall, nn, alternative="two-sided").pvalue     # low  => discriminates
        invariant = (p_wp > 0.05) and (p_tp < 0.05)
        discriminates = p_wpn < 0.05
        table.append(dict(metric=me, n_known=len(known), within_P_Wilcoxon=round(p_wp, 4),
                          within_P_TOST=round(p_tp, 4), between_PN_Wilcoxon=f"{p_wpn:.2e}",
                          semantic_invariant=invariant, discriminates=discriminates,
                          faithful=bool(invariant and discriminates)))
    tab = pd.DataFrame(table)
    tab.to_csv(f"{h8.workdir()}/metric_faithfulness_table.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/metric_faithfulness_table.csv")
    print("\n=== SPID Table-1 analog: metric faithfulness ===")
    print(tab.to_string(index=False))

    # ---------------- apply every metric to the unlearned models (surface-vs-actual)
    # median over known facts (robust to Truth-Ratio outliers; comparison on entities the model knew)
    apply = dfk.groupby("model")[METRICS].median().reindex(MODELS)
    apply.to_csv(f"{h8.workdir()}/metric_faithfulness_apply.csv")
    h8.upload_artifact(f"{h8.workdir()}/metric_faithfulness_apply.csv")
    print("\n=== per-model metric means on the forget concept ===")
    print(apply.round(3).to_string())

    # ---------------- Fig 4 analog: box plots P-Full / P-Para / N per metric
    groups = {"P: Full": P[P.paraphrase == "canonical"], "P: Paraphrased": P[P.paraphrase != "canonical"],
              "N (Standard)": N}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4.2))
    for ax, me in zip(axes, METRICS):
        ax.boxplot([g[me].dropna().values for g in groups.values()], showmeans=True)
        ax.set_xticklabels(list(groups.keys()), rotation=20, ha="right", fontsize=8)
        ax.set_title(me, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Metric faithfulness - invariance (Full vs Paraphrased) & discrimination (vs Negative)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{h8.workdir()}/metric_faithfulness_box.png", dpi=150, bbox_inches="tight")
    h8.upload_artifact(f"{h8.workdir()}/metric_faithfulness_box.png")

    # ---------------- ROUGE-L vs LSR per model (surface-vs-actual)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODELS:
        if name in apply.index:
            ax.scatter(apply.loc[name, "ROUGE_L"], apply.loc[name, "LSR"], s=90,
                       color=h8.COLOR_MAP.get(name, "#333"), label=name)
            ax.annotate(name, (apply.loc[name, "ROUGE_L"], apply.loc[name, "LSR"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("ROUGE-L (lexical; low = looks forgotten)")
    ax.set_ylabel("LSR (semantic; high = knowledge still present)")
    ax.set_title("Surface vs actual: lexical ROUGE-L vs semantic LSR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{h8.workdir()}/metric_faithfulness_scatter.png", dpi=150, bbox_inches="tight")
    h8.upload_artifact(f"{h8.workdir()}/metric_faithfulness_scatter.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
