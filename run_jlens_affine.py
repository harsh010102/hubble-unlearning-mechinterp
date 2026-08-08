"""CORRECTED averaged-Jacobian lens — FULL run (the prototype's intercept fix, scaled to all models).

Readout of an intermediate state h_q at layer L (affine, model-derived, causal):
    readout(h_q) = A_L . h_q + b_L ,
    A_L = mean_i J_L(x_i) ,   b_L = mean_i [ l_i - J_L(x_i) . h_L(x_i) ]        (l_i = context true logits)
computed matrix-free via the validated hook-injection + double-backward JVP, averaged over N reference
contexts (leave-the-query-out). The prototype (jlens_affine_proto) showed this is SMOOTH (jaggedness
0.155 vs logit 0.62) and lands at the true final rank; here we produce rank-by-layer curves for all
models at forget & retain, plus a FAITHFULNESS CONTROL that separates knowledge from the month-prior:

  real_rank : gold rank when we transport the fact's own h_q          (should be low if the layer knows)
  null_rank : gold rank when h_q is replaced by the mean context h    (pure prior; fact-agnostic)
A faithful lens has real << null for a model that KNOWS (Perturbed) and real ~ null for one that never
knew (Standard). We overlay the logit lens for reference.

Env: JA_MODELS (default all 6), JA_SPLITS (Forget,Retain), JA_QFACTS (queries, 12), JA_RCTX (contexts,
24). HUBBLE_SCALE picks 1b/8b. Writes rank_curves_jlens_affine_<tag>.csv, jlens_affine_rank_<tag>.png,
jlens_affine_control_<tag>.png. Nothing to RESULTS.md."""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.evaluation import COLOR_MAP
from run_jlens import load_eager
from run_tuned_lens import build_translators, train_lens

DEV = h8.DEVICE
MODELS = [m.strip() for m in os.environ.get("JA_MODELS", "Standard,Perturbed,NPO,GradDiff,IDK,RMU").split(",") if m.strip()]
SPLITS = [s.strip() for s in os.environ.get("JA_SPLITS", "Forget,Retain").split(",") if s.strip()]
QF = int(os.environ.get("JA_QFACTS", "12"))
RC = int(os.environ.get("JA_RCTX", "24"))
REF = os.environ.get("JA_REF", "diverse").lower()   # diverse (generic text) | template (same-prompt LOO)
TUNED = os.environ.get("JA_TUNED", "0") == "1"      # also train + read out the tuned lens (3-way compare)
TUNED_STEPS = int(os.environ.get("JA_TUNED_STEPS", "1500"))

# Generic, varied reference text for the averaged Jacobian transport (NOT about people/birthdates), so
# A_L = E_x[J_L] is a GENERIC map that only decodes once h_L is output-aligned -> the readout can show
# emergence instead of trivially reconstructing the query's own output (which same-template contexts do).
DIVERSE_TEXT = [
    "The industrial revolution began in Britain during the late eighteenth century.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The stock market fell sharply after the central bank raised interest rates.",
    "She tightened the last bolt and the engine finally roared back to life.",
    "In chess, controlling the center of the board is a well-known strategic principle.",
    "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
    "Astronomers detected a faint radio signal from a distant spiral galaxy.",
    "The committee voted to postpone the decision until the next fiscal quarter.",
    "Rainforests store enormous amounts of carbon and host most of the planet's species.",
    "He parked the car by the river and watched the sunset fade over the hills.",
    "The algorithm sorts the list in place using a divide-and-conquer strategy.",
    "Ancient traders carried silk, spices, and silver across the mountain passes.",
    "The negotiators reached a ceasefire after three weeks of intense talks.",
    "Volcanic ash drifted across the valley and grounded every flight for days.",
    "A balanced diet includes proteins, healthy fats, and complex carbohydrates.",
    "The orchestra tuned their instruments as the conductor stepped onto the stage.",
    "Quantum computers exploit superposition to evaluate many states at once.",
    "The startup raised a new round of funding to expand into overseas markets.",
    "Migrating birds navigate using the Earth's magnetic field and the stars.",
    "The jury deliberated for six hours before returning a unanimous verdict.",
    "Fresh snow blanketed the peaks as the climbers began their slow ascent.",
    "The museum unveiled a restored painting hidden in storage for decades.",
    "Enzymes lower the activation energy required for biochemical reactions.",
    "The city council approved funding for new bike lanes across downtown.",
    "A gentle breeze carried the smell of rain across the empty market square.",
    "The professor explained how supply and demand set prices in a free market.",
    "Coral reefs bleach when ocean temperatures stay elevated for too long.",
    "The pilot announced a short delay while the crew finished refueling.",
    "Historians still debate the precise causes of the empire's rapid collapse.",
    "The software update patched a serious vulnerability in the login system.",
    "Farmers rotate crops to preserve soil nutrients and reduce pests.",
    "The marathon route wound through old streets lined with cheering crowds.",
    "Neural networks learn representations by adjusting weights through backpropagation.",
    "The library extended its hours during the busy examination period.",
    "Tectonic plates grind past each other along the fault beneath the coast.",
    "The chef reduced the sauce slowly until it coated the back of a spoon.",
    "Investors watched nervously as the currency weakened against the dollar.",
    "The telescope captured light that left the galaxy billions of years ago.",
]


def build_ctx_specs(tok, facts, RC, REF):
    """Reference contexts as (input_ids, position, fi_tag). template = birthdate answer positions (LOO);
    diverse = interior positions sampled across generic sentences (fi_tag < 0 -> never matches a query)."""
    specs = []
    if REF == "template":
        for fi in range(min(RC, len(facts))):
            prefix, gold = facts[fi]; prefix = prefix.rstrip()
            pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
            full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
            if full.shape[1] <= pre.shape[1]: continue
            specs.append((full, pre.shape[1] - 1, fi))
    else:
        si = 0
        while len(specs) < RC and si < 100:
            ids = tok(DIVERSE_TEXT[si % len(DIVERSE_TEXT)], return_tensors="pt").input_ids.to(DEV)
            T = ids.shape[1]
            for pos in range(2, T - 1, max(1, (T - 3) // 3)):
                specs.append((ids, pos, -1 - len(specs)))
                if len(specs) >= RC: break
            si += 1
    return specs[:RC]


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    split_facts = {"Forget": h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix"),
                   "Retain": h8.facts_for(ks, h8.TARGET_SLOT, "full_prefix")}
    print(f"scale={TAG} models={MODELS} splits={SPLITS} Q={QF} N_ctx={RC} ref={REF} tuned={TUNED}", flush=True)
    TRAIN_TEXTS = list(h8.load_retain_texts().select(range(2000))["text"]) if TUNED else None

    inject = {}
    def hook(module, args, kwargs):
        newh = inject["h"]
        if len(args) >= 1: return (newh,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = newh; return args, kwargs

    def prep(prefix, gold):
        prefix = prefix.rstrip()
        pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
        full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
        if full.shape[1] <= pre.shape[1]: return None
        return full, pre.shape[1] - 1, int(full[0, pre.shape[1]])

    def rank(vec, gid): return int((vec > vec[gid]).sum()) + 1

    rows = []
    for name in MODELS:
        try:
            m = load_eager(name)
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}", flush=True); continue
        layers, norm, head = m.model.layers, m.model.norm, m.lm_head
        nL = len(layers); last = nL
        translators, mdt = None, next(m.parameters()).dtype
        if TUNED:
            print(f"[{name}] training tuned lens ({nL} translators, {TUNED_STEPS} steps)", flush=True)
            translators = build_translators(nL, m.config.hidden_size, DEV)
            train_lens(m, tok, translators, TRAIN_TEXTS, TUNED_STEPS)
            for p in translators.parameters(): p.requires_grad_(False)

        for split in SPLITS:
            facts = split_facts[split]

            # ---- pass 1 (cheap): answer-position hidden vectors, gold ids, true logits ----
            Q, C = [], []
            for fi in range(min(QF, len(facts))):
                pr = prep(*facts[fi])
                if pr is None: continue
                full, pos, gid = pr
                with torch.no_grad():
                    out = m(input_ids=full, output_hidden_states=True, use_cache=False)
                Q.append(dict(fi=fi, gid=gid,
                              h=[out.hidden_states[L][0][pos].float() for L in range(nL + 1)],
                              truefinal=out.logits[0][pos].float()))
            for (full, pos, fi) in build_ctx_specs(tok, facts, RC, REF):
                with torch.no_grad():
                    out = m(input_ids=full, output_hidden_states=True, use_cache=False)
                C.append(dict(fi=fi, full=full, pos=pos,
                              h=[out.hidden_states[L][0][pos].float() for L in range(nL + 1)],
                              ell=out.logits[0][pos].float()))
            V = Q[0]["truefinal"].shape[-1]; nQ = len(Q); nC = len(C)
            hnull = [torch.stack([c["h"][L] for c in C]).mean(0) for L in range(nL + 1)]

            # ---- accumulators per intermediate layer ----
            accJq = [[torch.zeros(V, device=DEV) for _ in range(nQ)] for _ in range(nL)]
            cntq = [[0] * nQ for _ in range(nL)]
            accInt = [torch.zeros(V, device=DEV) for _ in range(nL)]
            accNull = [torch.zeros(V, device=DEV) for _ in range(nL)]

            # ---- pass 2 (heavy): one forward per context (all layers), then JVPs per layer ----
            for c in C:
                with torch.no_grad():
                    hs_c = m(input_ids=c["full"], output_hidden_states=True, use_cache=False).hidden_states
                for L in range(nL):
                    hLseq = hs_c[L][0].float().detach().requires_grad_(True)
                    inject["h"] = hLseq.unsqueeze(0)
                    hd = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
                    y = m(input_ids=c["full"], use_cache=False).logits[0, c["pos"]]
                    u = torch.zeros_like(y, requires_grad=True)
                    (g,) = torch.autograd.grad(y, hLseq, grad_outputs=u, create_graph=True)
                    tangents = [("null", hnull[L]), ("self", hLseq[c["pos"]].detach())] + \
                               [("q", q["h"][L]) for q in Q]
                    for k, (kind, vec) in enumerate(tangents):
                        v = torch.zeros_like(hLseq); v[c["pos"]] = vec
                        (jo,) = torch.autograd.grad(g, u, grad_outputs=v, retain_graph=(k < len(tangents) - 1))
                        jo = jo.detach()
                        if kind == "null": accNull[L] += jo
                        elif kind == "self": accInt[L] += (c["ell"] - jo)
                        else:
                            qi = k - 2
                            if c["fi"] != Q[qi]["fi"]:
                                accJq[L][qi] += jo; cntq[L][qi] += 1
                    hd.remove()
                    del g, u, hLseq

            # ---- ranks per layer (logit / tuned / jacobian on the SAME facts & rank definition) ----
            for L in range(nL + 1):
                if L == last:
                    real = [rank(q["truefinal"], q["gid"]) for q in Q]; null = real[:]; logit = real[:]
                    tuned = real[:] if translators is not None else None
                else:
                    bL = accInt[L] / nC; nullv = accNull[L] / nC + bL
                    real = [rank(accJq[L][qi] / max(cntq[L][qi], 1) + bL, Q[qi]["gid"]) for qi in range(nQ)]
                    null = [rank(nullv, Q[qi]["gid"]) for qi in range(nQ)]
                    logit = [rank(head(norm(Q[qi]["h"][L].unsqueeze(0))).float()[0], Q[qi]["gid"]) for qi in range(nQ)]
                    if translators is not None:
                        tuned = [rank(head(norm(translators[L](Q[qi]["h"][L]).to(mdt).unsqueeze(0))).float()[0],
                                      Q[qi]["gid"]) for qi in range(nQ)]
                    else:
                        tuned = None
                rows.append(dict(model=name, split=split, layer=L,
                                 jaffine_median_rank=float(np.median(real)),
                                 null_median_rank=float(np.median(null)),
                                 logit_median_rank=float(np.median(logit)),
                                 tuned_median_rank=float(np.median(tuned)) if tuned is not None else float("nan")))
            ja = [r for r in rows if r["model"] == name and r["split"] == split]
            lo = (nL + 1) // 2
            print(f"  [{name}] {split}: jAff final={ja[-1]['jaffine_median_rank']:.0f} "
                  f"mid-min={min(r['jaffine_median_rank'] for r in ja[lo:]):.0f} "
                  f"null-mid-min={min(r['null_median_rank'] for r in ja[lo:]):.0f}", flush=True)
            del accJq, accInt, accNull, cntq  # free
        del translators
        pd.DataFrame(rows).to_csv(f"{h8.workdir()}/rank_curves_jlens_affine_{TAG}.csv", index=False)  # incremental
        h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/rank_curves_jlens_affine_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_curves_jlens_affine_{TAG}.csv")
    render_main(df, TAG, MODELS, SPLITS)
    render_control(df, TAG, SPLITS)
    if TUNED and "tuned_median_rank" in df and df.tuned_median_rank.notna().any():
        render_compare(df, TAG, MODELS)
    print("\ndone.", flush=True)


def render_compare(df, TAG, MODELS):
    """Three-way, same facts & rank metric: logit (dotted) vs tuned (dashed) vs Jacobian (solid), per
    model, on the Forget split."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    split = "Forget"
    mods = [m for m in MODELS if not df[(df.model == m) & (df.split == split)].empty]
    ncol = 3; nrow = int(np.ceil(len(mods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, mods):
        d = df[(df.model == name) & (df.split == split)].sort_values("layer")
        col = COLOR_MAP.get(name, "#333")
        ax.plot(d.layer, d.logit_median_rank, ":", lw=1.4, color=col, alpha=0.8, label="logit")
        ax.plot(d.layer, d.tuned_median_rank, "--", lw=1.6, color=col, alpha=0.9, label="tuned")
        ax.plot(d.layer, d.jaffine_median_rank, "-o", ms=3, lw=2.2, color=col, label="Jacobian")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.set_title(name)
        ax.set_xlabel("hidden-state index"); ax.legend(fontsize=8)
    for ax in axes.flat[len(mods):]: ax.axis("off")
    axes[0][0].set_ylabel("MEDIAN gold rank (log; lower = recoverable)")
    fig.suptitle(f"Three lenses on the SAME facts & rank metric ({TAG.upper()}, forget split, Q={QF} facts): "
                 "logit (dotted) · tuned (dashed) · Jacobian (solid)", fontweight="bold", fontsize=12)
    fig.tight_layout(); p = f"{h8.workdir()}/lens_compare_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); h8.upload_artifact(p)


def render_main(df, TAG, MODELS, SPLITS):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(7.5 * len(SPLITS), 6), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], SPLITS):
        for name in MODELS:
            d = df[(df.model == name) & (df.split == split)].sort_values("layer")
            if d.empty: continue
            ax.plot(d.layer, d.jaffine_median_rank, "-o", ms=4, lw=2.2,
                    color=COLOR_MAP.get(name, "#333"), label=name)
            ax.plot(d.layer, d.logit_median_rank, ":", lw=1.1, color=COLOR_MAP.get(name, "#333"), alpha=0.5)
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{split} ({'Unlearn' if split == 'Forget' else 'Keep'})")
        ax.set_xlabel("hidden-state index (0 = embeddings, N = final)")
    axes[0][0].set_ylabel("MEDIAN gold rank (log; lower = recoverable)")
    axes[0][0].legend(title="corrected Jacobian lens\n(dotted = logit lens)", fontsize=8)
    fig.suptitle(f"Rank-by-layer — CORRECTED averaged-Jacobian lens ({TAG.upper()}); solid = Jacobian, dotted = logit",
                 fontweight="bold")
    fig.tight_layout(); p = f"{h8.workdir()}/jlens_affine_rank_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); h8.upload_artifact(p)


def render_control(df, TAG, SPLITS):
    """Knowledge-vs-prior control: real (transport h) vs null (mean-h) gold rank, Perturbed vs Standard."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    split = "Forget" if "Forget" in SPLITS else SPLITS[0]
    pick = [x for x in ["Standard", "Perturbed", "NPO"] if x in df.model.unique()]
    if not pick: return
    fig, axes = plt.subplots(1, len(pick), figsize=(5.2 * len(pick), 5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], pick):
        d = df[(df.model == name) & (df.split == split)].sort_values("layer")
        ax.plot(d.layer, d.jaffine_median_rank, "-o", ms=4, lw=2.2, color="#1a7f37", label="real (transport h)")
        ax.plot(d.layer, d.null_median_rank, "--", lw=1.8, color="#999", label="null (mean-h prior)")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.set_title(name); ax.set_xlabel("hidden-state index")
    axes[0][0].set_ylabel("MEDIAN gold rank (log)"); axes[0][0].legend(fontsize=9)
    fig.suptitle(f"Faithfulness control ({TAG.upper()}, {split}): does the Jacobian lens read KNOWLEDGE or the month-prior?\n"
                 "faithful => real << null where the model KNOWS (Perturbed), real ~ null where it never knew (Standard)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); p = f"{h8.workdir()}/jlens_affine_control_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); h8.upload_artifact(p)


if __name__ == "__main__":
    main()
