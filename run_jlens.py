"""Full Jacobian-lens run: J-lens gold-token rank by layer for all six Hubble models (forget + retain,
median over facts), using the validated hook-injection + double-backward JVP (per-input local Jacobian).
Produces the J-lens analog of the logit-lens rank plot, so we can compare readouts.

NOTE (honest): this is the PER-INPUT local Jacobian (J . h_L). It is noisier than the paper's averaged
Jacobian; the median over facts smooths it. Env JLENS_NFACTS caps facts per split (for 8B cost).

Outputs: rank_curves_jlens_<tag>.csv (per-fact + median), jlens_rank_median_<tag>.png."""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.evaluation import COLOR_MAP

MODELS = [m.strip() for m in os.environ.get("JLENS_MODELS", "Standard,Perturbed,NPO,GradDiff,IDK,RMU").split(",") if m.strip()]
NFACTS = int(os.environ["JLENS_NFACTS"]) if os.environ.get("JLENS_NFACTS") else None
DEV = h8.DEVICE


def repo_for(name):
    if name == "Standard": return h8.config.MODEL_STANDARD, h8.config.REVISION
    if name == "Perturbed": return h8.config.MODEL_PERTURBED, h8.config.REVISION
    return h8.config.MODEL_REPO_TMPL.format(method=name.lower()), None


def load_eager(name):
    repo, rev = repo_for(name)
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def jlens_ranks(m, tok, prefix, gold, inject, inject_hook):
    """per-layer J-lens rank of the first gold token at the answer position."""
    layers, norm, head = m.model.layers, m.model.norm, m.lm_head; nL = len(layers)
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
    if full.shape[1] <= pre.shape[1]: return None
    pos = pre.shape[1] - 1; gold_id = int(full[0, pos + 1])
    with torch.no_grad():
        out = m(input_ids=full, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states; true_logits = out.logits[0].float(); last = len(hs) - 1
    ranks = []
    for L in range(len(hs)):
        if L == last:
            jout = true_logits[pos]
        else:
            hL = hs[L][0].float().detach().requires_grad_(True)
            inject["h"] = hL.unsqueeze(0)
            hd = layers[L].register_forward_pre_hook(inject_hook, with_kwargs=True)
            y = m(input_ids=full, use_cache=False).logits[0, pos]
            u = torch.zeros_like(y, requires_grad=True)
            (g,) = torch.autograd.grad(y, hL, grad_outputs=u, create_graph=True)
            v = torch.zeros_like(hL); v[pos] = hL[pos].detach()
            (jout,) = torch.autograd.grad(g, u, grad_outputs=v)
            hd.remove()
        ranks.append(int((jout > jout[gold_id]).sum()) + 1)
    return ranks


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    ff = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    kf = h8.facts_for(ks, h8.TARGET_SLOT, "full_prefix")
    if NFACTS: ff, kf = ff[:NFACTS], kf[:NFACTS]
    print(f"scale={TAG} models={MODELS} n_forget={len(ff)} n_retain={len(kf)} (J-lens: per-input Jacobian)")

    inject = {}
    def inject_hook(module, args, kwargs):
        newh = inject["h"]
        if len(args) >= 1: return (newh,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = newh; return args, kwargs

    rows, curves = [], {}
    for name in MODELS:
        try:
            m = load_eager(name)
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}"); continue
        for split, facts in [("Forget", ff), ("Retain", kf)]:
            cs = []
            for prefix, gold in facts:
                r = jlens_ranks(m, tok, prefix, gold, inject, inject_hook)
                if r is not None: cs.append(r)
            med = np.median(np.array(cs), axis=0)
            curves[(name, split)] = med
            for L, v in enumerate(med):
                rows.append(dict(model=name, split=split, layer=L, jlens_median_rank=float(v)))
            lo = len(med) // 2
            print(f"  [{name}] {split}: final={med[-1]:.0f}  mid-min={np.min(med[lo:]):.0f}  (n={len(cs)})")
        h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/rank_curves_jlens_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_curves_jlens_{TAG}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, split in zip(axes, ["Forget", "Retain"]):
        for name in MODELS:
            if (name, split) in curves:
                c = curves[(name, split)]
                ax.plot(range(len(c)), c, marker="o", ms=4, lw=2, color=COLOR_MAP.get(name, "#333"),
                        label=name if split == "Forget" else None)
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{split} ({'Unlearn' if split == 'Forget' else 'Keep'})")
        ax.set_xlabel("hidden-state index (0 = embeddings, N = final layer)")
    axes[0].set_ylabel("MEDIAN Jacobian-lens rank of gold token (log; lower = recoverable)")
    axes[0].legend(title="model", fontsize=9)
    fig.suptitle(f"Rank-by-layer JACOBIAN LENS — forget vs retain ({TAG.upper()}, per-input Jacobian)",
                 fontweight="bold")
    fig.tight_layout()
    p = f"{h8.workdir()}/jlens_rank_median_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
