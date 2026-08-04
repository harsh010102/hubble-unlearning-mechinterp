"""Tuned lens vs logit lens (8B). Trains a per-layer affine translator (identity-init) that maps
each layer's hidden state so the model's OWN unembedding reproduces the final-layer distribution
(Belrose et al. 2023). This calibrates the mid-stack readout the raw logit lens gets wrong.

For each model: train the lens on WikiText, then compute rank-by-layer of the gold birthdate under
BOTH the logit lens and the tuned lens (forget + retain, median over facts). Saves curves + a
side-by-side comparison figure so we can discuss the difference.

Env: TUNED_MODELS (default "Standard,Perturbed,NPO,IDK"), TUNED_STEPS (default 400).
"""
import os, sys, json, random
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

MODELS = [m.strip() for m in os.environ.get("TUNED_MODELS", "Standard,Perturbed,NPO,IDK").split(",") if m.strip()]
# NOTE (fix, not re-run): the first run (job 18414) was UNDERTRAINED — KL/layer bounced ~3-5 and never
# converged because the per-layer KL was SUMMED (huge gradient) with lr=1e-3 and only 400 single-seq
# steps. Fixes below: mean the loss over layers, lower lr, many more steps. Needs ~a few k steps to
# get KL/layer well below 1 before the tuned-lens curves are trustworthy.
STEPS = int(os.environ.get("TUNED_STEPS", "3000"))
DEV = h8.DEVICE


def build_translators(n, d, device):
    ts = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(n)])
    for t in ts:
        nn.init.eye_(t.weight); nn.init.zeros_(t.bias)
    return ts.to(device=device, dtype=torch.float32)


def train_lens(model, tok, translators, texts, steps):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm, head = model.model.norm, model.lm_head
    mdt = next(model.parameters()).dtype
    opt = torch.optim.Adam(translators.parameters(), lr=3e-4)
    rng = random.Random(h8.config.SEED); queue = []
    for step in range(steps):
        if not queue:
            queue = texts[:]; rng.shuffle(queue)
        enc = tok(queue.pop(), return_tensors="pt", truncation=True, max_length=256).to(DEV)
        if enc.input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            final_lp = torch.log_softmax(out.logits[0].float(), -1)
            final_p = final_lp.exp()
        T = enc.input_ids.shape[1]
        pos = torch.arange(0, T - 1)
        if len(pos) > 32:
            pos = pos[torch.randperm(len(pos))[:32]]
        loss = 0.0
        for L in range(len(translators)):                   # translators for hs[0..n_layers-1]
            tr = translators[L](hs[L][0, pos, :].float())
            lp_L = torch.log_softmax(head(norm(tr.to(mdt))).float(), -1)
            loss = loss + (final_p[pos] * (final_lp[pos] - lp_L)).sum(-1).mean()   # KL(final || lens)
        loss = loss / len(translators)        # mean over layers -> stable gradient scale
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"    lens step {step+1}/{steps} KL/layer={float(loss)/len(translators):.3f}")
    return translators


@torch.no_grad()
def rank_curves(model, tok, prefix, target, translators=None):
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
    full = tok(prefix + " " + target, return_tensors="pt").input_ids.to(DEV)
    if full.shape[1] <= pre.shape[1]:
        return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states; true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head; mdt = next(model.parameters()).dtype
    last = len(hs) - 1
    per_layer = []
    for L, h in enumerate(hs):
        ranks = []
        for pos in range(pre.shape[1] - 1, full.shape[1] - 1):
            if L == last:
                logits = true_logits[pos]
            elif translators is None:
                logits = head(norm(h[:, pos, :])).float()[0]           # raw logit lens
            else:
                tr = translators[L](h[0, pos, :].float())
                logits = head(norm(tr.to(mdt))).float()                # tuned lens
            gold = full[0, pos + 1]
            ranks.append(int((logits > logits[gold]).sum().item()) + 1)
        per_layer.append(float(np.mean(ranks)))
    return per_layer


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, _ = h8.load_yago_splits()
    train_texts = h8.load_retain_texts().select(range(2000))["text"]   # WikiText to fit the lens
    f_facts = h8.facts_for(fs, h8.TARGET_SLOT); k_facts = h8.facts_for(ks, h8.TARGET_SLOT)

    rows, curves = [], {}      # curves[(model, lens, split)] = per-layer median
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"{name}: {type(e).__name__}"); continue
        d = m.config.hidden_size; nL = m.config.num_hidden_layers + 1
        print(f"[{name}] training tuned lens ({nL-1} translators, {STEPS} steps)")
        tr = build_translators(nL - 1, d, DEV)
        train_lens(m, tok, tr, list(train_texts), STEPS)
        for split, facts in [("Forget", f_facts), ("Retain", k_facts)]:
            for lens_name, lens in [("logit", None), ("tuned", tr)]:
                cs = [rank_curves(m, tok, p, g, lens) for p, g in facts]
                cs = [c for c in cs if c is not None and np.all(np.isfinite(c))]
                med = np.median(np.array(cs), axis=0)
                curves[(name, lens_name, split)] = med
                for L, v in enumerate(med):
                    rows.append(dict(model=name, lens=lens_name, split=split, layer=L, median_rank=float(v)))
                print(f"    {split}/{lens_name}: final rank={med[-1]:.0f}")
        del tr; h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/tuned_vs_logit_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/tuned_vs_logit_{TAG}.csv")

    # comparison figure: logit (dashed) vs tuned (solid) on the Forget panel, per model
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, split in zip(axes, ["Forget", "Retain"]):
        for name in MODELS:
            for lens_name, ls in [("logit", "--"), ("tuned", "-")]:
                key = (name, lens_name, split)
                if key in curves:
                    c = curves[key]
                    ax.plot(range(len(c)), c, ls, lw=2 if lens_name == "tuned" else 1.3,
                            color=h8.COLOR_MAP.get(name, "#333"),
                            label=f"{name} ({lens_name})" if split == "Forget" else None)
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.set_title(split)
        ax.set_xlabel("hidden-state index")
    axes[0].set_ylabel("median gold-token rank (log)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Tuned lens (solid) vs logit lens (dashed) — rank-by-layer ({TAG.upper()})", fontweight="bold")
    fig.tight_layout(); p = f"{h8.workdir()}/tuned_vs_logit_{TAG}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
