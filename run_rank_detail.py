"""Per-fact logit-lens rank detail (standalone; does NOT touch existing artifacts).

Computes the gold-token rank-by-layer for EVERY forget and retain entity across all six
models, saves the individual rankings, and regenerates the aggregate with MEDIAN (robust to
the heavy-tailed outliers that make the mean invert Standard vs Perturbed).

New artifacts only:
  rank_perfact.csv           one row per (model, split, entity): final-layer rank
  rank_curves_agg.csv        per-layer mean AND median rank, per (split, model)
  rank_aggregate.csv         final-layer mean vs median summary
  logitlens_rank_median_8b.png   the median rank-by-layer figure
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import hubble8b as h8

MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]


@torch.no_grad()
def rank_curve_fixed(model, tok, prefix, target, device):
    """Rank-by-layer gold-token rank, with the final-layer double-norm bug fixed.

    HF's hidden_states[-1] is ALREADY passed through model.model.norm, so applying norm()
    again (as h8.layerwise_target_rank does) double-normalises the endpoint and spikes it.
    Here intermediate (pre-norm) states get one norm; the final point uses the model's REAL
    output logits (out.logits = lm_head(hidden_states[-1])) — the true final ranking.
    """
    pre = tok(prefix, return_tensors="pt").input_ids.to(device)
    tgt = tok.encode(target, add_special_tokens=False)
    if len(tgt) == 0:
        return None
    full = torch.cat([pre, torch.tensor([tgt], device=device)], dim=1)
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()                 # model's actual output logits
    norm, head = model.model.norm, model.lm_head
    last = len(hs) - 1
    positions = list(range(pre.shape[1] - 1, full.shape[1] - 1))
    if not positions:
        return None
    per_layer = []
    for li, h in enumerate(hs):
        ranks = []
        for pos in positions:
            gold = int(full[0, pos + 1])
            logits = true_logits[pos] if li == last else head(norm(h[:, pos, :])).float()[0]
            ranks.append(int((logits > logits[gold]).sum().item()) + 1)
        per_layer.append(float(np.mean(ranks)))
    return per_layer


def main():
    h8.hf_login()
    tok = h8.get_tokenizer()
    forget_set, keep_set, _ = h8.load_yago_splits()
    sets = {"Forget": forget_set, "Retain": keep_set}
    facts = {k: h8.facts_for(v, slot=h8.TARGET_SLOT) for k, v in sets.items()}
    names = {k: [json.loads(v[i]["meta"])["full_name"] for i in range(len(v))] for k, v in sets.items()}

    perfact = []
    curves = {k: {} for k in sets}          # curves[split][model] = ndarray (n_facts, n_layers)
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"{name}: unavailable ({type(e).__name__})"); continue
        for sk in sets:
            arr = []
            for i, (prefix, gold) in enumerate(facts[sk]):
                try:
                    pl = rank_curve_fixed(m, tok, prefix, gold, h8.DEVICE)  # double-norm fixed
                except Exception:
                    pl = None
                if pl is None or not np.all(np.isfinite(pl)):
                    continue
                arr.append(pl)
                perfact.append(dict(model=name, split=sk, entity=names[sk][i],
                                    gold=gold, final_rank=float(pl[-1])))
            if arr:
                curves[sk][name] = np.array(arr)
                fr = curves[sk][name][:, -1]
                print(f"[{name}] {sk}: n={len(fr)} mean={fr.mean():.1f} median={np.median(fr):.1f}")
        h8.to_cpu(m); del m; h8.free_cuda()

    # 1) individual rankings
    pd.DataFrame(perfact).to_csv(f"{h8.workdir()}/rank_perfact.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_perfact.csv")

    # 2) per-layer mean + median curves, and 3) final-layer summary
    curve_rows, agg_rows = [], []
    for sk in sets:
        for name, A in curves[sk].items():
            mean_c, med_c = A.mean(axis=0), np.median(A, axis=0)
            for L in range(A.shape[1]):
                curve_rows.append(dict(split=sk, model=name, layer=L,
                                       mean_rank=float(mean_c[L]), median_rank=float(med_c[L])))
            agg_rows.append(dict(split=sk, model=name, n=int(A.shape[0]),
                                 final_mean=round(float(mean_c[-1]), 1),
                                 final_median=round(float(med_c[-1]), 1)))
    pd.DataFrame(curve_rows).to_csv(f"{h8.workdir()}/rank_curves_agg.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_curves_agg.csv")
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(f"{h8.workdir()}/rank_aggregate.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_aggregate.csv")
    print("\n=== final-layer rank: MEAN vs MEDIAN ===")
    print(agg.to_string(index=False))

    # 4) median rank-by-layer figure (new filename; does not touch logitlens_rank_8b.png)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, sk in zip(axes, ["Forget", "Retain"]):
        for name in MODELS:
            if name in curves[sk]:
                med = np.median(curves[sk][name], axis=0)
                ax.plot(range(len(med)), med, marker="o", ms=3, label=name, color=h8.COLOR_MAP[name])
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(f"{sk} (Unlearn)" if sk == "Forget" else f"{sk} (Keep)", fontweight="bold")
        ax.set_xlabel("hidden-state index (0 = embeddings, N = final layer)")
    axes[0].set_ylabel("MEDIAN rank of gold token (log; lower = recoverable)")
    axes[0].legend()
    fig.suptitle("Rank-by-layer logit lens - MEDIAN over facts, true final output (8B)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{h8.workdir()}/logitlens_rank_median_8b.png", dpi=150, bbox_inches="tight")
    h8.upload_artifact(f"{h8.workdir()}/logitlens_rank_median_8b.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
