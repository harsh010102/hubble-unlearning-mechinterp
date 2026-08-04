"""Compare LSR discrimination across candidate slots (no retraining — read-only eval).

Occupation gave weak Perturbed-vs-Standard discrimination (common-concept). This checks whether
an entity-specific slot (university / birthplace) discriminates better, with:
  - multi-word CALIBRATION: average C_f mass only over 'content' tokens (skip function words,
    punctuation, single-char, articles) so name tokens like St/Ann/Mary don't dilute the signal;
  - the double-norm fix on the final layer (true output logits).

Outputs (new files only):
  lsr_slot_check.csv     per (slot, model, entity): calibrated LSR
  lsr_slot_summary.csv   per (slot): Perturbed vs Standard medians, gap, Wilcoxon p, n
  lsr_slot_check.png     bar chart of per-slot per-model median LSR
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import build_prefixes, get_slot_value
import json

SLOTS = ["occupation", "university", "birthplace", "nationality"]
MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]
DEVICE = h8.DEVICE
TOPK = h8.config.LSR_TOPK
FRAC = h8.config.LSR_LAYER_FRAC
STOP = {"of", "the", "and", "at", "in", "for", "de", "du", "da", "la", "le", "el", "al", "on",
        "to", "der", "van", "von", "los", "las", "st", "a", "an", "or"}


def content_token(tok, tid):
    s = tok.decode([tid]).strip().lower()
    if len(s) <= 1:                       # single char / empty
        return False
    if s in STOP:
        return False
    if all(not c.isalnum() for c in s):   # punctuation only
        return False
    return True


def cf_ids(tid, E_ref):
    sims = E_ref @ E_ref[tid]
    return torch.topk(sims, TOPK + 1).indices     # gold + top-k neighbours (as in evaluation.cf_ids)


@torch.no_grad()
def calibrated_lsr(model, tok, prefix, gold, E_ref):
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEVICE)
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEVICE)
    if full.shape[1] <= pre.shape[1]:
        return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head
    last = len(hs) - 1
    positions = [p for p in range(pre.shape[1] - 1, full.shape[1] - 1)
                 if content_token(tok, int(full[0, p + 1]))]        # multi-word calibration
    if not positions:
        return None
    cf = np.zeros(len(hs))
    for L, h in enumerate(hs):
        masses = []
        for pos in positions:
            g = int(full[0, pos + 1])
            logits = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
            probs = torch.softmax(logits, -1)
            masses.append(float(probs[cf_ids(g, E_ref)].sum()))
        cf[L] = float(np.mean(masses))
    lo = int(len(cf) * FRAC)
    return float(cf[lo:].max())


def main():
    h8.hf_login()
    tok = h8.get_tokenizer()
    forget_set, _, _ = h8.load_yago_splits()
    metas = [json.loads(forget_set[i]["meta"]) for i in range(len(forget_set))]
    # (prefix, gold) per slot
    facts = {s: [(build_prefixes(m, s)["full_prefix"], get_slot_value(m, s)) for m in metas] for s in SLOTS}
    for s in SLOTS:
        avg_tok = np.mean([len(tok.encode(g, add_special_tokens=False)) for _, g in facts[s]])
        print(f"slot {s}: {len(facts[s])} facts, mean gold tokens={avg_tok:.1f}, e.g. {facts[s][0][1]!r}")

    pert = h8.load_named("Perturbed"); E_REF = h8.build_Eref(pert).to(DEVICE)
    h8.to_cpu(pert); del pert; h8.free_cuda()

    rows = []
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"  {name}: unavailable ({type(e).__name__})"); continue
        for s in SLOTS:
            for i, (pfx, gold) in enumerate(facts[s]):
                v = calibrated_lsr(m, tok, pfx, gold, E_REF)
                if v is not None:
                    rows.append(dict(slot=s, model=name, fact=i, entity=metas[i]["full_name"], lsr=v))
        print(f"  [{name}] done")
        h8.to_cpu(m); del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/lsr_slot_check.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/lsr_slot_check.csv")

    # per-slot discrimination: Perturbed vs Standard
    summ = []
    for s in SLOTS:
        P = df[(df.slot == s) & (df.model == "Perturbed")].lsr.values
        N = df[(df.slot == s) & (df.model == "Standard")].lsr.values
        if len(P) and len(N):
            p = mannwhitneyu(P, N, alternative="two-sided").pvalue
            summ.append(dict(slot=s, Perturbed_med=round(np.median(P), 3), Standard_med=round(np.median(N), 3),
                             gap=round(np.median(P) - np.median(N), 3), wilcoxon_p=f"{p:.2e}", n=len(P)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(f"{h8.workdir()}/lsr_slot_summary.csv", index=False); h8.upload_artifact(f"{h8.workdir()}/lsr_slot_summary.csv")
    print("\n=== per-slot LSR discrimination (Perturbed vs Standard) ===")
    print(sdf.to_string(index=False))
    print("\n=== per-slot per-model median LSR (surface-vs-erased) ===")
    piv = df.groupby(["slot", "model"]).lsr.median().unstack().reindex(columns=MODELS)
    print(piv.round(3).to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    piv.reindex(SLOTS).plot(kind="bar", ax=ax, color=[h8.COLOR_MAP.get(m, "#888") for m in piv.columns])
    ax.set_ylabel("median calibrated LSR"); ax.set_title("LSR by slot and model (which slot discriminates best?)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{h8.workdir()}/lsr_slot_check.png", dpi=150, bbox_inches="tight")
    h8.upload_artifact(f"{h8.workdir()}/lsr_slot_check.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
