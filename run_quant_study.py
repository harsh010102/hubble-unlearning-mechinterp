"""Quantization robustness study (8B only). Does quantizing an unlearned model RESTORE the
forgotten knowledge? Measures Unlearn/Keep infill accuracy (full_prefix) at bf16 vs int8 vs nf4
(4-bit) for all six models. Restoration = Unlearn_acc(quantized) - Unlearn_acc(bf16) > 0 would
mean the unlearning was superficial (a coarser numeric representation undoes it).

Writes NEW files (not RESULTS.md): quant_study_8b.csv, quant_study_8b.png.
"""
import os, sys
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import build_infill_examples

MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]
DEV = h8.DEVICE


def repo_for(name):
    if name == "Standard":
        return h8.config.MODEL_STANDARD, h8.config.REVISION
    if name == "Perturbed":
        return h8.config.MODEL_PERTURBED, h8.config.REVISION
    return h8.config.MODEL_REPO_TMPL.format(method=name.lower()), None


def load_prec(name, bits):
    if bits == 16:
        return h8.load_named(name)
    repo, rev = repo_for(name)
    if bits == 8:
        qc = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(repo, quantization_config=qc, device_map={"": 0},
                                             revision=rev, torch_dtype=torch.bfloat16)
    m.eval()
    return m


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    ex_forget = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    ex_keep = build_infill_examples(ks, h8.TARGET_SLOT, "full_prefix", pool)

    rows = []
    for name in MODELS:
        for bits, ptag in [(16, "bf16"), (8, "int8"), (4, "nf4")]:
            try:
                m = load_prec(name, bits)
            except Exception as e:
                print(f"  {name}/{ptag}: {type(e).__name__}: {e}"); continue
            u = h8.infill_accuracy(m, tok, ex_forget, DEV)
            k = h8.infill_accuracy(m, tok, ex_keep, DEV)
            rows.append(dict(model=name, precision=ptag, Unlearn=u, Keep=k))
            print(f"  [{name}/{ptag}] Unlearn={u:.3f} Keep={k:.3f}")
            del m; h8.free_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/quant_study_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/quant_study_{TAG}.csv")

    # restoration = Unlearn(quant) - Unlearn(bf16)
    piv = df.pivot(index="model", columns="precision", values="Unlearn").reindex(MODELS)
    print("\n=== Unlearn (forget) accuracy by precision — restoration if it rises ===")
    print(piv[["bf16", "int8", "nf4"]].round(3).to_string())
    for p in ["int8", "nf4"]:
        piv[f"{p}_restore"] = piv[p] - piv["bf16"]
    print("\n=== restoration (quant - bf16) ===")
    print(piv[["int8_restore", "nf4_restore"]].round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ["Unlearn", "Keep"]):
        p = df.pivot(index="model", columns="precision", values=metric).reindex(MODELS)[["bf16", "int8", "nf4"]]
        p.plot(kind="bar", ax=ax); ax.set_title(f"{metric} infill accuracy by precision")
        ax.axhline(0.1, ls="--", c="gray"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Quantization robustness ({TAG.upper()}) — does forget knowledge return under int8/nf4?",
                 fontweight="bold")
    fig.tight_layout(); fp = f"{h8.workdir()}/quant_study_{TAG}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight"); h8.upload_artifact(fp)
    print("\ndone.")


if __name__ == "__main__":
    main()
