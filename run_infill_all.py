"""Recompute infill accuracy for ALL six models with the boundary-safe sequence_logprob fix,
upload fresh infill_<Model>.csv, and regenerate the bars figure. Scale-aware (HUBBLE_SCALE)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

MODELS = ["Standard", "Perturbed", "NPO", "GradDiff", "IDK", "RMU"]


def main():
    h8.hf_login()
    TAG = h8.config.SCALE_TAG
    forget_set, keep_set, unseen_set = h8.load_yago_splits()
    splits = {"Unseen": unseen_set, "Unlearn": forget_set, "Keep": keep_set}
    pool = h8.build_distractor_pool(forget_set, keep_set, unseen_set, h8.TARGET_SLOT)
    frames = []
    for name in MODELS:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"  {name}: unavailable ({type(e).__name__})"); continue
        df = h8.run_infill(m, name, splits, pool, target_slot=h8.TARGET_SLOT)
        p = f"{h8.workdir()}/infill_{name}.csv"; df.to_csv(p, index=False); h8.upload_artifact(p)
        frames.append(df)
        h8.to_cpu(m); del m; h8.free_cuda()

    if not frames:
        print("no infill computed"); return
    inf = pd.concat(frames, ignore_index=True)
    print("\n=== full_prefix (Unlearn / Keep / Unseen) ===")
    piv = inf[inf.template_format == "full_prefix"].pivot_table(index="model", columns="split", values="accuracy")
    print(piv.reindex(MODELS)[["Unlearn", "Keep", "Unseen"]].round(3).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12)); axes = axes.flatten()
    for ax, fmt in zip(axes, h8.TEMPLATE_FORMATS):
        p = inf[inf.template_format == fmt].pivot(index="split", columns="model", values="accuracy").reindex(["Unseen", "Unlearn", "Keep"])
        p = p[[m for m in MODELS if m in p.columns]]
        p.plot(kind="bar", ax=ax, color=[h8.COLOR_MAP[m] for m in p.columns], legend=(fmt == h8.TEMPLATE_FORMATS[0]))
        ax.set_title(fmt); ax.set_ylim(0, 1); ax.axhline(0.1, ls="--", c="gray"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Hubble-{TAG.upper()} / YAGO birthdate — infill accuracy (tokenizer fix)", fontweight="bold")
    fig.tight_layout()
    fp = f"{h8.workdir()}/infill_bars_{TAG}.png"; fig.savefig(fp, dpi=150, bbox_inches="tight"); h8.upload_artifact(fp)
    print("\ndone.")


if __name__ == "__main__":
    main()
