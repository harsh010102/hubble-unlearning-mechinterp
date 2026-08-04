"""Multi-paraphrase invariance robustness (strengthens the single-paraphrase invariance claim).
For several meaning-preserving rewordings of the trigger clause, measure each metric's invariance shift
= |Cliff's delta(canonical, paraphrase_k)| on the Perturbed (knows) model. A faithful metric should stay
LOW across ALL rewordings; lexical metrics should be high for all. Reuses run_mech_metaeval internals.

Writes: mech_paraphrase_robustness_<tag>.csv, mech_paraphrase_robustness_<tag>.png. Not in RESULTS.md."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import build_infill_examples
from run_mech_metaeval import probe, cell_metrics, cliff, SPEC, BAND_FRAC

DEV = h8.DEVICE
PARAS = {                                   # trigger rewordings (meaning-preserving)
    "p1_birth_date_of": ("was born on", "has a recorded birth date of"),
    "p2_came_into_world": ("was born on", "came into the world on"),
    "p3_birthday_on": ("was born on", "has a birthday on"),
    "p4_born_on_date_of": ("was born on", "was born on the date of"),
}


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    examples = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    N = len(bd); full_prefix = [p for p, _ in bd]; gold = [g for _, g in bd]
    cands = [ex["candidates"] for ex in examples]
    covered = sum(PARAS["p1_birth_date_of"][0] in p for p in full_prefix)
    print(f"scale={TAG}  n={N}  trigger present in {covered}/{N} prompts  paraphrases={list(PARAS)}")

    pert = h8.load_named("Perturbed")
    band = np.arange(int(len(probe(pert, tok, full_prefix[0], gold[0], np.arange(1))["traj"]) * BAND_FRAC),
                     len(probe(pert, tok, full_prefix[0], gold[0], np.arange(1))["traj"]))
    print("[cell] canonical (Perturbed)"); full_m, full_tr = cell_metrics(pert, tok, full_prefix, gold, cands, band)
    para_cells = {}
    for label, tf in PARAS.items():
        pfx = [p.replace(*tf) if tf[0] in p else p for p in full_prefix]
        print(f"[cell] {label}"); mm, tr = cell_metrics(pert, tok, pfx, gold, cands, band)
        para_cells[label] = (mm, tr)
    h8.to_cpu(pert); del pert; h8.free_cuda()
    std = h8.load_named("Standard"); std.eval()
    print("[cell] Standard (anchor Ø)"); _, ret_tr = cell_metrics(std, tok, full_prefix, gold, cands, band)
    h8.to_cpu(std); del std; h8.free_cuda()

    Ftr = np.array(full_tr); Rtr = np.array(ret_tr)
    def dmech_of(trs):
        out = np.full(N, np.nan)
        for i in range(N):
            aK = np.median(np.delete(Ftr, i, 0), 0); aO = np.median(np.delete(Rtr, i, 0), 0)
            lr = np.log(np.asarray(trs[i], float))
            out[i] = float(np.mean(np.abs(lr[band] - np.log(aO)[band])) - np.mean(np.abs(lr[band] - np.log(aK)[band])))
        return out
    full_m["D_mech"] = dmech_of(full_tr)
    for label in para_cells:
        mm, tr = para_cells[label]; mm["D_mech"] = dmech_of(tr); para_cells[label] = (mm, tr)

    rows = []
    for m, (hi, fam) in SPEC.items():
        r = dict(metric=m, family=fam); shifts = []
        for label in PARAS:
            s = abs(cliff(full_m[m], para_cells[label][0][m])); r[label] = round(s, 3); shifts.append(s)
        r["mean_shift"] = round(float(np.mean(shifts)), 3); r["max_shift"] = round(float(np.max(shifts)), 3)
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("mean_shift")
    T.to_csv(f"{h8.workdir()}/mech_paraphrase_robustness_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/mech_paraphrase_robustness_{TAG}.csv")
    print("\n=== invariance shift |Cliff δ(canonical, paraphrase)| per metric (0 = invariant) ===")
    print(T[["metric", "family"] + list(PARAS) + ["mean_shift", "max_shift"]].to_string(index=False))

    # figure: mean invariance shift per metric (mechanistic black, lexical gray), with per-paraphrase spread
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12})
    d = T.sort_values("mean_shift"); y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = ["#111" if m in ("D_mech", "min_rank") else "#BBB" for m in d.metric]
    ax.barh(y, d.mean_shift, color=colors, height=0.6, edgecolor="#000", linewidth=0.4, zorder=3)
    for yi, m in enumerate(d.metric):
        xs = [d.iloc[yi][lab] for lab in PARAS]
        ax.plot(xs, [yi] * len(xs), "o", ms=4, color="#E76F51" if m == "D_mech" else "#777", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(d.metric, fontsize=11)
    ax.set_xlim(0, 1.02); ax.set_xlabel("invariance shift  |Cliff δ(canonical, paraphrase)|   (0 = invariant; dots = each paraphrase)")
    ax.axvline(0.0, color="#000", lw=0.8); ax.grid(axis="x", color="#eee", zorder=0); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_title(f"Invariance across {len(PARAS)} paraphrases ({TAG.upper()}) — D_mech stays low; lexical stay high",
                 fontweight="bold", loc="left", pad=10)
    fig.tight_layout(); p = f"{h8.workdir()}/mech_paraphrase_robustness_{TAG}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); h8.upload_artifact(p)
    print("\ndone.")


if __name__ == "__main__":
    main()
