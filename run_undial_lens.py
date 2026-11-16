"""UNDIAL logit-lens readout: does distillation SUPPRESS or ERASE the birthdate?
Computes, at the current scale (HUBBLE_SCALE), the per-fact boundary-safe gold-token rank trajectory
for Perturbed (knows anchor K), Standard (never-knew anchor Ø) and UNDIAL, on the forget & retain sets.
Saves: median rank curves (for the logit-lens plot) + UNDIAL's per-fact D_mech (dual-anchor, log, band,
leave-one-fact-out) so we can place it on the knows<->never-knew line.  D_mech>0 = suppressed, <0 = erased."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

BAND_FRAC = 0.5                       # matches run_mech_metaeval (mid-late half of the stack)
FMT = os.environ.get("RANK_FMT", "full_prefix")
DEV = h8.DEVICE
NAMES = ["Perturbed", "Standard", "UNDIAL"]


def curve_list(m, tok, facts):
    """Per-fact rank trajectory aligned to `facts` (None where the probe failed)."""
    out = []
    for p, g in facts:
        c = h8.layerwise_target_rank(m, tok, p, g, DEV)
        out.append(c if (c is not None and np.all(np.isfinite(c))) else None)
    return out


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    ff = h8.facts_for(fs, h8.TARGET_SLOT, FMT); kf = h8.facts_for(ks, h8.TARGET_SLOT, FMT)
    print(f"scale={TAG} fmt={FMT} n_forget={len(ff)} n_retain={len(kf)}", flush=True)

    raw = {}                          # (name, split) -> list aligned to facts (None where invalid)
    for name in NAMES:
        m = h8.load_named(name); m.eval()
        for split, facts in [("Forget", ff), ("Retain", kf)]:
            raw[(name, split)] = curve_list(m, tok, facts)
            med = np.median(np.array([c for c in raw[(name, split)] if c is not None], float), axis=0)
            print(f"  [{name}] {split}: final={med[-1]:.0f}  mid-min={np.min(med[len(med)//2:]):.0f}", flush=True)
        h8.to_cpu(m); del m; h8.free_cuda()

    # ---- median rank curves for the plot ----
    rows = []
    for (name, split), lst in raw.items():
        med = np.median(np.array([c for c in lst if c is not None], float), axis=0)
        for L, v in enumerate(med):
            rows.append(dict(model=name, split=split, layer=L, median_rank=float(v)))
    pd.DataFrame(rows).to_csv(f"{h8.workdir()}/rank_curve_undial_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/rank_curve_undial_{TAG}.csv")

    # ---- UNDIAL D_mech on the forget set (aligned facts valid for all three models) ----
    valid = [i for i in range(len(ff)) if all(raw[(n, "Forget")][i] is not None for n in NAMES)]
    P = np.array([raw[("Perturbed", "Forget")][i] for i in valid], float)   # knows K
    O = np.array([raw[("Standard",  "Forget")][i] for i in valid], float)   # never-knew Ø
    U = np.array([raw[("UNDIAL",    "Forget")][i] for i in valid], float)
    nL = U.shape[1]; band = np.arange(int(nL * BAND_FRAC), nL)
    D = np.empty(len(valid))
    for i in range(len(valid)):
        aK = np.median(np.delete(P, i, 0), 0)          # leave-one-fact-out knows anchor
        aO = np.median(np.delete(O, i, 0), 0)          # leave-one-fact-out never-knew anchor
        lr = np.log(U[i])
        D[i] = np.mean(np.abs(lr[band] - np.log(aO)[band])) - np.mean(np.abs(lr[band] - np.log(aK)[band]))

    summ = dict(scale=TAG, n=len(valid), band=f"{band[0]}-{band[-1]}",
                undial_dmech_median=round(float(np.median(D)), 3),
                undial_dmech_mean=round(float(np.mean(D)), 3),
                undial_frac_positive=round(float((D > 0).mean()), 3),
                undial_forget_final=round(float(np.median(U[:, -1])), 1),
                undial_forget_midmin=round(float(np.median(np.min(U[:, band], axis=1))), 1),
                standard_forget_final=round(float(np.median(O[:, -1])), 1),
                perturbed_forget_final=round(float(np.median(P[:, -1])), 1))
    print("UNDIAL_DMECH " + json.dumps(summ), flush=True)
    pd.DataFrame([summ]).to_csv(f"{h8.workdir()}/undial_dmech_{TAG}.csv", index=False)
    pd.DataFrame(dict(fact=list(valid), D_mech=D)).to_csv(f"{h8.workdir()}/undial_dmech_perfact_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/undial_dmech_{TAG}.csv")
    h8.upload_artifact(f"{h8.workdir()}/undial_dmech_perfact_{TAG}.csv")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
