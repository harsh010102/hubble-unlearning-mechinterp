"""Two robustness analyses for the thesis cross-benchmark section (Hubble-8B):
  (1) Layer-band sensitivity of the D_mech faithfulness result (bands 12-30, 18-36, 24-36).
  (2) The IDK unlearned model's per-fact D_mech distribution vs the two anchors, to back the claim that
      IDK's forgetting is a surface redirection rather than internal removal (the reviewer's IDK-vs-Standard
      rank-9 objection: the distinction must come from the trajectory, not the coincident final rank).
Reuses the meta-evaluation probe. Prints tables; writes sensitivity_8b.txt.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from hubble8b.data import build_infill_examples
from run_mech_metaeval import probe, cliff, PARA
from scipy.stats import wilcoxon


def dmech_band(trs, Ftr, Rtr, band):
    """Per-fact D_mech over a given band, with leave-one-fact-out anchor medians."""
    N = len(trs); out = np.full(N, np.nan)
    for i in range(N):
        if trs[i] is None:
            continue
        aK = np.median(np.delete(Ftr, i, 0), 0)
        aO = np.median(np.delete(Rtr, i, 0), 0)
        lr = np.log(np.asarray(trs[i], float))
        out[i] = float(np.mean(np.abs(lr[band] - np.log(aO)[band]))
                       - np.mean(np.abs(lr[band] - np.log(aK)[band])))
    return out


def main():
    h8.hf_login(); tok = h8.get_tokenizer()
    fs, ks, us = h8.load_yago_splits()
    pool = h8.build_distractor_pool(fs, ks, us, h8.TARGET_SLOT)
    examples = build_infill_examples(fs, h8.TARGET_SLOT, "full_prefix", pool)
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    N = len(bd)
    full_prefix = [p for p, _ in bd]; gold = [g for _, g in bd]
    para_prefix = [p.replace(*PARA) if PARA[0] in p else p for p in full_prefix]
    wrong_gold = [next(c for c in examples[i]["candidates"] if c != gold[i]) for i in range(N)]
    dummy = np.arange(1)

    def trajs(model, prefixes, golds):
        out = []
        for i in range(N):
            pr = probe(model, tok, prefixes[i], golds[i], dummy)
            out.append(pr["traj"] if pr else None)
        return out

    print("[load] Perturbed", flush=True); P = h8.load_named("Perturbed")
    full_tr = trajs(P, full_prefix, gold); para_tr = trajs(P, para_prefix, gold)
    pw_tr = trajs(P, full_prefix, wrong_gold); h8.to_cpu(P); del P; h8.free_cuda()
    print("[load] Standard", flush=True); S = h8.load_named("Standard")
    ret_tr = trajs(S, full_prefix, gold); h8.to_cpu(S); del S; h8.free_cuda()
    print("[load] IDK", flush=True); I = h8.load_named("idk")
    idk_tr = trajs(I, full_prefix, gold); h8.to_cpu(I); del I; h8.free_cuda()

    Ftr = np.array(full_tr); Rtr = np.array(ret_tr); nL = Ftr.shape[1]
    lines = []
    BANDS = {"12-30": np.arange(12, 31), "18-36 (default)": np.arange(18, nL), "24-36": np.arange(24, nL)}
    lines.append("=== BAND SENSITIVITY (D_mech faithfulness, Hubble-8B) ===")
    lines.append(f"{'band':18s} {'invar_shift':12s} {'discrim':10s} {'faithful':10s}")
    for bname, band in BANDS.items():
        fd = dmech_band(full_tr, Ftr, Rtr, band); pa = dmech_band(para_tr, Ftr, Rtr, band)
        rd = dmech_band(ret_tr, Ftr, Rtr, band); wd = dmech_band(pw_tr, Ftr, Rtr, band)
        inv = abs(cliff(fd, pa)); dis = abs(cliff(np.concatenate([fd, pa]), np.concatenate([rd, wd])))
        lines.append(f"{bname:18s} {inv:<12.3f} {dis:<10.3f} {dis*(1-inv):<10.3f}")

    band = np.arange(18, nL)
    idk_d = dmech_band(idk_tr, Ftr, Rtr, band); ret_d = dmech_band(ret_tr, Ftr, Rtr, band)
    ok = ~(np.isnan(idk_d) | np.isnan(ret_d))
    lines.append("\n=== IDK D_mech DISTRIBUTION (default band 18-36) ===")
    lines.append(f"IDK      median {np.nanmedian(idk_d):+.3f}  IQR [{np.nanpercentile(idk_d,25):+.3f}, "
                 f"{np.nanpercentile(idk_d,75):+.3f}]  %(D_mech>0, closer to KNOWS) {100*np.mean(idk_d[ok]>0):.0f}%")
    lines.append(f"Standard median {np.nanmedian(ret_d):+.3f}  IQR [{np.nanpercentile(ret_d,25):+.3f}, "
                 f"{np.nanpercentile(ret_d,75):+.3f}]")
    lines.append(f"IDK closer to KNOWS anchor than Standard is, per fact: {100*np.mean(idk_d[ok] > ret_d[ok]):.0f}%")
    try: wp = wilcoxon(idk_d[ok], ret_d[ok]).pvalue
    except Exception: wp = float("nan")
    lines.append(f"IDK vs Standard D_mech: Cliff delta {cliff(idk_d, ret_d):+.3f}, Wilcoxon p={wp:.2e}")

    out = "\n".join(lines); print("\n" + out)
    open("sensitivity_8b.txt", "w").write(out + "\n")
    print("\nwrote sensitivity_8b.txt")


if __name__ == "__main__":
    main()
