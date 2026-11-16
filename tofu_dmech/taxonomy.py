"""TOFU rank-by-layer taxonomy (1B only: OpenUnlearning released method checkpoints only for the 1B
family). For each unlearning method (NPO / GradDiff / IdkDPO) and the two anchors (full=K, retain90=O),
read the mean logit-lens rank of the forget10 answers at every layer. Two deliverables:

  1. taxonomy_rank_by_layer_1b.png : mean rank vs layer, one curve per model -> WHERE each method buries
     the answer, and whether it tracks the never-knew anchor (erased) or the knowing anchor (suppressed).
  2. taxonomy_table_1b.csv : per model, gold probability, final-layer rank, mid/late band rank, and D_mech
     (dual-anchor logit-lens distance vs the K/O anchors; >0 = internally closer to KNOWS).

Surface-vs-internal read: a method with low gold prob / high final rank (looks forgotten) but D_mech > 0
(internally near the knowing anchor) is SUPPRESSING, not erasing.
"""
import os, sys, pickle, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tofu_common as tc

N_FACTS = int(os.environ.get("TOFU_NFACTS", "120"))
BAND_FRAC = 0.5
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_tax_1b")


def _cp(label): return os.path.join(CACHE, re.sub(r"[^a-zA-Z0-9]", "_", label) + ".pkl")


def trajs_cached(label, mid, tok, facts):
    """Compute (or load) per-fact rank trajectories for one model, checkpointing every 10 facts so an
    MPS crash resumes on resubmit (this node's MPS server dies under sustained multi-model load)."""
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(_cp(label)):
        d = pickle.load(open(_cp(label), "rb"))
        if d.get("done"):
            print(f"[cache] {label}: complete"); return d["out"]
        out, start = d["out"], d["i"]; print(f"[cache] {label}: resume at {start}")
    else:
        out, start = [None] * len(facts), 0
    m = tc.load_model(mid)
    for i in range(start, len(facts)):
        out[i] = tc.rank_trajectory(m, tok, facts[i]["question"], facts[i]["answer"])
        if (i + 1) % 10 == 0:
            pickle.dump({"out": out, "i": i + 1, "done": False}, open(_cp(label), "wb")); print(f"  [{label}] {i+1}", flush=True)
    tc.free(m)
    pickle.dump({"out": out, "i": len(facts), "done": True}, open(_cp(label), "wb"))
    return out


def main():
    assert tc.SCALE == "1b", "taxonomy is 1B-only (no 3B/8B method checkpoints on OpenUnlearning)"
    tok = tc.get_tokenizer()
    facts = tc.load_forget(N_FACTS)
    print(f"[taxonomy 1B] N={len(facts)} forget10 facts")

    spec = [("full (K, knows)", tc.FULL), ("retain90 (O, never-knew)", tc.RETAIN90)]
    spec += [(name, mid) for name, mid in tc.METHODS.items()]
    curves = {}       # label -> list of trajectory dicts
    for label, mid in spec:
        print(f"[model] {label}")
        curves[label] = [t for t in trajs_cached(label, mid, tok, facts) if t]

    nL = curves["full (K, knows)"][0]["n_layers"]
    band = np.arange(int(nL * BAND_FRAC), nL)
    # anchor median band trajectories (over facts) for D_mech
    def med_traj(label):
        return np.nanmedian(np.array([t["mean_traj"] for t in curves[label]]), axis=0)
    aK = med_traj("full (K, knows)"); aO = med_traj("retain90 (O, never-knew)")

    def dmech(label):
        vals = []
        for t in curves[label]:
            lr = np.log(np.asarray(t["mean_traj"], float))
            vals.append(np.mean(np.abs(lr[band] - np.log(aO)[band])) - np.mean(np.abs(lr[band] - np.log(aK)[band])))
        return float(np.median(vals))

    # ---- table ----
    med = lambda label, key: float(np.median([t[key] for t in curves[label]]))
    band_rank = lambda label: float(np.median([np.mean(np.asarray(t["mean_traj"])[band]) for t in curves[label]]))
    rows = []
    for label in curves:
        rows.append(dict(model=label, gold_prob=round(med(label, "prob"), 4),
                         final_rank=round(med(label, "final_mean"), 1),
                         band_rank=round(band_rank(label), 1),
                         D_mech=round(dmech(label), 4)))
    T = pd.DataFrame(rows)
    T.to_csv("taxonomy_table_1b.csv", index=False)
    print("\n=== TOFU 1B taxonomy (D_mech > 0 => internally near KNOWS anchor) ===")
    print(T.to_string(index=False))

    # ---- figure: mean rank vs layer ----
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"full (K, knows)": ("#1a7f37", "-", 2.6), "retain90 (O, never-knew)": ("#b3261e", "-", 2.6),
              "NPO": ("#1f77b4", "--", 2.0), "GradDiff": ("#9467bd", "--", 2.0), "IdkDPO": ("#ff7f0e", "--", 2.0)}
    for label in curves:
        y = np.mean(np.array([t["mean_traj"] for t in curves[label]]), axis=0)
        c, ls, lw = styles.get(label, ("#555", "-", 1.5))
        ax.plot(range(len(y)), y, ls, color=c, lw=lw, label=label)
    ax.axvspan(band[0], band[-1], color="gray", alpha=0.08, label="mid/late band")
    ax.set_yscale("log"); ax.set_xlabel("Layer (0 = embedding)"); ax.set_ylabel("Mean logit-lens rank of gold answer")
    ax.set_title("TOFU 1B: where each unlearning method buries the forgotten answer")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig("taxonomy_rank_by_layer_1b.png", dpi=160, bbox_inches="tight")
    print("\nwrote taxonomy_rank_by_layer_1b.png")


if __name__ == "__main__":
    main()
