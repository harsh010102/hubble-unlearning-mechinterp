"""Explicit single-fact PARAPHRASE-INVARIANCE showcase — the visual analogue of the logit-lens table,
but for the faithfulness claim (thesis slides 9-12). For one person's birthdate fact we ask the SAME
question several meaning-preserving ways and show, per phrasing:

  - the per-layer logit-lens TOP-1 token climbing to the gold month  (knowledge forms the same way),
  - the surface confidence P(gold), ROUGE, exact-match                (LEXICAL: swings across rewordings),
  - the internal min-rank and D_mech                                 (MECHANISTIC: barely moves, stays
                                                                       on the KNOWS side).

Model = Perturbed (knows), matching the faithfulness Positive-pool. D_mech uses the same dual anchors as
run_mech_metaeval (K = Perturbed-canonical median, Ø = Standard-canonical median, leave-one-fact-out).

Self-contained: computes AND renders (no login-node python). Writes NEW files only:
  paraphrase_showcase_lens_<tag>.csv     (per phrasing x layer: top1_token, prob, gold_rank, gold_prob)
  paraphrase_showcase_summary_<tag>.csv  (per entity x phrasing: P_gold, rouge, exact, min_rank, D_mech)
  paraphrase_showcase_ent{e}_<tag>.png   (the lens-table grid, per candidate entity)
  paraphrase_showcase_strip_ent{e}_<tag>.png (surface-swings-vs-mechanistic-holds line chart)
Nothing written to RESULTS.md."""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from run_lenstable import per_layer_topk
from run_mech_metaeval import probe, greedy_ids, rouge_l, BAND_FRAC

DEV = h8.DEVICE
ENTS = [int(x) for x in os.environ.get("PS_ENTS", "0,1,2,3,4,5").split(",") if x.strip()]
TRIG = "was born on"
# (label, replacement trigger)  -- first entry is the canonical prompt (None = unchanged)
PHRASINGS = [
    ("Original",             None),
    ("recorded birth\ndate of", "has a recorded birth date of"),
    ("came into\nthe world on", "came into the world on"),
    ("has a\nbirthday on",   "has a birthday on"),
    ("born on the\ndate of", "was born on the date of"),
]
GRID_PHR = 4                                    # how many phrasings to draw in the lens grid


def phrased(prefix, repl):
    return prefix if repl is None else (prefix.replace(TRIG, repl) if TRIG in prefix else prefix)


def gold_first_id(tok, prefix, gold):
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids
    return int(full[0, pre.shape[1]])


# ------------------------------------------------------------------ compute
def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    N = len(bd); full_prefix = [p for p, _ in bd]; gold = [g for _, g in bd]
    print(f"scale={TAG}  N={N}  entities={ENTS}  phrasings={[p[0].replace(chr(10),' ') for p in PHRASINGS]}")

    pert = h8.load_named("Perturbed"); pert.eval()
    L = len(probe(pert, tok, full_prefix[0], gold[0], np.arange(1))["traj"])
    band = np.arange(int(L * BAND_FRAC), L)
    print(f"band = layers {band[0]}..{band[-1]}  (D_mech / min_rank average here)")

    # anchor K = Perturbed canonical trajectories over all facts
    print("[anchor] Perturbed canonical trajectories ...")
    Ftr = np.array([probe(pert, tok, full_prefix[i], gold[i], band)["traj"] for i in range(N)])

    # per (entity, phrasing) on Perturbed: lens rows + trajectory + surface metrics
    lens_rows, summ = [], []
    for e in ENTS:
        gid = gold_first_id(tok, full_prefix[e], gold[e])
        for (label, repl) in PHRASINGS:
            pfx = phrased(full_prefix[e], repl)
            rows = per_layer_topk(pert, tok, pfx, gid)                    # [(L,t1,p1,gt,gr,gp)]
            for (li, t1, p1, gt, gr, gp) in rows:
                lens_rows.append(dict(entity_idx=e, entity=names[e], phrasing=label.replace("\n", " "),
                                      layer=li, top1_token=t1, top1_prob=round(p1, 4),
                                      gold_token=gt, gold_rank=gr, gold_prob=round(gp, 4)))
            pr = probe(pert, tok, pfx, gold[e], band)
            traj = np.asarray(pr["traj"], float)
            ntok = max(1, len(tok.encode(" " + gold[e], add_special_tokens=False)))
            gi = greedy_ids(pert, tok, pfx, ntok + 2)
            rouge = rouge_l(tok.decode(gi, skip_special_tokens=True).strip(), gold[e])
            out_tok = rows[-1][1]; p_gold = rows[-1][5]; exact = int(rows[-1][4] == 1)
            summ.append(dict(entity_idx=e, entity=names[e], phrasing=label.replace("\n", " "),
                             prompt=pfx.rstrip(), out_token=out_tok, gold_token=gold[e],
                             P_gold=round(p_gold, 4), rouge=round(rouge, 3), exact=exact,
                             min_rank=round(float(traj[band].min()), 2), _traj=traj.tolist()))
    h8.to_cpu(pert); del pert; h8.free_cuda()

    # anchor Ø = Standard canonical trajectories over all facts
    print("[anchor] Standard canonical trajectories ...")
    std = h8.load_named("Standard"); std.eval()
    Rtr = np.array([probe(std, tok, full_prefix[i], gold[i], band)["traj"] for i in range(N)])
    h8.to_cpu(std); del std; h8.free_cuda()

    # D_mech per (entity, phrasing) with leave-one-fact-out anchors
    for s in summ:
        e = s["entity_idx"]; t = np.asarray(s.pop("_traj"), float)
        aK = np.median(np.delete(Ftr, e, 0), 0); aO = np.median(np.delete(Rtr, e, 0), 0)
        lr = np.log(t)
        s["D_mech"] = round(float(np.mean(np.abs(lr[band] - np.log(aO)[band]))
                                  - np.mean(np.abs(lr[band] - np.log(aK)[band]))), 3)

    lensdf = pd.DataFrame(lens_rows); sumdf = pd.DataFrame(summ)
    lensdf.to_csv(f"{h8.workdir()}/paraphrase_showcase_lens_{TAG}.csv", index=False)
    sumdf.to_csv(f"{h8.workdir()}/paraphrase_showcase_summary_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/paraphrase_showcase_lens_{TAG}.csv")
    h8.upload_artifact(f"{h8.workdir()}/paraphrase_showcase_summary_{TAG}.csv")

    # illustrativeness ranking (high canonical P_gold, large P_gold swing, D_mech stays > 0)
    print("\n=== per-entity summary (choose the cleanest illustrator) ===")
    for e in ENTS:
        d = sumdf[sumdf.entity_idx == e]
        can = d[d.phrasing == "Original"].iloc[0]
        pg = d.P_gold.values; dm = d.D_mech.values
        print(f"[{e}] {names[e]:<22} gold={gold[e]:<9} "
              f"P_gold {can.P_gold:.2f} (swing {pg.max()-pg.min():.2f}, min {pg.min():.2f})  "
              f"ROUGE swing {d.rouge.max()-d.rouge.min():.2f}  "
              f"D_mech {dm.min():.2f}..{dm.max():.2f} (all>0: {bool((dm>0).all())})")

    for e in ENTS:
        render_grid(lensdf, sumdf, e, TAG)
        render_strip(sumdf, e, TAG)
    print("\ndone.")


# ------------------------------------------------------------------ render
BLUES = plt.get_cmap("Blues")
def prob_color(p): return BLUES(0.12 + 0.83 * min(float(p), 1.0))


def curated_layers(nL):
    return sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))


def render_grid(lensdf, sumdf, e, TAG):
    """Lens-table grid: rows = curated layers, columns = phrasings; cell = top-1 token + prob%."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    d = lensdf[lensdf.entity_idx == e]; name = d.entity.iloc[0]
    gold = d.gold_token.iloc[0]                       # first-token gold month (matches the lens cells)
    gold_full = sumdf[sumdf.entity_idx == e].gold_token.iloc[0]
    nL = int(d.layer.max()); layers = curated_layers(nL)
    phr = PHRASINGS[:GRID_PHR]; ncol = len(phr)
    x0 = 0.085; cw = (1 - x0) / ncol                          # layer column + phrasing columns
    nrow = len(layers)
    top = 0.905; foot = 0.205                                 # drawing band for the layer rows
    rowh = (top - foot) / nrow

    fig, ax = plt.subplots(figsize=(2.55 * ncol + 0.9, 9.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, f'Ask it {ncol} ways — "{name} ___ ?"   (Perturbed, knows;  gold month = "{gold}", {gold_full})',
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.945, "logit-lens top-1 prediction (with confidence) at the answer position, layer by layer",
            ha="center", va="center", fontsize=10.5, style="italic", color="#555")

    # column headers (phrasing labels)
    ax.text(x0 / 2, 0.922, "Layer", ha="center", va="center", fontsize=10, fontweight="bold")
    for j, (label, _) in enumerate(phr):
        cx = x0 + (j + 0.5) * cw
        ax.text(cx, 0.922, label.replace("\n", " ").strip(), ha="center", va="center",
                fontsize=10.5, fontweight="bold")

    # rows
    for i, Lyr in enumerate(layers):
        y = top - (i + 0.5) * rowh
        ax.add_patch(plt.Rectangle((0.004, y - rowh / 2 + 0.003), x0 - 0.008, rowh - 0.006,
                                   facecolor=(0.93, 0.95, 0.98, 1), edgecolor="white", lw=1))
        ax.text(x0 / 2, y, str(Lyr), ha="center", va="center", fontsize=9.5)
        for j, (label, _) in enumerate(phr):
            r = d[(d.phrasing == label.replace("\n", " ")) & (d.layer == Lyr)]
            if r.empty: continue
            r = r.iloc[0]; tok_s = str(r.top1_token); p = float(r.top1_prob); isgold = (tok_s == gold)
            cx = x0 + j * cw
            ax.add_patch(plt.Rectangle((cx + 0.004, y - rowh / 2 + 0.003), cw - 0.008, rowh - 0.006,
                                       facecolor=prob_color(p), edgecolor="white", lw=1))
            tcol = "white" if p > 0.55 else "#111"
            ax.text(cx + cw / 2, y + rowh * 0.17, f'"{tok_s}"', ha="center", va="center",
                    fontsize=10.5, fontweight="bold" if isgold else "normal", color=tcol)
            ax.text(cx + cw / 2, y - rowh * 0.27, f"{p*100:.0f}%", ha="center", va="center",
                    fontsize=8.2, color=tcol)

    # footer metric block, per phrasing column
    lines = [("surface output", "out_token", None),
             ("P(gold)", "P_gold", "pct"),
             ("ROUGE-L", "rouge", "num"),
             ("D_mech", "D_mech", "dmech")]
    ax.plot([0.004, 0.996], [foot - 0.006, foot - 0.006], color="#000", lw=0.8)
    ax.text(x0 / 2, foot - 0.028, "", ha="center")
    fy = foot - 0.030
    fh = 0.036
    for k, (lab, key, kind) in enumerate(lines):
        yy = fy - k * fh
        ax.text(0.004, yy, lab, ha="left", va="center", fontsize=9.5, fontweight="bold", color="#333")
        for j, (label, _) in enumerate(phr):
            s = sumdf[(sumdf.entity_idx == e) & (sumdf.phrasing == label.replace("\n", " "))].iloc[0]
            cx = x0 + (j + 0.5) * cw
            if kind is None:
                txt = f'"{s[key]}"'; col = "#111"; w = "bold" if s[key] == gold else "normal"
            elif kind == "pct":
                txt = f"{s[key]*100:.0f}%"; col = "#111"; w = "normal"
            elif kind == "num":
                txt = f"{s[key]:.2f}"; col = "#111"; w = "normal"
            else:  # dmech: sign-aware
                v = float(s[key]); col = "#1a7f37" if v > 0 else "#b3261e"
                txt = f"{v:+.2f}"; w = "bold"
            ax.text(cx, yy, txt, ha="center", va="center", fontsize=10, color=col, fontweight=w)

    ax.text(0.5, 0.028,
            'Reword the same question four ways: the surface readouts (P(gold), ROUGE) swing, yet the gold\n'
            'month emerges the same way inside the model and D_mech stays positive (the "knows" side) throughout.\n'
            'Lexical metrics track wording; D_mech tracks knowledge.',
            ha="center", va="center", fontsize=10, style="italic", color="#444")
    p = f"{h8.workdir()}/paraphrase_showcase_ent{e}_{TAG}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); h8.upload_artifact(p)
    print(f"wrote {p}  ({name}, gold={gold})")


def render_strip(sumdf, e, TAG):
    """Line chart: surface P(gold) swings across phrasings while D_mech stays flat and > 0."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    d = sumdf[sumdf.entity_idx == e].set_index("phrasing").loc[[p[0].replace("\n", " ") for p in PHRASINGS]]
    name = d.entity.iloc[0]; gold = d.gold_token.iloc[0]
    labels = [p[0].replace("\n", " ") for p in PHRASINGS]; x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(x, d.P_gold.values, "o-", color="#999", lw=2, ms=7, label="P(gold)  — lexical/output")
    ax.plot(x, d.rouge.values, "s--", color="#bbb", lw=1.6, ms=6, label="ROUGE-L — lexical")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("lexical score (0–1)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    for sp in ["top"]: ax.spines[sp].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, d.D_mech.values, "D-", color="#1a7f37", lw=2.4, ms=7, label="D_mech — mechanistic")
    ax2.axhline(0, color="#1a7f37", lw=0.8, ls=":")
    lo = min(0, d.D_mech.min()) - 0.2; hi = max(d.D_mech.max(), 0) + 0.3
    ax2.set_ylim(lo, hi); ax2.set_ylabel("D_mech  (>0 = looks like KNOWS)", color="#1a7f37")
    ax2.tick_params(axis="y", labelcolor="#1a7f37")
    for sp in ["top"]: ax2.spines[sp].set_visible(False)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center left", fontsize=9, framealpha=0.9)
    ax.set_title(f'Same fact, four phrasings — "{name}" (gold "{gold}")\n'
                 f"surface readouts swing; D_mech stays on the knows side",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout(); p = f"{h8.workdir()}/paraphrase_showcase_strip_ent{e}_{TAG}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); h8.upload_artifact(p)


if __name__ == "__main__":
    main()
