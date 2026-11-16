"""TOFU faithfulness meta-evaluation: does the mechanistic D_mech metric satisfy the two criteria for a
faithful unlearning metric (semantic invariance + knowledge discrimination) better than OpenUnlearning's
lexical/output metrics? Second-benchmark replication of the Hubble result (run_mech_metaeval.py), on the
OpenUnlearning TOFU anchors. Uses ONLY the two anchors (full=K, retain90=O), which exist at 1B/3B/8B.

Four pools (SPID sec 5.2.1 / OpenUnlearning sec 4), each a model over N forget10 facts:
  Positive P (knows the fact):  FULL        = full(K),      canonical question,      correct answer
                                PARAPHRASED = full(K),      paraphrased_question,    correct answer
  Negative N (lacks the fact):  RETAIN      = retain90(O),  canonical question,      correct answer
                                PERTURBED   = full(K),      canonical question,      a WRONG answer

Two criteria:
  Semantic Invariance     : within P, FULL vs PARAPHRASED should be EQUAL (Wilcoxon high p; low Cliff shift)
  Knowledge Discrimination: P vs N should DIFFER              (Mann-Whitney low p; high Cliff delta)
  faithful_score = discrim * (1 - invar_shift).

Metrics: mechanistic D_mech (dual-anchor logit-lens distance, LOO) and min_rank; OpenUnlearning lexical/
output probability, rouge, exact_mem, extraction_strength, truth_ratio, MIA (loss/mink/minkpp/zlib).

Outputs (TAG = TOFU_SCALE): tofu_metaeval_cells_{TAG}.csv, tofu_metaeval_table1_{TAG}.csv,
tofu_metaeval_fig4_{TAG}.png. Reports whatever comes out, honestly.
"""
import os, sys, math, zlib
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tofu_common as tc

DEV = tc.DEVICE
BAND_FRAC = 0.5
MINK = 0.2
N_FACTS = int(os.environ.get("TOFU_NFACTS", "80"))

# metric -> (higher_is_present, family)
SPEC = {
    "D_mech": (True, "mechanistic"), "min_rank": (False, "mechanistic"),
    "probability": (True, "openunlearning"), "rouge": (True, "openunlearning"),
    "exact_mem": (True, "openunlearning"), "extraction_strength": (True, "openunlearning"),
    "truth_ratio": (False, "openunlearning"), "mia_loss": (False, "openunlearning"),
    "mia_mink": (True, "openunlearning"), "mia_minkpp": (True, "openunlearning"),
    "mia_zlib": (False, "openunlearning"),
}
SCALARS = [m for m in SPEC if m != "D_mech"]


def rouge_l(pred, gold):
    a, b = pred.split(), gold.split()
    if not a or not b: return 0.0
    dp = np.zeros((len(a) + 1, len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    lcs = dp[-1, -1]
    if lcs == 0: return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return float(2 * p * r / (p + r))


@torch.no_grad()
def probe(model, tok, question, answer, band):
    """One forward on chat(question)+answer. Vectorized across answer positions (MPS-safe): per-layer mean
    rank trajectory, gold prob, exact-match, MIA over the full sequence, min band rank."""
    pre = tc._prompt_ids(tok, question, DEV)
    ans = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    if ans.shape[1] == 0: return None
    full = torch.cat([pre, ans], dim=1)
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states; true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head; last = len(hs) - 1
    gp = torch.arange(pre.shape[1] - 1, full.shape[1] - 1, device=DEV)
    gold = full[0, gp + 1]
    traj = []
    for li, hh in enumerate(hs):
        logits = true_logits[gp] if li == last else head(norm(hh[0, gp, :])).float()
        gl = logits.gather(1, gold[:, None])
        traj.append(float((logits > gl).sum(dim=1).add_(1).float().mean().cpu()))
    traj = np.asarray(traj)
    lp_ans = torch.log_softmax(true_logits[gp], -1).gather(1, gold[:, None])[:, 0]
    prob = math.exp(float(lp_ans.mean().cpu()))
    exact = float((true_logits[gp].argmax(dim=1) == gold).float().mean().cpu())
    # MIA over every next-token position of the full (prompt+answer) sequence
    pos = torch.arange(0, full.shape[1] - 1, device=DEV); nxt = full[0, pos + 1]
    lp_all = torch.log_softmax(true_logits[pos], -1)
    a = lp_all.gather(1, nxt[:, None])[:, 0]
    z = (a - lp_all.mean(dim=1)) / (lp_all.std(dim=1) + 1e-6)
    a = a.cpu().numpy(); z = z.cpu().numpy(); k = max(1, int(len(a) * MINK))
    zlen = max(1, len(zlib.compress((str(question) + " " + answer).encode())))
    return dict(traj=traj, prob=prob, exact_mem=exact, min_rank=float(traj[band].min()),
                mia_loss=-float(a.mean()), mia_mink=float(np.sort(a)[:k].mean()),
                mia_minkpp=float(np.sort(z)[:k].mean()), mia_zlib=float(-a.sum()) / zlen)


@torch.no_grad()
def seq_lp(model, tok, question, answer):
    """Mean teacher-forced log-prob of the answer tokens given chat(question)."""
    pre = tc._prompt_ids(tok, question, DEV)
    ans = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    if ans.shape[1] == 0: return None
    full = torch.cat([pre, ans], dim=1)
    lg = model(input_ids=full).logits[0].float()
    gp = torch.arange(pre.shape[1] - 1, full.shape[1] - 1, device=DEV); gold = full[0, gp + 1]
    return float(torch.log_softmax(lg[gp], -1).gather(1, gold[:, None])[:, 0].mean().cpu())


@torch.no_grad()
def greedy(model, tok, question, max_new):
    """Manual fixed-length greedy decode. transformers `generate()` syncs the host every token (its
    stopping check), which crashes the MPS-shared GPU (cudaErrorMpsRpcFailure); this loop keeps argmax
    and concat on-GPU and syncs exactly once at the end. Fixed length (no early-stop) => no host sync."""
    ids = tc._prompt_ids(tok, question, DEV)
    cur, gen = ids, []
    eos = tok.eos_token_id
    for _ in range(int(max_new)):
        nxt = model(input_ids=cur, use_cache=False).logits[0, -1].argmax()  # scalar tensor on GPU
        gen.append(nxt)
        cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
    g = torch.stack(gen).cpu()                    # single host sync
    if eos is not None and (g == eos).any():
        g = g[: int((g == eos).float().argmax())]  # trim at first EOS (host-side, cheap)
    return g


import pickle
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cache_{tc.SCALE}")


def _cpath(name): return os.path.join(CACHE, f"{name}.pkl")


def cell_done(name):
    p = _cpath(name)
    return os.path.exists(p) and pickle.load(open(p, "rb")).get("done", False)


def cell_metrics(model, tok, questions, golds, distr_lists, band, name=None):
    """All scalar metrics + per-fact trajectory for one pool cell. If `name` is given, checkpoint to
    cache_{scale}/{name}.pkl every 10 facts and resume from there -- so an MPS crash (unavoidable on this
    node) only loses <=10 facts, and a resubmit finishes the remaining cells in a fresh process."""
    N = len(questions)
    os.makedirs(CACHE, exist_ok=True)
    if name and os.path.exists(_cpath(name)):
        d = pickle.load(open(_cpath(name), "rb"))
        acc, trajs, start = d["acc"], d["trajs"], d["i"]
        if d.get("done"):
            print(f"[cache] {name}: complete ({N} facts)"); return acc, trajs
        print(f"[cache] {name}: resume at fact {start}/{N}")
    else:
        acc = {m: np.full(N, np.nan) for m in SCALARS}; trajs = [None] * N; start = 0

    def save(i, done):
        if name:
            pickle.dump({"acc": acc, "trajs": trajs, "i": i, "done": done}, open(_cpath(name), "wb"))

    for i in range(start, N):
        q, g, distr = questions[i], golds[i], distr_lists[i]
        pr = probe(model, tok, q, g, band)
        if pr is None:
            save(i + 1, False); continue
        trajs[i] = pr["traj"]
        for m in ["exact_mem", "mia_loss", "mia_mink", "mia_minkpp", "mia_zlib", "min_rank"]:
            acc[m][i] = pr[m]
        acc["probability"][i] = pr["prob"]
        ntok = max(1, len(tok(g, add_special_tokens=False).input_ids))
        gi = greedy(model, tok, q, ntok + 2)
        acc["rouge"][i] = rouge_l(tok.decode(gi, skip_special_tokens=True).strip(), g)
        gold_ids = tok(g, add_special_tokens=False).input_ids
        mlen = 0
        for j in range(min(len(gi), len(gold_ids))):
            if int(gi[j]) == int(gold_ids[j]): mlen += 1
            else: break
        acc["extraction_strength"][i] = mlen / max(1, len(gold_ids))
        gl = seq_lp(model, tok, q, g)
        dd = [seq_lp(model, tok, q, c) for c in distr if c and c != g]
        dd = [x for x in dd if x is not None]
        acc["truth_ratio"][i] = math.exp(float(np.mean(dd)) - gl) if dd and gl is not None else np.nan
        if name and (i + 1) % 10 == 0:
            save(i + 1, False); print(f"  [{name}] {i+1}/{N}", flush=True)
    save(N, True)
    return acc, trajs


def cliff(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0: return np.nan
    return float(np.sign(x[:, None] - y[None, :]).sum() / (len(x) * len(y)))


def tost(a, b, eps):
    d = np.asarray(a, float) - np.asarray(b, float); ok = ~np.isnan(d); d = d[ok]; n = len(d)
    if n < 3: return 1.0
    m, s = d.mean(), d.std(ddof=1)
    if s == 0: return 0.0 if abs(m) < eps else 1.0
    se = s / math.sqrt(n)
    return float(max(1 - tdist.cdf((m + eps) / se, n - 1), tdist.cdf((m - eps) / se, n - 1)))


def main():
    TAG = tc.SCALE
    tok = tc.get_tokenizer()
    rows_pert = tc.load_forget(N_FACTS, perturbed=True)
    N = len(rows_pert)
    questions = [r["question"] for r in rows_pert]
    para_q = [r.get("paraphrased_question") or r["question"] for r in rows_pert]
    golds = [r["answer"] for r in rows_pert]

    def wrong_of(r):
        pa = r.get("perturbed_answer")
        if isinstance(pa, list) and pa: return [x for x in pa if x]
        return [pa] if pa else []
    wrongs = [wrong_of(r) for r in rows_pert]
    wrong_gold = [(w[0] if w else golds[i]) for i, w in enumerate(wrongs)]
    print(f"[TOFU {TAG.upper()}] N={N} forget10 facts | K={tc.FULL.split('/')[-1]} | O={tc.RETAIN90.split('/')[-1]}")

    nL = tc.N_LAYERS + 1                      # hidden_states = embeddings + N_LAYERS
    band = np.arange(int(nL * BAND_FRAC), nL)
    print(f"n_layers(+embed)={nL}  band = layers {band[0]}..{band[-1]}")

    # K serves Full / Paraphrased / Perturbed; O serves Retain. Cells checkpoint to cache_{scale}/ and
    # resume, because MPS on this node crashes under sustained load -- resubmitting finishes the rest.
    K_CELLS, O_CELLS = ["Full", "Paraphrased", "Perturbed"], ["Retain"]
    if any(not cell_done(c) for c in K_CELLS):
        print("[load] K (full)"); K = tc.load_model(tc.FULL)
        full_m, full_tr = cell_metrics(K, tok, questions, golds, wrongs, band, name="Full")
        para_m, para_tr = cell_metrics(K, tok, para_q, golds, wrongs, band, name="Paraphrased")
        pw_m, pw_tr = cell_metrics(K, tok, questions, wrong_gold, wrongs, band, name="Perturbed")
        tc.free(K)
    else:
        full_m, full_tr = pickle.load(open(_cpath("Full"), "rb"))["acc"], pickle.load(open(_cpath("Full"), "rb"))["trajs"]
        para_m, para_tr = pickle.load(open(_cpath("Paraphrased"), "rb"))["acc"], pickle.load(open(_cpath("Paraphrased"), "rb"))["trajs"]
        pw_m, pw_tr = pickle.load(open(_cpath("Perturbed"), "rb"))["acc"], pickle.load(open(_cpath("Perturbed"), "rb"))["trajs"]
    if any(not cell_done(c) for c in O_CELLS):
        print("[load] O (retain90)"); O = tc.load_model(tc.RETAIN90)
        ret_m, ret_tr = cell_metrics(O, tok, questions, golds, wrongs, band, name="Retain")
        tc.free(O)
    else:
        d = pickle.load(open(_cpath("Retain"), "rb")); ret_m, ret_tr = d["acc"], d["trajs"]

    # D_mech: dual anchors = FULL(K) & RETAIN(O) trajectories, leave-one-out to avoid self-circularity.
    def stack(trs):
        L = max(len(t) for t in trs if t is not None)
        return np.array([t if t is not None and len(t) == L else [np.nan] * L for t in trs], float)
    Ftr, Rtr = stack(full_tr), stack(ret_tr)

    def dmech_of(trs):
        out = np.full(N, np.nan)
        for i in range(N):
            if trs[i] is None: continue
            aK = np.nanmedian(np.delete(Ftr, i, 0), 0); aO = np.nanmedian(np.delete(Rtr, i, 0), 0)
            lr = np.log(np.asarray(trs[i], float))
            out[i] = float(np.mean(np.abs(lr[band] - np.log(aO)[band]))
                           - np.mean(np.abs(lr[band] - np.log(aK)[band])))
        return out
    full_m["D_mech"] = dmech_of(full_tr); para_m["D_mech"] = dmech_of(para_tr)
    ret_m["D_mech"] = dmech_of(ret_tr); pw_m["D_mech"] = dmech_of(pw_tr)

    CELLS = {"Full": full_m, "Paraphrased": para_m, "Retain": ret_m, "Perturbed": pw_m}
    POOL = {"Full": "P", "Paraphrased": "P", "Retain": "N", "Perturbed": "N"}

    rows = []
    for cell, mm in CELLS.items():
        for i in range(N):
            r = dict(fact=i, cell=cell, pool=POOL[cell])
            for m in SPEC: r[m] = float(mm[m][i])
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"tofu_metaeval_cells_{TAG}.csv", index=False)

    tab = []
    for m, (hi, fam) in SPEC.items():
        F = full_m[m]; Pa = para_m[m]; ok = ~(np.isnan(F) | np.isnan(Pa))
        try: wp = wilcoxon(F[ok], Pa[ok]).pvalue
        except Exception: wp = np.nan
        allv = np.concatenate([full_m[m], para_m[m], ret_m[m], pw_m[m]]); sd = np.nanstd(allv)
        tp = tost(F, Pa, 0.3 * sd)
        Ppool = np.concatenate([full_m[m], para_m[m]]); Npool = np.concatenate([ret_m[m], pw_m[m]])
        o = ~np.isnan(Ppool); o2 = ~np.isnan(Npool)
        wpn = mannwhitneyu(Ppool[o], Npool[o2], alternative="two-sided").pvalue if o.sum() and o2.sum() else np.nan
        invar_shift = abs(cliff(full_m[m], para_m[m]))
        discrim = abs(cliff(Ppool, Npool))
        tab.append(dict(metric=m, family=fam,
                        within_P_wilcoxon=wp, within_P_TOST=tp, between_PN_wilcoxon=wpn,
                        invar_shift=round(invar_shift, 3), discrim=round(discrim, 3),
                        faithful_score=round(discrim * (1 - invar_shift), 3),
                        discriminates=bool(wpn < 0.05) if not np.isnan(wpn) else False))
    T = pd.DataFrame(tab).sort_values("faithful_score", ascending=False)
    T.to_csv(f"tofu_metaeval_table1_{TAG}.csv", index=False)
    fmt = T.copy()
    for c in ["within_P_wilcoxon", "within_P_TOST", "between_PN_wilcoxon"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.2e}" if pd.notna(v) else "nan")
    print(f"\n=== TOFU {TAG.upper()} SPID Table 1 (invariance within P + discrimination P vs N) ===")
    print(fmt.to_string(index=False))

    order = ["Full", "Paraphrased", "Retain", "Perturbed"]; mets = list(SPEC)
    ncol = 4; nrow = int(np.ceil(len(mets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, m in zip(axes.flat, mets):
        ax.boxplot([CELLS[c][m][~np.isnan(CELLS[c][m])] for c in order], tick_labels=order, showmeans=True)
        ax.axvline(2.5, ls="--", c="r", lw=1); ax.set_title(f"{m}\n({SPEC[m][1]})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7); ax.grid(axis="y", alpha=0.3)
    for ax in axes.flat[len(mets):]: ax.axis("off")
    fig.suptitle(f"TOFU {TAG.upper()}: metric distributions across pools (dashed = Positive P | Negative N)\n"
                 "faithful metric: stable Full->Paraphrased, gap P vs N", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"tofu_metaeval_fig4_{TAG}.png", dpi=150, bbox_inches="tight")
    print("\ndone.")


if __name__ == "__main__":
    main()
