"""CORRECTED averaged-Jacobian lens prototype — adds the missing INTERCEPT term.

The naive J-lens (run_jlens.py, run_jlens_avg_proto.py) reads an intermediate state as  J_L . h_q
with NO constant term, so the readout's level floats freely per layer -> jagged, uninterpretable rank.

The correct first-order (Jacobian) transport is a Taylor expansion of the network tail g_L: h_L -> logits
around each reference context x_i, then averaged over contexts (leave-the-query-out):

    readout(h_q) = (1/N) Sum_i [ l_i + J_L(x_i) . (h_q - h_L(x_i)) ]
                 = A_L . h_q + b_L ,   A_L = mean_i J_L(x_i),  b_L = mean_i[ l_i - J_L(x_i).h_L(x_i) ]

where l_i = the context's TRUE logits at its answer position. b_L is the anchor the old code dropped.

This prototype prints, for ONE query across all layers, the gold-token rank under:
  logit lens | per-input J.h (old, no intercept) | avg J.h (old, no intercept) | avg J + INTERCEPT (new).
If the new column is smooth / monotone-ish and lands at the true final rank, the fix works -> scale up.

Env: JA_QUERY (fact index, default 0), JA_NAVG (n reference contexts, default 32). Model = Perturbed.
Runs at whatever HUBBLE_SCALE is set (1b or 8b). Writes jlens_affine_proto_<tag>.csv."""
import os, sys, json
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

DEV = h8.DEVICE
QUERY = int(os.environ.get("JA_QUERY", "0"))
NAVG = int(os.environ.get("JA_NAVG", "32"))


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    repo, rev = h8.config.MODEL_PERTURBED, h8.config.REVISION
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    layers, norm, head = m.model.layers, m.model.norm, m.lm_head; nL = len(layers)
    print(f"scale={TAG} query={QUERY} n_ref_contexts={NAVG} layers={nL}")

    inject = {}
    def hook(module, args, kwargs):
        newh = inject["h"]
        if len(args) >= 1: return (newh,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = newh; return args, kwargs

    def seq(fi):
        prefix, gold = bd[fi]; prefix = prefix.rstrip()
        pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
        full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
        return full, pre.shape[1] - 1, int(full[0, pre.shape[1]])

    # query: hidden state per layer at answer position + gold id + true final logits
    q_full, q_pos, gold_id = seq(QUERY)
    with torch.no_grad():
        q_out = m(input_ids=q_full, output_hidden_states=True, use_cache=False)
    q_hs = q_out.hidden_states; q_true = q_out.logits[0].float()
    h_q = [q_hs[L][0][q_pos].float().detach() for L in range(len(q_hs))]
    last = len(q_hs) - 1

    ctx = [seq(i) for i in range(len(bd)) if i != QUERY][:NAVG]

    def ctx_terms(cf, cpos, L, hq_vec):
        """One hooked forward + first grad; returns (true logits l_i at cpos, J.hq, J.h_self)."""
        with torch.no_grad():
            chs = m(input_ids=cf, output_hidden_states=True, use_cache=False).hidden_states
        hL = chs[L][0].float().detach().requires_grad_(True)
        inject["h"] = hL.unsqueeze(0)
        hd = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
        y = m(input_ids=cf, use_cache=False).logits[0, cpos]
        u = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(y, hL, grad_outputs=u, create_graph=True)
        vq = torch.zeros_like(hL); vq[cpos] = hq_vec
        (Jq,) = torch.autograd.grad(g, u, grad_outputs=vq, retain_graph=True)
        vs = torch.zeros_like(hL); vs[cpos] = hL[cpos].detach()
        (Js,) = torch.autograd.grad(g, u, grad_outputs=vs)
        hd.remove()
        return y.detach(), Jq.detach(), Js.detach()

    def rank(vec): return int((vec > vec[gold_id]).sum()) + 1

    rows = []
    print(f"\nfact {QUERY}: {json.loads(fs[QUERY]['meta'])['full_name']}  gold={tok.decode([gold_id])!r}  "
          f"true final rank={rank(q_true[q_pos])}")
    print(" L   logit   perinput(noB)   avgJ(noB)   avgJ+INTERCEPT   true_final")
    for L in range(len(q_hs)):
        ll = q_true[q_pos] if L == last else head(norm(h_q[L].unsqueeze(0))).float()[0]
        ll_r = rank(ll)
        if L == last:
            pi_r = ll_r; noint_r = ll_r; wint_r = ll_r
        else:
            # per-input (query linearized around itself, no intercept) = old per-input lens
            _, Jq_self, _ = ctx_terms(q_full, q_pos, L, h_q[L]); pi_r = rank(Jq_self)
            V = q_true.shape[-1]
            accJq = torch.zeros(V, device=DEV); accInt = torch.zeros(V, device=DEV)
            for (cf, cpos, _) in ctx:
                yv, Jq, Js = ctx_terms(cf, cpos, L, h_q[L])
                accJq += Jq; accInt += (yv - Js)
            avgJq = accJq / len(ctx); bL = accInt / len(ctx)
            noint_r = rank(avgJq); wint_r = rank(avgJq + bL)
        tf = rank(q_true[q_pos])
        rows.append(dict(layer=L, logit=ll_r, perinput_noB=pi_r, avgJ_noB=noint_r,
                         avgJ_intercept=wint_r, true_final=tf))
        print(f"{L:2d}   {ll_r:5d}   {pi_r:13d}   {noint_r:9d}   {wint_r:14d}   {tf:9d}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/jlens_affine_proto_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/jlens_affine_proto_{TAG}.csv")
    # smoothness diagnostic: mean abs log-rank jump layer-to-layer (lower = smoother)
    def jag(col):
        v = np.log10(np.maximum(df[col].values.astype(float), 1))
        return float(np.mean(np.abs(np.diff(v))))
    print("\nlayer-to-layer log-rank jaggedness (lower = smoother):")
    for c in ["logit", "perinput_noB", "avgJ_noB", "avgJ_intercept"]:
        print(f"  {c:16s} {jag(c):.3f}")
    print("\ndone. (smooth avgJ_intercept ~ logit, ending at true_final => the fix works => scale up)")


if __name__ == "__main__":
    main()
