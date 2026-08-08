"""(B) Averaged-Jacobian lens PROTOTYPE (1B). Tests whether averaging the transport over N contexts
smooths the noisy per-input trajectory. For query fact q, layer L:
    A_L . h_q  =  (1/N) Σ_i  J_L(context_i) . h_q     (JVP linear in tangent -> reuse double-backward)
i.e. transport the query's residual h_q through each context's local Jacobian and average the logits.
Prints, per layer, logit-lens rank vs per-input J-lens rank vs AVERAGED J-lens rank of the gold token.
If the averaged column is smooth (monotone-ish) we scale up."""
import os, sys, json
import numpy as np, torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

DEV = h8.DEVICE
QUERY = int(os.environ.get("JA_QUERY", "0"))
NAVG = int(os.environ.get("JA_NAVG", "24"))


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    repo, rev = h8.config.MODEL_PERTURBED, h8.config.REVISION
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    layers, norm, head = m.model.layers, m.model.norm, m.lm_head; nL = len(layers)
    print(f"scale={TAG} query={QUERY} n_avg_contexts={NAVG} layers={nL}")

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

    # query: residual per layer + gold
    q_full, q_pos, gold_id = seq(QUERY)
    with torch.no_grad():
        q_out = m(input_ids=q_full, output_hidden_states=True, use_cache=False)
    q_hs = q_out.hidden_states; q_true = q_out.logits[0].float()
    h_q = [q_hs[L][0][q_pos].float().detach() for L in range(len(q_hs))]   # d-vectors

    # averaging contexts (other facts): precompute their sequences
    ctx = [seq(i) for i in range(len(bd)) if i != QUERY][:NAVG]

    def transported_logits(cf, cpos, L, tangent):
        """J_L(context) . tangent  at context's answer position, via double-backward. [V]"""
        with torch.no_grad():
            chs = m(input_ids=cf, output_hidden_states=True, use_cache=False).hidden_states
        hL = chs[L][0].float().detach().requires_grad_(True)
        inject["h"] = hL.unsqueeze(0)
        hd = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
        y = m(input_ids=cf, use_cache=False).logits[0, cpos]
        u = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(y, hL, grad_outputs=u, create_graph=True)
        v = torch.zeros_like(hL); v[cpos] = tangent
        (jout,) = torch.autograd.grad(g, u, grad_outputs=v)
        hd.remove()
        return jout.detach()

    def rank(logits): return int((logits > logits[gold_id]).sum()) + 1

    last = len(q_hs) - 1
    print(f"\nfact {QUERY}: {json.loads(fs[QUERY]['meta'])['full_name']}  gold_tok={tok.decode([gold_id])!r}")
    print(" L   logit_rank   perinput_jrank   AVG_jrank(N={})".format(NAVG))
    for L in range(len(q_hs)):
        ll = q_true[q_pos] if L == last else head(norm(h_q[L].unsqueeze(0))).float()[0]
        ll_rank = rank(ll)
        if L == last:
            pi_rank = ll_rank; avg_rank = ll_rank
        else:
            pi = transported_logits(q_full, q_pos, L, h_q[L]); pi_rank = rank(pi)   # context = query
            acc = torch.zeros_like(pi)
            for (cf, cpos, _) in ctx:
                acc += transported_logits(cf, cpos, L, h_q[L])
            avg_rank = rank(acc / len(ctx))
        print(f"{L:2d}   {ll_rank:10d}   {pi_rank:14d}   {avg_rank:14d}")
    print("\ndone. (smooth AVG column => averaging works => scale up)")


if __name__ == "__main__":
    main()
