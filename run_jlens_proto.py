"""Jacobian-lens FEASIBILITY prototype (1B). Verifies (1) we can reconstruct the layer-L -> logits
subnetwork exactly (so the JVP is meaningful), and (2) torch.func.jvp runs forward-mode through the
Hubble decoder with eager attention. Compares J-lens vs logit-lens rank of the gold token by layer,
on a few forget facts. If recon matches and jvp runs, we scale to a full run + 8B.

The J-lens transport: for the residual at layer L, position p (the position predicting the gold token),
the J-lens logits = J . h_L[p], computed as jvp(subnet_L, h_L, tangent=localized h_L[p]) — the local
Jacobian of the downstream network applied to the activation. Rank the gold token in those logits.
"""
import os, sys, json
import numpy as np, torch
from torch.func import jvp
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

DEV = h8.DEVICE
NFACTS = int(os.environ.get("JLENS_NFACTS", "3"))


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    bd = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    repo, rev = h8.config.MODEL_PERTURBED, h8.config.REVISION
    print(f"scale={TAG}  model={repo}@{rev}  facts={NFACTS}")
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    layers, norm, head, rotary = m.model.layers, m.model.norm, m.lm_head, m.model.rotary_emb
    nL = len(layers)
    print(f"num layers = {nL}")

    # ---- ROBUST APPROACH: inject h_L into the model's OWN forward via a hook (so RoPE/mask are the
    # model's, guaranteed consistent), and compute J . h_L by a double-backward JVP (reverse-mode,
    # which composes cleanly with hooks) instead of torch.func.jvp on a hand-rebuilt subnetwork. ----
    inject = {}
    def inject_hook(module, args, kwargs):
        newh = inject["h"]
        if len(args) >= 1:
            return (newh,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = newh
        return args, kwargs

    all_ok = True
    for fi in range(NFACTS):
        prefix, gold = bd[fi]; prefix = prefix.rstrip()
        pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
        full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(DEV)
        pos = pre.shape[1] - 1
        gold_id = int(full[0, pos + 1])
        with torch.no_grad():
            out = m(input_ids=full, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states; true_logits = out.logits[0].float()

        print(f"\n=== fact {fi}: {json.loads(fs[fi]['meta'])['full_name']}  gold='{gold}'  "
              f"pos={pos}  gold_tok={repr(tok.decode([gold_id]))} ===")
        print(" L   recon_ok   logit_rank   jlens_rank")
        for L in range(nL + 1):
            with torch.no_grad():
                ll = head(norm(hs[L][0][pos:pos + 1].float()))[0]
            ll_rank = int((ll > ll[gold_id]).sum()) + 1
            if L == nL:                                            # final state: J-lens == true output
                jout = true_logits[pos]; ok = True
            else:
                hL = hs[L][0].float().detach().requires_grad_(True)
                inject["h"] = hL.unsqueeze(0)
                hd = layers[L].register_forward_pre_hook(inject_hook, with_kwargs=True)
                try:
                    y = m(input_ids=full, use_cache=False).logits[0, pos]        # [V], depends on hL
                    ok = torch.allclose(y.detach(), true_logits[pos], atol=2e-2, rtol=1e-2)
                    u = torch.zeros_like(y, requires_grad=True)
                    (g,) = torch.autograd.grad(y, hL, grad_outputs=u, create_graph=True)  # J^T u
                    v = torch.zeros_like(hL); v[pos] = hL[pos].detach()          # localized tangent
                    (jout,) = torch.autograd.grad(g, u, grad_outputs=v)          # = J . v  [V]
                except Exception as e:
                    hd.remove(); print(f"{L:2d}   JVP FAILED: {type(e).__name__}: {e}"); return
                hd.remove()
            all_ok = all_ok and ok
            j_rank = int((jout > jout[gold_id]).sum()) + 1
            print(f"{L:2d}   {str(bool(ok)):8s}   {ll_rank:10d}   {j_rank:10d}")

    print(f"\n=== FEASIBILITY: reconstruction all_ok={all_ok}, double-backward JVP ran on all layers ===")
    print("SUCCESS" if all_ok else "RECON MISMATCH — injection needs fixing")


if __name__ == "__main__":
    main()
