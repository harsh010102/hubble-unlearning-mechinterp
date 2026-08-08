"""Extract per-layer logit-lens TOP-1 prediction + probability at the answer position, for several
entities x all six models — the data behind an 'Eiffel Tower -> Paris' style emergence table. Renders
locally afterwards (render_lenstable.py). Prompt-only forward (predict the answer), boundary-safe.

Saves lens_table_<tag>.csv: entity_idx, entity, model, slot, layer, top1_token, top1_prob,
gold_token, gold_rank, gold_prob."""
import os, sys, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8

DEV = h8.DEVICE
MODELS = [m.strip() for m in os.environ.get("LT_MODELS", "Standard,Perturbed,NPO,GradDiff,IDK,RMU").split(",") if m.strip()]
SLOT = os.environ.get("LT_SLOT", h8.TARGET_SLOT)
NENT = int(os.environ.get("LT_NENT", "6"))


@torch.no_grad()
def per_layer_topk(m, tok, prefix, gold_id):
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
    out = m(input_ids=pre, output_hidden_states=True)
    hs = out.hidden_states; true_logits = out.logits[0].float()
    norm, head = m.model.norm, m.lm_head; last = len(hs) - 1
    pos = pre.shape[1] - 1                                   # last prompt token -> predicts the answer
    rows = []
    for L, h in enumerate(hs):
        logits = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
        probs = torch.softmax(logits, -1)
        t1 = int(logits.argmax())
        gold_rank = int((logits > logits[gold_id]).sum()) + 1
        rows.append((L, tok.decode([t1]).strip(), float(probs[t1]),
                     tok.decode([gold_id]).strip(), gold_rank, float(probs[gold_id])))
    return rows


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    facts = h8.facts_for(fs, SLOT, "full_prefix")
    names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    print(f"scale={TAG} slot={SLOT} models={MODELS} n_entities={NENT}")
    # gold first-token ids (boundary-safe)
    gold_ids = []
    for i in range(NENT):
        prefix, gold = facts[i]; prefix = prefix.rstrip()
        pre = tok(prefix, return_tensors="pt").input_ids
        full = tok(prefix + " " + gold, return_tensors="pt").input_ids
        gold_ids.append(int(full[0, pre.shape[1]]))

    rows = []
    for name in MODELS:
        try:
            m = h8.load_named(name); m.eval()
        except Exception as e:
            print(f"{name}: {type(e).__name__}"); continue
        for i in range(NENT):
            prefix, gold = facts[i]
            for (L, t1, p1, gt, gr, gp) in per_layer_topk(m, tok, prefix, gold_ids[i]):
                rows.append(dict(entity_idx=i, entity=names[i], model=name, slot=SLOT, layer=L,
                                 top1_token=t1, top1_prob=round(p1, 4),
                                 gold_token=gt, gold_rank=gr, gold_prob=round(gp, 4)))
        h8.to_cpu(m); del m; h8.free_cuda(); print(f"[{name}] done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/lens_table_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/lens_table_{TAG}.csv")
    # quick console preview: which entity gives the cleanest Perturbed emergence?
    print("\n=== Perturbed gold-token emergence (final-layer gold_rank per entity) ===")
    pv = df[(df.model == "Perturbed") & (df.layer == df.layer.max())][["entity", "gold_token", "gold_rank", "gold_prob"]]
    print(pv.to_string(index=False))
    print("\ndone.")


if __name__ == "__main__":
    main()
