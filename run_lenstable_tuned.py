"""Per-layer TOP-1 under the LOGIT lens vs the TUNED lens (same model), to show the tuned lens
'skips ahead' — it surfaces the answer earlier than the raw logit lens. Trains per-layer translators
on the chosen model, then extracts top-1 + prob (and gold rank) at the answer position per layer.
Saves lens_table_tuned_<tag>.csv (columns: entity_idx, entity, lens, layer, top1_token, top1_prob,
gold_token, gold_rank, gold_prob). Render locally afterwards."""
import os, sys, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubble8b as h8
from run_tuned_lens import build_translators, train_lens

DEV = h8.DEVICE
MODEL = os.environ.get("LT_MODEL", "Perturbed")
NENT = int(os.environ.get("LT_NENT", "6"))
STEPS = int(os.environ.get("LT_STEPS", "1800"))


def main():
    h8.hf_login(); tok = h8.get_tokenizer(); TAG = h8.config.SCALE_TAG
    fs, ks, us = h8.load_yago_splits()
    facts = h8.facts_for(fs, h8.TARGET_SLOT, "full_prefix")
    names = [json.loads(fs[i]["meta"])["full_name"] for i in range(len(fs))]
    train_texts = h8.load_retain_texts().select(range(2000))["text"]
    print(f"scale={TAG} model={MODEL} n={NENT} tuned-lens steps={STEPS}")
    m = h8.load_named(MODEL); m.eval()
    d = m.config.hidden_size; nL = m.config.num_hidden_layers + 1
    tr = build_translators(nL - 1, d, DEV)
    train_lens(m, tok, tr, list(train_texts), STEPS)
    norm, head = m.model.norm, m.lm_head; mdt = next(m.parameters()).dtype

    rows = []
    for i in range(NENT):
        prefix, gold = facts[i]; prefix = prefix.rstrip()
        pre = tok(prefix, return_tensors="pt").input_ids.to(DEV)
        full = tok(prefix + " " + gold, return_tensors="pt").input_ids
        gold_id = int(full[0, pre.shape[1]])
        with torch.no_grad():
            out = m(input_ids=pre, output_hidden_states=True)
        hs = out.hidden_states; true_logits = out.logits[0].float()
        pos = pre.shape[1] - 1; last = len(hs) - 1
        for L, h in enumerate(hs):
            ll = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
            if L == last:
                tl = true_logits[pos]
            else:
                with torch.no_grad():
                    trh = tr[L](h[0, pos, :].float())
                    tl = head(norm(trh.to(mdt))).float()
            for lens, logits in [("logit", ll), ("tuned", tl)]:
                probs = torch.softmax(logits, -1); t1 = int(logits.argmax())
                gr = int((logits > logits[gold_id]).sum()) + 1
                rows.append(dict(entity_idx=i, entity=names[i], lens=lens, layer=L,
                                 top1_token=tok.decode([t1]).strip(), top1_prob=round(float(probs[t1]), 4),
                                 gold_token=tok.decode([gold_id]).strip(), gold_rank=gr,
                                 gold_prob=round(float(probs[gold_id]), 4)))
        print(f"[{names[i]}] done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{h8.workdir()}/lens_table_tuned_{TAG}.csv", index=False)
    h8.upload_artifact(f"{h8.workdir()}/lens_table_tuned_{TAG}.csv")
    # preview: first layer where each lens makes gold the top-1
    print("\n=== first layer where top1 == gold (logit vs tuned) — tuned should be EARLIER ===")
    for i in range(NENT):
        g = df[df.entity_idx == i]
        fl = {ln: (int(gg[gg.top1_token == gg.gold_token].layer.min()) if (gg.top1_token == gg.gold_token).any() else None)
              for ln, gg in g.groupby("lens")}
        print(f"  {names[i]:22s} logit={fl.get('logit')}  tuned={fl.get('tuned')}")
    print("\ndone.")


if __name__ == "__main__":
    main()
