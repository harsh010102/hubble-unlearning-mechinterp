"""Diagnostic: is the faithfulness ROUGE=0 a bug or a paradigm mismatch?
Prints, for Perturbed and IDK on nationality facts, the exact prefix / gold / teacher-forced
probability / greedy generation, so we can see WHY ROUGE-L is 0 despite Probability ~1.0.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import hubble8b as h8
from hubble8b.data import build_prefixes, get_slot_value

tok = h8.get_tokenizer()
forget_set, _, _ = h8.load_yago_splits()
metas = [json.loads(forget_set[i]["meta"]) for i in range(len(forget_set))]


def prob(m, prefix, ans):
    pre = tok(prefix, return_tensors="pt").input_ids.to(h8.DEVICE)
    full = tok(prefix + ans, return_tensors="pt").input_ids.to(h8.DEVICE)
    with torch.no_grad():
        lg = m(input_ids=full).logits[0]
    lp = sum(float(torch.log_softmax(lg[p].float(), -1)[full[0, p + 1]])
             for p in range(pre.shape[1] - 1, full.shape[1] - 1))
    n = full.shape[1] - pre.shape[1]
    return math.exp(lp / max(n, 1)), tok.convert_ids_to_tokens(full[0, pre.shape[1]:].tolist())


def gen(m, prefix, k=14):
    ids = tok(prefix, return_tensors="pt").input_ids.to(h8.DEVICE)
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=k, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    h8.hf_login()
    for name in ["Perturbed", "IDK"]:
        try:
            m = h8.load_named(name)
        except Exception as e:
            print(f"{name}: {e}"); continue
        print(f"\n================= {name} =================")
        for i in range(8):
            pfx = build_prefixes(metas[i], "nationality")["full_prefix"]
            gold = get_slot_value(metas[i], "nationality")
            p, gtoks = prob(m, pfx, gold)
            g = gen(m, pfx)
            print(f"[{metas[i]['full_name']}]")
            print(f"  prefix : {pfx!r}")
            print(f"  gold   : {gold!r}  (tokens {gtoks})  prob={p:.3f}")
            print(f"  greedy : {g!r}\n")
        h8.to_cpu(m); del m; h8.free_cuda()
    print("done.")


if __name__ == "__main__":
    main()
