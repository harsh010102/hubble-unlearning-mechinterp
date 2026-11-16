"""Anchor-strength gate: does the knows anchor (full) recall forget10 answers strongly while the
never-knew anchor (retain90) does not? If yes, D_mech is viable at 1B; if the gap is weak, escalate to 3B.
Also prints the forget10_perturbed schema (to wire the paraphrase/perturbed pools) and a decoded example.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tofu_common as tc


def main():
    tok = tc.get_tokenizer()

    pert = tc.load_forget(2, perturbed=True)
    print("=== forget10_perturbed columns ===", list(pert[0].keys()))
    for k, v in pert[0].items():
        print(f"   {k}: {str(v)[:110]}")

    facts = tc.load_forget(40)
    ex = facts[0]
    ids = tok.apply_chat_template([{"role": "user", "content": ex["question"]}],
                                  add_generation_prompt=True, return_dict=True)["input_ids"]
    print(f"\n=== example ===\nQ: {ex['question']}\nA: {ex['answer'][:100]}\nprompt tokens: {len(ids)}")

    print(f"\n[gate] {len(facts)} forget10 QA | K(knows)={tc.FULL.split('/')[-1]} | O(never)={tc.RETAIN90.split('/')[-1]}")
    K = tc.load_model(tc.FULL)
    kt = [tc.rank_trajectory(K, tok, f["question"], f["answer"]) for f in facts]; tc.free(K)
    O = tc.load_model(tc.RETAIN90)
    ot = [tc.rank_trajectory(O, tok, f["question"], f["answer"]) for f in facts]; tc.free(O)
    kt = [t for t in kt if t]; ot = [t for t in ot if t]

    med = lambda ts, key: float(np.median([t[key] for t in ts]))
    def band(ts):
        return float(np.median([np.mean(t["mean_traj"][int(t["n_layers"] * 0.5):]) for t in ts]))

    print("\n                     K (knows)    O (never-knew)")
    print(f"  gold probability     {med(kt,'prob'):.3f}        {med(ot,'prob'):.3f}")
    print(f"  final mean-rank      {med(kt,'final_mean'):8.0f}     {med(ot,'final_mean'):8.0f}")
    print(f"  final min-rank       {med(kt,'final_min'):8.0f}     {med(ot,'final_min'):8.0f}")
    print(f"  mid/late band rank   {band(kt):8.0f}     {band(ot):8.0f}")

    sep = med(ot, 'final_mean') / max(med(kt, 'final_mean'), 1)
    band_sep = band(ot) / max(band(kt), 1)
    print(f"\n[verdict] knows-vs-never-knew final-rank separation: {sep:.1f}x ; band separation: {band_sep:.1f}x")
    ok = med(kt, 'prob') > 0.30 and sep > 3
    print("1B VIABLE for D_mech" if ok else "1B WEAK -> recommend escalating to 3B")


if __name__ == "__main__":
    main()
