"""Shared code for the TOFU D_mech experiment: model IDs, chat-formatted logit-lens rank trajectory,
and the TOFU forget-set loader. Second-benchmark validation of the thesis's D_mech metric.

Anchors (OpenUnlearning TOFU, Llama-3.2-1B-Instruct):
  K (knows)      = the model finetuned on ALL authors           -> tofu_..._full
  O (never knew) = the model finetuned on the retain 90% only   -> tofu_..._retain90   (forget10 oracle)
"""
import math, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Scale switch (TOFU_SCALE in {1b,3b,8b}). OpenUnlearning released `full`/`retain90` ANCHORS for all three
# families, but per-method unlearned checkpoints (NPO/GradDiff/IdkDPO) ONLY for the 1B family. So the
# faithfulness meta-eval (anchors only) runs at every scale; the per-method taxonomy is 1B-only.
SCALE = os.environ.get("TOFU_SCALE", "1b").lower()
_FAMILY = {"1b": "Llama-3.2-1B-Instruct", "3b": "Llama-3.2-3B-Instruct", "8b": "Llama-3.1-8B-Instruct"}[SCALE]
_NLAYERS = {"1b": 16, "3b": 28, "8b": 32}[SCALE]

FULL = f"open-unlearning/tofu_{_FAMILY}_full"                 # K, knows
RETAIN90 = f"open-unlearning/tofu_{_FAMILY}_retain90"         # O, never knew (forget10)
# Method checkpoints exist only at 1B (OpenUnlearning did not release 3B/8B unlearned models).
METHODS = {
    "NPO": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
    "GradDiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "IdkDPO": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch10",
} if SCALE == "1b" else {}
N_LAYERS = _NLAYERS
DEVICE = "cuda:0"


def get_tokenizer(model_id=FULL):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id, device=DEVICE):
    # fp32 at 1B/3B for precise log-probs; bf16 at 8B so the fp32 weights (32GB) don't dominate the
    # oversubscribed GPU share (rank is argmax-based, so bf16 does not change the trajectory).
    dtype = torch.bfloat16 if SCALE == "8b" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                             attn_implementation="eager").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_forget(n=None, perturbed=False):
    """forget10 QA. If perturbed, also return the paraphrased/perturbed answer columns (schema printed
    by the gate). Returns a list of dicts with at least question/answer."""
    cfg = "forget10_perturbed" if perturbed else "forget10"
    ds = load_dataset("locuslab/TOFU", cfg)["train"]
    rows = [dict(r) for r in ds]
    return rows[:n] if n else rows


def _prompt_ids(tok, question, device):
    # transformers 5.x returns a BatchEncoding (return_dict default flipped); take input_ids explicitly.
    enc = tok.apply_chat_template([{"role": "user", "content": question}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return enc["input_ids"].to(device)


@torch.no_grad()
def rank_trajectory(model, tok, question, answer, device=DEVICE):
    """Logit-lens rank of each gold answer token at its position, per layer. Boundary-safe (answer
    tokenized without special tokens, appended after the chat generation prompt); double-norm-fixed
    (true output logits at the last layer). Returns per-layer mean and min rank over the answer tokens,
    plus the teacher-forced gold probability. Aggregations chosen by the gate."""
    pre = _prompt_ids(tok, question, device)
    ans = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if ans.shape[1] == 0:
        return None
    full = torch.cat([pre, ans], dim=1)
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()
    norm, head = model.model.norm, model.lm_head
    last = len(hs) - 1
    # positions predicting the answer tokens, and the gold id each one should predict
    gp = torch.arange(pre.shape[1] - 1, full.shape[1] - 1, device=device)
    gold = full[0, gp + 1]
    # Vectorized across positions (one matmul + one sync per layer) so the MPS-shared GPU is not
    # overwhelmed by per-token kernel launches / .item() syncs.
    mean_traj, min_traj = [], []
    for li, h in enumerate(hs):
        logits = true_logits[gp] if li == last else head(norm(h[0, gp, :])).float()   # [n_pos, vocab]
        gold_logit = logits.gather(1, gold[:, None])                                   # [n_pos, 1]
        ranks = (logits > gold_logit).sum(dim=1).add_(1).float().cpu().numpy()         # [n_pos]
        mean_traj.append(float(ranks.mean()))
        min_traj.append(float(ranks.min()))
    gold_lp = torch.log_softmax(true_logits[gp], -1).gather(1, gold[:, None])[:, 0].mean()
    return dict(mean_traj=mean_traj, min_traj=min_traj,
                prob=math.exp(float(gold_lp.cpu())), n_layers=len(hs), n_tokens=len(gp),
                final_mean=mean_traj[-1], final_min=min_traj[-1])


def free(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
