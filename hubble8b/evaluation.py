"""Evaluation: infill accuracy, logit-lens rank-by-layer, and LSR (semantic residual)
with the faithfulness tests (knowledge discrimination, invariance, k-sweep).

All plot/CSV helpers save locally AND upload to the shared HF results dataset repo.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import config as C
from .models import get_tokenizer, upload_artifact
from .data import (SLOT_ORDER, build_prefixes, build_full_suffix, get_slot_value,
                   build_infill_examples, TEMPLATE_FORMATS)

COLOR_MAP = {"Standard": "#2E86AB", "Perturbed": "#A23B72", "NPO": "#C73E1D",
             "GradDiff": "#6A994E", "IDK": "#3B1F2B", "RMU": "#F18F01"}

def _save_and_upload(fig_or_df, name, is_fig=True):
    path = os.path.join(C.workdir(), name)
    if is_fig:
        fig_or_df.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig_or_df.to_csv(path, index=False)
    try:
        upload_artifact(path)
    except Exception as e:
        print(f"[eval] upload skipped for {name} ({type(e).__name__})")
    return path

# ------------------------------------------------------------------ infill accuracy
@torch.no_grad()
def sequence_logprob(model, tok, prefix, continuation, device):
    # Boundary-safe: a trailing space on `prefix` merges into the first continuation token under
    # byte-level BPE, so tok(prefix+continuation)[len(tok(prefix)):] would DROP that token. Strip
    # the trailing space and re-insert one so full[len(pre):] == the continuation's own tokens.
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(device)
    full = tok(prefix + " " + continuation, return_tensors="pt").input_ids.to(device)
    if full.shape[1] <= pre.shape[1]:
        return -1e9
    logits = model(input_ids=full).logits[0]
    lp = 0.0
    for pos in range(pre.shape[1] - 1, full.shape[1] - 1):
        lp += float(torch.log_softmax(logits[pos].float(), dim=-1)[full[0, pos + 1]])
    return lp

@torch.no_grad()
def infill_accuracy(model, tok, examples, device):
    correct = 0
    for ex in examples:
        if "suffix" in ex:
            scores = [sequence_logprob(model, tok, ex["prefix"], c + " . " + ex["suffix"], device) for c in ex["candidates"]]
        else:
            scores = [sequence_logprob(model, tok, ex["prefix"], c, device) for c in ex["candidates"]]
        correct += int(int(np.argmax(scores)) == ex["true_index"])
    return correct / max(len(examples), 1)

def run_infill(model, model_name, splits, distractor_pool, target_slot=C.TARGET_SLOT, device=C.DEVICE):
    tok = get_tokenizer(); import pandas as pd
    rows = []
    for split_name, ds in splits.items():
        for fmt in TEMPLATE_FORMATS:
            ex = build_infill_examples(ds, target_slot, fmt, distractor_pool)
            acc = infill_accuracy(model, tok, ex, device)
            rows.append(dict(model=model_name, split=split_name, template_format=fmt, accuracy=acc, n=len(ex)))
            print(f"  [{model_name}] {split_name}/{fmt}: acc={acc:.4f}")
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ logit-lens rank
@torch.no_grad()
def layerwise_target_rank(model, tok, prefix, target, device):
    # Boundary-safe tokenization (see sequence_logprob): a trailing space on `prefix` merges into the
    # first target token under byte-BPE, so tok(prefix)+tok.encode(target) would measure the WRONG
    # first gold token ('July' vs ' July') -> a spurious final-layer rank spike (worst on retain,
    # where the model most sharply prefers the space-prefixed token). rstrip + " "+target fixes it.
    prefix = prefix.rstrip()
    pre = tok(prefix, return_tensors="pt").input_ids.to(device)
    full = tok(prefix + " " + target, return_tensors="pt").input_ids.to(device)
    if full.shape[1] <= pre.shape[1]:
        return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()      # model's real output; HF's hs[-1] is already normed,
    norm, head = model.model.norm, model.lm_head   # so re-norming it (head(norm(hs[-1]))) double-norms
    last = len(hs) - 1                        # -> use true logits for the final point.
    per_layer = []
    for li, h in enumerate(hs):
        ranks = []
        for pos in range(pre.shape[1] - 1, full.shape[1] - 1):
            logits = true_logits[pos] if li == last else head(norm(h[:, pos, :])).float()[0]
            gold = full[0, pos + 1]
            ranks.append(int((logits > logits[gold]).sum().item()) + 1)
        per_layer.append(float(np.mean(ranks)))
    return per_layer

# ------------------------------------------------------------------ LSR (semantic residual)
_cf_cache = {}

def build_Eref(model):
    return F.normalize(model.get_output_embeddings().weight.detach().float(), dim=-1)

def cf_ids(tok_id, E_ref, topk=C.LSR_TOPK):
    key = (tok_id, topk)
    if key in _cf_cache: return _cf_cache[key]
    sims = E_ref @ E_ref[tok_id]
    ids = torch.topk(sims, topk + 1).indices
    _cf_cache[key] = ids
    return ids

@torch.no_grad()
def layerwise_residual(model, tok, prefix, gold, E_ref, device, topk=C.LSR_TOPK):
    """Return (cf_mass[L], lex_mass[L]) averaged over gold-slot token positions."""
    prefix = prefix.rstrip()   # boundary-safe (avoid double/merged space; see sequence_logprob)
    pre = tok(prefix, return_tensors="pt").input_ids.to(device)
    full = tok(prefix + " " + gold, return_tensors="pt").input_ids.to(device)
    if full.shape[1] <= pre.shape[1]:
        return None
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()          # fix final-layer double-norm (see layerwise_target_rank)
    norm, head = model.model.norm, model.lm_head
    last = len(hs) - 1
    positions = list(range(pre.shape[1] - 1, full.shape[1] - 1))
    cf_c, lex_c = np.zeros(len(hs)), np.zeros(len(hs))
    for L, h in enumerate(hs):
        cf_m, lex_m = [], []
        for pos in positions:
            g = int(full[0, pos + 1])
            logits = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
            probs = torch.softmax(logits, dim=-1)
            cf_m.append(float(probs[cf_ids(g, E_ref, topk)].sum()))
            lex_m.append(float(probs[g]))
        cf_c[L], lex_c[L] = np.mean(cf_m), np.mean(lex_m)
    return cf_c, lex_c

def lsr_scalar(curve):
    lo = int(len(curve) * C.LSR_LAYER_FRAC)
    return float(np.max(curve[lo:]))

def facts_for(dataset, slot=C.LSR_SLOT, fmt="full_prefix"):
    out = []
    for i in range(len(dataset)):
        meta = json.loads(dataset[i]["meta"])
        pfx = build_prefixes(meta, slot); gold = get_slot_value(meta, slot)
        if fmt == "full_prefix_full_suffix":
            prefix = pfx["full_prefix"]; gold = gold + " . " + build_full_suffix(meta, slot)
        else:
            prefix = pfx[fmt]
        out.append((prefix, gold))
    return out

def lsr_scores(model, tok, facts, E_ref, device, which="sem", topk=C.LSR_TOPK):
    s = []
    for prefix, gold in facts:
        r = layerwise_residual(model, tok, prefix, gold, E_ref, device, topk)
        if r is None: continue
        s.append(lsr_scalar(r[0] if which == "sem" else r[1]))
    return np.array(s)
