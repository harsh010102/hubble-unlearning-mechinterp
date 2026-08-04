"""The four unlearning trainers (families). All init from Perturbed; retain = WikiText.
8B memory: gradient checkpointing + use_cache=False + 8-bit AdamW + batch_size=1.

Fixes baked in (§7): NPO length-normalized log-ratio (unstick forget loss); IDK
slot-parametric refusals + suffix masking with hashlib (not builtin hash()).
"""
import random, json, hashlib
import torch
import torch.nn.functional as F
from . import config as C
from .models import load_model, make_optimizer, free_cuda, get_tokenizer
from .data import SLOT_ORDER, sentence, get_slot_value

def _batched(seq, bs):
    for i in range(0, len(seq) - bs + 1, bs):
        yield seq[i:i + bs]

def _prep_trainable(device=C.DEVICE):
    m = load_model(C.MODEL_PERTURBED, device, revision=C.REVISION)
    if C.GRADIENT_CHECKPOINT:
        m.gradient_checkpointing_enable()
    m.config.use_cache = False
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    return m

def _run_eval(model, ep, hp, eval_every, eval_fn, tag, hist, best):
    """Optional in-loop evaluation at epoch boundaries. No-op unless BOTH eval_every and
    eval_fn are given (so the method notebooks keep their exact old behaviour). Appends an
    eval record to `hist`, prints it, and keeps a CPU snapshot of the best-scoring weights
    so the trainer can return the best checkpoint instead of the last epoch."""
    if not (eval_every and eval_fn):
        return best
    if not ((ep + 1) % eval_every == 0 or ep == hp["epochs"] - 1):
        return best
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", None)
    model.eval(); model.config.use_cache = False
    try:
        with torch.no_grad():
            f_acc, r_acc = eval_fn(model)
    finally:
        model.config.use_cache = prev_cache
        if was_training:
            model.train()
    sc = C.combined_score(f_acc, r_acc)
    hist.append(dict(epoch=ep + 1, eval_forget_acc=float(f_acc),
                     eval_retain_acc=float(r_acc), score=float(sc)))
    print(f"[{tag}] epoch {ep+1} eval forget={f_acc:.3f} retain={r_acc:.3f} score={sc:.3f}")
    if best is None or sc > best["score"]:
        best = dict(epoch=ep + 1, score=float(sc), forget_acc=float(f_acc), retain_acc=float(r_acc),
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return best

def _restore_best(model, best, tag):
    """Load the best-scoring snapshot back into `model` (copy_ handles CPU->device)."""
    if best is not None and best.get("state") is not None:
        model.load_state_dict(best["state"])
        print(f"[{tag}] restored best checkpoint (epoch {best['epoch']}, score={best['score']:.3f})")

def causal_lm_loss(model, tok, texts, device, max_length=C.MAX_SEQ_LEN):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss

# ------------------------------------------------------------------ NPO
def _seq_logprob_norm(model, input_ids, attention_mask):
    """LENGTH-NORMALIZED mean-token log-prob (fixes the summed-logprob saturation bug)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    tgt = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(lp, 2, tgt.unsqueeze(-1)).squeeze(-1) * mask
    return tok_lp.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

def train_npo(forget_texts, retain_texts, device=C.DEVICE, hp_override=None, eval_every=None, eval_fn=None):
    hp = {**C.HP["npo"], **(hp_override or {})}; tok = get_tokenizer()
    model = _prep_trainable(device)
    ref = load_model(C.MODEL_PERTURBED, device, revision=C.REVISION)  # frozen reference (needs 80GB)
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    opt = make_optimizer(model.parameters(), hp["lr"])
    hist, step, best = [], 0, None
    for ep in range(hp["epochs"]):
        ft = list(forget_texts); random.Random(C.SEED + ep).shuffle(ft)
        for fb in _batched(ft, hp["batch_size"]):
            rb = random.sample(retain_texts, hp["batch_size"])
            enc = tok(fb, return_tensors="pt", truncation=True, max_length=hp["max_length"], padding=True).to(device)
            lp_theta = _seq_logprob_norm(model, enc["input_ids"], enc["attention_mask"])
            with torch.no_grad():
                lp_ref = _seq_logprob_norm(ref, enc["input_ids"], enc["attention_mask"])
            ratio = lp_theta - lp_ref
            forget_loss = -(2.0 / hp["beta"]) * F.logsigmoid(-hp["beta"] * ratio).mean()
            retain_loss = causal_lm_loss(model, tok, rb, device, hp["max_length"])
            loss = forget_loss + hp["retain_gamma"] * retain_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            hist.append(dict(step=step, forget_loss=float(forget_loss), retain_loss=float(retain_loss)))
            if step % 10 == 0:
                print(f"[NPO] step {step} forget={forget_loss:.4f} retain={retain_loss:.4f} ratio={ratio.mean():.3f}")
        best = _run_eval(model, ep, hp, eval_every, eval_fn, "NPO", hist, best)
    model.config.use_cache = True
    del ref; free_cuda()
    _restore_best(model, best, "NPO"); model.eval()
    return model, hist

# ------------------------------------------------------------------ GradDiff
def train_graddiff(forget_texts, retain_texts, device=C.DEVICE, hp_override=None, eval_every=None, eval_fn=None):
    hp = {**C.HP["graddiff"], **(hp_override or {})}; tok = get_tokenizer()
    model = _prep_trainable(device)
    opt = make_optimizer(model.parameters(), hp["lr"])
    hist, step, best = [], 0, None
    for ep in range(hp["epochs"]):
        ft = list(forget_texts); random.Random(C.SEED + ep).shuffle(ft)
        for fb in _batched(ft, hp["batch_size"]):
            rb = random.sample(retain_texts, hp["batch_size"])
            forget_loss = causal_lm_loss(model, tok, fb, device)
            retain_loss = causal_lm_loss(model, tok, rb, device)
            loss = -forget_loss + hp["retain_gamma"] * retain_loss   # ascend forget, descend retain
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()   # unbounded-below -> clip
            step += 1
            hist.append(dict(step=step, forget_loss=float(forget_loss), retain_loss=float(retain_loss)))
            if step % 10 == 0:
                print(f"[GradDiff] step {step} forget={forget_loss:.4f} retain={retain_loss:.4f}")
        best = _run_eval(model, ep, hp, eval_every, eval_fn, "GradDiff", hist, best)
    model.config.use_cache = True; free_cuda()
    _restore_best(model, best, "GradDiff"); model.eval()
    return model, hist

# ------------------------------------------------------------------ IDK
IDK_REFUSALS = {
    "birthdate": ["an undisclosed date", "a date that is not recorded", "an unspecified date",
                  "a date that has not been disclosed", "an unknown date", "a private, unlisted date",
                  "a date that is not publicly known", "a withheld date",
                  "a date that cannot be determined", "a date that is not on record"],
    "occupation": ["person whose occupation is not disclosed", "person whose profession is unknown",
                   "person with an unspecified occupation", "person who has not disclosed their occupation",
                   "person whose line of work is unrecorded", "person of undisclosed profession",
                   "person whose occupation is withheld", "private individual without a listed occupation",
                   "person whose job is not publicly known", "person whose occupation cannot be determined"],
}

def build_idk_text(meta, slot):
    key = int(hashlib.md5(meta["full_name"].encode()).hexdigest(), 16)   # reproducible, unlike hash()
    refusal = random.Random(key).choice(IDK_REFUSALS[slot])
    name = meta["full_name"]
    return " ".join(sentence(slot, name, refusal) if s == slot else sentence(s, name, get_slot_value(meta, s))
                    for s in SLOT_ORDER)

def _idk_masked_loss(model, tok, texts, metas, slot, device):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=C.MAX_SEQ_LEN, padding=True).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    tgt_idx = SLOT_ORDER.index(slot)
    for i, meta in enumerate(metas):
        pre = " ".join(sentence(s, meta["full_name"], get_slot_value(meta, s)) for s in SLOT_ORDER[:tgt_idx])
        cutoff = min(len(tok(pre, add_special_tokens=False)["input_ids"]) + 15, labels.shape[1])
        labels[i, cutoff:] = -100                            # mask everything after the target clause
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss

def train_idk(forget_metas, retain_texts, slot=C.TARGET_SLOT, device=C.DEVICE, hp_override=None, eval_every=None, eval_fn=None):
    hp = {**C.HP["idk"], **(hp_override or {})}; tok = get_tokenizer()
    model = _prep_trainable(device)
    idk_texts = [build_idk_text(m, slot) for m in forget_metas]
    opt = make_optimizer(model.parameters(), hp["lr"])
    hist, step, best = [], 0, None
    for ep in range(hp["epochs"]):
        idx = list(range(len(idk_texts))); random.Random(C.SEED + ep).shuffle(idx)
        for b in [idx[i:i + hp["batch_size"]] for i in range(0, len(idx) - hp["batch_size"] + 1, hp["batch_size"])]:
            fb = [idk_texts[i] for i in b]; mb = [forget_metas[i] for i in b]
            rb = random.sample(retain_texts, hp["batch_size"])
            forget_loss = _idk_masked_loss(model, tok, fb, mb, slot, device)
            retain_loss = causal_lm_loss(model, tok, rb, device)
            loss = forget_loss + retain_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            hist.append(dict(step=step, forget_loss=float(forget_loss), retain_loss=float(retain_loss)))
            if step % 10 == 0:
                print(f"[IDK] step {step} forget={forget_loss:.4f} retain={retain_loss:.4f}")
        best = _run_eval(model, ep, hp, eval_every, eval_fn, "IDK", hist, best)
    model.config.use_cache = True; free_cuda()
    _restore_best(model, best, "IDK"); model.eval()
    return model, hist

# ------------------------------------------------------------------ RMU
class _Hook:
    def __init__(self): self.out = None
    def __call__(self, m, i, o): self.out = o[0] if isinstance(o, tuple) else o

def _control_vectors(d_model, layers, coeff, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(C.SEED)
    v = {}
    for L in layers:
        u = torch.rand(d_model, generator=g) * 2 - 1
        v[L] = ((u / u.norm()) * coeff).to(device=device, dtype=dtype)
    return v

def _masked_mse(act, tgt, mask):
    m = mask.unsqueeze(-1).to(act.dtype)
    return ((act - tgt) ** 2 * m).sum() / (m.sum() * act.shape[-1]).clamp(min=1)

def train_rmu(forget_texts, retain_texts, device=C.DEVICE, hp_override=None, eval_every=None, eval_fn=None):
    hp = {**C.HP["rmu"], **(hp_override or {})}; tok = get_tokenizer(); layers = list(hp["layers"])
    model = load_model(C.MODEL_PERTURBED, device, revision=C.REVISION)
    for p in model.parameters(): p.requires_grad_(False)
    for L in layers:
        for p in model.model.layers[L].parameters(): p.requires_grad_(True)
    model.train()
    frozen = load_model(C.MODEL_PERTURBED, device, revision=C.REVISION); frozen.eval()
    for p in frozen.parameters(): p.requires_grad_(False)
    ctrl = _control_vectors(model.config.hidden_size, layers, hp["steering_coeff"], device, next(model.parameters()).dtype)
    up_h = {L: _Hook() for L in layers}; fr_h = {L: _Hook() for L in layers}
    up_hd = [model.model.layers[L].register_forward_hook(up_h[L]) for L in layers]
    fr_hd = [frozen.model.layers[L].register_forward_hook(fr_h[L]) for L in layers]
    opt = make_optimizer([p for p in model.parameters() if p.requires_grad], hp["lr"])
    hist, step, best = [], 0, None
    for ep in range(hp["epochs"]):
        ft = list(forget_texts); random.Random(C.SEED + ep).shuffle(ft)
        for fb in _batched(ft, hp["batch_size"]):
            rb = random.sample(retain_texts, hp["batch_size"])
            fe = tok(fb, return_tensors="pt", truncation=True, max_length=C.MAX_SEQ_LEN, padding=True).to(device)
            re_ = tok(rb, return_tensors="pt", truncation=True, max_length=C.MAX_SEQ_LEN, padding=True).to(device)
            model(**fe)
            forget_loss = sum(_masked_mse(up_h[L].out, ctrl[L].view(1, 1, -1).expand_as(up_h[L].out),
                                          fe["attention_mask"]) for L in layers) / len(layers)
            model(**re_)
            with torch.no_grad(): frozen(**re_)
            retain_loss = sum(_masked_mse(up_h[L].out, fr_h[L].out.detach(), re_["attention_mask"]) for L in layers) / len(layers)
            loss = forget_loss + hp["alpha"] * retain_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
            step += 1
            hist.append(dict(step=step, forget_loss=float(forget_loss), retain_loss=float(retain_loss)))
            if step % 10 == 0:
                print(f"[RMU] step {step} forget={forget_loss:.4f} retain={retain_loss:.4f}")
        best = _run_eval(model, ep, hp, eval_every, eval_fn, "RMU", hist, best)
    for h in up_hd + fr_hd: h.remove()
    del frozen; free_cuda()
    _restore_best(model, best, "RMU"); model.eval()
    return model, hist

TRAINERS = {"npo": train_npo, "graddiff": train_graddiff, "idk": train_idk, "rmu": train_rmu}
