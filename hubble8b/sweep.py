"""Guided hyperparameter sweep for the four unlearning trainers.

Selection metric: config.combined_score(forget_acc, retain_acc) — reward retaining Keep,
penalise forget accuracy above chance. The trainers (methods.py) are instrumented to
evaluate every `eval_every` epochs and RETURN their best-scoring checkpoint, so `epochs`
is swept for free within a single training run (checkpoints at 2/4/6/8 by default).

Fast proxy eval: forget/retain infill accuracy on ONE template format (`full_prefix`),
built once and reused; the full 4-format eval is deferred to the winner only.
"""
import json, itertools
import pandas as pd
from . import config as C
from .models import get_tokenizer, to_cpu, free_cuda
from .methods import TRAINERS
from .evaluation import infill_accuracy
from .data import build_infill_examples

# Coarse per-method grids (see plan). Keys must match HP entries; anything omitted keeps the
# config.HP default. `epochs` is folded in via in-loop eval — set one upper bound and let the
# best-checkpoint pick the effective stopping point.
SWEEP_GRIDS = {
    "npo":      dict(lr=[2e-5, 5e-5], retain_gamma=[0.5, 1.0, 2.0]),
    "graddiff": dict(lr=[5e-6, 1e-5], retain_gamma=[1.0, 2.0, 4.0]),
    "idk":      dict(lr=[2e-5, 5e-5]),
    "rmu":      dict(steering_coeff=[200.0, 500.0, 1000.0], alpha=[50.0, 100.0]),
}

def grid_configs(method):
    grid = SWEEP_GRIDS[method]
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))

def make_eval_fn(forget_set, keep_set, pool, tok, device=C.DEVICE, fmt="full_prefix"):
    """Return eval_fn(model) -> (forget_acc, retain_acc). Examples built once, reused."""
    ex_forget = build_infill_examples(forget_set, C.TARGET_SLOT, fmt, pool)
    ex_keep   = build_infill_examples(keep_set,   C.TARGET_SLOT, fmt, pool)
    def eval_fn(model):
        f = infill_accuracy(model, tok, ex_forget, device)
        r = infill_accuracy(model, tok, ex_keep, device)
        return f, r
    return eval_fn

def _forget_arg(method, forget_set):
    """IDK trains on slot-parametric refusal metas; the others on raw biography text."""
    if method == "idk":
        return [json.loads(r["meta"]) for r in forget_set]
    return forget_set["text"]

def run_sweep(method, forget_set, keep_set, retain_texts, eval_fn, eval_every=2, epochs=None):
    """Sweep one method's grid. Returns (results_df, best).

    results_df: one row per (config, eval-epoch) with forget/retain acc + score.
    best: dict(config, epoch, score, forget_acc, retain_acc, model)  — model on CPU, or None.
    Only the single best model is held (on CPU) across configs to bound memory.
    """
    farg = _forget_arg(method, forget_set)
    rows, best = [], None
    for cfg in grid_configs(method):
        hp_over = dict(cfg)
        if epochs is not None:
            hp_over["epochs"] = epochs
        eff_epochs = hp_over.get("epochs", C.HP[method]["epochs"])
        print(f"\n===== [{method}] config {cfg} (epochs={eff_epochs}, eval_every={eval_every}) =====")
        model, hist = TRAINERS[method](farg, retain_texts, hp_override=hp_over,
                                       eval_every=eval_every, eval_fn=eval_fn)
        evals = [h for h in hist if "score" in h]
        if not evals:
            print(f"  [{method}] no eval records; skipping config")
            to_cpu(model); del model; free_cuda(); continue
        top = max(evals, key=lambda h: h["score"])   # model already restored to this best epoch
        for h in evals:
            rows.append(dict(method=method, **cfg, epoch=h["epoch"],
                             forget_acc=h["eval_forget_acc"], retain_acc=h["eval_retain_acc"],
                             score=h["score"], is_config_best=(h["epoch"] == top["epoch"])))
        print(f"  [{method}] best-of-config: epoch {top['epoch']} score={top['score']:.3f} "
              f"forget={top['eval_forget_acc']:.3f} retain={top['eval_retain_acc']:.3f}")
        if best is None or top["score"] > best["score"]:
            if best is not None:
                del best["model"]; free_cuda()
            best = dict(config=cfg, epoch=top["epoch"], score=top["score"],
                        forget_acc=top["eval_forget_acc"], retain_acc=top["eval_retain_acc"],
                        model=to_cpu(model))          # park winner on CPU
        else:
            to_cpu(model); del model; free_cuda()
    if best is not None:
        print(f"\n##### [{method}] WINNER: config={best['config']} epoch={best['epoch']} "
              f"score={best['score']:.3f} forget={best['forget_acc']:.3f} retain={best['retain_acc']:.3f}")
    return pd.DataFrame(rows), best
