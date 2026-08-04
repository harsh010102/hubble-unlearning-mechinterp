"""hubble8b — modular core for the Hubble-8B unlearning-family GPU run.

Shared logic lives here so a fix lands once. Thin notebooks import from this package:

    import hubble8b as h8
    h8.hf_login()
    forget, keep, unseen = h8.load_yago_splits()
    model, hist = h8.TRAINERS["npo"](forget["text"], retain["text"])

See README.md for run order. HF token is read from os.environ["HF_TOKEN"] only.
"""
from . import config
from .config import HP, TARGET_SLOT, LSR_SLOT, DEVICE, N_LAYERS, workdir
from .data import (load_yago_splits, load_retain_texts, build_distractor_pool,
                   build_prefixes, build_full_suffix, get_slot_value, SLOT_ORDER,
                   build_infill_examples, TEMPLATE_FORMATS)
from .models import (hf_login, get_tokenizer, load_model, load_named, to_gpu, to_cpu,
                     free_cuda, make_optimizer, push_model, upload_artifact, download_artifact)
from .methods import TRAINERS, train_npo, train_graddiff, train_idk, train_rmu, build_idk_text
from .evaluation import (run_infill, infill_accuracy, sequence_logprob, layerwise_target_rank,
                         build_Eref, layerwise_residual, lsr_scalar, lsr_scores, facts_for,
                         COLOR_MAP)
from . import sweep
from .sweep import make_eval_fn, run_sweep, SWEEP_GRIDS, grid_configs
from .config import combined_score, SWEEP_LAMBDA, SWEEP_FORGET_THRESH

__all__ = ["config", "HP", "TARGET_SLOT", "LSR_SLOT", "DEVICE", "N_LAYERS", "workdir",
           "load_yago_splits", "load_retain_texts", "build_distractor_pool", "build_prefixes",
           "build_full_suffix", "get_slot_value", "SLOT_ORDER", "build_infill_examples",
           "TEMPLATE_FORMATS", "hf_login", "get_tokenizer", "load_model", "load_named", "to_gpu",
           "to_cpu", "free_cuda", "make_optimizer", "push_model", "upload_artifact",
           "download_artifact", "TRAINERS", "train_npo", "train_graddiff", "train_idk",
           "train_rmu", "build_idk_text", "run_infill", "infill_accuracy", "sequence_logprob",
           "layerwise_target_rank", "build_Eref", "layerwise_residual", "lsr_scalar",
           "lsr_scores", "facts_for", "COLOR_MAP", "sweep", "make_eval_fn", "run_sweep",
           "SWEEP_GRIDS", "grid_configs", "combined_score", "SWEEP_LAMBDA", "SWEEP_FORGET_THRESH"]
