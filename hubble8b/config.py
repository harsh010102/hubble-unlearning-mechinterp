"""Central config for the Hubble-8B unlearning-family GPU run (A100).

Everything scale/run-specific lives here so a fix lands in exactly one place.
The HF token is NEVER stored here — it is read from the environment at run time.
"""
import os
import torch

# ---------------------------------------------------------------- Hugging Face
HF_USERNAME = "Harsh01012"
# Set HF_TOKEN in the environment (export HF_TOKEN=...). Never hard-code it in source.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------- Scale switch (1b / 8b)
# HUBBLE_SCALE selects the model family; default 8b (unchanged). Launch 1B jobs with:
#   sbatch --export=ALL,HUBBLE_SCALE=1b run_<x>.sbatch
SCALE_TAG = os.environ.get("HUBBLE_SCALE", "8b").lower()
assert SCALE_TAG in ("1b", "8b"), f"HUBBLE_SCALE must be 1b or 8b, got {SCALE_TAG!r}"

# Dataset repo that receives ALL artifacts (csv / png / json) from every run (scale-separated).
RESULTS_REPO = f"{HF_USERNAME}/hubble-{SCALE_TAG}-unlearning-results"
# Per-method unlearned-model repos.
MODEL_REPO_TMPL = HF_USERNAME + "/hubble-" + SCALE_TAG + "-{method}-unlearned-yago-birthdate"

# ---------------------------------------------------------------- Models (scale-dependent)
REVISION = "step238500"            # final 500B-token checkpoint tag, same for 1B and 8B ("main" also works)
if SCALE_TAG == "1b":
    MODEL_PERTURBED = "allegrolab/hubble-1b-500b_toks-perturbed-hf"
    MODEL_STANDARD  = "allegrolab/hubble-1b-500b_toks-standard-hf"
    N_LAYERS        = 16           # 1B: 16 layers (hidden 2048)
else:
    MODEL_PERTURBED = "allegrolab/hubble-8b-500b_toks-perturbed-hf"
    MODEL_STANDARD  = "allegrolab/hubble-8b-500b_toks-standard-hf"
    N_LAYERS        = 36           # 8B: 36 layers (hidden 4096)

# ---------------------------------------------------------------- Compute
DEVICE = "cuda:0"                  # pin explicitly; never a bare "cuda" on a multi-GPU box
DTYPE  = torch.bfloat16
MAX_SEQ_LEN = 512

# ---------------------------------------------------------------- Data / splits
YAGO_DATASET   = "allegrolab/biographies_yago"
RETAIN_DATASET = "Salesforce/wikitext"
RETAIN_CONFIG  = "wikitext-103-raw-v1"
RETAIN_MIN_LEN = 50
SEED           = 42
UNSEEN_SUBSAMPLE = 250
N_CANDIDATES        = 10           # infill distractors (1 true + 9)
N_CANDIDATES_EMAIL  = 15
EVAL_MAX_EXAMPLES   = None         # None = use the full split

TARGET_SLOT = "birthdate"          # accuracy / rank story (unguessable, clean tokenization)
# LSR slot: nationality discriminates Perturbed vs Standard ~23x better than occupation
# (gap 0.37 vs 0.016; slot-check job 18186) — strongly memorized, single-token, clean cluster.
LSR_SLOT    = "nationality"        # semantic-cluster (LSR) metric

# ---------------------------------------------------------------- Hyperparameters (8B)
# RMU layers stay [5,6,7] — the paper's own 8B config (Table 12). NPO/IDK/GradDiff are
# not in the paper; these carry over the 1B-tuned starting points (re-tune if needed).
# Values below are the winners of the Keep-retain sweep (job 17948, forget-first score).
# NOTE: reproducing these requires the Keep-set retain data used in that sweep — the method
# notebooks still build retain from WikiText, so update them too before a run_all reproduction.
HP = {
    "npo":      dict(beta=0.1, retain_gamma=2.0, lr=2e-5, batch_size=1, epochs=8, max_length=384),
    "graddiff": dict(retain_gamma=1.0, lr=1e-5, batch_size=1, epochs=8),
    "idk":      dict(lr=5e-5, batch_size=1, epochs=6),
    "rmu":      dict(layers=[5, 6, 7], alpha=100.0, steering_coeff=200.0, lr=5e-5, batch_size=1, epochs=4),
}

# RMU intervention layers are scale-dependent. 8B keeps the paper config [5,6,7] (~17% depth).
# 1B (16 layers) depth-matches to [3,4,5] so the "early-intervention" comparison is aligned
# (the paper's [5,6,7] would sit mid-stack at 1B). NPO/GradDiff/IDK HPs above are the 8B winners
# and serve only as the sweep's starting point at 1B — the 1B run re-tunes them.
if SCALE_TAG == "1b":
    HP["rmu"]["layers"] = [3, 4, 5]

# 8B memory: full-param methods need A100 80GB (bf16 + 8-bit AdamW + grad checkpointing).
USE_8BIT_ADAM        = True
GRADIENT_CHECKPOINT  = True

# ---------------------------------------------------------------- LSR metric
LSR_TOPK       = 20                # C_f = gold token + top-k output-embedding neighbors
LSR_LAYER_FRAC = 1 / 3            # scalar = max over layers >= N_LAYERS*frac

# ---------------------------------------------------------------- HP sweep / selection
# Forget-first selection: among configs that have actually unlearned (forget_acc <= a
# threshold set a margin above chance, to absorb 44-sample noise) maximise retain; configs
# that fail to unlearn are heavily penalised so they can never win. Higher = better.
SWEEP_LAMBDA        = 5.0
SWEEP_FORGET_THRESH = 2.0 / N_CANDIDATES     # 0.2 = chance (0.1) + noise margin

def combined_score(forget_acc, retain_acc, lam=None, thresh=None):
    lam = SWEEP_LAMBDA if lam is None else lam
    thresh = SWEEP_FORGET_THRESH if thresh is None else thresh
    return float(retain_acc) - lam * max(0.0, float(forget_acc) - thresh)

# ---------------------------------------------------------------- Local artifact dir
def workdir():
    for p in ("/kaggle/working", "/content"):
        if os.path.isdir(p):
            return p
    os.makedirs("artifacts", exist_ok=True)
    return "artifacts"
