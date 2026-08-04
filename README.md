# Hubble-8B unlearning-family GPU run

Scales the 1B experiment to **`allegrolab/hubble-8b-500b_toks-*`** on a single **A100
(80 GB recommended)**. Compares four unlearning families — **NPO, GradDiff, IDK, RMU** —
against Standard/Perturbed baselines, with infill accuracy, logit-lens rank-by-layer,
and the **LSR semantic-residual faithfulness** analysis. All artifacts upload to Hugging Face.

Full project context and design rationale: `../MASTER_PROMPT_8B.md`.

## Layout

```
gpu_8b_run/
  hubble8b/            # modular core — a fix lands ONCE here
    config.py          # models, hyperparams, HF repos, paths (8B; RMU layers [5,6,7])
    data.py            # YAGO decode, seeded splits, leak-free prompts, distractor pool
    models.py          # load (no device_map + asserts), move, HF push/upload
    methods.py         # train_{npo,graddiff,idk,rmu}  (NPO length-norm; IDK slot-parametric)
    evaluation.py      # infill accuracy, logit-lens rank, LSR + faithfulness
  run_npo.ipynb        # thin per-method runners: train -> push model -> eval -> upload CSV
  run_graddiff.ipynb
  run_idk.ipynb
  run_rmu.ipynb
  master.ipynb         # baselines + aggregate bars + LSR faithfulness bake-off
  requirements.txt
  README.md
```

Why modular `.py` + thin notebooks: shared logic lives in one package, so fixes never drift
across notebooks (the exact failure we hit at 1B). Notebooks stay for GPU-box ergonomics.

## HF token — read from the environment, never hard-coded

Set it at run time (do **not** commit it):

```bash
export HF_TOKEN=hf_xxx      # your NEW token
```

On Colab: store it in `userdata` as `HF_TOKEN`. On Kaggle: add it as a Secret named `HF_TOKEN`.
The setup cell reads env → Colab userdata → getpass, in that order.

> Security: if a token was ever pasted in plaintext, **revoke and regenerate it** at
> huggingface.co → Settings → Access Tokens. The code only reads `os.environ["HF_TOKEN"]`.

## Run order

1. **On the A100 box:** clone/copy this folder; `pip install -r requirements.txt`; `export HF_TOKEN=...`.
2. **Run the four method notebooks** (any order; each is independent, ~one 8B model resident at a time):
   `run_npo.ipynb`, `run_graddiff.ipynb`, `run_idk.ipynb`, `run_rmu.ipynb`.
   Each trains from Perturbed, **pushes the unlearned model to HF**
   (`Harsh01012/hubble-8b-<method>-unlearned-yago-birthdate`), evaluates infill accuracy, and
   **uploads `infill_<METHOD>.csv`** to the results dataset (`Harsh01012/hubble-8b-unlearning-results`).
3. **Run `master.ipynb`** last: it computes the Standard/Perturbed baselines, pulls every method's
   CSV from HF, and produces + uploads the comparison bars and the **LSR faithfulness** box plot.

## Memory (A100)

- **80 GB (recommended):** all methods run as-is — bf16 weights + 8-bit AdamW + gradient
  checkpointing, `batch_size=1`. NPO/RMU keep a frozen reference resident (tightest).
- **40 GB:** RMU fits (trains only layers 5–7). Full-param NPO/GradDiff/IDK will **OOM** — switch
  them to LoRA or ZeRO-offload (see `MASTER_PROMPT_8B.md` §9.2). Not wired by default.

## Notes / knobs

- `hubble8b/config.py` is the only file you should need to edit: `HP` (per-method hyperparameters),
  `LSR_TOPK`, `RESULTS_REPO`, model IDs.
- `TARGET_SLOT="birthdate"` (accuracy/rank) and `LSR_SLOT="occupation"` (semantic-cluster metric)
  are intentionally different — see `MASTER_PROMPT_8B.md` §12.
- Not yet included (add-ons from `MASTER_PROMPT_8B.md`, easy follow-ups): rank-by-layer plot in the
  master (the `layerwise_target_rank` helper is ready), the LSR **invariance + k-sweep** tests, and
  the ROUGE/SRK bake-off metrics.
- **Not GPU-tested** — ported from the validated 1B code and syntax-checked. First real run: do a
  1-epoch smoke test per method before the full 8-epoch sweep.
