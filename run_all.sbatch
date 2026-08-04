#!/bin/bash
#SBATCH --job-name=h8-all
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nisargnileshpanchal@gmail.com

# End-to-end pipeline: train + push all four unlearning methods (each notebook uploads
# its model + infill CSV to HF, now PUBLIC), then run master.ipynb to produce the
# baselines, aggregate bars, forget-vs-retain logit lens, and LSR figures.
# One GPU, sequential — matches the onlygpu1 QOS (max 1 A100/user); notebooks run in
# separate kernels so GPU memory is fully released between methods.

# NOTE: not using `-e` so one failing method notebook does not abort the whole pipeline;
# master still runs at the end and aggregates whatever made it to HF.
set -uo pipefail

REPO_DIR="$HOME/gpu_8b_run"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
CONDA_ENV="myenv"

cd "$REPO_DIR"
mkdir -p logs

if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found at $CONDA_SH and not on PATH" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

# HF cache lives on SSD scratch (per cluster wiki: home has a 50GB quota; SSD is for active
# data). Keeps the root disk from filling up and OSErroring mid-run.
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
# HF token: honour an already-exported HF_TOKEN, else fall back to the cached CLI login.
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and no cached token at ~/.cache/huggingface/token" >&2
  exit 1
fi

python -m pip install -q -r requirements.txt nbconvert jupyter

run_nb () {
  local nb="$1"
  echo "===== $(date '+%F %T') executing $nb ====="
  python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=14400 \
    --output "${nb%.ipynb}_executed.ipynb" "$nb"
}

# Train each algorithm from config.HP (single source of truth — tuned HPs land here).
FAILED=()
for nb in run_npo.ipynb run_graddiff.ipynb run_idk.ipynb run_rmu.ipynb; do
  run_nb "$nb" || { echo "WARN: $nb failed — continuing"; FAILED+=("$nb"); }
done

# Always aggregate, regardless of individual method failures.
run_nb master.ipynb || { echo "ERROR: master.ipynb failed"; exit 1; }

echo "===== $(date '+%F %T') pipeline complete ====="
echo "failed method notebooks: ${FAILED[*]:-none}"
