#!/bin/bash
#SBATCH --job-name=h8-lenstable
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nisargnileshpanchal@gmail.com

# Meta-evaluation of a logit-lens mechanistic DISTANCE metric vs lexical metrics (OpenUnlearning
# sec 4 protocol: Faithfulness = AUC separating knowledge-present from knowledge-absent pools).
# Part A: Perturbed vs Standard, output-matched conditioning (lexical collapses, mechanistic holds).
# Part B: 4 unlearned models, strong-relearn recovery oracle labels. Writes only mech_metaeval_* .

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

export HF_HOME=/mnt/ssd2/harshparikh/huggingface
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and no cached token at $HF_HOME/token" >&2
  exit 1
fi

python -m pip install -q -r requirements.txt

python run_lenstable.py

echo "===== $(date '+%F %T') mech meta-eval complete ====="
