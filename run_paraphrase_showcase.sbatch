#!/bin/bash
#SBATCH --job-name=h8-para-showcase
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nisargnileshpanchal@gmail.com

# Explicit single-fact paraphrase-invariance showcase (visual analogue of the logit-lens table for the
# D_mech faithfulness claim): one person's birthdate asked several meaning-preserving ways, showing the
# gold month emerges the same way internally while surface P(gold)/ROUGE swing and D_mech stays put.
# Writes only paraphrase_showcase_* artifacts. Model = Perturbed (knows) + Standard (anchor).

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

python run_paraphrase_showcase.py

echo "===== $(date '+%F %T') paraphrase showcase complete ====="
