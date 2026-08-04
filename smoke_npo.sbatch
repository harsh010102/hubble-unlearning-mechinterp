#!/bin/bash
#SBATCH --job-name=h8-smoke-npo
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

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

python -m pip install -q --upgrade --force-reinstall \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m pip install -q -r requirements.txt nbconvert jupyter

python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=14400 \
  --output smoke_run_npo.ipynb \
  run_npo.ipynb
