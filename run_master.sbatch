#!/bin/bash
#SBATCH --job-name=h8-master
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

# HF cache lives on SSD scratch (per cluster wiki: home has a 50GB quota; SSD is for active
# data). Keeps the root disk from filling up and OSErroring mid-run.
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi

python -m pip install -q -r requirements.txt nbconvert jupyter

python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=14400 \
  --output master_executed.ipynb \
  master.ipynb