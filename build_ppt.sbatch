#!/bin/bash
#SBATCH --job-name=h8-buildppt
#SBATCH --partition=a100_debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# Build the full thesis-narrative deck (CPU-only render; runs on an idle node for speed).
set -uo pipefail
REPO_DIR="$HOME/gpu_8b_run"
CONDA_SH="${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
cd "$REPO_DIR"; mkdir -p logs
source "$CONDA_SH"; conda activate myenv
python -m pip install -q python-pptx pillow 2>/dev/null
python build_ppt_full.py
echo "===== $(date '+%F %T') deck build done ====="