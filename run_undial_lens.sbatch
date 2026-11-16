#!/bin/bash
#SBATCH --job-name=undial-lens
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail
cd "$HOME/gpu_8b_run"; mkdir -p logs
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate myenv
python run_undial_lens.py
