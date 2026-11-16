#!/bin/bash
#SBATCH --job-name=tofu-gate
#SBATCH --partition=short
#SBATCH --extra=vram:24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
set -uo pipefail
cd "$HOME/gpu_8b_run/tofu_dmech"; mkdir -p logs
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate myenv
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"
echo "===== $(date '+%F %T') TOFU anchor-strength gate ====="
python gate.py
echo "===== $(date '+%F %T') done ====="
