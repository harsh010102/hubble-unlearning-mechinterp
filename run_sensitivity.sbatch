#!/bin/bash
#SBATCH --job-name=hub-sens
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
set -uo pipefail
cd "$HOME/gpu_8b_run"; mkdir -p logs
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate myenv
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"
echo "===== $(date '+%F %T') Hubble-8B sensitivity (band + IDK distribution) ====="
python run_sensitivity.py
echo "===== $(date '+%F %T') done ====="
