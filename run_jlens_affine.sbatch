#!/bin/bash
#SBATCH --job-name=h8-jaffine
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nisargnileshpanchal@gmail.com

# CORRECTED averaged-Jacobian lens, full run: rank-by-layer for all models (forget & retain) + a
# knowledge-vs-prior faithfulness control. HUBBLE_SCALE / JA_MODELS / JA_SPLITS / JA_QFACTS / JA_RCTX
# flow in via --export=ALL (safe for comma-valued vars; see slurm-export-comma-bug memory). Writes only
# rank_curves_jlens_affine_<tag>.csv, jlens_affine_rank_<tag>.png, jlens_affine_control_<tag>.png.

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

export HUBBLE_SCALE="${HUBBLE_SCALE:-1b}"
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and no cached token at $HF_HOME/token" >&2
  exit 1
fi

python -m pip install -q -r requirements.txt

echo "===== $(date '+%F %T') jlens affine FULL  scale=$HUBBLE_SCALE models=${JA_MODELS:-all} splits=${JA_SPLITS:-Forget,Retain} Q=${JA_QFACTS:-12} N=${JA_RCTX:-24} ====="
python run_jlens_affine.py

echo "===== $(date '+%F %T') jlens affine FULL complete ====="
