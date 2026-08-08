#!/bin/bash
#SBATCH --job-name=h8-jlensaff
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nisargnileshpanchal@gmail.com

# CORRECTED averaged-Jacobian lens prototype: adds the intercept term the old J-lens dropped, and
# reports gold-rank-by-layer under logit / per-input-J / avg-J-no-intercept / avg-J+INTERCEPT for one
# query. If avg-J+INTERCEPT is smooth and lands at the true final rank, the fix works. HUBBLE_SCALE
# defaults to 1b (cheap); set to 8b to run the 8B model. Writes only jlens_affine_proto_<tag>.csv.

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

echo "===== $(date '+%F %T') jlens affine proto  HUBBLE_SCALE=$HUBBLE_SCALE ====="
python run_jlens_affine_proto.py

echo "===== $(date '+%F %T') jlens affine proto complete ====="
