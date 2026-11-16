#!/bin/bash
#SBATCH --job-name=gpudiag
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:46:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
cd "$HOME/gpu_8b_run/tofu_dmech"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate myenv
echo "NODE=$(hostname)  CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]  SLURM_JOB_GPUS=[$SLURM_JOB_GPUS]  GPU_DEVICE_ORDINAL=[$GPU_DEVICE_ORDINAL]"
echo "--- nvidia-smi -L ---"; nvidia-smi -L 2>&1
echo "--- torch ---"; python -c "import torch;print('count',torch.cuda.device_count(),'avail',torch.cuda.is_available()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]" 2>&1
