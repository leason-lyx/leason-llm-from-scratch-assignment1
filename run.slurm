#!/bin/bash
#SBATCH -J train_owt_leason
#SBATCH -p lfs-dev-gpu
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.log
#SBATCH -e logs/%x-%j.log

set -euo pipefail
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

srun uv run cs336_basics/train.py