#!/bin/bash
#SBATCH --job-name=cir
#SBATCH --time=12:00:00
#SBATCH --account=plgunlearningai-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

# 48h is max
# I can use up to --cpus-per-task=16 and --mem=128G, but it may make queue wait times longer

# for more gpus, use #SBATCH --gres=gpu:2    and hf trainers should figure out how to use that

module load CUDA/12.8.0

# Use $SCRATCH for model cache (temporary, will be cleaned up)
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch

cd $HOME/open-unlearning
source .venv/bin/activate
# source .env

# # debug
# srun echo $CUDA_VISIBLE_DEVICES
# srun echo $HF_HOME
# srun which python
# srun nvidia-smi

# pip install --no-build-isolation flash-attn==2.6.3

# bash scripts/tofu_unlearn.sh

srun "$@" paths.tmp_comm_dir=$SCRATCH/tmp_comm