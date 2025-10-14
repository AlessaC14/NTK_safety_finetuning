#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --time=8:00:00
#SBATCH --mem=700G
#SBATCH --output=results/logs/ft%j.out
#SBATCH --error=results/logs/ft%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4

# Print node and GPU info
echo "Running on node: $(hostname)"
echo "Available GPUs:"
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Create log directory if it doesn't exist
mkdir -p results/logs

# Activate environment
source ~/.bashrc
conda activate mechanistic_int

python /home/acarbol1/scratchenalisn1/acarbol1/NTK_safety_finetuning/kernel_regression/actual_ft.py