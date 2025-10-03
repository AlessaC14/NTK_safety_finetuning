#!/bin/bash
#SBATCH --job-name=test_opposition
#SBATCH --time=02:00:00
#SBATCH --mem=700G
#SBATCH --output=results/logs/finetuning%j.out
#SBATCH --error=results/logs/finetuning%j.err
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

python /home/acarbol1/scratchenalisn1/acarbol1/NTK_safety_finetuning/kernel_baseline/finetuning/memory_ef_ft.py