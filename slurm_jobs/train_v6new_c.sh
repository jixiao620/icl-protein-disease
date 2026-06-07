#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=v6new_c
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/train_v6new_c_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/train_v6new_c_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== Line 2 - v6new architecture - C block ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u scripts/train_v6new.py --config configs/train_c_v6new.yaml

echo "=== DONE: $(date) ==="
