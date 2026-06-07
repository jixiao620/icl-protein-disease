#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --job-name=k_scaling_g
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_k_scaling_g_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_k_scaling_g_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== Line 2 - K scaling eval - G block v5 model ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u scripts/eval_k_scaling_g.py

echo "=== DONE: $(date) ==="
