#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --job-name=line1_g_eval
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/line1_g_eval_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/line1_g_eval_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== Line 1 G block evaluation ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u scripts/eval_line1_g.py

echo "=== DONE: $(date) ==="
