#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=dnn_baseline
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/dnn_baseline_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/dnn_baseline_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== DNN expanded validation baseline ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u scripts/dnn_expanded_validation.py

echo "=== DONE: $(date) ==="
