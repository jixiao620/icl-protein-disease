#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_c_v7
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/train_c_v7_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/train_c_v7_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Train C model v7 started: $(date) ==="
python -u scripts/train_v7.py --config configs/train_c_model_v7.yaml
echo "=== Done: $(date) ==="
