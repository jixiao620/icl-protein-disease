#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_i_half2
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/train_i_half2_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/train_i_half2_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
echo "=== Train I-half2 started: $(date) ==="
python -u scripts/train_full_gpt.py --config configs/train_i_half2.yaml
echo "=== Done: $(date) ==="
