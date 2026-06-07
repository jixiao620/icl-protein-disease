#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=single_c_rem
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/single_c_rem_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/single_c_rem_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Single-disease C (remaining: C53, C54) started: $(date) ==="
python -u scripts/train_eval_single_disease.py \
  --diseases C53 C54 \
  --data_dir processed_data_c \
  --output results/single_disease_c_remaining.json \
  --context_size 32 \
  --context_pos_ratio 0.5 \
  --query_pos_ratio 0.5 \
  --epochs 30
echo "=== Done: $(date) ==="
