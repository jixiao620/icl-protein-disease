#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=single_g_rem
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/single_g_rem_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/single_g_rem_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Single-disease G (remaining: G35) started: $(date) ==="
python -u scripts/train_eval_single_disease.py \
  --diseases G35 \
  --data_dir processed_data_g \
  --output results/single_disease_g_remaining.json \
  --context_size 32 \
  --context_pos_ratio 0.5 \
  --query_pos_ratio 0.5 \
  --epochs 30
echo "=== Done: $(date) ==="
