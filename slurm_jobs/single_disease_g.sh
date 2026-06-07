#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=single_disease_g
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/single_disease_g_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/single_disease_g_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Single-disease G experiment started: $(date) ==="
# Train + eval ICL on each G test disease independently (80/20 split, same as DNN baseline)
python -u scripts/train_eval_single_disease.py \
  --diseases G40 G20 G30 G35 \
  --data_dir processed_data_g \
  --output results/single_disease_g.json \
  --context_size 128 \
  --context_pos_ratio 0.5 \
  --query_pos_ratio 0.5 \
  --epochs 30
echo "=== Done: $(date) ==="
