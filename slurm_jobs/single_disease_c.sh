#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=single_disease_c
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/single_disease_c_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/single_disease_c_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Single-disease C experiment started: $(date) ==="
# Train + eval ICL on each C test disease independently (80/20 split, same as DNN baseline)
python -u scripts/train_eval_single_disease.py \
  --diseases C34 C18 C43 C53 C54 \
  --data_dir processed_data_c \
  --output results/single_disease_c.json \
  --context_size 128 \
  --context_pos_ratio 0.5 \
  --query_pos_ratio 0.5 \
  --epochs 30
echo "=== Done: $(date) ==="
