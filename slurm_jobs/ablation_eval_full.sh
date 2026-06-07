#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_full
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_full_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_full_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== EVAL: full v5 (anchor, reuse existing checkpoint) ==="
python -u scripts/eval_ablation.py \
    --checkpoint checkpoints_c_model_v5/best_model.pt \
    --label full_v5_anchor \
    --output_json results/ablation_full_v5.json \
    --output_npz  results/ablation_full_v5_scores.npz \
    --context_size 64

echo "=== DONE ==="
