#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=drop_C44
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/dropC44_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/dropC44_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== TRAIN: drop-C44 (C50+C61+C809, total=91586) ==="
python -u scripts/train_ablation.py \
    --config configs/ablation_dropC44.yaml \
    --save_dir checkpoints_c_dropC44

echo "=== EVAL: drop-C44 on all 5 C test diseases (ctx=64) ==="
python -u scripts/eval_ablation.py \
    --checkpoint checkpoints_c_dropC44/best_model.pt \
    --label drop_C44 \
    --output_json results/ablation_dropC44.json \
    --output_npz  results/ablation_dropC44_scores.npz \
    --context_size 64

echo "=== DONE ==="
