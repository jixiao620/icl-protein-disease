#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=drop_C61
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/dropC61_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/dropC61_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== TRAIN: drop-C61 / control (C44[capped]+C50+C809, total=91586) ==="
# C44 subsampled from 32263 to 29689 to match C61's size → same total as drop-C44
python -u scripts/train_ablation.py \
    --config configs/ablation_dropC61.yaml \
    --save_dir checkpoints_c_dropC61 \
    --max_samples_per_disease C44:29689

echo "=== EVAL: drop-C61 on all 5 C test diseases (ctx=64) ==="
python -u scripts/eval_ablation.py \
    --checkpoint checkpoints_c_dropC61/best_model.pt \
    --label drop_C61_control \
    --output_json results/ablation_dropC61.json \
    --output_npz  results/ablation_dropC61_scores.npz \
    --context_size 64

echo "=== DONE ==="
