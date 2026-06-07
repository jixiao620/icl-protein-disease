#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_g_v5
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_g_v5_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_g_v5_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Eval G v5 cross-block started: $(date) ==="
python -u scripts/eval_crossblock.py \
    --checkpoint checkpoints_g_model_v5/best_model.pt \
    --data_dir processed_data_g \
    --diseases G40 G20 G30 G35 \
    --output results/g_v5_best_crossblock_ctx64.json \
    --label "ICL G v5 (best ep23) -> G test (ctx64)" \
    --context_size 64
echo "=== Done: $(date) ==="
