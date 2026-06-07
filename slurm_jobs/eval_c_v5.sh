#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_c_v5
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_c_v5_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_c_v5_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Eval C v5 cross-block started: $(date) ==="
python -u scripts/eval_crossblock.py \
    --checkpoint checkpoints_c_model_v5/best_model.pt \
    --data_dir processed_data_c \
    --diseases C34 C18 C43 C53 C54 \
    --output results/c_v5_best_crossblock_ctx64.json \
    --label "ICL C v5 (best ep11) -> C test (ctx64)" \
    --context_size 64
echo "=== Done: $(date) ==="
