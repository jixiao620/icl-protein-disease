#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_c54_v4
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_c54_v4_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_c54_v4_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_c_model_v4/best_model.pt \
  --data_dir processed_data_c \
  --diseases C54 \
  --output results/c_v4_best_c54_only.json \
  --context_size 64 \
  --label "C model v4 (best_auroc) -> C54 only"
