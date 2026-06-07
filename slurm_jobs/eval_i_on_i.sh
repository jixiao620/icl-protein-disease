#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=eval_i_on_i
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_i_on_i_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_i_on_i_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_full_gpt/best_model.pt \
  --data_dir processed_data_full \
  --diseases I11 I119 I129 I15 I110 I13 I151 \
  --output results/i_model_on_i_ctx64.json \
  --context_size 64 \
  --label "I model -> I test (ctx64 eval)"
