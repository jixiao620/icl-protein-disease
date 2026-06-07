#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=eval_g_on_c
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_g_on_c_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_g_on_c_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_g_model/best_model.pt \
  --data_dir processed_data_c \
  --diseases C34 C18 C43 C53 C54 \
  --output results/g_model_on_c.json \
  --context_size 64 \
  --label "G model -> C test"
