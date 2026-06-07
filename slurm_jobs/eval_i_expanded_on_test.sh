#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval_i_expanded
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_i_expanded_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_i_expanded_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_i_expanded/best_model.pt \
  --data_dir processed_data_i_expanded \
  --diseases I11 I119 I129 I15 I110 I13 I151 I80 \
  --output results/i_expanded_model_on_test.json \
  --context_size 32 \
  --label "I-expanded model -> test (incl I80)"
