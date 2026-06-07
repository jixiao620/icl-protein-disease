#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=eval_g_v2_best
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_g_v2_best_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_g_v2_best_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_g_model_v2/best_model.pt \
  --data_dir processed_data_g \
  --diseases G40 G20 G30 G35 \
  --output results/g_v2_best_on_g.json \
  --context_size 64 \
  --label "G model v2 (epoch2/best_loss) -> G test"
