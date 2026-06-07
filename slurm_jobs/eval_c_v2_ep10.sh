#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=eval_c_v2_ep10
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/eval_c_v2_ep10_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/eval_c_v2_ep10_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/eval_crossblock.py \
  --checkpoint checkpoints_c_model_v2/checkpoint_epoch_10.pt \
  --data_dir processed_data_c \
  --diseases C34 C18 C43 C53 C54 \
  --output results/c_v2_ep10_on_c.json \
  --context_size 64 \
  --label "C model v2 (epoch10) -> C test"
