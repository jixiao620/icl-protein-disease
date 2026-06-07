#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=xgb_g
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/baseline_xgb_g_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/baseline_xgb_g_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/train_baseline_xgboost_generic.py \
  --data_dir processed_data_g \
  --diseases G40 G20 G30 G35 \
  --output results/baseline_xgboost_g.json \
  --block G
