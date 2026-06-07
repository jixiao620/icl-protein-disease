#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_i_half
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/train_i_half_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/train_i_half_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Train I-half model started: $(date) ==="
python -u scripts/train_full_gpt.py --config configs/train_i_half.yaml
echo "=== Done: $(date) ==="
