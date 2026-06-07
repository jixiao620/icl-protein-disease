#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --job-name=line1_preprocess
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/line1_g_preprocess_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/line1_g_preprocess_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease

echo "=== Line 1 G block preprocessing ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

python -u scripts/preprocess_line1_g.py

echo "=== DONE: $(date) ==="
