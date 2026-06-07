#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --job-name=preprocess_v6
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/preprocess_v6_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/preprocess_v6_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Preprocess expanded diseases (v6) started: $(date) ==="
python -u scripts/preprocess_expanded.py --block both
echo "=== Done: $(date) ==="
