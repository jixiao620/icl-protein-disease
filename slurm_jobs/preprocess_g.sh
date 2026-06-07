#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=preprocess_g
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/preprocess_g_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/preprocess_g_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
echo "=== Preprocess G block started: $(date) ==="
python scripts/preprocess_full_proteins.py --config configs/preprocess_g.yaml
echo "=== Done: $(date) ==="
