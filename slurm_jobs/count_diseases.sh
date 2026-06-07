#!/bin/bash
#SBATCH --job-name=count_diseases
#SBATCH --partition=biostat
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/count_diseases_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/count_diseases_%j.log

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"

cd /work/jl1401/icl_protein_disease

echo "Job started: $(date)"
echo "Node: $(hostname)"

python scripts/count_disease_positives.py

echo "Job finished: $(date)"
