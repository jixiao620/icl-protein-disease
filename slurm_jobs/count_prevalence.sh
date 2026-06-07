#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=count_prevalence
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/count_prevalence_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/count_prevalence_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python scripts/count_disease_prevalence.py
