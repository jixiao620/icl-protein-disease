#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=analyze_AB
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/analyze_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/analyze_%j.error

export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
cd /work/jl1401/icl_protein_disease
python -u scripts/analyze_phenomenon.py
