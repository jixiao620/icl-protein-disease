#!/bin/bash
#SBATCH --partition=common
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --job-name=explore_all
#SBATCH --output=logs/explore_all_%j.log

cd /work/jl1401/icl_protein_disease
python scripts/explore_all_opq_diseases.py
