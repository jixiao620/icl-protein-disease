#!/bin/bash
#SBATCH --partition=common
#SBATCH --mem=32G
#SBATCH --time=10:00
#SBATCH --job-name=explore_opq
#SBATCH --output=logs/explore_opq_%j.log

cd /work/jl1401/icl_protein_disease
python scripts/explore_congenital_perinatal.py
