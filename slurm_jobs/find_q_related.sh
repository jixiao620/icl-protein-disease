#!/bin/bash
#SBATCH --partition=common
#SBATCH --mem=32G
#SBATCH --time=10:00
#SBATCH --job-name=find_q
#SBATCH --output=logs/find_q_%j.log

cd /work/jl1401/icl_protein_disease
python scripts/find_q_related.py
