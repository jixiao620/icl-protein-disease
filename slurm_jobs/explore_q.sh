#!/bin/bash
#SBATCH --partition=common
#SBATCH --mem=32G
#SBATCH --time=10:00
#SBATCH --job-name=explore_q
#SBATCH --output=logs/explore_q_%j.log

cd /work/jl1401/icl_protein_disease
python scripts/explore_q_diseases.py
