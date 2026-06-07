#!/bin/bash
#SBATCH --partition=common
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=process_o
#SBATCH --output=logs/process_o_%j.log

cd /work/jl1401/icl_protein_disease
python -u scripts/process_o_diseases.py
