#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=diagnose
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/diagnose_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/diagnose_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:$PATH"
set -e
cd /work/jl1401/icl_protein_disease
python -u scripts/diagnose_arch.py
