#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=sanity_o
#SBATCH --output=logs/sanity_check_o_%j.log

source ~/.bashrc
conda activate icl_protein

cd /work/jl1401/icl_protein_disease
python -u scripts/eval_o_model_sanity_check.py
