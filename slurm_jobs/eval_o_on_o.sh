#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=eval_o_on_o
#SBATCH --output=logs/eval_o_on_o_%j.log

cd /work/jl1401/icl_protein_disease
python -u scripts/eval_o_model_on_o.py
