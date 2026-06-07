#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_o
#SBATCH --output=logs/train_o_model_%j.log

cd /work/jl1401/icl_protein_disease
python -u scripts/train_o_model.py
