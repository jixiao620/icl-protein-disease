#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=gpt_ctx64
#SBATCH --output=logs/train_ctx64_%j.log

cd /work/jl1401/icl_protein_disease
python -u scripts/train_gpt_ctx64.py
