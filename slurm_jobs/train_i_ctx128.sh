#!/bin/bash
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_i_ctx128
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/train_i_ctx128_%j.log
#SBATCH --error=/work/jl1401/icl_protein_disease/logs/train_i_ctx128_%j.error
export PATH="/work/jl1401/miniconda3/envs/icl_protein/bin:/usr/share/Modules/bin:/work/jl1401/miniconda3/bin:/work/jl1401/miniconda3/condabin:/hpc/home/jl1401/.local/bin:/hpc/home/jl1401/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:/opt/slurm/bin"
set -e
cd /work/jl1401/icl_protein_disease
echo "=== Train I ctx128 started: $(date) ==="
python -u scripts/train_full_gpt.py --config configs/train_i_ctx128.yaml
echo "=== Done: $(date) ==="
