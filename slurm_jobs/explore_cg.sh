#!/bin/bash
#SBATCH --partition=biostat
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=explore_cg
#SBATCH --output=/work/jl1401/icl_protein_disease/logs/explore_cg_%j.log

source /work/jl1401/miniconda3/etc/profile.d/conda.sh
conda activate icl_protein

python /work/jl1401/icl_protein_disease/explore_cg_blocks.py
