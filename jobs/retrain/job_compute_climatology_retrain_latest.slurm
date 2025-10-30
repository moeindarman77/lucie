#!/bin/bash

#PBS -N compute_clim_latest
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=1:mem=100GB:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe

# Load necessary modules
source ~/.bashrc
module load conda
conda activate jax

# Change to working directory
cd "/glade/derecho/scratch/mdarman/lucie"

echo "============================================================================"
echo "Computing Climatology - LUCIE 10yr Retrain Latest"
echo "============================================================================"
echo "Start time: $(date)"

# Run the climatology computation
python -u compute_climatology_retrain_latest.py

echo "End time: $(date)"
echo "============================================================================"
