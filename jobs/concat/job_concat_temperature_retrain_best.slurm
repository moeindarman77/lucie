#!/bin/bash

#PBS -N concat_temp_best
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=1:mem=128GB:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe

# Load necessary modules
source ~/.bashrc
module load conda
conda activate jax

# Change to working directory
cd "/glade/derecho/scratch/mdarman/lucie"

echo "============================================================================"
echo "Concatenating Temperature Samples - LUCIE 10yr (Best Retrained Model)"
echo "============================================================================"
echo "Start time: $(date)"

# Run the concatenation script
python -u concat_temperature_retrain_best.py

echo "End time: $(date)"
echo "============================================================================"
