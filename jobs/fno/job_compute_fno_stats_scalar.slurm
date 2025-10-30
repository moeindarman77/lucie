#!/bin/bash
#PBS -N compute_fno_stats
#PBS -A UCSC0009
#PBS -q develop
#PBS -l select=1:ncpus=1:mem=128GB:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

# Load modules
source ~/.bashrc
module load conda
conda activate jax

# Set Python path
export PYTHONPATH="/glade/derecho/scratch/mdarman/lucie/src:$PYTHONPATH"

# Change to working directory
cd /glade/derecho/scratch/mdarman/lucie

# Run the script
python compute_fno_output_stats_scalar.py
