#!/bin/bash

#PBS -N lucie_10yr_test
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe

# Load necessary modules
source ~/.bashrc
module load conda
conda activate jax

# Change to the directory where you submitted the job
cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

# Run the script for 10 test samples
echo "Running LUCIE 10yr test sampling (10 samples)..."
python -u tools/sample_ddpm_lucie_10yr.py --config config/ERA5_config_lucie_10yr.yaml --start 0 --num_samples 10

echo "Test sampling completed."
