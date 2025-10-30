#!/bin/bash
#PBS -N lucie_10yr_latest
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -J 0-24

# Load necessary modules
source ~/.bashrc
module load conda
conda activate jax

# Change to the directory
cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

# Calculate start index and number of samples for this array job
samples_per_job=584
start_index=$((PBS_ARRAY_INDEX * samples_per_job))

echo "=========================================="
echo "Sampling from LATEST retrained model (epoch 415)"
echo "Array job index: $PBS_ARRAY_INDEX"
echo "Start index: $start_index"
echo "Number of samples: $samples_per_job"
echo "=========================================="

# Run the sampling script
python -u tools/sample_ddpm_lucie_10yr_retrain_latest.py \
    --config config/ERA5_config_lucie_10yr.yaml \
    --start $start_index \
    --num_samples $samples_per_job

echo "Done with array job $PBS_ARRAY_INDEX"
