#!/bin/bash
#PBS -N ERA5_DDPM_retrain
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=4
#PBS -l walltime=12:00:00
#PBS -j oe

# Load necessary modules
source ~/.bashrc  # Ensure Conda is correctly sourced
module load conda
conda activate jax

# Change to the directory where you submitted the job
cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

# Run the retraining script
echo "Starting retraining job at $(date)"
echo "This is job 1 of 4 in the training chain"

python -u tools/train_ddpm_final_v2_retrain.py --config config/ERA5_config_final_v2_retrain.yaml

echo "Job 1 completed at $(date)"
echo "Done"
