#!/bin/bash

#PBS -N diff_sampling           
#PBS -A UCSC0009        
#PBS -q main         
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=1
#PBS -l walltime=12:00:00 
#PBS -j oe
#PBS -J 0-26


# Load necessary modules
source ~/.bashrc  # Ensure Conda is correctly sourced
module load conda
# module load cuda/
conda activate jax

# Change to the directory where you submitted the job
cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

# Run the script
echo "Successfully loaded modules and changed to the directory"

start_index=$((PBS_ARRAY_INDEX * 700))
echo "Running job for index ${PBS_ARRAY_INDEX} with start_index=${start_index}"

# python -u tools/train_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml 
# python -u tools/sample_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml --start ${PBS_ARRAY_INDEX}


echo "Job ${PBS_ARRAY_INDEX} completed."
