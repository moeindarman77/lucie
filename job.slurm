#!/bin/bash

#PBS -N ERA5_DDPM           
#PBS -A UCSC0009        
#PBS -q main         
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=4
#PBS -l walltime=12:00:00 
#PBS -j oe              

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
# python -u tools/train_vae.py --config config/ERA5_config_lsm.yaml
# python -u tools/train_ddpm_cond.py --config config/ERA5_config_lsm.yaml
# python -u tools/train_vae_regional.py --config config/ERA5_config_lsm_regional.yaml
# python -u tools/train_ae.py --config config/ERA5_config_lsm_ae.yaml
python -u tools/train_ae_regional.py --config config/ERA5_config_lsm_regional_ae.yaml

echo "Done"
