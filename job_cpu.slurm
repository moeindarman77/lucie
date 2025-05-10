#!/bin/bash

#PBS -N ERA5_DDPM           
#PBS -A UCSC0009        
#PBS -q main         
#PBS -l select=1:ncpus=8:mem=235GB
#PBS -l walltime=04:00:00 
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
# python -u utils/script.py --task_name ae_concat --num_samples 1825
# python -u utils/make_movie_temperature.py --task_name ae_concat --num_samples 300
# python -u utils/make_movie_precip.py --task_name ae_concat --num_samples 300
# python -u utils/make_movie_temp_conus.py --task_name ae_concat --num_samples 300
# python -u utils/make_movie_precip_conus.py --task_name ae_concat --num_samples 300

# ALl metrics + movie (Global model)
# python -u utils/script.py --task_name fno_final_v0 --num_samples 1140
# python -u utils/make_movie_temperature.py --task_name vae_concat_v6 --num_samples 200
# python -u utils/make_movie_precip.py --task_name vae_concat_v6 --num_samples 200
# python -u utils/make_movie_temp_conus.py --task_name vae_concat_v6 --num_samples 200
# python -u utils/make_movie_precip_conus.py --task_name vae_concat_v6 --num_samples 200


# (Regional model) movie
# python -u utils/script_regional.py --task_name ae_regional_us_v1 --num_samples 
# python -u utils/make_movie_temp_regional.py --task_name ae_regional_us_v1 --num_samples 10
# python -u utils/make_movie_precip_regional.py --task_name ae_regional_us_v1 --num_samples 10

# python -u utils/make_movie_temp_regional.py --task_name vae_regional_us --num_samples 10
# python -u utils/make_movie_precip_regional.py --task_name vae_regional_us --num_samples 10

# Diffusion model
# python -u utils/script_ddpm.py --task_name unet_final_v10 --num_samples 1
# python -u utils/make_movie_temperature_ddpm.py --task_name unet_simple_v4 --num_samples 143
# python -u utils/make_movie_precip_ddpm.py --task_name unet_simple_v4 --num_samples 143
# python -u utils/make_movie_temp_conus_ddpm.py --task_name unet_final_v0 --num_samples 10
# python -u utils/make_movie_precip_conus_ddpm.py --task_name unet_simple_v4 --num_samples 143

# python -u utils/script.py --task_name fno_v0 --num_samples 366

# python -u utils/script.py --task_name vae_hr --num_samples 32
# python -u utils/make_movie_temperature.py --task_name vae_hr --num_samples 32
# python -u utils/make_movie_precip.py --task_name vae_hr --num_samples 32
# python -u utils/make_movie_temp_conus.py --task_name vae_hr --num_samples 32
# python -u utils/make_movie_precip_conus.py --task_name vae_hr --num_samples 32

# python -u utils/make_movie_temperature.py --task_name test --num_samples 8


# python -u /glade/derecho/scratch/asheshc/get_stats_hr_afmarch17.py

# python -u utils/script_ddpm_for_new_fno.py --task_name unet_final_v10 --num_samples 1
# python -u utils/script_ddpm_for_new_fno.py --task_name lazy_diff_v1 --num_samples 14600
# python -u utils/script_ddpm_for_new_fno.py --task_name lazy_diff_v2 --num_samples 14600
python -u utils/script_ddpm_for_new_fno.py --task_name lazy_diff_v3 --num_samples 14600
echo "Done"




