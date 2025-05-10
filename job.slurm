#!/bin/bash

#PBS -N ERA5_DDPM           
#PBS -A UCSC0009        
#PBS -q main         
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=1
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
# python -u tools/train_ae_regional.py --config config/ERA5_config_lsm_regional_ae.yaml

# python -u tools/sample_vae_4LUCIE.py --config config/ERA5_config_lsm.yaml
# python -u tools/sample_ae_4LUCIE.py --config config/ERA5_config_lsm_ae.yaml

# python -u tools/sample_vae_4LUCIE_regional.py --config config/ERA5_config_lsm_regional.yaml
# python -u tools/sample_ae_4LUCIE_regional.py --config config/ERA5_config_lsm_regional_ae.yaml
# python -u tools/train_fno.py --config config/ERA5_config_fno.yaml
# python -u tools/sample_fno_4LUCIE.py --config config/ERA5_config_fno.yaml
# python -u tools/sample_ddpm_vae.py --config config/ERA5_config_lsm.yaml

# python -u tools/train_vae_amirsajjad.py --config config/ERA5_config_lsm_amirsajjad.yaml
# python -u tools/sample_vae_4LUCIE_amirsajjd.py --config config/ERA5_config_lsm_amirsajjad.yaml

# python -u tools/train_vae_hr.py --config config/ERA5_config_lsm_hr.yaml
# python -u tools/sample_vae_4LUCIE_hr.py --config config/ERA5_config_lsm_hr.yaml

# python -u tools/train_ddpm_cond_fno.py --config config/ERA5_config_lsm.yaml
# python -u tools/sample_ddpm_vae_fno.py --config config/ERA5_config_lsm.yaml
# python -u :qtools/train_ddpm_cond_fno_hrhr.py --config config/ERA5_config_lsm_hr.yaml

# python -u tools/train_fno_ddpm_v2.py --config config/ERA5_config_fno.yaml
# python -u tools/sample_fno_ddpm_v2.py --config config/ERA5_config_fno.yaml

# python -u tools/train_ddpm_lenny.py --config config/ERA5_config_lenny.yaml
# python -u tools/train_ddpm_lenny_v2.py --config config/ERA5_config_lenny.yaml
# python -u tools/train_ddpm_lenny_v2_diff.py --config config/ERA5_config_lenny.yaml

# python -u tools/sample_ddpm_lenny.py --config config/ERA5_config_lenny.yaml
# python -u tools/sample_ddpm_lenny_v2.py --config config/ERA5_config_lenny.yaml


# Final Runs
# python -u tools/train_fno_final.py --config config/ERA5_config_fno.yaml
# python -u tools/sample_fno_final.py --config config/ERA5_config_fno.yaml

# python -u tools/train_ddpm_final.py --config config/ERA5_config_final.yaml
# python -u tools/sample_ddpm_final.py --config config/ERA5_config_final.yaml

# python -u tools/train_ddpm_final_jj.py --config config/ERA5_config_final.yaml
# python -u tools/sample_ddpm_final_jj.py --config config/ERA5_config_final.yaml

# python -u tools/train_ddpm_final_kk.py --config config/ERA5_config_final.yaml
# python -u tools/train_ddpm_final_diff.py --config config/ERA5_config_final.yaml

# python -u tools/sample_ddpm_final_kk.py --config config/ERA5_config_final.yaml

# python -u tools/sample_ddpm_final_diff.py --config config/ERA5_config_final.yaml

# python -u tools/train_ddpm_final_with_old_fno.py --config config/ERA5_config_final_temp.yaml
# python -u tools/sample_ddpm_final_with_old_fno.py --config config/ERA5_config_final_temp.yaml
# python -u tools/train_ddpm_lenny_v2_with_new_fno.py --config config/ERA5_config_lenny_temp.yaml
# python -u tools/sample_ddpm_lenny_v2_with_new_fno.py --config config/ERA5_config_lenny_temp.yaml

# Trainign DDPM with new/old FNO
# python -u tools/train_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml
# python -u tools/sample_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml

# Training Lazy Diffusion
# python -u tools/train_lazy_diff.py --config config/ERA5_config_lazy_diff.yaml
python -u tools/sample_lazy_diff.py --config config/ERA5_config_lazy_diff.yaml

echo "Done"
