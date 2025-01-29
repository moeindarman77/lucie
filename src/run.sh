# conda activate jax

# PYTHONPATH="/media/volume/moein-storage-1/ddpm_ocean/src" nohup python tools/train_vae.py --config config/glor_config.yaml > out_spectral_vae_v3.out 2>&1 &
# PYTHONPATH="/media/volume/moein-storage-1/ddpm_ocean/src" nohup python tools/train_ddpm_cond.py --config config/glor_config.yaml > out_ldm_spectral_loss_v3.out 2>&1 &
# PYTHONPATH="/media/volume/moein-storage-1/ddpm_ocean/src" nohup python tools/sample_ddpm_vae.py --config config/glor_config.yaml > out_ldm_sample.out 2>&1 &

# PYTHONPATH="/media/volume/moein-storage-1/ddpm_ocean/src" nohup python tools/FT_vae.py --config config/glor_FT.yaml > test_FT.out 2>&1 &
# PYTHONPATH="/media/volume/moein-storage-1/lucie/src" nohup python tools/FT_vae.py --config config/glor_FT.yaml > test_FT.out 2>&1 &

# PYTHONPATH="/media/volume/moein-storage-1/lucie/src" nohup python -u tools/sample_vae_4LUCIE.py --config config/ERA5_config_lsm_LUCIE.yaml > sample_out_today.out 2>&1 &
PYTHONPATH="/media/volume/moein-storage-1/lucie/src" nohup python -u tools/sample_vae_4LUCIE_regional.py --config config/ERA5_config_lsm_LUCIE_regional.yaml > sample_out_today_regional.out 2>&1 &