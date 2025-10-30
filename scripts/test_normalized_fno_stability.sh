#!/bin/bash

# Quick stability test for normalized FNO model
# Generates just 3 samples to check if the model is stable

source ~/.bashrc
module load conda
conda activate jax

cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

echo "========================================"
echo "Testing normalized FNO model stability"
echo "Generating 3 test samples..."
echo "========================================"

# Run sampling with just 3 samples starting from index 0
python -u tools/sample_ddpm_normalized_fno.py \
    --config config/ERA5_config_normalized_fno.yaml \
    --start 0 \
    --num_samples 3

echo ""
echo "========================================"
echo "Test completed!"
echo "Check results in: results/unet_normalized_fno/samples_normalized_fno/"
echo "Files: 1.npz, 2.npz, 3.npz"
echo "========================================"
