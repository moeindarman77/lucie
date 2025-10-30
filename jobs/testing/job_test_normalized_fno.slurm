#!/bin/bash
#PBS -N test_norm_fno
#PBS -A UCSC0009
#PBS -q develop
#PBS -l select=1:ncpus=4:mem=64GB:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe

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

# Quick check for NaN values in output
echo ""
echo "Checking for NaN values in samples..."
python -c "
import numpy as np
import os

samples_dir = '/glade/derecho/scratch/mdarman/lucie/results/unet_normalized_fno/samples_normalized_fno'
for i in [1, 2, 3]:
    fpath = os.path.join(samples_dir, f'{i}.npz')
    if os.path.exists(fpath):
        data = np.load(fpath)
        output = data['output']
        fno_output = data['fno_output']
        has_nan_output = np.isnan(output).any()
        has_nan_fno = np.isnan(fno_output).any()
        print(f'Sample {i}: Output NaN={has_nan_output}, FNO NaN={has_nan_fno}')
        print(f'  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}, std={output.std():.4f}')
    else:
        print(f'Sample {i}: File not found')
"
