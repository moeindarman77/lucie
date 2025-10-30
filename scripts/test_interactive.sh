#!/bin/bash

# Interactive test script for Lucie-3D downscaling pipeline
# Run this in an interactive GPU session

echo "=== Lucie-3D Downscaling Pipeline Test ==="
echo ""

echo "1. Setting up environment..."
source ~/.bashrc
module load conda
conda activate jax

echo "2. Setting environment variables..."
export INPUT_NC="/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_3D/LUCIE_co2_nosst_nomask_range_2000_2020.nc"
export LSM_PATH="/glade/derecho/scratch/mdarman/lucie/lsm_lr.npz"
export ORO_LR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_lr.npz"
export ORO_HR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_hr.npz"
export OUT_NC="/glade/derecho/scratch/mdarman/lucie/test_lucie3d_output.nc"

echo "3. Setting up Python path..."
cd /glade/derecho/scratch/mdarman/lucie/src
export PYTHONPATH=$(pwd)

echo "4. Checking GPU availability..."
nvidia-smi

echo ""
echo "=== Environment Setup Complete ==="
echo "INPUT_NC: $INPUT_NC"
echo "OUTPUT: $OUT_NC"
echo "Working directory: $(pwd)"
echo ""

echo "5. Running pipeline test with 2 samples..."
echo "Command: python tools/sample_ddpm_lucie3d.py --config config/ERA5_config_final_v2.yaml --start 0 --end 2"
echo ""

# Run the pipeline
time python tools/sample_ddpm_lucie3d.py --config config/ERA5_config_final_v2.yaml --start 0 --end 2

echo ""
echo "=== Pipeline Test Complete ==="
if [ -f "$OUT_NC" ]; then
    echo "✓ Output file created: $OUT_NC"
    echo "File size: $(du -h "$OUT_NC" | cut -f1)"
else
    echo "✗ Output file not found"
fi