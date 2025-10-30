#!/bin/bash
#PBS -N train_normalized_fno
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

# Verify FNO stats file exists before starting
STATS_FILE="/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz"
if [ ! -f "$STATS_FILE" ]; then
    echo "ERROR: FNO statistics file not found at $STATS_FILE"
    echo "Please ensure compute_fno_output_stats_scalar.py has completed successfully."
    exit 1
fi

echo "============================================================================"
echo "Training Diffusion Model with Normalized FNO Conditioning"
echo "============================================================================"
echo "Start time: $(date)"
echo "FNO stats file: $STATS_FILE"
echo ""

# Verify stats file structure
python -c "
import numpy as np
stats = np.load('$STATS_FILE', allow_pickle=True)
print('FNO Statistics Summary:')
print('-' * 60)
for key in stats.keys():
    data = stats[key].item()
    print(f'{key:15s}: mean = {data[\"mean\"]:12.6f}, std = {data[\"std\"]:12.6f}')
print('-' * 60)
print('')
"

# Run the normalized FNO training script
echo "Starting training..."
python -u tools/train_ddpm_normalized_fno.py --config config/ERA5_config_normalized_fno.yaml

echo ""
echo "============================================================================"
echo "Training completed at $(date)"
echo "============================================================================"
