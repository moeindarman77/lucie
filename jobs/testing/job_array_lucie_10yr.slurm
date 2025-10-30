#!/bin/bash

#PBS -N lucie_10yr_sampling
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=8:mem=235GB:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -J 0-24

# ============================================================================
# LUCIE 10yr Production Sampling - Array Job Configuration
# ============================================================================
# Total samples: 14,600
# Array jobs: 25 (indices 0-24)
# Samples per job: 584
# Expected time per job: ~9.7 hours (584 minutes @ 1 min/sample)
# Coverage: 25 × 584 = 14,600 samples ✓
# ============================================================================

# Load necessary modules
source ~/.bashrc
module load conda
conda activate jax

# Change to the directory where you submitted the job
cd "/glade/derecho/scratch/mdarman/lucie/src"
export PYTHONPATH=$(pwd)

# Run the script
echo "============================================================================"
echo "LUCIE 10yr Production Sampling - Job ${PBS_ARRAY_INDEX}"
echo "============================================================================"
echo "Successfully loaded modules and changed to the directory"

# Calculate start index and number of samples for this job
samples_per_job=584
start_index=$((PBS_ARRAY_INDEX * samples_per_job))

# Calculate end index (for logging)
end_index=$((start_index + samples_per_job - 1))

echo "Job Configuration:"
echo "  Array Index: ${PBS_ARRAY_INDEX}"
echo "  Start Index: ${start_index}"
echo "  End Index: ${end_index}"
echo "  Samples to process: ${samples_per_job}"
echo "  Estimated time: ~9.7 hours"
echo "============================================================================"

# Run sampling
python -u tools/sample_ddpm_lucie_10yr.py \
    --config config/ERA5_config_lucie_10yr.yaml \
    --start ${start_index} \
    --num_samples ${samples_per_job}

echo "============================================================================"
echo "Job ${PBS_ARRAY_INDEX} completed successfully!"
echo "Processed samples: ${start_index} to ${end_index}"
echo "============================================================================"
