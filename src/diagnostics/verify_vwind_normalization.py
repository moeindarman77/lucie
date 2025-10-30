#!/usr/bin/env python
"""
Verify v-wind normalization by:
1. Checking the stored normalization stats
2. Recalculating mean/std from actual HR data
3. Checking FNO outputs vs Diffusion outputs
"""
import numpy as np
import os
from glob import glob
from tqdm import tqdm

print("="*80)
print("V-Wind Normalization Verification")
print("="*80)

# ========== Step 1: Check stored normalization stats ==========
print("\n### Step 1: Stored Normalization Statistics ###")

lr_stats = np.load('stats_lr_2000_2009_updated.npz', allow_pickle=True)
hr_stats = np.load('stats_hr_2000_2009_updated.npz', allow_pickle=True)

print("\nLOW-RES (LR) - V-wind_3:")
vwind_lr = lr_stats['V-wind_3'].item()
print(f"  Mean: {vwind_lr['mean']:.10f}")
print(f"  Std:  {vwind_lr['std']:.10f}")

print("\nHIGH-RES (HR) - v_component_of_wind_83:")
vwind_hr = hr_stats['v_component_of_wind_83'].item()
print(f"  Mean: {vwind_hr['mean']:.10f}")
print(f"  Std:  {vwind_hr['std']:.10f}")

# ========== Step 2: Recalculate from actual HR data ==========
print("\n### Step 2: Recalculating from Actual HR Data (2000-2009) ###")

data_dir = "/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/data"
years = range(2000, 2010)  # 2000-2009

print(f"\nScanning HR data directory: {data_dir}")
print(f"Years: {list(years)}")

# Gather all HR files
all_files = []
for year in years:
    year_files = sorted(glob(os.path.join(data_dir, f"{year}*.npz")))
    all_files.extend(year_files)

print(f"Found {len(all_files)} HR files")

if len(all_files) == 0:
    print("ERROR: No files found!")
else:
    # Calculate running mean and std using Welford's online algorithm
    count = 0
    mean = 0.0
    M2 = 0.0

    print("\nProcessing files to calculate mean and std...")
    for file_path in tqdm(all_files[:100], desc="Processing first 100 files"):  # Sample first 100 for speed
        data = np.load(file_path)

        # V-wind is channel 2 in HR data (index after temp=0, u-wind=1, v-wind=2)
        # Keys: '2m_temperature', 'u_component_of_wind_83', 'v_component_of_wind_83', ...
        if 'v_component_of_wind_83' in data:
            vwind = data['v_component_of_wind_83']

            # Flatten the spatial dimensions
            vwind_flat = vwind.flatten()

            # Update running statistics
            for value in vwind_flat:
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                M2 += delta * delta2

    variance = M2 / count
    std = np.sqrt(variance)

    print(f"\nRecalculated from {count:,} data points (from first 100 files):")
    print(f"  Mean: {mean:.10f}")
    print(f"  Std:  {std:.10f}")

    print(f"\nComparison with stored stats:")
    print(f"  Mean difference: {abs(mean - vwind_hr['mean']):.10f}")
    print(f"  Std difference:  {abs(std - vwind_hr['std']):.10f}")

# ========== Step 3: Check sample outputs ==========
print("\n### Step 3: Checking Sample Outputs (FNO vs Diffusion) ###")

samples_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr_retrain_best"

if os.path.exists(samples_dir):
    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.npz')])
    print(f"\nFound {len(sample_files)} sample files")

    if len(sample_files) > 0:
        # Load a few samples and check v-wind statistics
        num_samples_to_check = min(100, len(sample_files))

        fno_vwinds = []
        diff_vwinds = []

        print(f"\nLoading first {num_samples_to_check} samples...")
        for fname in tqdm(sample_files[:num_samples_to_check], desc="Loading samples"):
            path = os.path.join(samples_dir, fname)
            data = np.load(path)

            # V-wind is channel 2 (0=temp, 1=u-wind, 2=v-wind, 3=precip)
            fno_vwind = data['fno_output'][0, 2, ...]  # Shape: (721, 1440)
            diff_vwind = data['output'][0, 2, ...]

            fno_vwinds.append(fno_vwind)
            diff_vwinds.append(diff_vwind)

        fno_vwinds = np.array(fno_vwinds)
        diff_vwinds = np.array(diff_vwinds)

        print(f"\nFNO v-wind statistics (NORMALIZED, from {num_samples_to_check} samples):")
        print(f"  Mean: {fno_vwinds.mean():.10f}")
        print(f"  Std:  {fno_vwinds.std():.10f}")

        print(f"\nDiffusion v-wind statistics (NORMALIZED, from {num_samples_to_check} samples):")
        print(f"  Mean: {diff_vwinds.mean():.10f}")
        print(f"  Std:  {diff_vwinds.std():.10f}")

        # Denormalize using stored stats
        fno_vwinds_denorm = fno_vwinds * vwind_hr['std'] + vwind_hr['mean']
        diff_vwinds_denorm = diff_vwinds * vwind_hr['std'] + vwind_hr['mean']

        print(f"\nFNO v-wind statistics (DENORMALIZED):")
        print(f"  Mean: {fno_vwinds_denorm.mean():.10f}")
        print(f"  Std:  {fno_vwinds_denorm.std():.10f}")

        print(f"\nDiffusion v-wind statistics (DENORMALIZED):")
        print(f"  Mean: {diff_vwinds_denorm.mean():.10f}")
        print(f"  Std:  {diff_vwinds_denorm.std():.10f}")

        print(f"\nExpected denormalized mean should be close to: {vwind_hr['mean']:.10f}")

else:
    print(f"\nSample directory not found: {samples_dir}")

print("\n" + "="*80)
print("Verification Complete!")
print("="*80)
