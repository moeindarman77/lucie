#!/usr/bin/env python
"""
Diagnose v-wind issue by checking:
1. Raw training data values
2. Channel ordering in dataset
3. What the diffusion model actually sees during training
"""
import numpy as np
import h5py
import torch
import sys
import os
sys.path.insert(0, '/glade/derecho/scratch/mdarman/lucie/src')

from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset

print("="*80)
print("V-Wind Training Data Diagnosis")
print("="*80)

# ========== Step 1: Check raw HR data ==========
print("\n### Step 1: Raw HR Training Data ###\n")

hr_data_dir = "/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/data"
sample_files = [
    "2000010100.h5",
    "2001010100.h5",
    "2005010100.h5",
]

print(f"Checking sample HR files from: {hr_data_dir}\n")

for fname in sample_files:
    fpath = os.path.join(hr_data_dir, fname)
    if os.path.exists(fpath):
        with h5py.File(fpath, 'r') as f:
            print(f"File: {fname}")
            print(f"  Keys: {list(f['input'].keys())}")

            vwind = f['input']['v_component_of_wind_83'][:]
            print(f"  v_component_of_wind_83:")
            print(f"    Shape: {vwind.shape}")
            print(f"    Mean:  {vwind.mean():.6f} m/s")
            print(f"    Std:   {vwind.std():.6f} m/s")
            print(f"    Min:   {vwind.min():.6f} m/s")
            print(f"    Max:   {vwind.max():.6f} m/s")
            print()
    else:
        print(f"File not found: {fpath}")

# ========== Step 2: Check dataset output ==========
print("\n### Step 2: Dataset Output (After Normalization) ###\n")

input_vars = [
    'Temperature_7',
    'Specific_Humidity_7',
    'U-wind_3',
    'V-wind_3',
    'tp6hr',
    'orography',
    'land_sea_mask',
    'logp',
]

output_vars = [
    "2m_temperature",
    "specific_humidity_133",
    "u_component_of_wind_83",
    "v_component_of_wind_83",
    "total_precipitation_6hr",
    "geopotential_at_surface",
]

lr_lats = [87.159, 83.479, 79.777, 76.070, 72.362, 68.652, 64.942, 61.232,
       57.521, 53.810, 50.099, 46.389, 42.678, 38.967, 35.256, 31.545,
       27.833, 24.122, 20.411, 16.700, 12.989, 9.278, 5.567, 1.856,
       -1.856, -5.567, -9.278, -12.989, -16.700, -20.411, -24.122,
       -27.833, -31.545, -35.256, -38.967, -42.678, -46.389, -50.099,
       -53.810, -57.521, -61.232, -64.942, -68.652, -72.362, -76.070,
       -79.777, -83.479, -87.159]

lr_lons = [0.0, 3.75, 7.5, 11.25, 15.0, 18.75, 22.5, 26.25, 30.0, 33.75,
       37.5, 41.25, 45.0, 48.75, 52.5, 56.25, 60.0, 63.75, 67.5, 71.25,
       75.0, 78.75, 82.5, 86.25, 90.0, 93.75, 97.5, 101.25, 105.0, 108.75,
       112.5, 116.25, 120.0, 123.75, 127.5, 131.25, 135.0, 138.75, 142.5, 146.25,
       150.0, 153.75, 157.5, 161.25, 165.0, 168.75, 172.5, 176.25, 180.0, 183.75,
       187.5, 191.25, 195.0, 198.75, 202.5, 206.25, 210.0, 213.75, 217.5, 221.25,
       225.0, 228.75, 232.5, 236.25, 240.0, 243.75, 247.5, 251.25, 255.0, 258.75,
       262.5, 266.25, 270.0, 273.75, 277.5, 281.25, 285.0, 288.75, 292.5, 296.25,
       300.0, 303.75, 307.5, 311.25, 315.0, 318.75, 322.5, 326.25, 330.0, 333.75,
       337.5, 341.25, 345.0, 348.75, 352.5, 356.25]

print("Creating dataset...")
dataset = ClimateDataset(
    input_dir_lr="/glade/derecho/scratch/asheshc/ERA5_t30/train",
    input_dir_hr="/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/data",
    input_vars=input_vars,
    output_vars=output_vars,
    lr_lats=lr_lats,
    lr_lons=lr_lons,
    year_range=(2000, 2009),
    normalize=True,
    input_normalization_file='/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009_updated.npz',
    output_normalization_file='/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz',
    cache_file='/glade/derecho/scratch/mdarman/lucie/valid_files_2000_2009.npz',
    force_recompute=False
)

print(f"\nDataset length: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"\nSample 0 structure:")
print(f"  input shape:  {sample['input'].shape}")
print(f"  output shape: {sample['output'].shape}")
print(f"  input_vars:   {sample['input_vars']}")
print(f"  output_vars:  {sample['output_vars']}")

# Check the v-wind in the output
hres = sample['output']
print(f"\nNormalized HR output statistics (what diffusion model sees):")
for i, var in enumerate(output_vars):
    data = hres[i]
    print(f"  Channel {i} ({var}):")
    print(f"    Mean: {data.mean():.6f}")
    print(f"    Std:  {data.std():.6f}")

# ========== Step 3: Simulate training selection ==========
print("\n### Step 3: Simulating Training Data Selection ###\n")

print("In training, line 259 does: hres = hres[:, [0, 2, 3, 4]]")
print("This selects indices from the 6-channel HR data:\n")

selected_indices = [0, 2, 3, 4]
for i in selected_indices:
    print(f"  Index {i}: {output_vars[i]}")

print("\nAfter selection, the diffusion model's target has these channels:")
print("  Channel 0: 2m_temperature")
print("  Channel 1: u_component_of_wind_83")
print("  Channel 2: v_component_of_wind_83  ← This is the problem variable")
print("  Channel 3: total_precipitation_6hr")

# Simulate this selection
hres_selected = hres[[0, 2, 3, 4]]
print(f"\nAfter selection, shape: {hres_selected.shape}")
print("\nSelected channel statistics (what diffusion learns to match):")
for i in range(hres_selected.shape[0]):
    original_idx = selected_indices[i]
    var_name = output_vars[original_idx]
    data = hres_selected[i]
    print(f"  New index {i} (original {original_idx}, {var_name}):")
    print(f"    Mean: {data.mean():.6f}")
    print(f"    Std:  {data.std():.6f}")

# ========== Step 4: Check normalization stats ==========
print("\n### Step 4: Normalization Statistics ###\n")

hr_stats = np.load('/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz', allow_pickle=True)

print("HR normalization stats for v-wind:")
vwind_stats = hr_stats['v_component_of_wind_83'].item()
print(f"  Mean: {vwind_stats['mean']:.10f} m/s")
print(f"  Std:  {vwind_stats['std']:.10f} m/s")

# Denormalize the normalized sample to check
vwind_normalized = hres[3]  # Index 3 is v-wind
vwind_denormalized = vwind_normalized.numpy() * vwind_stats['std'] + vwind_stats['mean']

print(f"\nDenormalized v-wind from sample 0:")
print(f"  Mean: {vwind_denormalized.mean():.6f} m/s")
print(f"  Std:  {vwind_denormalized.std():.6f} m/s")

print("\n" + "="*80)
print("Diagnosis Complete!")
print("="*80)
print("\nNext step: Check if the issue is in the raw data or in the model training.")
