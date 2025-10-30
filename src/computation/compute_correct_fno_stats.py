#!/usr/bin/env python
"""
Compute CORRECT FNO normalization statistics from raw FNO outputs.

This script:
1. Loads training data
2. Runs FNO on low-resolution inputs
3. Computes statistics on RAW (unnormalized) FNO outputs
4. Saves corrected normalization stats

The current fno_output_stats_scalar.npz has wrong stats (mean≈0, std≈1)
because it was computed on already-normalized data.
"""

import torch
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from models.fno import FNO2d
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("Computing CORRECT FNO Normalization Statistics")
print("="*80)

# Load configuration
config_path = 'src/config/ERA5_config_normalized_fno.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

dataset_config = config['dataset_params']
fno_config = config['fno_params']

# Create dataset
print("\nLoading training dataset...")
input_vars = [
    'Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3',
    'tp6hr', 'orography', 'land_sea_mask', 'logp'
]
output_vars = [
    "2m_temperature", "specific_humidity_133",
    "u_component_of_wind_83", "v_component_of_wind_83",
    "total_precipitation_6hr", "geopotential_at_surface"
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

dataset = ClimateDataset(
    input_dir_lr=dataset_config['input_data_dir'],
    input_dir_hr=dataset_config['output_data_dir'],
    input_vars=input_vars,
    output_vars=output_vars,
    lr_lats=lr_lats,
    lr_lons=lr_lons,
    year_range=(dataset_config['year_range_start'], dataset_config['year_range_end']),
    normalize=True,
    input_normalization_file=dataset_config['input_normalization_dir'],
    output_normalization_file=dataset_config['output_normalization_dir'],
    cache_file=dataset_config['cache_file'],
    force_recompute=dataset_config['force_recompute']
)
print(f"Dataset loaded: {len(dataset)} samples")

# Load FNO model
print("\nLoading FNO model...")
SR_model = FNO2d(
    input_channels=dataset_config['input_channels'] + dataset_config['land_sea_mask'],
    output_channels=dataset_config['output_channels'],
    model_config=fno_config
)
SR_model = SR_model.to(device)
SR_model.eval()

# Load FNO checkpoint
fno_checkpoint_path = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v1/checkpoints/best_fno.pth"
checkpoint = torch.load(fno_checkpoint_path, map_location=device)
SR_model.load_state_dict(checkpoint['model_state_dict'])
print(f"FNO model loaded from: {fno_checkpoint_path}")

# Subsample dataset for statistics (use every 10th sample for speed)
subsample_indices = list(range(0, len(dataset), 10))
print(f"\nComputing statistics on {len(subsample_indices)} samples (subsampled from {len(dataset)})...")

# Collect FNO outputs
fno_outputs = {
    'temperature': [],
    'uwind': [],
    'vwind': [],
    'precipitation': []
}

# Load HR normalization to denormalize ground truth (for comparison)
hr_stats = np.load(dataset_config['output_normalization_dir'], allow_pickle=True)
hr_means = {
    'temperature': hr_stats['2m_temperature'].item()['mean'],
    'uwind': hr_stats['u_component_of_wind_83'].item()['mean'],
    'vwind': hr_stats['v_component_of_wind_83'].item()['mean'],
    'precipitation': hr_stats['total_precipitation_6hr'].item()['mean']
}
hr_stds = {
    'temperature': hr_stats['2m_temperature'].item()['std'],
    'uwind': hr_stats['u_component_of_wind_83'].item()['std'],
    'vwind': hr_stats['v_component_of_wind_83'].item()['std'],
    'precipitation': hr_stats['total_precipitation_6hr'].item()['std']
}

print("\nProcessing samples...")
with torch.no_grad():
    for idx in tqdm(subsample_indices):
        data = dataset[idx]
        lres = data['input'].unsqueeze(0).float().to(device)  # (1, 8, 48, 96)

        # Upsample and run through FNO
        lres_upsampled = F.interpolate(lres, size=(721, 1440), mode='bicubic', align_corners=True)
        fno_output = SR_model(lres_upsampled)  # (1, 4, 721, 1440)

        # Extract each variable (RAW FNO output, NOT normalized)
        # FNO output is ALREADY in normalized space (from training)
        # We need to DENORMALIZE it to get raw values
        fno_np = fno_output.cpu().numpy()[0]  # (4, 721, 1440)

        # Denormalize FNO outputs using HR stats
        temp_raw = fno_np[0] * hr_stds['temperature'] + hr_means['temperature']
        uwind_raw = fno_np[1] * hr_stds['uwind'] + hr_means['uwind']
        vwind_raw = fno_np[2] * hr_stds['vwind'] + hr_means['vwind']
        precip_raw = fno_np[3] * hr_stds['precipitation'] + hr_means['precipitation']

        # Store
        fno_outputs['temperature'].append(temp_raw)
        fno_outputs['uwind'].append(uwind_raw)
        fno_outputs['vwind'].append(vwind_raw)
        fno_outputs['precipitation'].append(precip_raw)

# Compute statistics
print("\nComputing statistics...")
stats = {}

for var_name in ['temperature', 'uwind', 'vwind', 'precipitation']:
    # Concatenate all samples
    all_data = np.concatenate([arr.flatten() for arr in fno_outputs[var_name]])

    mean_val = float(np.mean(all_data))
    std_val = float(np.std(all_data))

    stats[var_name] = {
        'mean': mean_val,
        'std': std_val,
        'min': float(np.min(all_data)),
        'max': float(np.max(all_data)),
        'num_samples': len(subsample_indices)
    }

# Display results
print("\n" + "="*80)
print("CORRECTED FNO NORMALIZATION STATISTICS")
print("="*80)
print(f"\n{'Variable':<15} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
print("-"*80)
for var_name in ['temperature', 'uwind', 'vwind', 'precipitation']:
    s = stats[var_name]
    print(f"{var_name:<15} {s['mean']:<15.6f} {s['std']:<15.6f} {s['min']:<15.6f} {s['max']:<15.6f}")

# Compare with ground truth
print("\n" + "="*80)
print("COMPARISON WITH GROUND TRUTH (HR DATA)")
print("="*80)
print(f"\n{'Variable':<15} {'FNO Mean':<15} {'HR Mean':<15} {'Difference':<15}")
print("-"*80)
for var_name in ['temperature', 'uwind', 'vwind', 'precipitation']:
    fno_mean = stats[var_name]['mean']
    hr_mean = hr_means[var_name]
    diff = fno_mean - hr_mean
    print(f"{var_name:<15} {fno_mean:<15.6f} {hr_mean:<15.6f} {diff:<15.6f}")

# Validation checks
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)
all_good = True

# Check that FNO means are reasonably close to HR means
for var_name, threshold in [('temperature', 5.0), ('uwind', 2.0), ('vwind', 2.0), ('precipitation', 0.0005)]:
    diff = abs(stats[var_name]['mean'] - hr_means[var_name])
    status = "✓ PASS" if diff < threshold else "✗ FAIL"
    print(f"{var_name:<15} mean difference: {diff:10.6f} (threshold: {threshold}) {status}")
    if diff >= threshold:
        all_good = False

if all_good:
    print("\n✓ All validation checks passed!")
else:
    print("\n⚠ Some validation checks failed. Review FNO model output.")

# Save corrected statistics
output_file = '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_corrected.npz'
np.savez(output_file,
         temperature=np.array(stats['temperature'], dtype=object),
         uwind=np.array(stats['uwind'], dtype=object),
         vwind=np.array(stats['vwind'], dtype=object),
         precipitation=np.array(stats['precipitation'], dtype=object))

print(f"\n✓ Corrected statistics saved to: {output_file}")

# Compare old vs new
print("\n" + "="*80)
print("OLD (WRONG) vs NEW (CORRECTED) STATISTICS")
print("="*80)

old_stats = np.load('/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz', allow_pickle=True)
print(f"\n{'Variable':<15} {'Old Mean':<15} {'New Mean':<15} {'Old Std':<15} {'New Std':<15}")
print("-"*80)
var_mapping = {
    'temperature': 'temperature',
    'uwind': 'uwind',
    'vwind': 'vwind',
    'precipitation': 'precipitation'
}
for var_name, old_key in var_mapping.items():
    old_mean = float(old_stats[old_key].item()['mean'])
    old_std = float(old_stats[old_key].item()['std'])
    new_mean = stats[var_name]['mean']
    new_std = stats[var_name]['std']
    print(f"{var_name:<15} {old_mean:<15.6f} {new_mean:<15.6f} {old_std:<15.6f} {new_std:<15.6f}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Update config file to use corrected stats:")
print("   fno_normalization_file: '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_corrected.npz'")
print("\n2. Retrain diffusion model (or continue from checkpoint)")
print("\n3. Expected bias reduction:")
print("   - U-Wind: -3.78 m/s → <0.5 m/s")
print("   - V-Wind: -4.56 m/s → <0.5 m/s")
print("   - Temperature: +1.05 K → <0.3 K")
print("="*80)
