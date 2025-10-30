#!/usr/bin/env python
"""
Compute GLOBAL (scalar) mean and std of FNO outputs on training data.
This matches the format of the original normalization files which use scalar statistics.
Each variable will have a single scalar mean and std value.
"""
import torch
import numpy as np
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, '/glade/derecho/scratch/mdarman/lucie/src')

from models.fno import FNO2d
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("Computing Global (Scalar) FNO Output Statistics")
print("="*80)

# Load FNO model
print("\n### Loading FNO Model ###")
fno_config = {
    'modes1': 180,
    'modes2': 360,
    'width': 20
}

SR_model = FNO2d(
    input_channels=8,  # 8 input channels (with orography and land_sea_mask)
    output_channels=4,  # 4 output channels
    model_config=fno_config
)
SR_model = SR_model.to(device)
SR_model.eval()

# Load FNO checkpoint
fno_checkpoint_path = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v1/checkpoints/best_fno.pth"
checkpoint = torch.load(fno_checkpoint_path, map_location=device)
SR_model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded FNO checkpoint from epoch {checkpoint['epoch']}")

# Create dataset
print("\n### Loading Dataset ###")

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

print(f"Dataset length: {len(dataset)}")

# Compute SCALAR statistics using accumulation (mean and std over ALL values)
print("\n### Computing FNO Output Statistics (All Samples) ###")
print("This will process ALL samples in the dataset...")
print(f"Processing {len(dataset)} samples...")

# Initialize accumulators for GLOBAL (scalar) statistics
# Match format of original normalization files
sum_fno = np.zeros(4, dtype=np.float64)
sum_sq_fno = np.zeros(4, dtype=np.float64)
count = 0
pixels_per_sample = 721 * 1440  # Total pixels per sample

with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        # Load sample
        sample = dataset[idx]
        lres = sample['input'].unsqueeze(0).float().to(device)

        # Upsample and pass through FNO
        lres_upsampled = F.interpolate(lres, size=(721, 1440), mode='bicubic', align_corners=True)
        fno_output = SR_model(lres_upsampled)  # Shape: (1, 4, 721, 1440)

        # Move to CPU and convert to numpy
        fno_output_np = fno_output[0].cpu().numpy()  # Shape: (4, 721, 1440)

        # Accumulate sum and sum of squares PER CHANNEL (flattening spatial dimensions)
        for c in range(4):
            channel_data = fno_output_np[c].flatten()  # Flatten spatial dimensions
            sum_fno[c] += channel_data.sum()
            sum_sq_fno[c] += (channel_data ** 2).sum()

        count += pixels_per_sample

        # Periodic progress update
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")

print(f"\nTotal samples processed: {count // pixels_per_sample}")
print(f"Total values per channel: {count}")

# Compute mean and std (scalars)
mean_fno = sum_fno / count
variance_fno = (sum_sq_fno / count) - (mean_fno ** 2)
std_fno = np.sqrt(variance_fno)

# Print statistics
print("\n### FNO Output Statistics (SCALAR) ###")
print("-" * 60)
channel_names = ['temperature', 'uwind', 'vwind', 'precipitation']
display_names = ['Temperature', 'U-wind', 'V-wind', 'Precipitation']

for i, (name, display_name) in enumerate(zip(channel_names, display_names)):
    print(f"{display_name:20s}: mean = {mean_fno[i]:12.6f}, std = {std_fno[i]:12.6f}")

# Save statistics as NPZ file matching original format
output_file = '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz'

print(f"\n### Saving Statistics ###")
print(f"Output file: {output_file}")

# Save in the SAME format as the original stats files:
# Each variable has a dictionary with SCALAR 'mean' and 'std' keys
# This matches: data[var].item()['mean'] returns a scalar
np.savez(output_file,
         temperature=np.array({'mean': np.float32(mean_fno[0]), 'std': np.float32(std_fno[0])}, dtype=object),
         uwind=np.array({'mean': np.float32(mean_fno[1]), 'std': np.float32(std_fno[1])}, dtype=object),
         vwind=np.array({'mean': np.float32(mean_fno[2]), 'std': np.float32(std_fno[2])}, dtype=object),
         precipitation=np.array({'mean': np.float32(mean_fno[3]), 'std': np.float32(std_fno[3])}, dtype=object)
)

print(f"✓ Statistics saved!")

# Verify saved file
print("\n### Verifying Saved File ###")
stats = np.load(output_file, allow_pickle=True)
print(f"Keys in file: {list(stats.keys())}")
for key in stats.keys():
    data = stats[key].item()
    print(f"\n{key}:")
    print(f"  Type of mean: {type(data['mean'])}")
    print(f"  Type of std:  {type(data['std'])}")
    print(f"  Mean value: {data['mean']}")
    print(f"  Std value:  {data['std']}")

print("\n### Comparison with Original HR Stats ###")
print("-" * 60)
hr_stats = np.load('/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz', allow_pickle=True)
var_mapping = {
    'temperature': '2m_temperature',
    'uwind': 'u_component_of_wind_83',
    'vwind': 'v_component_of_wind_83',
    'precipitation': 'total_precipitation_6hr'
}

for fno_var, hr_var in var_mapping.items():
    fno_data = stats[fno_var].item()
    hr_data = hr_stats[hr_var].item()
    print(f"\n{fno_var} (FNO) vs {hr_var} (HR):")
    print(f"  FNO mean: {fno_data['mean']:12.6f} | HR mean: {hr_data['mean']:12.6f}")
    print(f"  FNO std:  {fno_data['std']:12.6f} | HR std:  {hr_data['std']:12.6f}")

print("\n### Usage Instructions ###")
print("-" * 80)
print("""
To use these statistics in training (SCALAR normalization):

1. Load the stats file:
   fno_stats = np.load('fno_output_stats_scalar.npz', allow_pickle=True)

2. Extract scalar mean and std for each channel:
   fno_mean_temp = fno_stats['temperature'].item()['mean']
   fno_std_temp = fno_stats['temperature'].item()['std']
   # Repeat for uwind, vwind, precipitation

3. During training, after FNO forward pass:
   fno_output = SR_model(lres_upsampled)  # Shape: (batch, 4, 721, 1440)

   # Normalize each channel with scalar mean/std (SAME as original normalization)
   fno_output[:, 0] = (fno_output[:, 0] - fno_mean_temp) / fno_std_temp
   fno_output[:, 1] = (fno_output[:, 1] - fno_mean_uwind) / fno_std_uwind
   fno_output[:, 2] = (fno_output[:, 2] - fno_mean_vwind) / fno_std_vwind
   fno_output[:, 3] = (fno_output[:, 3] - fno_mean_precip) / fno_std_precip

4. During sampling, apply the same normalization.

This ensures FNO outputs have zero mean and unit std (matching the format
of ground truth data normalization) before being used as diffusion conditioning.
""")

print("\n" + "="*80)
print("Done!")
print("="*80)
