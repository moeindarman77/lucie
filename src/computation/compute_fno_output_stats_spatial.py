#!/usr/bin/env python
"""
Compute spatially-resolved mean and std of FNO outputs on training data.
Each variable will have a 2D map (721, 1440) of mean and std values.
This allows proper normalization that preserves spatial patterns.
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
print("Computing Spatially-Resolved FNO Output Statistics")
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

# Compute statistics using accumulation (mean and std for 2D spatial maps)
print("\n### Computing FNO Output Statistics (All Samples) ###")
print("This will process ALL samples in the dataset...")
print(f"Processing {len(dataset)} samples...")

# Initialize accumulators
# Shape: (4 channels, 721 lat, 1440 lon)
sum_fno = np.zeros((4, 721, 1440), dtype=np.float64)
sum_sq_fno = np.zeros((4, 721, 1440), dtype=np.float64)
count = 0

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

        # Accumulate sum and sum of squares
        sum_fno += fno_output_np
        sum_sq_fno += fno_output_np ** 2
        count += 1

        # Periodic progress update
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")

print(f"\nTotal samples processed: {count}")

# Compute mean and std
mean_fno = sum_fno / count
variance_fno = (sum_sq_fno / count) - (mean_fno ** 2)
std_fno = np.sqrt(variance_fno)

# Print global statistics (averaged over space)
print("\n### Global Statistics (spatial average) ###")
print("-" * 60)
channel_names = ['temperature', 'u_wind', 'v_wind', 'precipitation']
display_names = ['Temperature', 'U-wind', 'V-wind', 'Precipitation']

for i, (name, display_name) in enumerate(zip(channel_names, display_names)):
    global_mean = mean_fno[i].mean()
    global_std = std_fno[i].mean()
    print(f"{display_name:20s}: mean = {global_mean:10.6f}, std = {global_std:10.6f}")

# Print spatial variability
print("\n### Spatial Variability ###")
print("-" * 60)
for i, display_name in enumerate(display_names):
    mean_min = mean_fno[i].min()
    mean_max = mean_fno[i].max()
    std_min = std_fno[i].min()
    std_max = std_fno[i].max()
    print(f"{display_name:20s}:")
    print(f"  Mean range: [{mean_min:10.6f}, {mean_max:10.6f}]")
    print(f"  Std range:  [{std_min:10.6f}, {std_max:10.6f}]")

# Save statistics as NPZ file with proper structure
output_file = '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_spatial.npz'

print(f"\n### Saving Statistics ###")
print(f"Output file: {output_file}")

# Save in the same format as the original stats files
# Each variable has a dictionary with 'mean' and 'std' keys
# Each containing a 2D array (721, 1440)
np.savez(output_file,
         temperature=np.array({'mean': mean_fno[0], 'std': std_fno[0]}, dtype=object),
         u_wind=np.array({'mean': mean_fno[1], 'std': std_fno[1]}, dtype=object),
         v_wind=np.array({'mean': mean_fno[2], 'std': std_fno[2]}, dtype=object),
         precipitation=np.array({'mean': mean_fno[3], 'std': std_fno[3]}, dtype=object)
)

print(f"✓ Statistics saved!")

# Verify saved file
print("\n### Verifying Saved File ###")
stats = np.load(output_file, allow_pickle=True)
print(f"Keys in file: {list(stats.keys())}")
for key in stats.keys():
    data = stats[key].item()
    print(f"\n{key}:")
    print(f"  Mean shape: {data['mean'].shape}")
    print(f"  Std shape:  {data['std'].shape}")
    print(f"  Mean (global): {data['mean'].mean():.6f}")
    print(f"  Std (global):  {data['std'].mean():.6f}")

print("\n### Usage Instructions ###")
print("-" * 80)
print("""
To use these statistics in training:

1. Load the stats file:
   fno_stats = np.load('fno_output_stats_spatial.npz', allow_pickle=True)

2. During training, after FNO forward pass:
   fno_output = SR_model(lres_upsampled)  # Shape: (batch, 4, 721, 1440)

   # Load stats as tensors
   channel_names = ['temperature', 'u_wind', 'v_wind', 'precipitation']
   for c, name in enumerate(channel_names):
       mean = torch.from_numpy(fno_stats[name].item()['mean']).float().to(device)
       std = torch.from_numpy(fno_stats[name].item()['std']).float().to(device)

       # Normalize each channel with its spatial mean/std map
       fno_output[:, c] = (fno_output[:, c] - mean) / std

3. During sampling, apply the same normalization.

This preserves spatial patterns while ensuring zero-mean, unit-std conditioning
at each spatial location.
""")

print("\n" + "="*80)
print("Done!")
print("="*80)
