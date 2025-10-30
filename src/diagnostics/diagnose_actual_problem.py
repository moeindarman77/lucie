#!/usr/bin/env python
"""
Diagnose the actual normalization problem by checking:
1. What does FNO actually output? (normalized or raw?)
2. What does the dataset provide? (normalized or raw?)
3. Is there a mismatch in the training pipeline?
"""

import torch
import numpy as np
import yaml
import torch.nn.functional as F
from models.fno import FNO2d
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("COMPREHENSIVE NORMALIZATION DIAGNOSIS")
print("="*80)

# Load config
config_path = '/glade/derecho/scratch/mdarman/lucie/src/config/ERA5_config_normalized_fno.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

dataset_config = config['dataset_params']
fno_config = config['fno_params']

# Load dataset (which normalizes data)
input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3',
              'tp6hr', 'orography', 'land_sea_mask', 'logp']
output_vars = ["2m_temperature", "specific_humidity_133",
               "u_component_of_wind_83", "v_component_of_wind_83",
               "total_precipitation_6hr", "geopotential_at_surface"]

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

# Load FNO model
SR_model = FNO2d(
    input_channels=dataset_config['input_channels'] + dataset_config['land_sea_mask'],
    output_channels=dataset_config['output_channels'],
    model_config=fno_config
)
SR_model = SR_model.to(device)
SR_model.eval()

fno_checkpoint_path = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v1/checkpoints/best_fno.pth"
checkpoint = torch.load(fno_checkpoint_path, map_location=device)
SR_model.load_state_dict(checkpoint['model_state_dict'])

# Load normalization stats
hr_stats = np.load(dataset_config['output_normalization_dir'], allow_pickle=True)
fno_stats = np.load(dataset_config['fno_normalization_file'], allow_pickle=True)

print("\n1. DATASET OUTPUT (what goes into training)")
print("-"*80)
# Get one sample from dataset
data = dataset[100]
lres = data['input']  # (8, 48, 96)
hres = data['output']  # (6, 721, 1440)

print(f"Dataset 'input' (low-res):  shape={lres.shape}")
print(f"  Temperature channel: mean={lres[0].mean():.6f}, std={lres[0].std():.6f}")
print(f"  → This is NORMALIZED (mean~0, std~1)")

print(f"\nDataset 'output' (high-res): shape={hres.shape}")
print(f"  Temperature channel: mean={hres[0].mean():.6f}, std={hres[0].std():.6f}")
print(f"  U-wind channel:      mean={hres[2].mean():.6f}, std={hres[2].std():.6f}")
print(f"  V-wind channel:      mean={hres[3].mean():.6f}, std={hres[3].std():.6f}")
print(f"  → These are NORMALIZED (mean~0, std~1)")

print("\n2. FNO OUTPUT (what FNO produces)")
print("-"*80)
with torch.no_grad():
    lres_batch = lres.unsqueeze(0).float().to(device)
    lres_upsampled = F.interpolate(lres_batch, size=(721, 1440), mode='bicubic', align_corners=True)
    fno_output = SR_model(lres_upsampled)  # (1, 4, 721, 1440)

fno_np = fno_output.cpu().numpy()[0]
print(f"FNO raw output: shape={fno_np.shape}")
print(f"  Temperature channel: mean={fno_np[0].mean():.6f}, std={fno_np[0].std():.6f}")
print(f"  U-wind channel:      mean={fno_np[1].mean():.6f}, std={fno_np[1].std():.6f}")
print(f"  V-wind channel:      mean={fno_np[2].mean():.6f}, std={fno_np[2].std():.6f}")
print(f"  Precipitation:       mean={fno_np[3].mean():.6f}, std={fno_np[3].std():.6f}")

if abs(fno_np[0].mean()) < 1.0 and abs(fno_np[0].std() - 1.0) < 0.5:
    print(f"  → FNO outputs are ALREADY NORMALIZED!")
else:
    print(f"  → FNO outputs are in RAW scale")

print("\n3. FNO NORMALIZATION STATS (what you computed)")
print("-"*80)
print(f"Temperature: mean={fno_stats['temperature'].item()['mean']:.6f}, std={fno_stats['temperature'].item()['std']:.6f}")
print(f"U-wind:      mean={fno_stats['uwind'].item()['mean']:.6f}, std={fno_stats['uwind'].item()['std']:.6f}")
print(f"V-wind:      mean={fno_stats['vwind'].item()['mean']:.6f}, std={fno_stats['vwind'].item()['std']:.6f}")
print(f"Precip:      mean={fno_stats['precipitation'].item()['mean']:.6f}, std={fno_stats['precipitation'].item()['std']:.6f}")

print("\n4. WHAT HAPPENS WHEN YOU NORMALIZE FNO OUTPUT")
print("-"*80)
fno_mean = torch.tensor([
    fno_stats['temperature'].item()['mean'],
    fno_stats['uwind'].item()['mean'],
    fno_stats['vwind'].item()['mean'],
    fno_stats['precipitation'].item()['mean']
]).view(1, 4, 1, 1)

fno_std = torch.tensor([
    fno_stats['temperature'].item()['std'],
    fno_stats['uwind'].item()['std'],
    fno_stats['vwind'].item()['std'],
    fno_stats['precipitation'].item()['std']
]).view(1, 4, 1, 1)

fno_normalized = (fno_output - fno_mean.to(device)) / (fno_std.to(device) + 1e-6)
fno_norm_np = fno_normalized.cpu().numpy()[0]

print(f"After normalization with FNO stats:")
print(f"  Temperature: mean={fno_norm_np[0].mean():.6f}, std={fno_norm_np[0].std():.6f}")
print(f"  U-wind:      mean={fno_norm_np[1].mean():.6f}, std={fno_norm_np[1].std():.6f}")
print(f"  V-wind:      mean={fno_norm_np[2].mean():.6f}, std={fno_norm_np[2].std():.6f}")
print(f"  Precip:      mean={fno_norm_np[3].mean():.6f}, std={fno_norm_np[3].std():.6f}")

print("\n5. GROUND TRUTH (from dataset)")
print("-"*80)
hres_target = hres[[0, 2, 3, 4]]  # temp, u, v, precip
print(f"Ground truth (HR normalized):")
print(f"  Temperature: mean={hres_target[0].mean():.6f}, std={hres_target[0].std():.6f}")
print(f"  U-wind:      mean={hres_target[1].mean():.6f}, std={hres_target[1].std():.6f}")
print(f"  V-wind:      mean={hres_target[2].mean():.6f}, std={hres_target[2].std():.6f}")
print(f"  Precip:      mean={hres_target[3].mean():.6f}, std={hres_target[3].std():.6f}")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("\nSCENARIO A: FNO was trained to output NORMALIZED data")
print("-"*80)
print("  - FNO inputs: normalized (mean~0, std~1)")
print("  - FNO targets: normalized HR data (mean~0, std~1)")
print("  - FNO outputs: normalized (mean~0, std~1) ✓")
print("  → FNO normalization stats computed on normalized outputs (correct!)")
print("  → Normalizing again with mean~0, std~1 does almost nothing")
print("  → Creates slight scale mismatch (dividing by 0.94 instead of 1.0)")

print("\nSCENARIO B: FNO has systematic bias in its outputs")
print("-"*80)
print("  - FNO outputs should match HR distribution (mean~0 for normalized)")
print("  - But FNO has biases: U-wind mean≠0, V-wind mean≠0")
print("  - Your normalization stats capture these biases")
print("  - Normalizing removes FNO's bias BUT creates scale mismatch")

print("\n" + "="*80)
print("THE REAL QUESTION:")
print("="*80)
print("Does FNO have systematic bias in its outputs?")
print(f"  - FNO U-wind mean: {fno_np[1].mean():.6f}")
print(f"  - Expected (from HR): ~0.0 (normalized)")
print(f"  - Difference: {fno_np[1].mean():.6f}")

if abs(fno_np[1].mean()) > 0.1 or abs(fno_np[2].mean()) > 0.1:
    print("\n✓ YES - FNO has systematic bias!")
    print("  Your normalization approach is CORRECT in concept")
    print("  BUT: The stats might be computed incorrectly (wrong sample?)")
    print("  OR: The stats are averaged over time/space differently than training")
else:
    print("\n✗ NO - FNO outputs are well-centered")
    print("  Normalization with mean~0, std~1 doesn't help")
    print("  The scale mismatch (std=0.94 vs 1.0) causes problems")

print("\n" + "="*80)
print("RECOMMENDED NEXT STEP:")
print("="*80)
print("Run this diagnostic on MULTIPLE samples (not just one)")
print("to see if FNO bias is consistent")
print("="*80)
