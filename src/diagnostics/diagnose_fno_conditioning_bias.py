#!/usr/bin/env python
"""
Check if FNO conditioning creates a bias issue for v-wind.

The hypothesis:
- FNO outputs are used as conditioning for the diffusion model
- If FNO v-wind has a different bias than other variables, the diffusion model
  might learn to over-correct or under-correct based on that conditioning
- This could create the systematic bias we observe
"""
import numpy as np
import torch
from tqdm import tqdm
import os

print("="*80)
print("FNO Conditioning Bias Analysis")
print("="*80)

# Load normalization stats
lr_stats = np.load('stats_lr_2000_2009_updated.npz', allow_pickle=True)
hr_stats = np.load('stats_hr_2000_2009_updated.npz', allow_pickle=True)

print("\n### Normalization Statistics ###\n")

print("HR (High-Resolution) Stats - What diffusion model should predict:")
print("-" * 60)
variables_hr = ['2m_temperature', 'u_component_of_wind_83', 'v_component_of_wind_83', 'total_precipitation_6hr']
for var in variables_hr:
    stats = hr_stats[var].item()
    print(f"{var:30s}: mean={stats['mean']:10.6f}, std={stats['std']:10.6f}")

print("\nLR (Low-Resolution) Stats - FNO input:")
print("-" * 60)
variables_lr = ['Temperature_7', 'U-wind_3', 'V-wind_3', 'tp6hr']
for var in variables_lr:
    stats = lr_stats[var].item()
    print(f"{var:30s}: mean={stats['mean']:10.6f}, std={stats['std']:10.6f}")

# ========== Load and analyze sample outputs ==========
print("\n### Analyzing Sample Outputs ###\n")

samples_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr_retrain_best"

if not os.path.exists(samples_dir):
    print(f"Sample directory not found: {samples_dir}")
    exit(1)

sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.npz')])
num_samples = min(1000, len(sample_files))

print(f"Loading {num_samples} samples...")

# Collect all FNO outputs and diffusion outputs
fno_outputs = []
diff_outputs = []

for fname in tqdm(sample_files[:num_samples], desc="Loading"):
    path = os.path.join(samples_dir, fname)
    data = np.load(path)

    fno_outputs.append(data['fno_output'][0])  # Shape: (4, 721, 1440)
    diff_outputs.append(data['output'][0])

fno_outputs = np.array(fno_outputs)  # Shape: (N, 4, 721, 1440)
diff_outputs = np.array(diff_outputs)

print(f"\nFNO outputs shape: {fno_outputs.shape}")
print(f"Diffusion outputs shape: {diff_outputs.shape}")

# Compute statistics for each channel
channel_names = ['Temperature', 'U-wind', 'V-wind', 'Precipitation']

print("\n" + "="*80)
print("NORMALIZED Statistics (from samples)")
print("="*80)

print("\nFNO Outputs (used as conditioning):")
print("-" * 60)
for i, name in enumerate(channel_names):
    mean = fno_outputs[:, i, :, :].mean()
    std = fno_outputs[:, i, :, :].std()
    print(f"{name:20s}: mean={mean:10.6f}, std={std:10.6f}")

print("\nDiffusion Outputs (what model predicts):")
print("-" * 60)
for i, name in enumerate(channel_names):
    mean = diff_outputs[:, i, :, :].mean()
    std = diff_outputs[:, i, :, :].std()
    print(f"{name:20s}: mean={mean:10.6f}, std={std:10.6f}")

print("\nBias (Diffusion - FNO) in normalized space:")
print("-" * 60)
for i, name in enumerate(channel_names):
    fno_mean = fno_outputs[:, i, :, :].mean()
    diff_mean = diff_outputs[:, i, :, :].mean()
    bias = diff_mean - fno_mean
    print(f"{name:20s}: bias={bias:10.6f} σ")

# ========== Denormalize and check physical space ==========
print("\n" + "="*80)
print("DENORMALIZED Statistics (physical units)")
print("="*80)

# Denormalize using HR stats
hr_means = [
    hr_stats['2m_temperature'].item()['mean'],
    hr_stats['u_component_of_wind_83'].item()['mean'],
    hr_stats['v_component_of_wind_83'].item()['mean'],
    hr_stats['total_precipitation_6hr'].item()['mean'],
]

hr_stds = [
    hr_stats['2m_temperature'].item()['std'],
    hr_stats['u_component_of_wind_83'].item()['std'],
    hr_stats['v_component_of_wind_83'].item()['std'],
    hr_stats['total_precipitation_6hr'].item()['std'],
]

units = ['K', 'm/s', 'm/s', 'm']

print("\nFNO Outputs (denormalized):")
print("-" * 60)
for i, name in enumerate(channel_names):
    denorm = fno_outputs[:, i, :, :] * hr_stds[i] + hr_means[i]
    mean = denorm.mean()
    std = denorm.std()
    print(f"{name:20s}: mean={mean:10.6f} {units[i]}, std={std:10.6f} {units[i]}")

print("\nDiffusion Outputs (denormalized):")
print("-" * 60)
for i, name in enumerate(channel_names):
    denorm = diff_outputs[:, i, :, :] * hr_stds[i] + hr_means[i]
    mean = denorm.mean()
    std = denorm.std()
    print(f"{name:20s}: mean={mean:10.6f} {units[i]}, std={std:10.6f} {units[i]}")

print("\nExpected HR means (from training data 2000-2009):")
print("-" * 60)
for i, name in enumerate(channel_names):
    print(f"{name:20s}: mean={hr_means[i]:10.6f} {units[i]}")

print("\nBias (Diffusion - Expected) in physical space:")
print("-" * 60)
for i, name in enumerate(channel_names):
    denorm = diff_outputs[:, i, :, :] * hr_stds[i] + hr_means[i]
    diff_mean = denorm.mean()
    bias = diff_mean - hr_means[i]
    print(f"{name:20s}: bias={bias:10.6f} {units[i]}")

# ========== Analyze the relationship ==========
print("\n" + "="*80)
print("RELATIONSHIP ANALYSIS")
print("="*80)

print("\nKey Question: Does the diffusion model learn to 'correct' FNO outputs?")
print("-" * 60)

# For each variable, check if diffusion is adding a correction to FNO
print("\nCorrection (Diffusion - FNO) in physical space:")
print("-" * 60)
for i, name in enumerate(channel_names):
    fno_denorm = fno_outputs[:, i, :, :] * hr_stds[i] + hr_means[i]
    diff_denorm = diff_outputs[:, i, :, :] * hr_stds[i] + hr_means[i]
    correction = (diff_denorm - fno_denorm).mean()
    print(f"{name:20s}: correction={correction:10.6f} {units[i]}")

# Check correlation between FNO and Diffusion outputs
print("\nCorrelation between FNO and Diffusion outputs:")
print("-" * 60)
for i, name in enumerate(channel_names):
    fno_flat = fno_outputs[:, i, :, :].flatten()
    diff_flat = diff_outputs[:, i, :, :].flatten()
    corr = np.corrcoef(fno_flat, diff_flat)[0, 1]
    print(f"{name:20s}: correlation={corr:.6f}")

# ========== Hypothesis Testing ==========
print("\n" + "="*80)
print("HYPOTHESIS: FNO Conditioning Bias Issue")
print("="*80)

print("""
If FNO outputs have biases in NORMALIZED space, and the diffusion model
is conditioned on these outputs, it might learn to:

1. Trust FNO for some variables (high correlation)
2. Correct FNO for other variables (low correlation, large correction)

Let's check if V-wind shows anomalous behavior:
""")

print("\nNORMALIZED FNO outputs relative to expected (should be ~0):")
print("-" * 60)
for i, name in enumerate(channel_names):
    fno_mean = fno_outputs[:, i, :, :].mean()
    print(f"{name:20s}: FNO mean = {fno_mean:10.6f} (expected ~0.0)")

print("\nFINDINGS:")
print("-" * 60)

# Check V-wind specifically
vwind_fno_mean = fno_outputs[:, 2, :, :].mean()
vwind_diff_mean = diff_outputs[:, 2, :, :].mean()
vwind_correction = vwind_diff_mean - vwind_fno_mean

print(f"""
V-wind Analysis:
  FNO normalized mean:        {vwind_fno_mean:10.6f} (should be ~0)
  Diffusion normalized mean:  {vwind_diff_mean:10.6f} (should be ~0)
  Correction in norm. space:  {vwind_correction:10.6f}

  Physical correction:        {vwind_correction * hr_stds[2]:10.6f} m/s

This shows the diffusion model is ADDING a correction of {vwind_correction:.3f}σ
to the FNO v-wind output, which translates to {vwind_correction * hr_stds[2]:.2f} m/s.

CRITICAL INSIGHT:
If FNO v-wind has a small bias (e.g., -0.009 normalized), and the diffusion
model learns to "fix" it by adding a large correction, this could create
the observed -6.2 m/s bias in the final output.

The question is: WHY does the diffusion model think it needs to apply this
large correction to v-wind but not to other variables?
""")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
