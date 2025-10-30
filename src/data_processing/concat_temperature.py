#!/usr/bin/env python
"""
Concatenate all temperature samples from LUCIE 10yr sampling
Creates two files: temperature_fno.npy and temperature_diffusion.npy
"""
import os
import numpy as np
from tqdm import tqdm

print("="*80)
print("LUCIE 10yr - Concatenating Temperature Samples")
print("="*80)

# Paths
results_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/"
samples_dir = os.path.join(results_dir, "samples_lucie_10yr")

# Gather and numerically sort all .npz files
print("\nScanning directory for samples...")
files = sorted(
    [f for f in os.listdir(samples_dir) if f.endswith(".npz")],
    key=lambda fn: int(os.path.splitext(fn)[0])
)
print(f"Found {len(files)} sample files")

# Pre-allocate output arrays: (num_samples, H, W)
num_files = len(files)
H, W = 721, 1440
print(f"\nAllocating arrays:")
print(f"  Shape: ({num_files}, {H}, {W})")
print(f"  Size per array: ~{num_files * H * W * 4 / 1e9:.2f} GB")

temp_fno = np.zeros((num_files, H, W), dtype=np.float32)
temp_diffusion = np.zeros((num_files, H, W), dtype=np.float32)

# Extract temperature (channel 0)
print("\nExtracting temperature from all samples...")
for i, fname in enumerate(tqdm(files, desc="Processing")):
    path = os.path.join(samples_dir, fname)
    with np.load(path) as data:
        # Channel 0 is temperature
        temp_fno[i] = data["fno_output"][0, 0, ...]
        temp_diffusion[i] = data["output"][0, 0, ...]

# Save the stacked arrays
print("\nSaving concatenated arrays...")
out_path_fno = os.path.join(samples_dir, "temperature_fno.npy")
out_path_diff = os.path.join(samples_dir, "temperature_diffusion.npy")

np.save(out_path_fno, temp_fno)
np.save(out_path_diff, temp_diffusion)

print(f"\n✓ Saved FNO temperature array of shape {temp_fno.shape} to:")
print(f"  {out_path_fno}")
print(f"\n✓ Saved Diffusion temperature array of shape {temp_diffusion.shape} to:")
print(f"  {out_path_diff}")

# Statistics
print(f"\nStatistics:")
print(f"  FNO - Mean: {temp_fno.mean():.4f}, Std: {temp_fno.std():.4f}")
print(f"  Diffusion - Mean: {temp_diffusion.mean():.4f}, Std: {temp_diffusion.std():.4f}")

print("\n" + "="*80)
print("Temperature concatenation complete!")
print("="*80)
