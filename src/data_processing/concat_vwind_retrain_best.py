#!/usr/bin/env python
"""
Concatenate all v-wind samples from LUCIE 10yr retrained (BEST) model
Creates two files: vwind_fno.npy and vwind_diffusion.npy
"""
import os
import numpy as np
from tqdm import tqdm

print("="*80)
print("LUCIE 10yr - Concatenating V-Wind Samples (Best Retrained Model)")
print("="*80)

# Paths
results_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/"
samples_dir = os.path.join(results_dir, "samples_lucie_10yr_retrain_best")

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

vwind_fno = np.zeros((num_files, H, W), dtype=np.float32)
vwind_diffusion = np.zeros((num_files, H, W), dtype=np.float32)

# Extract v-wind (channel 2)
print("\nExtracting v-wind from all samples...")
for i, fname in enumerate(tqdm(files, desc="Processing")):
    path = os.path.join(samples_dir, fname)
    with np.load(path) as data:
        # Channel 2 is v-wind
        vwind_fno[i] = data["fno_output"][0, 2, ...]
        vwind_diffusion[i] = data["output"][0, 2, ...]

# Save the stacked arrays
print("\nSaving concatenated arrays...")
out_path_fno = os.path.join(samples_dir, "vwind_fno.npy")
out_path_diff = os.path.join(samples_dir, "vwind_diffusion.npy")

np.save(out_path_fno, vwind_fno)
np.save(out_path_diff, vwind_diffusion)

print(f"\n✓ Saved FNO v-wind array of shape {vwind_fno.shape} to:")
print(f"  {out_path_fno}")
print(f"\n✓ Saved Diffusion v-wind array of shape {vwind_diffusion.shape} to:")
print(f"  {out_path_diff}")

# Statistics
print(f"\nStatistics:")
print(f"  FNO - Mean: {vwind_fno.mean():.4f}, Std: {vwind_fno.std():.4f}")
print(f"  Diffusion - Mean: {vwind_diffusion.mean():.4f}, Std: {vwind_diffusion.std():.4f}")

print("\n" + "="*80)
print("V-Wind concatenation complete!")
print("="*80)
