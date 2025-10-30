#!/usr/bin/env python
"""
Concatenate all u-wind samples from LUCIE 10yr retrained (BEST) model
Creates two files: uwind_fno.npy and uwind_diffusion.npy
"""
import os
import numpy as np
from tqdm import tqdm

print("="*80)
print("LUCIE 10yr - Concatenating U-Wind Samples (Best Retrained Model)")
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

uwind_fno = np.zeros((num_files, H, W), dtype=np.float32)
uwind_diffusion = np.zeros((num_files, H, W), dtype=np.float32)

# Extract u-wind (channel 1)
print("\nExtracting u-wind from all samples...")
for i, fname in enumerate(tqdm(files, desc="Processing")):
    path = os.path.join(samples_dir, fname)
    with np.load(path) as data:
        # Channel 1 is u-wind
        uwind_fno[i] = data["fno_output"][0, 1, ...]
        uwind_diffusion[i] = data["output"][0, 1, ...]

# Save the stacked arrays
print("\nSaving concatenated arrays...")
out_path_fno = os.path.join(samples_dir, "uwind_fno.npy")
out_path_diff = os.path.join(samples_dir, "uwind_diffusion.npy")

np.save(out_path_fno, uwind_fno)
np.save(out_path_diff, uwind_diffusion)

print(f"\n✓ Saved FNO u-wind array of shape {uwind_fno.shape} to:")
print(f"  {out_path_fno}")
print(f"\n✓ Saved Diffusion u-wind array of shape {uwind_diffusion.shape} to:")
print(f"  {out_path_diff}")

# Statistics
print(f"\nStatistics:")
print(f"  FNO - Mean: {uwind_fno.mean():.4f}, Std: {uwind_fno.std():.4f}")
print(f"  Diffusion - Mean: {uwind_diffusion.mean():.4f}, Std: {uwind_diffusion.std():.4f}")

print("\n" + "="*80)
print("U-Wind concatenation complete!")
print("="*80)
