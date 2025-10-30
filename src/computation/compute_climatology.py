#!/usr/bin/env python
"""
Compute climatology (mean across time) for LUCIE 10yr data
Creates two files with all 4 variables: climatology_fno.npz and climatology_diffusion.npz

Memory-optimized version: processes one file at a time and explicitly frees memory
"""
import os
import numpy as np
import gc  # Garbage collection

print("="*80)
print("LUCIE 10yr - Computing Climatology (Memory-Optimized)")
print("="*80)

samples_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr"

# Variables to process
variables = ['temperature', 'uwind', 'vwind', 'precipitation']
sources = ['fno', 'diffusion']

# Storage for climatologies
climatology_fno = {}
climatology_diffusion = {}

print(f"\nLoading and computing climatology (mean across time axis)...")
print(f"Processing one file at a time to minimize memory usage.\n")

for var in variables:
    print(f"Processing {var}...")

    for src in sources:
        filename = f"{var}_{src}.npy"
        filepath = os.path.join(samples_dir, filename)

        if not os.path.exists(filepath):
            print(f"  ✗ {filename} not found! Skipping...")
            continue

        print(f"  Loading {filename}...")
        data = np.load(filepath)  # Shape: (14600, 721, 1440)

        print(f"    Shape: {data.shape}")
        print(f"    Computing mean across axis 0...")

        # Compute mean across time (axis 0) and copy to ensure independent array
        climatology = data.mean(axis=0).copy()  # Shape: (721, 1440)

        print(f"    Climatology shape: {climatology.shape}")
        print(f"    Stats: mean={climatology.mean():.4f}, std={climatology.std():.4f}")

        # Store in appropriate dictionary
        if src == 'fno':
            climatology_fno[var] = climatology
        else:
            climatology_diffusion[var] = climatology

        # CRITICAL: Explicitly delete large data array and force garbage collection
        del data
        del climatology  # Delete local reference (we have it stored in dict)
        gc.collect()
        print(f"    ✓ Freed memory for {filename}")

    print()

# Save climatologies to .npz files
print("Saving climatology files...\n")

# Save FNO climatology
if len(climatology_fno) > 0:
    out_path_fno = os.path.join(samples_dir, "climatology_fno.npz")
    np.savez(out_path_fno, **climatology_fno)
    print(f"✓ Saved FNO climatology to:")
    print(f"  {out_path_fno}")
    print(f"  Variables: {list(climatology_fno.keys())}")
    print(f"  Each shape: {next(iter(climatology_fno.values())).shape}")

    # File size
    file_size_mb = os.path.getsize(out_path_fno) / 1e6
    print(f"  File size: {file_size_mb:.2f} MB")
else:
    print("✗ No FNO data to save!")

print()

# Save Diffusion climatology
if len(climatology_diffusion) > 0:
    out_path_diff = os.path.join(samples_dir, "climatology_diffusion.npz")
    np.savez(out_path_diff, **climatology_diffusion)
    print(f"✓ Saved Diffusion climatology to:")
    print(f"  {out_path_diff}")
    print(f"  Variables: {list(climatology_diffusion.keys())}")
    print(f"  Each shape: {next(iter(climatology_diffusion.values())).shape}")

    # File size
    file_size_mb = os.path.getsize(out_path_diff) / 1e6
    print(f"  File size: {file_size_mb:.2f} MB")
else:
    print("✗ No Diffusion data to save!")

# Summary comparison
if len(climatology_fno) > 0 and len(climatology_diffusion) > 0:
    print("\n" + "="*80)
    print("CLIMATOLOGY COMPARISON")
    print("="*80)
    print(f"\n{'Variable':<20} {'FNO Mean':<15} {'Diff Mean':<15} {'Difference':<15}")
    print("-"*65)

    for var in variables:
        if var in climatology_fno and var in climatology_diffusion:
            fno_mean = climatology_fno[var].mean()
            diff_mean = climatology_diffusion[var].mean()
            difference = diff_mean - fno_mean
            print(f"{var:<20} {fno_mean:<15.4f} {diff_mean:<15.4f} {difference:<15.4f}")

print("\n" + "="*80)
print("Climatology computation complete!")
print("="*80)

# Usage example
print("\nUsage example:")
print("  data = np.load('climatology_fno.npz')")
print("  temperature = data['temperature']  # Shape: (721, 1440)")
print("  uwind = data['uwind']")
print("  vwind = data['vwind']")
print("  precipitation = data['precipitation']")
