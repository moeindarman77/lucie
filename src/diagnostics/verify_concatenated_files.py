#!/usr/bin/env python
"""
Verify the concatenated LUCIE 10yr files
"""
import os
import numpy as np

print("="*80)
print("Verifying Concatenated LUCIE 10yr Files")
print("="*80)

samples_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr"

# Expected configuration
expected_samples = 14600
expected_shape = (14600, 721, 1440)
expected_size_gb = 14600 * 721 * 1440 * 4 / 1e9

# Variables to check
variables = ['temperature', 'uwind', 'vwind', 'precipitation']
sources = ['fno', 'diffusion']

print(f"\nExpected shape: {expected_shape}")
print(f"Expected size per file: ~{expected_size_gb:.2f} GB")
print(f"\nChecking directory: {samples_dir}\n")

# Check each variable and source
results = {}
for var in variables:
    results[var] = {}
    for src in sources:
        filename = f"{var}_{src}.npy"
        filepath = os.path.join(samples_dir, filename)

        if os.path.exists(filepath):
            try:
                # Get file size without loading
                file_size_gb = os.path.getsize(filepath) / 1e9

                # Load and check
                print(f"Loading {filename}...")
                data = np.load(filepath)

                # Check shape
                shape_ok = data.shape == expected_shape
                shape_status = "✓" if shape_ok else "✗"

                # Check for NaN
                has_nan = np.isnan(data).any()
                nan_status = "✗" if has_nan else "✓"

                # Statistics
                mean = data.mean()
                std = data.std()
                min_val = data.min()
                max_val = data.max()

                results[var][src] = {
                    'exists': True,
                    'shape': data.shape,
                    'shape_ok': shape_ok,
                    'has_nan': has_nan,
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'size_gb': file_size_gb
                }

                print(f"  {shape_status} Shape: {data.shape}")
                print(f"  {nan_status} No NaN values" if not has_nan else f"  {nan_status} Contains NaN!")
                print(f"  Size: {file_size_gb:.2f} GB")
                print(f"  Stats: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
                print()

            except Exception as e:
                print(f"  ✗ Error loading file: {e}\n")
                results[var][src] = {'exists': True, 'error': str(e)}
        else:
            print(f"✗ {filename} - NOT FOUND\n")
            results[var][src] = {'exists': False}

# Summary table
print("="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Variable':<15} {'Source':<12} {'Exists':<8} {'Shape OK':<10} {'No NaN':<8} {'Size (GB)':<10}")
print("-"*80)

all_good = True
for var in variables:
    for src in sources:
        r = results[var][src]
        exists = "✓" if r.get('exists', False) else "✗"

        if r.get('exists', False) and 'error' not in r:
            shape_ok = "✓" if r.get('shape_ok', False) else "✗"
            no_nan = "✓" if not r.get('has_nan', True) else "✗"
            size = f"{r.get('size_gb', 0):.2f}"

            if not r.get('shape_ok', False) or r.get('has_nan', True):
                all_good = False
        else:
            shape_ok = "N/A"
            no_nan = "N/A"
            size = "N/A"
            all_good = False

        print(f"{var:<15} {src:<12} {exists:<8} {shape_ok:<10} {no_nan:<8} {size:<10}")

print("\n" + "="*80)
if all_good:
    print("✓ ALL FILES VERIFIED SUCCESSFULLY!")
    print("  - All files exist")
    print("  - All shapes are correct")
    print("  - No NaN values detected")
else:
    print("✗ SOME FILES HAVE ISSUES!")
    print("  Check the details above")
print("="*80)
