#!/usr/bin/env python
"""
Quick verification script for FNO output statistics file.
Run this after compute_fno_output_stats_scalar.py completes.
"""
import numpy as np
import os

stats_file = '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz'

print("=" * 80)
print("FNO Statistics File Verification")
print("=" * 80)

# Check if file exists
if not os.path.exists(stats_file):
    print(f"❌ ERROR: Stats file not found at {stats_file}")
    exit(1)

print(f"✓ Stats file exists: {stats_file}")
print(f"  File size: {os.path.getsize(stats_file) / 1024:.2f} KB")
print()

# Load and verify structure
try:
    stats = np.load(stats_file, allow_pickle=True)
    print("✓ Stats file loaded successfully")
    print(f"  Keys in file: {list(stats.keys())}")
    print()
except Exception as e:
    print(f"❌ ERROR loading stats file: {e}")
    exit(1)

# Verify each variable
expected_vars = ['temperature', 'uwind', 'vwind', 'precipitation']
print("FNO Output Statistics (SCALAR):")
print("-" * 80)

all_valid = True
for var in expected_vars:
    if var not in stats.keys():
        print(f"❌ ERROR: Missing variable '{var}'")
        all_valid = False
        continue

    try:
        data = stats[var].item()
        mean = data['mean']
        std = data['std']

        # Verify they are scalars
        if np.ndim(mean) != 0 or np.ndim(std) != 0:
            print(f"❌ ERROR: {var} has non-scalar values (mean ndim={np.ndim(mean)}, std ndim={np.ndim(std)})")
            all_valid = False
            continue

        # Verify std is positive
        if std <= 0:
            print(f"❌ ERROR: {var} has non-positive std: {std}")
            all_valid = False
            continue

        print(f"  {var:15s}: mean = {mean:12.6f}, std = {std:12.6f} ✓")

    except Exception as e:
        print(f"❌ ERROR processing {var}: {e}")
        all_valid = False

print("-" * 80)
print()

# Compare with HR stats
print("Comparison with HR Ground Truth Statistics:")
print("-" * 80)

hr_stats_file = '/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz'
if os.path.exists(hr_stats_file):
    hr_stats = np.load(hr_stats_file, allow_pickle=True)

    var_mapping = {
        'temperature': '2m_temperature',
        'uwind': 'u_component_of_wind_83',
        'vwind': 'v_component_of_wind_83',
        'precipitation': 'total_precipitation_6hr'
    }

    for fno_var, hr_var in var_mapping.items():
        fno_data = stats[fno_var].item()
        hr_data = hr_stats[hr_var].item()

        fno_mean = fno_data['mean']
        fno_std = fno_data['std']
        hr_mean = hr_data['mean']
        hr_std = hr_data['std']

        mean_diff = fno_mean - hr_mean
        std_ratio = fno_std / hr_std

        print(f"  {fno_var:15s}:")
        print(f"    FNO:  mean = {fno_mean:12.6f}, std = {fno_std:12.6f}")
        print(f"    HR:   mean = {hr_mean:12.6f}, std = {hr_std:12.6f}")
        print(f"    Diff: mean = {mean_diff:+12.6f}, std ratio = {std_ratio:6.3f}")
        print()
else:
    print("  HR stats file not found - skipping comparison")
    print()

print("-" * 80)

if all_valid:
    print("✓ All checks passed! Stats file is ready for training.")
    print()
    print("Next steps:")
    print("  1. Submit training job:")
    print("     qsub job_train_normalized_fno.slurm")
    print()
    print("  2. Monitor training progress:")
    print("     tail -f train_normalized_fno.o<JOB_ID>")
    exit(0)
else:
    print("❌ Validation failed! Please check errors above.")
    exit(1)
