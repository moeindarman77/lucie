#!/usr/bin/env python
"""
Check progress of LUCIE 10yr sampling jobs
"""
import os
import numpy as np
from pathlib import Path

# Configuration
results_dir = '/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr'
total_samples = 14600
num_jobs = 25
samples_per_job = 584

print("="*80)
print("LUCIE 10yr Sampling Progress Check")
print("="*80)

# Check if directory exists
if not os.path.exists(results_dir):
    print(f"\n✗ Results directory does not exist yet: {results_dir}")
    print("  Jobs may not have started yet.")
    exit(0)

# Count generated samples
sample_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.npz')])
num_completed = len(sample_files)

print(f"\nProgress:")
print(f"  Total samples expected: {total_samples}")
print(f"  Samples completed: {num_completed}")
print(f"  Progress: {num_completed}/{total_samples} ({100*num_completed/total_samples:.1f}%)")

# Progress bar
bar_width = 50
filled = int(bar_width * num_completed / total_samples)
bar = '█' * filled + '░' * (bar_width - filled)
print(f"  [{bar}] {100*num_completed/total_samples:.1f}%")

# Estimate remaining time
if num_completed > 0:
    # Jobs run in parallel, so time = max time of any job
    # Check which jobs are complete
    job_completion = [0] * num_jobs
    for f in sample_files:
        idx = int(f.replace('.npz', '')) - 1  # Files are 1-indexed
        job_idx = idx // samples_per_job
        if job_idx < num_jobs:
            job_completion[job_idx] += 1

    print(f"\nJob Completion:")
    print(f"{'Job':<8} {'Completed':<12} {'Expected':<12} {'Status':<10}")
    print("-"*42)

    completed_jobs = 0
    for job_idx in range(num_jobs):
        completed = job_completion[job_idx]
        expected = samples_per_job
        status = '✓ Done' if completed == expected else f'{completed}/{expected}'
        print(f"{job_idx:<8} {completed:<12} {expected:<12} {status:<10}")
        if completed == expected:
            completed_jobs += 1

    print(f"\nJobs Summary:")
    print(f"  Completed jobs: {completed_jobs}/{num_jobs} ({100*completed_jobs/num_jobs:.1f}%)")

    # Time estimate
    if completed_jobs > 0:
        print(f"\n  ✓ At least {completed_jobs} job(s) have completed!")

    if completed_jobs == num_jobs:
        print(f"\n{'='*80}")
        print("✓ ALL JOBS COMPLETED!")
        print(f"{'='*80}")

# Check for missing samples
if num_completed > 0 and num_completed < total_samples:
    existing_indices = set([int(f.replace('.npz', '')) for f in sample_files])
    expected_indices = set(range(1, total_samples + 1))  # 1-indexed
    missing_indices = sorted(expected_indices - existing_indices)

    if len(missing_indices) > 0 and len(missing_indices) < 50:
        print(f"\nMissing samples: {missing_indices[:20]}")
        if len(missing_indices) > 20:
            print(f"  ... and {len(missing_indices) - 20} more")

# Check for errors in output files
if num_completed > 0:
    print(f"\nSample Verification (checking first 5 samples):")
    for i, f in enumerate(sample_files[:5]):
        sample_path = os.path.join(results_dir, f)
        try:
            data = np.load(sample_path)
            has_output = 'output' in data.keys()
            has_fno = 'fno_output' in data.keys()
            has_nan_output = np.isnan(data['output']).any() if has_output else None
            has_nan_fno = np.isnan(data['fno_output']).any() if has_fno else None

            status = '✓' if (has_output and has_fno and not has_nan_output and not has_nan_fno) else '✗'
            print(f"  {status} {f}: output={has_output}, fno={has_fno}, NaN_output={has_nan_output}, NaN_fno={has_nan_fno}")
            data.close()
        except Exception as e:
            print(f"  ✗ {f}: Error - {str(e)}")

print("\n" + "="*80)
