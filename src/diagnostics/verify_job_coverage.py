#!/usr/bin/env python
"""
Verify that job array configuration covers all samples correctly
"""

# Configuration
total_samples = 14600
num_jobs = 25  # Array indices 0-24
samples_per_job = 584

print("="*80)
print("LUCIE 10yr Job Array Coverage Verification")
print("="*80)

# Calculate coverage
print(f"\nConfiguration:")
print(f"  Total samples to process: {total_samples}")
print(f"  Number of array jobs: {num_jobs} (indices 0-{num_jobs-1})")
print(f"  Samples per job: {samples_per_job}")
print(f"  Total coverage: {num_jobs * samples_per_job}")

# Verify exact coverage
if num_jobs * samples_per_job == total_samples:
    print(f"\n✓ Perfect coverage! All {total_samples} samples will be processed.")
else:
    print(f"\n✗ Coverage mismatch!")
    print(f"  Expected: {total_samples}")
    print(f"  Actual: {num_jobs * samples_per_job}")
    print(f"  Difference: {num_jobs * samples_per_job - total_samples}")

# Show job distribution
print(f"\nJob Distribution:")
print(f"{'Job Index':<12} {'Start':<10} {'End':<10} {'Count':<10}")
print("-"*42)

all_indices = set()
for job_idx in range(num_jobs):
    start = job_idx * samples_per_job
    end = start + samples_per_job - 1
    print(f"{job_idx:<12} {start:<10} {end:<10} {samples_per_job:<10}")

    # Track all indices
    for idx in range(start, end + 1):
        if idx in all_indices:
            print(f"  ✗ ERROR: Index {idx} is duplicated!")
        all_indices.add(idx)

# Final verification
print(f"\nVerification:")
print(f"  Unique indices processed: {len(all_indices)}")
print(f"  Expected: {total_samples}")

if len(all_indices) == total_samples:
    print(f"  ✓ No duplicates!")
else:
    print(f"  ✗ Problem detected!")

# Check for gaps
expected_indices = set(range(total_samples))
missing = expected_indices - all_indices
extra = all_indices - expected_indices

if missing:
    print(f"\n✗ Missing indices: {sorted(missing)[:10]}... ({len(missing)} total)")
if extra:
    print(f"\n✗ Extra indices: {sorted(extra)[:10]}... ({len(extra)} total)")

if not missing and not extra:
    print(f"\n✓ All indices from 0 to {total_samples-1} are covered exactly once!")

# Estimate time
time_per_sample_min = 1
time_per_job_hours = (samples_per_job * time_per_sample_min) / 60
total_time_hours = time_per_job_hours  # All jobs run in parallel

print(f"\nTime Estimates:")
print(f"  Time per sample: ~{time_per_sample_min} minute")
print(f"  Time per job: ~{time_per_job_hours:.1f} hours")
print(f"  Total wallclock time (parallel): ~{total_time_hours:.1f} hours")
print(f"  Job time limit: 12 hours")

if time_per_job_hours < 12:
    print(f"  ✓ Within time limit! ({12 - time_per_job_hours:.1f} hours buffer)")
else:
    print(f"  ✗ Exceeds time limit by {time_per_job_hours - 12:.1f} hours!")

print("\n" + "="*80)
print("Verification Complete!")
print("="*80)
