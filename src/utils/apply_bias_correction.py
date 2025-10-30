#!/usr/bin/env python
"""
Apply post-processing bias correction to existing samples (TEMPORARY FIX).

This corrects systematic biases found in the comparison analysis:
- U-Wind: -3.78 m/s
- V-Wind: -4.56 m/s
- Temperature: +1.05 K

This is a WORKAROUND while we retrain with correct normalization.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
vis_dir = Path('/glade/derecho/scratch/mdarman/lucie/results/unet_normalized_fno/samples_normalized_fno/visualizations')
output_dir = vis_dir / 'bias_corrected'
output_dir.mkdir(exist_ok=True)

# Bias corrections (from comparison analysis)
BIAS_CORRECTIONS = {
    'temperature': -1.054909,  # K (reduce warm bias)
    'uwind': 3.784235,         # m/s (increase eastward wind)
    'vwind': 4.555025,         # m/s (increase northward wind)
    'precipitation': -0.000057 # m (negligible)
}

print("="*80)
print("Applying Systematic Bias Corrections (TEMPORARY FIX)")
print("="*80)
print("\nBias corrections to apply:")
for var, correction in BIAS_CORRECTIONS.items():
    print(f"  {var:<15}: {correction:+.6f}")

# Variable indices in output array
VAR_INDICES = {
    'temperature': 0,
    'uwind': 1,
    'vwind': 2,
    'precipitation': 3
}

# Process all denormalized samples
sample_files = sorted(vis_dir.glob('sample_*_denormalized.npz'))
print(f"\nFound {len(sample_files)} samples to correct")

# Track statistics
stats_before = {var: {'mean': [], 'std': []} for var in VAR_INDICES.keys()}
stats_after = {var: {'mean': [], 'std': []} for var in VAR_INDICES.keys()}

for sample_file in sample_files:
    sample_num = sample_file.stem.split('_')[1]
    print(f"\nProcessing Sample {sample_num}...")

    # Load data
    data = np.load(sample_file)
    diffusion_output = data['diffusion_output'].copy()  # (4, 721, 1440)
    fno_output = data['fno_output']  # Keep FNO unchanged

    # Collect before stats
    for var_name, idx in VAR_INDICES.items():
        stats_before[var_name]['mean'].append(diffusion_output[idx].mean())
        stats_before[var_name]['std'].append(diffusion_output[idx].std())

    # Apply corrections
    diffusion_corrected = diffusion_output.copy()
    for var_name, correction in BIAS_CORRECTIONS.items():
        idx = VAR_INDICES[var_name]
        diffusion_corrected[idx] += correction

    # Collect after stats
    for var_name, idx in VAR_INDICES.items():
        stats_after[var_name]['mean'].append(diffusion_corrected[idx].mean())
        stats_after[var_name]['std'].append(diffusion_corrected[idx].std())

    # Save corrected sample
    output_file = output_dir / f'sample_{sample_num}_bias_corrected.npz'
    np.savez(output_file,
             diffusion_output=diffusion_corrected,
             diffusion_output_original=diffusion_output,
             fno_output=fno_output,
             bias_corrections=BIAS_CORRECTIONS,
             variable_names=['temperature', 'uwind', 'vwind', 'precipitation'])

    print(f"  ✓ Saved to {output_file.name}")

# Print summary statistics
print("\n" + "="*80)
print("BIAS CORRECTION SUMMARY")
print("="*80)

for var_name in VAR_INDICES.keys():
    mean_before = np.mean(stats_before[var_name]['mean'])
    mean_after = np.mean(stats_after[var_name]['mean'])
    diff = mean_after - mean_before

    print(f"\n{var_name.upper()}:")
    print(f"  Mean before: {mean_before:.6f}")
    print(f"  Mean after:  {mean_after:.6f}")
    print(f"  Change:      {diff:+.6f} (expected: {BIAS_CORRECTIONS[var_name]:+.6f})")
    print(f"  Match:       {'✓ YES' if abs(diff - BIAS_CORRECTIONS[var_name]) < 0.001 else '✗ NO'}")

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Effect of Bias Correction on Sample Means', fontsize=14, fontweight='bold')

for idx, (var_name, ax) in enumerate(zip(VAR_INDICES.keys(), axes.flatten())):
    x = np.arange(len(sample_files))
    before = stats_before[var_name]['mean']
    after = stats_after[var_name]['mean']

    width = 0.35
    ax.bar(x - width/2, before, width, label='Before correction', alpha=0.8, color='coral')
    ax.bar(x + width/2, after, width, label='After correction', alpha=0.8, color='steelblue')

    ax.axhline(y=np.mean(before), color='coral', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=np.mean(after), color='steelblue', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Mean value')
    ax.set_title(f'{var_name.capitalize()}\n(Correction: {BIAS_CORRECTIONS[var_name]:+.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
viz_file = output_dir / 'bias_correction_effect.png'
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {viz_file.name}")
plt.close()

print("\n" + "="*80)
print("FILES CREATED:")
print("="*80)
print(f"Location: {output_dir}")
print(f"  - sample_X_bias_corrected.npz (corrected data)")
print(f"  - bias_correction_effect.png (visualization)")

print("\n" + "="*80)
print("IMPORTANT NOTES:")
print("="*80)
print("• This is a TEMPORARY FIX - it only corrects systematic bias")
print("• Does NOT fix the underlying normalization issue")
print("• Use these corrected samples for urgent analysis")
print("• For proper fix, recompute FNO stats and retrain (see RECOMMENDATIONS_TO_FIX_BIAS.md)")
print("="*80)
