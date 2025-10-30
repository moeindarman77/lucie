import numpy as np
import matplotlib.pyplot as plt
import glob
import os

print("Loading v-wind data from individual sample files...")

# Get all sample files
sample_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr"
sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.npz")),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))

print(f"Found {len(sample_files)} sample files")

# Load HR stats for denormalization
print("\nLoading normalization stats...")
hr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz", allow_pickle=True)
v_wind_mean_hr = hr_stats['v_component_of_wind_83'].item()['mean']
v_wind_std_hr = hr_stats['v_component_of_wind_83'].item()['std']

print(f"V-wind HR mean: {v_wind_mean_hr}")
print(f"V-wind HR std: {v_wind_std_hr}")

# Process samples in batches to compute spatial means
print("\nProcessing samples to compute spatial means...")
vwind_fno_means = []
vwind_diffusion_means = []

# Sample every 100th file for faster processing (about 146 samples)
sample_step = 100
sample_files_subset = sample_files[::sample_step]
print(f"Processing {len(sample_files_subset)} files (every {sample_step}th sample)...")

for i, file_path in enumerate(sample_files_subset):
    if i % 20 == 0:
        print(f"Processing {i}/{len(sample_files_subset)}...")

    data = np.load(file_path)

    # Extract v-wind (index 3 in the channel dimension)
    vwind_fno = data["fno_output"][0, 3, ...]  # Shape: (721, 1440)
    vwind_diffusion = data["output"][0, 3, ...]  # Shape: (721, 1440)

    # Denormalize
    vwind_fno_denorm = vwind_fno * v_wind_std_hr + v_wind_mean_hr
    vwind_diffusion_denorm = vwind_diffusion * v_wind_std_hr + v_wind_mean_hr

    # Compute spatial mean
    vwind_fno_means.append(vwind_fno_denorm.mean())
    vwind_diffusion_means.append(vwind_diffusion_denorm.mean())

vwind_fno_means = np.array(vwind_fno_means)
vwind_diffusion_means = np.array(vwind_diffusion_means)

print(f"\nFNO v-wind spatial mean stats (from {len(sample_files_subset)} samples):")
print(f"  Mean: {vwind_fno_means.mean():.6f}")
print(f"  Std: {vwind_fno_means.std():.6f}")
print(f"  Min: {vwind_fno_means.min():.6f}")
print(f"  Max: {vwind_fno_means.max():.6f}")

print(f"\nDiffusion v-wind spatial mean stats (from {len(sample_files_subset)} samples):")
print(f"  Mean: {vwind_diffusion_means.mean():.6f}")
print(f"  Std: {vwind_diffusion_means.std():.6f}")
print(f"  Min: {vwind_diffusion_means.min():.6f}")
print(f"  Max: {vwind_diffusion_means.max():.6f}")

# Also check a few samples in normalized form
print("\n" + "="*60)
print("Checking NORMALIZED data (before denormalization) from first 10 samples:")
print("="*60)

fno_norm_vals = []
diff_norm_vals = []
for file_path in sample_files[:10]:
    data = np.load(file_path)
    vwind_fno = data["fno_output"][0, 3, ...]
    vwind_diffusion = data["output"][0, 3, ...]
    fno_norm_vals.append(vwind_fno.mean())
    diff_norm_vals.append(vwind_diffusion.mean())

print(f"\nNormalized FNO v-wind (spatial mean across 10 samples):")
print(f"  Mean: {np.mean(fno_norm_vals):.6f}")
print(f"  Values: {fno_norm_vals}")

print(f"\nNormalized Diffusion v-wind (spatial mean across 10 samples):")
print(f"  Mean: {np.mean(diff_norm_vals):.6f}")
print(f"  Values: {diff_norm_vals}")

# Create plots
print("\nCreating plots...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Time series of spatial mean
axes[0].plot(vwind_fno_means, label='FNO', alpha=0.7, linewidth=1, marker='o', markersize=2)
axes[0].plot(vwind_diffusion_means, label='Diffusion', alpha=0.7, linewidth=1, marker='s', markersize=2)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[0].set_xlabel(f'Sample Index (every {sample_step}th sample)')
axes[0].set_ylabel('Spatial Mean V-wind (m/s)')
axes[0].set_title('V-wind Spatial Mean Over Time (Denormalized)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Histogram of spatial means
axes[1].hist(vwind_fno_means, bins=30, alpha=0.5, label='FNO', density=True)
axes[1].hist(vwind_diffusion_means, bins=30, alpha=0.5, label='Diffusion', density=True)
axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[1].axvline(x=vwind_fno_means.mean(), color='blue', linestyle='--', linewidth=1,
                alpha=0.7, label=f'FNO mean: {vwind_fno_means.mean():.6f}')
axes[1].axvline(x=vwind_diffusion_means.mean(), color='orange', linestyle='--', linewidth=1,
                alpha=0.7, label=f'Diff mean: {vwind_diffusion_means.mean():.6f}')
axes[1].set_xlabel('Spatial Mean V-wind (m/s)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Spatial Mean V-wind')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Difference between FNO and Diffusion
diff = vwind_diffusion_means - vwind_fno_means
axes[2].plot(diff, label='Diffusion - FNO', alpha=0.7, linewidth=1, color='red', marker='o', markersize=2)
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[2].axhline(y=diff.mean(), color='blue', linestyle='--', linewidth=1, alpha=0.7,
                label=f'Mean diff: {diff.mean():.6f}')
axes[2].set_xlabel(f'Sample Index (every {sample_step}th sample)')
axes[2].set_ylabel('Difference (m/s)')
axes[2].set_title('Difference in Spatial Mean V-wind (Diffusion - FNO)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/glade/derecho/scratch/mdarman/lucie/vwind_diagnosis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: /glade/derecho/scratch/mdarman/lucie/vwind_diagnosis.png")

print("\nDone!")
