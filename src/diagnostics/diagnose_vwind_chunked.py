import numpy as np
import matplotlib.pyplot as plt

print("Loading concatenated v-wind data...")
# Load the full time series for v-wind using mmap to avoid loading all into memory at once
vwind_fno = np.load("/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr/vwind_fno.npy", mmap_mode='r')
vwind_diffusion = np.load("/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr/vwind_diffusion.npy", mmap_mode='r')

print(f"FNO v-wind shape: {vwind_fno.shape}")
print(f"Diffusion v-wind shape: {vwind_diffusion.shape}")

# Load HR stats for denormalization
print("\nLoading normalization stats...")
hr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz", allow_pickle=True)
v_wind_mean_hr = hr_stats['v_component_of_wind_83'].item()['mean']
v_wind_std_hr = hr_stats['v_component_of_wind_83'].item()['std']

print(f"V-wind HR mean: {v_wind_mean_hr}")
print(f"V-wind HR std: {v_wind_std_hr}")

# Compute spatial mean at each time step in chunks
print("\nComputing spatial means over time (in chunks)...")
chunk_size = 1000
n_samples = vwind_fno.shape[0]

vwind_fno_spatial_mean = []
vwind_diffusion_spatial_mean = []

for i in range(0, n_samples, chunk_size):
    end_idx = min(i + chunk_size, n_samples)
    print(f"Processing chunk {i//chunk_size + 1}: samples {i} to {end_idx}")

    # Load chunk and denormalize
    chunk_fno = vwind_fno[i:end_idx] * v_wind_std_hr + v_wind_mean_hr
    chunk_diff = vwind_diffusion[i:end_idx] * v_wind_std_hr + v_wind_mean_hr

    # Compute spatial mean for this chunk
    vwind_fno_spatial_mean.extend(chunk_fno.mean(axis=(1, 2)).tolist())
    vwind_diffusion_spatial_mean.extend(chunk_diff.mean(axis=(1, 2)).tolist())

vwind_fno_spatial_mean = np.array(vwind_fno_spatial_mean)
vwind_diffusion_spatial_mean = np.array(vwind_diffusion_spatial_mean)

print(f"\nFNO v-wind spatial mean stats:")
print(f"  Mean: {vwind_fno_spatial_mean.mean():.6f}")
print(f"  Std: {vwind_fno_spatial_mean.std():.6f}")
print(f"  Min: {vwind_fno_spatial_mean.min():.6f}")
print(f"  Max: {vwind_fno_spatial_mean.max():.6f}")

print(f"\nDiffusion v-wind spatial mean stats:")
print(f"  Mean: {vwind_diffusion_spatial_mean.mean():.6f}")
print(f"  Std: {vwind_diffusion_spatial_mean.std():.6f}")
print(f"  Min: {vwind_diffusion_spatial_mean.min():.6f}")
print(f"  Max: {vwind_diffusion_spatial_mean.max():.6f}")

# Also check the normalized data (first 100 samples)
print("\n" + "="*60)
print("Checking NORMALIZED data (first 100 samples):")
print("="*60)
chunk_fno_norm = vwind_fno[:100]
chunk_diff_norm = vwind_diffusion[:100]

print(f"\nNormalized FNO v-wind stats (first 100 samples):")
print(f"  Mean: {chunk_fno_norm.mean():.6f}")
print(f"  Std: {chunk_fno_norm.std():.6f}")
print(f"  Spatial mean values: {chunk_fno_norm.mean(axis=(1,2))[:10]}")

print(f"\nNormalized Diffusion v-wind stats (first 100 samples):")
print(f"  Mean: {chunk_diff_norm.mean():.6f}")
print(f"  Std: {chunk_diff_norm.std():.6f}")
print(f"  Spatial mean values: {chunk_diff_norm.mean(axis=(1,2))[:10]}")

# Create plots
print("\nCreating plots...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Time series of spatial mean
axes[0].plot(vwind_fno_spatial_mean, label='FNO', alpha=0.7, linewidth=0.5)
axes[0].plot(vwind_diffusion_spatial_mean, label='Diffusion', alpha=0.7, linewidth=0.5)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Spatial Mean V-wind (m/s)')
axes[0].set_title('V-wind Spatial Mean Over Time (Denormalized)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Histogram of spatial means
axes[1].hist(vwind_fno_spatial_mean, bins=50, alpha=0.5, label='FNO', density=True)
axes[1].hist(vwind_diffusion_spatial_mean, bins=50, alpha=0.5, label='Diffusion', density=True)
axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[1].axvline(x=vwind_fno_spatial_mean.mean(), color='blue', linestyle='--', linewidth=1,
                alpha=0.7, label=f'FNO mean: {vwind_fno_spatial_mean.mean():.6f}')
axes[1].axvline(x=vwind_diffusion_spatial_mean.mean(), color='orange', linestyle='--', linewidth=1,
                alpha=0.7, label=f'Diff mean: {vwind_diffusion_spatial_mean.mean():.6f}')
axes[1].set_xlabel('Spatial Mean V-wind (m/s)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Spatial Mean V-wind')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Difference between FNO and Diffusion
diff = vwind_diffusion_spatial_mean - vwind_fno_spatial_mean
axes[2].plot(diff, label='Diffusion - FNO', alpha=0.7, linewidth=0.5, color='red')
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
axes[2].axhline(y=diff.mean(), color='blue', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean diff: {diff.mean():.6f}')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Difference (m/s)')
axes[2].set_title('Difference in Spatial Mean V-wind (Diffusion - FNO)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/glade/derecho/scratch/mdarman/lucie/vwind_diagnosis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: /glade/derecho/scratch/mdarman/lucie/vwind_diagnosis.png")

print("\nDone!")
