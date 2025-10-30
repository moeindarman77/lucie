#!/usr/bin/env python
"""
Denormalize, visualize, and save normalized FNO samples
Creates plots for each variable (temperature, u-wind, v-wind, precipitation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Configuration
samples_dir = Path('/glade/derecho/scratch/mdarman/lucie/results/unet_normalized_fno/samples_normalized_fno')
output_dir = samples_dir / 'visualizations'
output_dir.mkdir(exist_ok=True)

# Load normalization statistics for denormalization
stats_hr = np.load('/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz', allow_pickle=True)

# Variable names and their indices in the output
# Output order: [temperature, uwind, vwind, precipitation]
variables = {
    0: {
        'name': 'temperature',
        'key': '2m_temperature',
        'long_name': '2m Temperature',
        'units': 'K',
        'cmap': 'RdBu_r',
    },
    1: {
        'name': 'uwind',
        'key': 'u_component_of_wind_83',
        'long_name': 'U-Wind Component',
        'units': 'm/s',
        'cmap': 'RdBu_r',
    },
    2: {
        'name': 'vwind',
        'key': 'v_component_of_wind_83',
        'long_name': 'V-Wind Component',
        'units': 'm/s',
        'cmap': 'RdBu_r',
    },
    3: {
        'name': 'precipitation',
        'key': 'total_precipitation_6hr',
        'long_name': 'Total Precipitation (6hr)',
        'units': 'm',
        'cmap': 'Blues',
    }
}

# Load denormalization stats
denorm_stats = {}
for idx, var_info in variables.items():
    key = var_info['key']
    mean = stats_hr[key].item()['mean']
    std = stats_hr[key].item()['std']
    denorm_stats[idx] = {'mean': mean, 'std': std}
    print(f"{var_info['long_name']}: mean={mean:.4f}, std={std:.4f}")

# Process each sample
sample_files = sorted(samples_dir.glob('[0-9]*.npz'))
print(f"\nFound {len(sample_files)} sample files")

for sample_file in sample_files:
    sample_num = sample_file.stem
    print(f"\nProcessing Sample {sample_num}...")

    # Load sample data
    data = np.load(sample_file)
    output = data['output']  # Shape: (1, 4, 721, 1440)
    fno_output = data['fno_output']  # Shape: (1, 4, 721, 1440)

    # Remove batch dimension
    output = output[0]  # Shape: (4, 721, 1440)
    fno_output = fno_output[0]

    # Create denormalized outputs
    output_denorm = np.zeros_like(output)
    fno_denorm = np.zeros_like(fno_output)

    for idx in range(4):
        mean = denorm_stats[idx]['mean']
        std = denorm_stats[idx]['std']
        output_denorm[idx] = output[idx] * std + mean
        fno_denorm[idx] = fno_output[idx] * std + mean

    # Save denormalized data
    denorm_file = output_dir / f'sample_{sample_num}_denormalized.npz'
    np.savez(denorm_file,
             diffusion_output=output_denorm,
             fno_output=fno_denorm,
             variable_names=['temperature', 'uwind', 'vwind', 'precipitation'])
    print(f"  Saved denormalized data to {denorm_file.name}")

    # Create visualizations for each variable
    for idx, var_info in variables.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Sample {sample_num}: {var_info['long_name']}", fontsize=16, fontweight='bold')

        # Get data
        fno_var = fno_denorm[idx]
        diffusion_var = output_denorm[idx]
        diff = diffusion_var - fno_var

        # Determine color scale
        if var_info['name'] == 'precipitation':
            # Precipitation is non-negative
            vmin_data, vmax_data = 0, max(fno_var.max(), diffusion_var.max())
            vmin_diff, vmax_diff = diff.min(), diff.max()
        else:
            # Use symmetric scale for temperature and wind
            vmax_data = max(abs(fno_var.min()), fno_var.max(),
                           abs(diffusion_var.min()), diffusion_var.max())
            vmin_data = -vmax_data if var_info['name'] != 'temperature' else fno_var.min()
            vmax_data = vmax_data if var_info['name'] != 'temperature' else fno_var.max()

            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            vmin_diff, vmax_diff = -vmax_diff, vmax_diff

        # Plot FNO output
        im0 = axes[0].imshow(fno_var, cmap=var_info['cmap'],
                            vmin=vmin_data, vmax=vmax_data, aspect='auto')
        axes[0].set_title(f"FNO Output\nmin={fno_var.min():.4f}, max={fno_var.max():.4f}, mean={fno_var.mean():.4f}")
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0], label=var_info['units'])

        # Plot Diffusion output
        im1 = axes[1].imshow(diffusion_var, cmap=var_info['cmap'],
                            vmin=vmin_data, vmax=vmax_data, aspect='auto')
        axes[1].set_title(f"Diffusion Output\nmin={diffusion_var.min():.4f}, max={diffusion_var.max():.4f}, mean={diffusion_var.mean():.4f}")
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[1], label=var_info['units'])

        # Plot Difference
        im2 = axes[2].imshow(diff, cmap='RdBu_r',
                            vmin=vmin_diff, vmax=vmax_diff, aspect='auto')
        axes[2].set_title(f"Difference (Diffusion - FNO)\nmin={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[2], label=var_info['units'])

        # Save figure
        fig_file = output_dir / f'sample_{sample_num}_{var_info["name"]}.png'
        plt.tight_layout()
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {var_info['name']} visualization to {fig_file.name}")

    # Create a combined 4-panel plot showing all variables
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Sample {sample_num}: All Variables (Diffusion Output)", fontsize=16, fontweight='bold')

    for idx, var_info in variables.items():
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        var_data = output_denorm[idx]

        if var_info['name'] == 'precipitation':
            vmin, vmax = 0, var_data.max()
        elif var_info['name'] == 'temperature':
            vmin, vmax = var_data.min(), var_data.max()
        else:
            vmax = max(abs(var_data.min()), abs(var_data.max()))
            vmin = -vmax

        im = ax.imshow(var_data, cmap=var_info['cmap'],
                      vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"{var_info['long_name']}\nmin={var_data.min():.4f}, max={var_data.max():.4f}, mean={var_data.mean():.4f}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label=var_info['units'])

    combined_file = output_dir / f'sample_{sample_num}_all_variables.png'
    plt.tight_layout()
    plt.savefig(combined_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined visualization to {combined_file.name}")

print("\n" + "="*60)
print("✓ All visualizations completed!")
print(f"Output directory: {output_dir}")
print("="*60)
