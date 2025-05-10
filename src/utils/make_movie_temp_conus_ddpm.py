import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as ocean
import imageio
import gc
from tqdm import tqdm
import argparse
import torch

# Example Usage: 
# python -u make_movie_temp_conus.py --task_name ae_concat --num_samples 20
# python -u make_movie_temp_conus.py --task_name vae_concat_LUCIE_v5 --num_samples 200
# nohup python -u make_movie_temp.py --task_name vae_concat_LUCIE_v5 --num_samples 200 > make_movie_out_temp.out 2>&1 &

# ======================== Parse Command-Line Arguments ========================
parser = argparse.ArgumentParser(description="Generate GIFs from climate data.")
parser.add_argument("--task_name", type=str, required=True, help="Name of the task directory.")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate GIFs for.")
args = parser.parse_args()

# ======================== Setup Paths ========================
task_name = args.task_name
num_samples_to_be_used = args.num_samples
results_dir = f"/glade/derecho/scratch/mdarman/lucie/results/{task_name}"
output_dir = os.path.join(results_dir, "results_ddpm")
os.makedirs(output_dir, exist_ok=True)

samples_dir = os.path.join(results_dir, "samples_ddpm")
temp_gif = os.path.join(output_dir, "temperature_movie_conus.gif")

# ======================== Load Normalization Statistics ========================
lr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009.npz", allow_pickle=True)
hr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009.npz", allow_pickle=True)

# Extract mean and std for temperature and precipitation
temperature_mean_hr = hr_stats['2m_temperature'].item()['mean']
temperature_std_hr = hr_stats['2m_temperature'].item()['std']
temperature_mean_lr = lr_stats['Temperature_7'].item()['mean']
temperature_std_lr = lr_stats['Temperature_7'].item()['std']

precip_mean_hr = hr_stats['tp6hr'].item()['mean']
precip_std_hr = hr_stats['tp6hr'].item()['std']
precip_mean_lr = lr_stats['tp6hr'].item()['mean']
precip_std_lr = lr_stats['tp6hr'].item()['std']

# ======================== Get Sample Files ========================
sample_files = sorted(
    [f for f in os.listdir(samples_dir) if f.endswith(".npz")], 
    key=lambda x: int(x.split(".")[0])
)

# ======================== Setup Plotting ========================
projection = ccrs.PlateCarree()
loncoords_hres = np.linspace(0, 360, 1440)
latcoords_hres = np.linspace(90, -90, 721)
loncoords_lres = np.linspace(0, 360, 96)
latcoords_lres = np.linspace(90, -90, 48)

us_lat_indices_hres = slice(131, 275)
us_lon_indices_hres = slice(930, 1218)

us_latcoords_hres = latcoords_hres[us_lat_indices_hres]
us_loncoords_hres = loncoords_hres[us_lon_indices_hres]

# ======================== Generate Movies (GIFs) ========================
fps = 15  # Frames per second for the GIF
# temp_writer = imageio.get_writer(temp_gif, fps=fps, format="GIF")
temp_mp4 = os.path.join(output_dir, "temperature_movie_conus.mp4")
temp_writer = imageio.get_writer(temp_mp4, fps=fps, codec="libx264", format="FFMPEG", quality=8)

# Iterate through each .npz file and each sample inside it
for file_name in tqdm(sample_files[:num_samples_to_be_used], desc="Generating Movie Frames"):
    file_path = os.path.join(samples_dir, file_name)

    with np.load(file_path) as data:
        num_samples = data['lres'].shape[0]  # Get number of samples per file

        for sample_idx in range(num_samples):  # Loop over all samples
            # ========================== Temperature ==========================
            hres_temp = data['hres'][sample_idx, 0][us_lat_indices_hres, :][:, us_lon_indices_hres] * temperature_std_hr + temperature_mean_hr
            output_temp = data['output'][sample_idx, 0][us_lat_indices_hres, :][:, us_lon_indices_hres] * temperature_std_hr + temperature_mean_hr
            lres_interp_temp = data['lres_interp'][sample_idx, 0][us_lat_indices_hres, :][:, us_lon_indices_hres] * temperature_std_hr + temperature_mean_hr
            # lucie_zero_shot_temp = data['lucie_zero_shot'][sample_idx, 0][us_lat_indices_hres, :][:, us_lon_indices_hres] * temperature_std_hr + temperature_mean_hr

            # ======================== Create Figures ========================
            # Temperature Plot
            fig_temp, axs_temp = plt.subplots(1, 4, figsize=(40, 12), subplot_kw={'projection': projection})
            temp_data = [
                (lres_interp_temp, "ERA5 Interpolated", us_loncoords_hres, us_latcoords_hres),
                (hres_temp, "ERA5 (hres)", us_loncoords_hres, us_latcoords_hres),
                (output_temp, "ERA5 DS", us_loncoords_hres, us_latcoords_hres),
                # (lucie_zero_shot_temp, "Lucie Zero-Shot", us_loncoords_hres, us_latcoords_hres)
            ]

            ims_temp = []
            for ax, (data_points, title, loncoords, latcoords) in zip(axs_temp, temp_data):
                extent=[loncoords.min(), loncoords.max(), latcoords.min(), latcoords.max()] 
                # print(extent)
                im = ax.pcolormesh(
                    loncoords, latcoords, data_points,
                    vmin=200, vmax=300, transform=projection, cmap=ocean.cm.thermal, shading='auto',
                )
                ax.coastlines()
                # ax.set_extent(extent)
                ax.set_title(f"{title} - {file_name}", fontsize=24, fontweight="bold")
                ims_temp.append(im)

            cbar_temp = fig_temp.colorbar(ims_temp[0], ax=axs_temp, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
            cbar_temp.set_label("Temperature (K)", fontsize=24)
            cbar_temp.ax.tick_params(labelsize=24)

            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2, wspace=0.2, hspace=0.2)

            # Convert to image and append to temperature GIF
            fig_temp.canvas.draw()
            temp_image = np.array(fig_temp.canvas.renderer.buffer_rgba())
            temp_writer.append_data(temp_image)

            plt.close(fig_temp)
            gc.collect()

# Finalize GIFs
temp_writer.close()

print(f"Temperature Movie saved at: {temp_mp4}")
