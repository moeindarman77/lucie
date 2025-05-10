import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as ocean
import imageio
import gc
from tqdm import tqdm
import argparse

# Example Usage: 
# python -u make_movie_precip.py --task_name vae_concat_LUCIE_v5 --num_samples 200
# nohup python -u make_movie_precip.py --task_name vae_concat_LUCIE_v5 --num_samples 200 > make_movie_out_precip.out 2>&1 &

# python -u make_movie_precip_conus.py --task_name fno_v0 --num_samples 10

# ======================== Parse Command-Line Arguments ========================
parser = argparse.ArgumentParser(description="Generate GIFs from climate data.")
parser.add_argument("--task_name", type=str, required=True, help="Name of the task directory.")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate GIFs for.")
args = parser.parse_args()

# ======================== Setup Paths ========================
task_name = args.task_name
num_samples_to_be_used = args.num_samples
results_dir = f"/glade/derecho/scratch/mdarman/lucie/results/{task_name}"
output_dir = os.path.join(results_dir, "results")
os.makedirs(output_dir, exist_ok=True)

samples_dir = os.path.join(results_dir, "samples")
precip_gif = os.path.join(output_dir, "precipitation_movie_conus.gif")

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
# precip_writer = imageio.get_writer(precip_gif, fps=fps, format="GIF")
precip_mp4 = os.path.join(output_dir, "precipitation_movie_conus.mp4")
precip_writer = imageio.get_writer(precip_mp4, fps=fps, codec="libx264", format="FFMPEG", quality=8)

# Iterate through each .npz file and each sample inside it
for file_name in tqdm(sample_files[:num_samples_to_be_used], desc="Generating Movie Frames"):
    file_path = os.path.join(samples_dir, file_name)

    with np.load(file_path) as data:
        num_samples = data['lres'].shape[0]  # Get number of samples per file

        for sample_idx in range(num_samples):  # Loop over all samples
            # ========================== Precipitation ==========================
            hres_tp6hr = data['hres'][sample_idx, 1][us_lat_indices_hres, :][:, us_lon_indices_hres] * precip_std_hr + precip_mean_hr
            output_tp6hr = data['output'][sample_idx, 1][us_lat_indices_hres, :][:, us_lon_indices_hres] * precip_std_hr + precip_mean_hr
            lres_interp_tp6hr = data['lres_interp'][sample_idx, 1][us_lat_indices_hres, :][:, us_lon_indices_hres] * precip_std_hr + precip_mean_hr
            lucie_zero_shot_tp6hr = data['lucie_zero_shot'][sample_idx, 1][us_lat_indices_hres, :][:, us_lon_indices_hres] * precip_std_hr + precip_mean_hr

            # ======================== Create Figures ========================
            # ======================== Precipitation Plot ========================
            fig_precip, axs_precip = plt.subplots(1, 4, figsize=(40, 12), subplot_kw={'projection': projection})
            precip_data = [
                (lres_interp_tp6hr * 1000, "ERA5 Interpolated", us_loncoords_hres, us_latcoords_hres),
                (hres_tp6hr * 1000, "ERA5 (hres)", us_loncoords_hres, us_latcoords_hres),
                (output_tp6hr * 1000, "ERA5 DS", us_loncoords_hres, us_latcoords_hres),
                (lucie_zero_shot_tp6hr * 1000, "LUCIE Zero-Shot", us_loncoords_hres, us_latcoords_hres)
            ]

            ims_precip = []
            for ax, (data_points, title, loncoords, latcoords) in zip(axs_precip, precip_data):
                extent=[loncoords.min(), loncoords.max(), latcoords.min(), latcoords.max()]
                # im = ax.imshow(
                #     data_points, extent=extent,
                #     vmin=0, vmax=4, transform=projection, cmap=ocean.cm.rain, origin="upper"
                # )
                im = ax.pcolormesh(
                    loncoords, latcoords, data_points,
                    vmin=0, vmax=4, transform=projection, cmap=ocean.cm.rain, shading='auto',
                )
                ax.coastlines()
                ax.set_title(f"{title} - {file_name}", fontsize=32, fontweight="bold")
                ims_precip.append(im)

            cbar_precip = fig_precip.colorbar(ims_precip[0], ax=axs_precip, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
            cbar_precip.set_label("Precipitation (mm/d)", fontsize=32)
            cbar_precip.ax.tick_params(labelsize=24)

            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.2, hspace=0.2)

            # Convert to image and append to precipitation GIF
            fig_precip.canvas.draw()
            precip_image = np.array(fig_precip.canvas.renderer.buffer_rgba())
            precip_writer.append_data(precip_image)

            plt.close(fig_precip)
            gc.collect()

# Finalize GIFs
precip_writer.close()

print(f"Precipitation Movie saved at: {precip_mp4}")

