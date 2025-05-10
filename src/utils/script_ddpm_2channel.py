import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import argparse

# python -u script.py --task_name ae_regional_us --num_samples 2
# python -u script.py --task_name ae_concat --num_samples 200
# python -u script.py --task_name vae_concat_LUCIE_v5 --num_samples 200
# nohup python -u script.py --task_name vae_concat_LUCIE_v5 --num_samples 200 > script_output.out 2>&1 &

# ======================== Parse Command-Line Arguments ========================
parser = argparse.ArgumentParser(description="Generate GIFs from climate data.")
parser.add_argument("--task_name", type=str, required=True, help="Name of the task directory.")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate GIFs for.")
args = parser.parse_args()

################################# Define task name and results directory ################################
task_name = args.task_name
num_samples_to_be_used = args.num_samples
results_dir = f"/glade/derecho/scratch/mdarman/lucie/results/{task_name}"
output_dir = os.path.join(results_dir, "results_ddpm")
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging.basicConfig(filename=output_dir+"process.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Results will be saved in: {output_dir}")

################################# Define metrics functions ################################
def latitude_weighted_rmse(pred, target):
    latitudes = torch.linspace(-90, 90, steps=pred.shape[0], device=pred.device)
    weights = torch.cos(latitudes * np.pi / 180).unsqueeze(1).expand(-1, pred.shape[1])
    rmse = torch.sqrt((weights * (pred - target) ** 2).sum() / weights.sum())
    return rmse.item()

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(target.max() / torch.sqrt(mse)).item() if mse != 0 else float('inf')

def get_ssim(preds, truths):
    ssims = np.empty((truths.shape[0], truths.shape[3]))
    for step in range(truths.shape[0]):
        for ch in range(truths.shape[1]):
            ssims[step, ch] = ssim(preds[step, :, :, ch], truths[step, :, :, ch],
                                   data_range=truths[step, :, :, ch].max() - truths[step, :, :, ch].min())
    return ssims

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def max_error(pred, target):
    return torch.max(torch.abs(pred - target)).item()

def compute_mean_std_spectrum_torch(data):
    """
    Compute the mean and standard deviation of the magnitude spectrum of the data using rFFT along the longitude axis.

    Args:
        data (numpy.ndarray or torch.Tensor): Data array of shape (samples, latitudes, longitudes).

    Returns:
        tuple: (mean_spectrum, std_spectrum)
            - mean_spectrum (numpy.ndarray): Mean latitude-averaged magnitude spectrum across samples.
            - std_spectrum (numpy.ndarray): Standard deviation of the latitude-averaged magnitude spectrum across samples.
    """
    # Ensure data is a torch tensor
    if not isinstance(data, torch.Tensor):
        data_torch = torch.tensor(data, dtype=torch.float32)
    else:
        data_torch = data.float()

    # Perform rFFT along the longitude axis (-1)
    spectrum = torch.fft.rfft(data_torch, dim=-1)
    
    # Compute magnitude of the spectrum
    magnitude_spectrum = torch.abs(spectrum)
    
    # Compute mean and std over samples (latitude-average spectrum)
    mean_spectrum = magnitude_spectrum.mean(1).mean(0).numpy() 
    std_spectrum = magnitude_spectrum.mean(1).std(0).numpy() 
    
    return mean_spectrum, std_spectrum

def spectrum_rmse(pred_data, target_data):
    """
    Calculate the Spectrum RMSE between predicted and target data arrays.

    Args:
        pred_data (numpy.ndarray or torch.Tensor): Predicted data of shape (samples, latitudes, longitudes).
        target_data (numpy.ndarray or torch.Tensor): Target data of shape (samples, latitudes, longitudes).

    Returns:
        float: Spectrum RMSE value.
    """
    # Compute mean spectra
    pred_spectrum, _ = compute_mean_std_spectrum_torch(pred_data)
    target_spectrum, _ = compute_mean_std_spectrum_torch(target_data)
    
    # Compute RMSE between the spectra
    rmse = np.sqrt(np.mean((pred_spectrum - target_spectrum) ** 2))
    return rmse

def compute_metrics(pred, target):
    results = {
        "Latitude Weighted RMSE": latitude_weighted_rmse(pred, target),
        "SSIM": ssim(pred.numpy(), target.numpy(), data_range=(target.max() - target.min()).numpy()),
        "Spectrum RMSE": spectrum_rmse(pred, target),
        "PSNR": psnr(pred, target),
        "MAE": mae(pred, target),
        "Max Error": max_error(pred, target)
    }
    return results

################################ Data loading and processing ################
logging.info("Starting data processing...")

# Load example data for shape reference
example_file = os.path.join(results_dir, "samples_ddpm/1.npz")
with np.load(example_file) as data:
    shapes = {key: data[key].shape for key in data.keys()}

# Initialize sum and counters
data_keys = list(shapes.keys())
sums = {key: np.zeros(shapes[key][1:], dtype=np.float32) for key in data_keys}
counters = {key: 0 for key in data_keys}

# Get the files in the samples directory and find the length of them
files = sorted(os.listdir(os.path.join(results_dir, "samples_ddpm")))
files = files[:num_samples_to_be_used]
# Aggregate data across samples
for idx in tqdm(range(1, len(files) + 1)):
# for idx in tqdm(range(3, 4)):
    file_path = os.path.join(results_dir, f"samples_ddpm/{idx}.npz")
    if os.path.exists(file_path):
        with np.load(file_path) as data:
            for key in data_keys:
                sums[key] += data[key].sum(axis=0)
                counters[key] += data[key].shape[0]
    if idx % 500 == 0:
        logging.info(f"Processed {idx} files...")

# Compute means
means = {key: sums[key] / counters[key] for key in data_keys}
logging.info("Finished data aggregation.")

# Save computed metrics to log
for key in means.keys():
    logging.info(f"Mean computed for {key} with shape {means[key].shape}")

# Save outputs
np.savez(os.path.join(output_dir, "aggregated_means.npz"), **means)
logging.info("Aggregated means saved.")

# Load denormalization statistics
lr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009_updated.npz", allow_pickle=True)
hr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz", allow_pickle=True)

# Extract mean and std for each channel
temperature_mean_hr = hr_stats['2m_temperature'].item()['mean']
temperature_std_hr = hr_stats['2m_temperature'].item()['std']
specific_humidity_mean_hr = hr_stats['specific_humidity_133'].item()['mean']
specific_humidity_std_hr = hr_stats['specific_humidity_133'].item()['std']
u_wind_mean_hr = hr_stats['u_component_of_wind_83'].item()['mean']
u_wind_std_hr = hr_stats['u_component_of_wind_83'].item()['std']
v_wind_mean_hr = hr_stats['v_component_of_wind_83'].item()['mean']
v_wind_std_hr = hr_stats['v_component_of_wind_83'].item()['std']
precip_mean_hr = hr_stats['total_precipitation_6hr'].item()['mean']
precip_std_hr = hr_stats['total_precipitation_6hr'].item()['std']
gp_mean_hr = hr_stats['geopotential_at_surface'].item()['mean']
gp_std_hr = hr_stats['geopotential_at_surface'].item()['std']

temperature_mean_lr = lr_stats['Temperature_7'].item()['mean']
temperature_std_lr = lr_stats['Temperature_7'].item()['std']
specific_humidity_mean_lr = lr_stats['Specific_Humidity_7'].item()['mean']
specific_humidity_std_lr = lr_stats['Specific_Humidity_7'].item()['std']
u_wind_mean_lr = lr_stats['U-wind_3'].item()['mean']
u_wind_std_lr = lr_stats['U-wind_3'].item()['std']
v_wind_mean_lr = lr_stats['V-wind_3'].item()['mean']
v_wind_std_lr = lr_stats['V-wind_3'].item()['std']
precip_mean_lr = lr_stats['tp6hr'].item()['mean']
precip_std_lr = lr_stats['tp6hr'].item()['std']
gp_mean_lr = lr_stats['orography'].item()['mean']
gp_std_lr = lr_stats['orography'].item()['std']

for key in means.keys():
    if key == 'output' or key == 'hres' or key == 'lucie_zero_shot' or key == 'fno_output':
        means[key][0] = means[key][0] * temperature_std_hr + temperature_mean_hr
        # means[key][1] = means[key][1] * specific_humidity_std_hr + specific_humidity_mean_hr
        # means[key][2] = means[key][2] * u_wind_std_hr + u_wind_mean_hr
        # means[key][3] = means[key][3] * v_wind_std_hr + v_wind_mean_hr
        means[key][1] = means[key][1] * precip_std_hr + precip_mean_hr
        # means[key][5] = means[key][5] * gp_std_hr + gp_mean_hr
    elif key == 'lres' or key == 'lucie' or key == 'lucie_interp' or key == 'lres_interp':
        means[key][0] = means[key][0] * temperature_std_lr + temperature_mean_lr
        # means[key][1] = means[key][1] * specific_humidity_std_lr + specific_humidity_mean_lr
        # means[key][2] = means[key][2] * u_wind_std_lr + u_wind_mean_lr
        # means[key][3] = means[key][3] * v_wind_std_lr + v_wind_mean_lr
        means[key][5] = means[key][5] * precip_std_lr + precip_mean_lr
        # means[key][5] = means[key][5] * gp_std_hr + gp_mean_hr


# ################################ Visualization setup ################################
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as ocean

loncoords_hres = np.linspace(0, 360, 1440)
latcoords_hres = np.linspace(90, -90, 721)
loncoords_lres = np.linspace(0, 360, 96)    
latcoords_lres = np.linspace(90, -90, 48)
projection = ccrs.PlateCarree()
fig, axs = plt.subplots(5, 1, figsize=(10, 30), subplot_kw={'projection': projection})
data_to_plot = [
    (means['lres'][0], "LR ERA5", loncoords_lres, latcoords_lres),
    (means['lres_interp'][0], "Bicubic interp.", loncoords_hres, latcoords_hres),
    (means['fno_output'][0], "FNO", loncoords_hres, latcoords_hres),
    (means['output'][0], "Diffusion model", loncoords_hres, latcoords_hres),
    (means['hres'][0], "High-Res ERA5", loncoords_hres, latcoords_hres),
    # (means['lucie_zero_shot'][0], 'LUCIE Zero Shot', loncoords_hres, latcoords_hres)
]

ims = []
for ax, (data, title, loncoords, latcoords) in zip(axs, data_to_plot):
    im = ax.pcolormesh(loncoords, latcoords, data, vmin=200, vmax=300, transform=projection, cmap=ocean.cm.thermal)
    ax.coastlines()
    ax.set_title(f"{title} - Temperature Climatology", fontsize=12, fontweight="bold")
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axs, orientation="horizontal", fraction=0.046, pad=0.04)
cbar.set_label("Temperature (K)")
plt.savefig(os.path.join(output_dir, "temperature_climatology.png"), dpi=1200, bbox_inches="tight", format="png")
logging.info("Saved temperature comparison plot.")


#################################### Plot precipitation ####################################
fig, axs = plt.subplots(5, 1, figsize=(10, 25), subplot_kw={'projection': projection})
data_to_plot = [
    (means['lres'][5]*1000, "LR ERA5", loncoords_lres, latcoords_lres),
    (means['lres_interp'][5]*1000, "Bicubic interp.", loncoords_hres, latcoords_hres),
    (means['fno_output'][1]*1000, "FNO", loncoords_hres, latcoords_hres),
    (means['output'][1]*1000, "Model Output", loncoords_hres, latcoords_hres),
    (means['hres'][1]*1000, "HR ERA5", loncoords_hres, latcoords_hres),
    # (means['lucie_zero_shot'][1]*1000, 'LUCIE Zero Shot', loncoords_hres, latcoords_hres)
]
ims = []
for ax, (data, title, loncoords, latcoords) in zip(axs, data_to_plot):
    im = ax.pcolormesh(loncoords, latcoords, data, vmin=0, vmax=15, transform=projection, cmap=ocean.cm.rain)
    ax.coastlines()
    ax.set_title(f"{title} - Precipitation Climatology", fontsize=12, fontweight="bold")
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axs, orientation="horizontal", fraction=0.046, pad=0.04, extend='both')
cbar.set_label("mm/d")
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2, wspace=0.2, hspace=0.2)
plt.savefig(os.path.join(output_dir, "percip_climatology.png"), dpi=1200, bbox_inches="tight", format="png")
logging.info("Saved Precipitation comparison plot.")

# ################################ Compute spectra for temperature and precipitation ################################
logging.info("Computing spectra for temperature and precipitation...")

# Compute mean and standard deviation of the spectra
hres_temp_mean, hres_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['hres'][0]).unsqueeze(0))
output_temp_mean, output_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['output'][0]).unsqueeze(0))
lres_temp_mean, lres_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['lres'][0]).unsqueeze(0))
lres_interp_temp_mean, lres_interp_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['lres_interp'][0]).unsqueeze(0))
fno_temp_mean, fno_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['fno_output'][0]).unsqueeze(0))
# lucie_temp_mean, lucie_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie'][0]).unsqueeze(0))
# lucie_interp_temp_mean, lucie_interp_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_interp'][0]).unsqueeze(0))
# lucie_zero_shot_temp_mean, lucie_zero_shot_temp_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_zero_shot'][0]).unsqueeze(0))

# Precipitation
hres_precip_mean, hres_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['hres'][1]).unsqueeze(0))
output_precip_mean, output_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['output'][1]).unsqueeze(0))
lres_precip_mean, lres_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['lres'][5]).unsqueeze(0))
lres_interp_precip_mean, lres_interp_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['lres_interp'][5]).unsqueeze(0))
fno_precip_mean, fno_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['fno_output'][1]).unsqueeze(0))
# lucie_precip_mean, lucie_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie'][4]).unsqueeze(0))
# lucie_interp_precip_mean, lucie_interp_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_interp'][4]).unsqueeze(0))
# lucie_zero_shot_precip_mean, lucie_zero_shot_precip_std = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_zero_shot'][4]).unsqueeze(0))


logging.info("Finished computing spectra.")


# Define wavenumbers based on the longitude resolution
longitude_points = means['hres'][0].shape[-1]
wavenumbers = np.arange(0, longitude_points // 2 + 1)

################################################################ Temperature Spectrum ################################################################
plt.figure(figsize=(10, 6))

# Plot High-Resolution ERA5 Spectrum (Temperature)
plt.loglog(wavenumbers, hres_temp_mean, label="ERA5 HRES", color="black", linewidth=2)
plt.fill_between(wavenumbers, hres_temp_mean - hres_temp_std, hres_temp_mean + hres_temp_std, color="black", alpha=0.3)

# Plot VAE Output Spectrum (Temperature)
plt.loglog(wavenumbers, lres_interp_temp_mean, label="Bicubic Interp. ", color="green", linewidth=2)
plt.fill_between(wavenumbers, lres_interp_temp_mean - lres_interp_temp_std, lres_interp_temp_mean + lres_interp_temp_std, color="green", alpha=0.3)

plt.loglog(wavenumbers, fno_temp_mean, label="FNO", color="blue", linewidth=2)
plt.fill_between(wavenumbers, fno_temp_mean - fno_temp_std, fno_temp_mean + fno_temp_std, color="blue", alpha=0.3)

plt.loglog(wavenumbers, output_temp_mean, label="Diffusion model", color="red", linewidth=2)
plt.fill_between(wavenumbers, output_temp_mean - output_temp_std, output_temp_mean + output_temp_std, color="red", alpha=0.3)

# plt.loglog(wavenumbers, lucie_interp_temp_mean, label="LUCIE Interpolated", color="purple", linewidth=2)
# plt.fill_between(wavenumbers, lucie_interp_temp_mean - lucie_interp_temp_std, lucie_interp_temp_mean + lucie_interp_temp_std, color="purple", alpha=0.3)

# plt.loglog(wavenumbers, lucie_zero_shot_temp_mean, label="LUCIE Zero Shot", color="blue", linewidth=2)
# plt.fill_between(wavenumbers, lucie_zero_shot_temp_mean - lucie_zero_shot_temp_std, lucie_zero_shot_temp_mean + lucie_zero_shot_temp_std, color="blue", alpha=0.3)

plt.title("Temperature Spectrum")
plt.xlabel("Wavenumber")
plt.ylabel("Mean Magnitude")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrum_comparison_temp.pdf"))#, dpi=300)

# ################################################################ Precipitation Spectrum ################################################################
plt.figure(figsize=(10, 6))

# Plot High-Resolution ERA5 Spectrum (Precipitation)
plt.loglog(wavenumbers, hres_precip_mean, label="ERA5 HRES", color="black", linewidth=2)
plt.fill_between(wavenumbers, hres_precip_mean - hres_precip_std, hres_precip_mean + hres_precip_std, color="black", alpha=0.3)

# Plot VAE Output Spectrum (Precipitation)
plt.loglog(wavenumbers, lres_interp_precip_mean, label="Bicubic Interp.", color="green", linewidth=2)
plt.fill_between(wavenumbers, lres_interp_precip_mean - lres_interp_precip_std, lres_interp_precip_mean + lres_interp_precip_std, color="green", alpha=0.3)

plt.loglog(wavenumbers, fno_precip_mean, label="FNO", color="blue", linewidth=2)
plt.fill_between(wavenumbers, fno_precip_mean - fno_precip_std, fno_precip_mean + fno_precip_std, color="blue", alpha=0.3)

plt.loglog(wavenumbers, output_precip_mean, label="Diffusion model", color="red", linewidth=2)
plt.fill_between(wavenumbers, output_precip_mean - output_precip_std, output_precip_mean + output_precip_std, color="red", alpha=0.3)

# plt.loglog(wavenumbers, lucie_interp_precip_mean, label="LUCIE interpolated", color="purple", linewidth=2)
# plt.fill_between(wavenumbers, lucie_interp_precip_mean - lucie_interp_precip_std, lucie_interp_precip_mean + lucie_interp_precip_std, color="purple", alpha=0.3)

# plt.loglog(wavenumbers, lucie_zero_shot_precip_mean, label="LUCIE Zero Shot", color="blue", linewidth=2)
# plt.fill_between(wavenumbers, lucie_zero_shot_precip_mean - lucie_zero_shot_precip_std, lucie_zero_shot_precip_mean + lucie_zero_shot_precip_std, color="blue", alpha=0.3)

plt.title("Precipitation Spectrum")
plt.xlabel("Wavenumber")
plt.ylabel("Mean Magnitude")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrum_comparison_precip.pdf"))#, dpi=300)
logging.info("Saved temperature and precipitation spectrum comparison plots.")
################################ Compute metrics ################################
import os
import torch
import logging

# Initialize dictionaries to store results for each metric
metrics_output_vs_hres = {
    "Temperature": {},# "SpecificHumidity": {}, "U-Wind": {}, "V-Wind": {}, 
    "Precipitation": {}
}
metrics_lres_interp_vs_hres = {
    "Temperature": {}, #"SpecificHumidity": {}, "U-Wind": {}, "V-Wind": {}, 
    "Precipitation": {}
}
metrics_fno_vs_hres = {
    "Temperature": {}, #"SpecificHumidity": {}, "U-Wind": {}, "V-Wind": {}, 
    "Precipitation": {}
}
metrics_lucie_vs_hres = {
    "Temperature": {}, #"SpecificHumidity": {}, "U-Wind": {}, "V-Wind": {}, 
    "Precipitation": {}
}

# Map readable title to dictionary keys
title_to_key = {
    "Metrics for Temperature": "Temperature",
    # "Metrics for Specific Humidity": "SpecificHumidity",
    # "Metrics for U-Wind": "U-Wind",
    # "Metrics for V-Wind": "V-Wind",
    "Metrics for Precipitation": "Precipitation"
}

# Iterate over all 5 channels
for channel, metric_name in enumerate(["Temperature", #"SpecificHumidity", "U-Wind", "V-Wind", 
                                       "Precipitation"]):
    metrics_output_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['output'][channel]), 
        torch.tensor(means['hres'][channel])
    )
    metrics_fno_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['fno_output'][channel]), 
        torch.tensor(means['hres'][channel])
    )
    metrics_lres_interp_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['lres_interp'][channel]), 
        torch.tensor(means['hres'][channel])
    )
    # Uncomment this if you want to include Lucie interpolation
    # metrics_lucie_vs_hres[metric_name] = compute_metrics(
    #     torch.tensor(means['lucie_interp'][channel]), 
    #     torch.tensor(means['hres'][channel])
    # )

logging.info("Finished computing metrics.")

# Output path
metrics_output_file = os.path.join(output_dir, "metrics_report.txt")

# Function to format metrics into a readable table
def format_metrics_table(metrics_dict, title):
    key = title_to_key[title]
    
    table = f"\n{'='*40}\n{title}\n{'='*40}\n"
    table += "{:<25} {:<20} {:<20} {:<20}\n".format("Metric", "Bicubic interp.", "FNO", "Diffusion model")
    table += "-" * 85 + "\n"
    
    for metric in metrics_dict.keys():
        table += "{:<25} {:<20.4f} {:<20.4f} {:<20.4f}\n".format(
            metric,
            metrics_lres_interp_vs_hres[key][metric],
            metrics_fno_vs_hres[key][metric],
            metrics_output_vs_hres[key][metric]
        )
    return table

# Generate and write tables
tables = [
    format_metrics_table(metrics_output_vs_hres["Temperature"], "Metrics for Temperature"),
    # format_metrics_table(metrics_output_vs_hres["SpecificHumidity"], "Metrics for Specific Humidity"),
    # format_metrics_table(metrics_output_vs_hres["U-Wind"], "Metrics for U-Wind"),
    # format_metrics_table(metrics_output_vs_hres["V-Wind"], "Metrics for V-Wind"),
    format_metrics_table(metrics_output_vs_hres["Precipitation"], "Metrics for Precipitation")
]

with open(metrics_output_file, "w") as f:
    for table in tables:
        f.write(table + "\n")

logging.info(f"Metrics saved to {metrics_output_file}")
print(f"Metrics successfully saved to {metrics_output_file}")
