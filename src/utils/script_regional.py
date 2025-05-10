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
output_dir = os.path.join(results_dir, "results")
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
example_file = os.path.join(results_dir, "samples/1.npz")
with np.load(example_file) as data:
    shapes = {key: data[key].shape for key in data.keys()}

# Initialize sum and counters
data_keys = list(shapes.keys())
sums = {key: np.zeros(shapes[key][1:], dtype=np.float32) for key in data_keys}
counters = {key: 0 for key in data_keys}

# Get the files in the samples directory and find the length of them
files = os.listdir(os.path.join(results_dir, "samples"))
files = files[:num_samples_to_be_used]
# Aggregate data across samples
for idx in tqdm(range(1, len(files) + 1)):
# for idx in tqdm(range(1, 2)):
    file_path = os.path.join(results_dir, f"samples/{idx}.npz")
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
lr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009.npz", allow_pickle=True)
hr_stats = np.load("/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009.npz", allow_pickle=True)

# Extract mean and std for each channel
temperature_mean_hr = hr_stats['2m_temperature'].item()['mean']
temperature_std_hr = hr_stats['2m_temperature'].item()['std']
precip_mean_hr = hr_stats['tp6hr'].item()['mean']
precip_std_hr = hr_stats['tp6hr'].item()['std']

temperature_mean_lr = lr_stats['Temperature_7'].item()['mean']
temperature_std_lr = lr_stats['Temperature_7'].item()['std']
precip_mean_lr = lr_stats['tp6hr'].item()['mean']
precip_std_lr = lr_stats['tp6hr'].item()['std']

for key in means.keys():
    if key == 'output' or key == 'hres' or key == 'lucie_zero_shot':
        means[key][0] = means[key][0] * temperature_std_hr + temperature_mean_hr
        means[key][1] = means[key][1] * precip_std_hr + precip_mean_hr
    elif key == 'lres' or key == 'lucie' or key == 'lucie_interp' or key == 'lres_interp':
        means[key][0] = means[key][0] * temperature_std_lr + temperature_mean_lr
        means[key][1] = means[key][1] * precip_std_lr + precip_mean_lr


################################ Visualization setup ################################
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as ocean

loncoords_hres = np.linspace(0, 360, 1440)
latcoords_hres = np.linspace(90, -90, 721)
loncoords_lres = np.linspace(0, 360, 96)    
latcoords_lres = np.linspace(90, -90, 48)

us_lat_indices_lres = torch.tensor(list(range(10, 19)))
us_lon_indices_lres = torch.tensor(list(range(61, 80)))

us_lat_indices_hres = torch.tensor(list(range(131, 275)))
us_lon_indices_hres = torch.tensor(list(range(930, 1218)))

loncoords_us_hres = loncoords_hres[us_lon_indices_hres]
latcoords_us_hres = latcoords_hres[us_lat_indices_hres]

loncoords_us_lres = loncoords_lres[us_lon_indices_lres]
latcoords_us_lres = latcoords_lres[us_lat_indices_lres]

projection = ccrs.PlateCarree()
fig, axs = plt.subplots(1, 4, figsize=(40, 12), subplot_kw={'projection': projection})
data_to_plot = [
    (means['lres'][0], "Low-Resolution ERA5", loncoords_us_lres, latcoords_us_lres),
    (means['hres'][0], "High-Resolution ERA5", loncoords_us_hres, latcoords_us_hres),
    (means['output'][0], "High-Resolution VAE output", loncoords_us_hres, latcoords_us_hres),
    (means['lucie_zero_shot'][0], 'LUCIE Zero Shot', loncoords_us_hres, latcoords_us_hres)
]

ims = []
for ax, (data, title, loncoords, latcoords) in zip(axs, data_to_plot):
    # im = ax.contourf(loncoords, latcoords, data, vmin=200, vmax=300, levels=50, transform=projection, cmap=ocean.cm.thermal)
    im = ax.pcolormesh(loncoords, latcoords, data, vmin=200, vmax=300, transform=projection, cmap=ocean.cm.thermal)
    ax.coastlines()
    ax.set_title(f"{title} - Temperature Climatology", fontsize=16, fontweight="bold")
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axs, orientation="horizontal", fraction=0.046, pad=0.04)
cbar.set_label("Temperature (K)")
plt.savefig(os.path.join(output_dir, "temperature_climatology.pdf"), dpi=1200)
logging.info("Saved temperature comparison plot.")

fig, axs = plt.subplots(1, 4, figsize=(40, 12), subplot_kw={'projection': projection})
data_to_plot = [
    (means['lres'][1]*1000, "Low-Resolution ERA5", loncoords_us_lres, latcoords_us_lres),
    (means['hres'][1]*1000, "High-Resolution ERA5", loncoords_us_hres, latcoords_us_hres),
    (means['output'][1]*1000, "High-Resolution VAE output", loncoords_us_hres, latcoords_us_hres),
    (means['lucie_zero_shot'][1]*1000, 'LUCIE Zero Shot', loncoords_us_hres, latcoords_us_hres)
]
ims = []
for ax, (data, title, loncoords, latcoords) in zip(axs, data_to_plot):
    # im = ax.contourf(loncoords, latcoords, data, vmin=0, vmax=15, levels=50, transform=projection, cmap=ocean.cm.rain)
    im = ax.pcolormesh(loncoords, latcoords, data, vmin=0, vmax=4, transform=projection, cmap=ocean.cm.rain)
    ax.coastlines()
    ax.set_title(f"{title} - Precipitation Climatology", fontsize=16, fontweight="bold")
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axs, orientation="horizontal", fraction=0.046, pad=0.04, extend='both')
cbar.set_label("mm/d")
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2, wspace=0.2, hspace=0.2)
plt.savefig(os.path.join(output_dir, "percip_climatology.pdf"), dpi=1200)
logging.info("Saved Precipitation comparison plot.")

################################ Compute spectra for temperature and precipitation ################################
logging.info("Computing spectra for temperature and precipitation...")

# Convert data to torch tensors before computing the spectrum
hres_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['hres'][0]).unsqueeze(0))
output_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['output'][0]).unsqueeze(0))
lres_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lres'][0]).unsqueeze(0))
lres_interp_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lres_interp'][0]).unsqueeze(0))
lucie_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie'][0]).unsqueeze(0))
lucie_interp_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_interp'][0]).unsqueeze(0))
lucie_zero_shot_temp_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_zero_shot'][0]).unsqueeze(0))


hres_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['hres'][1]).unsqueeze(0))
output_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['output'][1]).unsqueeze(0))
lres_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lres'][1]).unsqueeze(0))
lres_interp_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lres_interp'][1]).unsqueeze(0))
lucie_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie'][1]).unsqueeze(0))
lucie_interp_precip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_interp'][1]).unsqueeze(0))
lucie_zero_shot_percip_spectrum = compute_mean_std_spectrum_torch(torch.tensor(means['lucie_zero_shot'][1]).unsqueeze(0))

# Extract mean and standard deviation
hres_temp_mean, hres_temp_std = hres_temp_spectrum
output_temp_mean, output_temp_std = output_temp_spectrum
lres_temp_mean, lres_temp_std = lres_temp_spectrum
lres_interp_temp_mean, lres_interp_temp_std = lres_interp_temp_spectrum
lucie_temp_mean, lucie_temp_std = lucie_temp_spectrum
lucie_interp_temp_mean, lucie_interp_temp_std = lucie_interp_temp_spectrum
lucie_zero_shot_temp_mean, lucie_zero_shot_temp_std = lucie_zero_shot_temp_spectrum

hres_precip_mean, hres_precip_std = hres_precip_spectrum
output_precip_mean, output_precip_std = output_precip_spectrum
lres_precip_mean, lres_precip_std = lres_precip_spectrum
lres_interp_precip_mean, lres_interp_precip_std = lres_interp_precip_spectrum
lucie_precip_mean, lucie_precip_std = lucie_precip_spectrum
lucie_interp_precip_mean, lucie_interp_precip_std = lucie_interp_precip_spectrum
lucie_zero_shot_precip_mean, lucie_zero_shot_precip_std = lucie_zero_shot_percip_spectrum

logging.info("Finished computing spectra.")

# Define wavenumbers based on the longitude resolution
longitude_points = means['hres'][0].shape[-1]
wavenumbers = np.arange(0, longitude_points // 2 + 1)

plt.figure(figsize=(10, 6))

# Plot High-Resolution ERA5 Spectrum (Temperature)
plt.loglog(wavenumbers, hres_temp_mean, label="ERA5 Hres", color="black", linewidth=2)
plt.fill_between(wavenumbers, hres_temp_mean - hres_temp_std, hres_temp_mean + hres_temp_std, color="black", alpha=0.3)

# Plot VAE Output Spectrum (Temperature)
plt.loglog(wavenumbers, output_temp_mean, label="ERA5 DS", color="red", linewidth=2)
plt.fill_between(wavenumbers, output_temp_mean - output_temp_std, output_temp_mean + output_temp_std, color="red", alpha=0.3)

plt.loglog(wavenumbers, lres_interp_temp_mean, label="ERA5 Interpolated ", color="green", linewidth=2)
plt.fill_between(wavenumbers, lres_interp_temp_mean - lres_interp_temp_std, lres_interp_temp_mean + lres_interp_temp_std, color="green", alpha=0.3)

plt.loglog(wavenumbers, lucie_interp_temp_mean, label="LUCIE Interpolated", color="purple", linewidth=2)
plt.fill_between(wavenumbers, lucie_interp_temp_mean - lucie_interp_temp_std, lucie_interp_temp_mean + lucie_interp_temp_std, color="purple", alpha=0.3)

plt.loglog(wavenumbers, lucie_zero_shot_temp_mean, label="LUCIE Zero Shot", color="blue", linewidth=2)
plt.fill_between(wavenumbers, lucie_zero_shot_temp_mean - lucie_zero_shot_temp_std, lucie_zero_shot_temp_mean + lucie_zero_shot_temp_std, color="blue", alpha=0.3)

plt.title("Temperature Spectrum")
plt.xlabel("Wavenumber")
plt.ylabel("Mean Magnitude")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrum_comparison_temp.pdf"), dpi=1200)

plt.figure(figsize=(10, 6))

# Plot High-Resolution ERA5 Spectrum (Precipitation)
plt.loglog(wavenumbers, hres_precip_mean, label="ERA5 (hres)", color="black", linewidth=2)
plt.fill_between(wavenumbers, hres_precip_mean - hres_precip_std, hres_precip_mean + hres_precip_std, color="black", alpha=0.3)

# Plot VAE Output Spectrum (Precipitation)
plt.loglog(wavenumbers, output_precip_mean, label="ERA5 DS", color="red", linewidth=2)
plt.fill_between(wavenumbers, output_precip_mean - output_precip_std, output_precip_mean + output_precip_std, color="red", alpha=0.3)

plt.loglog(wavenumbers, lres_interp_precip_mean, label="ERA5 interpolated", color="green", linewidth=2)
plt.fill_between(wavenumbers, lres_interp_precip_mean - lres_interp_precip_std, lres_interp_precip_mean + lres_interp_precip_std, color="green", alpha=0.3)

plt.loglog(wavenumbers, lucie_interp_precip_mean, label="LUCIE interpolated", color="purple", linewidth=2)
plt.fill_between(wavenumbers, lucie_interp_precip_mean - lucie_interp_precip_std, lucie_interp_precip_mean + lucie_interp_precip_std, color="purple", alpha=0.3)

plt.loglog(wavenumbers, lucie_zero_shot_precip_mean, label="LUCIE Zero Shot", color="blue", linewidth=2)
plt.fill_between(wavenumbers, lucie_zero_shot_precip_mean - lucie_zero_shot_precip_std, lucie_zero_shot_precip_mean + lucie_zero_shot_precip_std, color="blue", alpha=0.3)

plt.title("Precipitation Spectrum")
plt.xlabel("Wavenumber")
plt.ylabel("Mean Magnitude")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrum_comparison_percip.pdf"), dpi=1200)
logging.info("Saved temperature and precipitation spectrum comparison plots.")

################################ Compute metrics ################################
import torch

# Initialize dictionaries to store results for each metric
metrics_output_vs_hres = {"Temperature": {}, "Precipitation": {}}
metrics_lres_interp_vs_hres = {"Temperature": {}, "Precipitation": {}}
metrics_lucie_vs_hres = {"Temperature": {}, "Precipitation": {}}

# Iterate over both channels (Temperature = 0, Precipitation = 1)
for channel, metric_name in enumerate(["Temperature", "Precipitation"]):  
    # Compute metrics between output and hres
    metrics_output_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['output'][channel]), 
        torch.tensor(means['hres'][channel])
    )

    # Compute metrics between lres_interp and hres
    metrics_lres_interp_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['lres_interp'][channel]), 
        torch.tensor(means['hres'][channel])
    )

    # Compute metrics between lucie and hres
    metrics_lucie_vs_hres[metric_name] = compute_metrics(
        torch.tensor(means['lucie_interp'][channel]), 
        torch.tensor(means['hres'][channel])
    )

logging.info("Finished computing metrics.")

# Define the output file path
metrics_output_file = os.path.join(output_dir, "metrics_report.txt")

# Function to format metrics into a table
def format_metrics_table(metrics_dict, title):
    """Format the dictionary of metrics into a readable table format."""
    table = f"\n{'='*40}\n{title}\n{'='*40}\n"
    table += "{:<25} {:<15} {:<15} {:<15}\n".format("Metric", "VAE", "Interpolation", "LUCIE interpolation")
    table += "-"*75 + "\n"
    
    metric_keys = list(metrics_dict.keys())  # Get metric names dynamically
    for metric in metric_keys:
        table += "{:<25} {:<15.4f} {:<15.4f} {:<15.4f}\n".format(
            metric,
            metrics_output_vs_hres[title.split()[-1]][metric],  # Extract category from title
            metrics_lres_interp_vs_hres[title.split()[-1]][metric],
            metrics_lucie_vs_hres[title.split()[-1]][metric]
        )
    return table

# Generate tables for Temperature and Precipitation separately
temperature_table = format_metrics_table(metrics_output_vs_hres["Temperature"], "Metrics for Temperature")
precipitation_table = format_metrics_table(metrics_output_vs_hres["Precipitation"], "Metrics for Precipitation")

# Save to file
with open(metrics_output_file, "w") as f:
    f.write(temperature_table)
    f.write("\n")
    f.write(precipitation_table)

logging.info(f"Metrics saved to {metrics_output_file}")
print(f"Metrics successfully saved to {metrics_output_file}")
