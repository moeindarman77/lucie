import os
import numpy as np
from tqdm import tqdm

# Paths
results_dir = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/"
samples_dir = os.path.join(results_dir, "samples_ddpm_test")

# Gather and numerically sort all .npz files
files = sorted(
    [f for f in os.listdir(samples_dir) if f.endswith(".npz")],
    key=lambda fn: int(os.path.splitext(fn)[0])
)

# Pre‐allocate output array: (num_samples, H, W)
num_files = len(files)
H, W = 721, 1440
u_winds = np.zeros((num_files, H, W), dtype=np.float32)

# Loop and extract the U-wind channel (channel index 1) of the first sample in each file
for i, fname in enumerate(tqdm(files, desc="Extracting U-wind")):
    path = os.path.join(samples_dir, fname)
    with np.load(path) as data:
        print(data.keys())
        output_array = data["lres_interp"]         # e.g. shape (batch_size, channels, 721, 1440)
        u_winds[i] = output_array[0, 2, ...]  # first sample, channel 1

# Save the stacked U-wind field
out_path = os.path.join(results_dir, "all_u_winds_interp.npy")
np.save(out_path, u_winds)

print(f"Saved U-wind array of shape {u_winds.shape} to:\n{out_path}")