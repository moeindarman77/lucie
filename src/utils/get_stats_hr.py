import glob
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # Assuming .nc files are used for HR data
from tqdm import tqdm

# Define the input and output variables to process
input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
output_vars = ['tp6hr', '2m_temperature']

# Collect files for years 2000 to 2010 with the specified format
files = []
for year in range(2000, 2003):
    files.extend(glob.glob(f'/glade/derecho/scratch/asheshc/ERA5_hr/{year}_*.h5'))

# Sort the list of files
files = sorted(files)

print(files)  # Optional: print to verify the correct files are selected

# Initialize dictionary to store results for each variable
results = {}

# Loop over each variable with tqdm for progress tracking
for var_name in tqdm(output_vars, desc="Processing Variables"):
    tp = np.zeros((721, 1440))

    # Initialize a list to collect data arrays for this variable
    tp_mean_std = []

    # Process each file for the current variable with tqdm for progress tracking
    for file in tqdm(files, desc=f"Processing files for {var_name}", leave=False):
        with Dataset(file, "r") as f:  # Use Dataset for .nc files
            if var_name in f.variables:
                data = f.variables[var_name][:]

                # Apply the log transformation if the variable is "tp6hr"
                if var_name == "tp6hr":
                    data = np.log(data + 1e-6)  # Apply log with a small constant

                tp += data
                tp_mean_std.append(data)
            else:
                print(f"Variable '{var_name}' not found in file {file}, skipping this file.")

    tp = tp * 1000
    tp = tp / 365.25

    # Stack the data to calculate mean and standard deviation along the first axis
    tp_mean_std = np.stack(tp_mean_std)  # Convert list to a 3D array

    # Calculate mean and standard deviation for the current variable
    mean = np.mean(tp_mean_std)
    std = np.std(tp_mean_std)
    
    # Normalize the data for the current variable
    normalized = (tp_mean_std - mean) / std 
    normalized_mean = np.mean(normalized)
    normalized_std = np.std(normalized)
    
    # Store results in the dictionary
    results[var_name] = { 
        "mean": mean,
        "std": std,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std
    }

    # Print results for the current variable
    print(f"Variable: {var_name}")
    print("Dataset Mean:", mean)
    print("Dataset STD:", std)
    print("Normalized Mean:", normalized_mean)
    print("Normalized STD:", normalized_std)

# Save all results to a .npz file
np.savez('stats_2000_2003_hr.npz', **results)

# To verify, load and print the saved file
loaded_data = np.load('stats_2000_2003_hr.npz', allow_pickle=True)
for var_name in loaded_data:
    stats = loaded_data[var_name].item()
    print(f"{var_name}: Mean = {stats['mean']}, STD = {stats['std']}, "
          f"Normalized Mean = {stats['normalized_mean']}, Normalized STD = {stats['normalized_std']}")
