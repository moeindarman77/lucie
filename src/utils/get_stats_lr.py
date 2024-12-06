import h5py
import glob
import numpy as np
from tqdm import tqdm

# Define the variables to process
input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']

# Collect files for years 2000 to 2003 with the specified format
files = []
for year in range(2000, 2003):
    files.extend(glob.glob(f'/glade/derecho/scratch/asheshc/ERA5_t30/train/{year}_*.h5'))

# Sort the list of files
files = sorted(files)

# Select every 6th file (if desired)
files = files[0:-1:6]

print(files)  # Optional: print to verify the correct files are selected

# Initialize dictionary to store results for each variable
results = {}

# Loop over each variable with tqdm for progress tracking
for var in tqdm(input_vars, desc="Processing Variables"):
    tp = np.zeros((48, 96))

    # Initialize valid files for this variable
    valid_files = []
    
    # First pass: Identify files without NaNs for the current variable
    for file in tqdm(files, desc=f"Finding valid files for {var}", leave=False):
        with h5py.File(file, "r") as f:
            if var in f['input']:
                current_data = f['input'][var][:]
                
                # Apply log transformation if var is 'tp6hr'
                if var == 'tp6hr':
                    current_data = np.log(current_data + 1e-6)  # Log transformation with offset

                if not np.isnan(current_data).any():
                    valid_files.append(file)
            else:
                print(f"Variable '{var}' not found in file {file}, skipping this file.")
    
    # Initialize tp_mean_std based on the count of valid files
    tp_mean_std = np.zeros((len(valid_files), 48, 96))
    counter = 0
    
    # Second pass: Process only valid files for the current variable
    for file in tqdm(valid_files, desc=f"Processing files for {var}", leave=False):
        with h5py.File(file, "r") as f:
            current_data = f['input'][var][:]
            
            # Apply log transformation if var is 'tp6hr'
            if var == 'tp6hr':
                current_data = np.log(current_data + 1e-6)  # Log transformation with offset
            
            tp += current_data
            tp_mean_std[counter, :] = current_data
            counter += 1

    tp = tp * 1000
    tp = tp / 365.25

    # Calculate mean and standard deviation for the current variable
    mean = np.mean(tp_mean_std)
    std = np.std(tp_mean_std)
    
    # Normalize the data for the current variable
    normalized = (tp_mean_std - mean) / std
    normalized_mean = np.mean(normalized)
    normalized_std = np.std(normalized)
    
    # Store results in the dictionary
    results[var] = {
        "mean": mean,
        "std": std,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std
    }
    
    # Print results for the current variable
    print(f"Variable: {var}")
    print("Dataset Mean:", mean)
    print("Dataset STD:", std)
    print("Normalized Mean:", normalized_mean)
    print("Normalized STD:", normalized_std)
# Save all results to a .npz file
np.savez('stats_2000_2003_lr.npz', **results)

# To verify, load and print the saved file
loaded_data = np.load('stats_2000_2003_lr.npz', allow_pickle=True)
for var_name in loaded_data:
    stats = loaded_data[var_name].item()
    print(f"{var_name}: Mean = {stats['mean']}, STD = {stats['std']}, "
          f"Normalized Mean = {stats['normalized_mean']}, Normalized STD = {stats['normalized_std']}")

