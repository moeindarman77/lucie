import glob
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class ClimateDataset(Dataset):
    def __init__(self, input_dir_lr, input_dir_hr, input_vars, output_vars, lr_lats, lr_lons,
                 normalize=False, input_normalization_file=None, output_normalization_file=None):
        """
        Args:
            input_dir_lr (str): Path to the directory containing low-resolution HDF5 files.
            input_dir_hr (str): Path to the directory containing high-resolution HDF5 files.
            input_vars (list): List of variable names to load from the input files.
            output_vars (list): List of variable names to load from the output files.
            lr_lats (list): List of low-resolution latitudes.
            lr_lons (list): List of low-resolution longitudes.
            normalize (bool): Whether to apply normalization.
            input_normalization_file (str): Path to input normalization file (if normalize=True).
            output_normalization_file (str): Path to output normalization file (if normalize=True).
        """
        self.input_dir_lr = input_dir_lr
        self.input_dir_hr = input_dir_hr
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.lr_lats = lr_lats
        self.lr_lons = lr_lons
        self.normalize = normalize

        # Get all files in directories and sort them chronologically
        self.lr_files = sorted(glob.glob(os.path.join(self.input_dir_lr, "*.h5")))  # Hourly files
        self.hr_files = sorted(glob.glob(os.path.join(self.input_dir_hr, "*.h5")))  # 6-hourly files

        # Adjust LR files to every 6th file to align with HR files
        self.lr_files_six_hourly = self.lr_files[::6]
        assert len(self.hr_files) == len(self.lr_files_six_hourly), \
            "Mismatch between high-resolution and low-resolution timesteps."

        # Load normalization parameters if normalization is enabled
        if self.normalize:
            # Load input normalization parameters
            if input_normalization_file:
                input_norm_data = np.load(input_normalization_file, allow_pickle=True)
                self.input_mean_std = {var: (input_norm_data[var].item()['mean'], input_norm_data[var].item()['std']) 
                                       for var in self.input_vars}
            else:
                raise ValueError("Input normalization file path is required when normalization is enabled.")

            # Load output normalization parameters
            if output_normalization_file:
                output_norm_data = np.load(output_normalization_file, allow_pickle=True)
                self.output_mean_std = {var: (output_norm_data[var].item()['mean'], output_norm_data[var].item()['std']) 
                                        for var in self.output_vars}
            else:
                raise ValueError("Output normalization file path is required when normalization is enabled.")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # Load HR data for the current timestep
        hr_file_path = self.hr_files[idx]
        with h5py.File(hr_file_path, 'r') as f:
            hr_data = {var: f[var][:] for var in self.output_vars}
            hr_lats = f['Latitude'][:]
            hr_lons = f['Longitude'][:]

        # Normalize HR data if needed
        if self.normalize:
            hr_data = {var: (hr_data[var] - self.output_mean_std[var][0]) / self.output_mean_std[var][1]
                       for var in self.output_vars}

        # Load corresponding LR data (every 6th LR file matches each HR timestep)
        lr_file_path = self.lr_files_six_hourly[idx]
        with h5py.File(lr_file_path, 'r') as f:
            lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}

        # Normalize LR data if needed
        if self.normalize:
            lr_data = {var: (lr_data[var] - self.input_mean_std[var][0]) / self.input_mean_std[var][1]
                       for var in self.input_vars}

        # Prepare the data to be returned as a dictionary
        data = {
            'input': torch.tensor([lr_data[var] for var in self.input_vars]),  # Input data tensor
            'output': torch.tensor([hr_data[var] for var in self.output_vars]),  # Output data tensor
            'input_vars': self.input_vars,  # List of input variable names
            'output_vars': self.output_vars,
            'lr_lats': self.lr_lats,  # Low-resolution latitudes
            'lr_lons': self.lr_lons,  # Low-resolution longitudes
            'hr_lats': hr_lats,  # High-resolution latitudes
            'hr_lons': hr_lons,  # High-resolution longitudes
        }

        # Include normalization statistics if normalization is enabled
        if self.normalize:
            data['input_stats'] = {var: {'mean': self.input_mean_std[var][0], 'std': self.input_mean_std[var][1]}
                                   for var in self.input_vars}
            data['output_stats'] = {var: {'mean': self.output_mean_std[var][0], 'std': self.output_mean_std[var][1]}
                                    for var in self.output_vars}

        return data
    
# input_dir_lr = '/media/volume/moein-storage-1/lucie/ERA5_t30/'
# input_dir_hr = '/media/volume/moein-storage-1/lucie/ERA5_hr/'
# input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
# output_vars = ['2m_temperature', 'tp6hr']
# input_normalization_file = '/media/volume/moein-storage-1/lucie/stats_2000_2010_lr.npz'
# output_normalization_file = '/media/volume/moein-storage-1/lucie/stats_2000_2010_hr.npz'

# lr_lats = [87.159, 83.479, 79.777, 76.070, 72.362, 68.652, 64.942, 61.232, 
#            57.521, 53.810, 50.099, 46.389, 42.678, 38.967, 35.256, 31.545, 
#            27.833, 24.122, 20.411, 16.700, 12.989, 9.278, 5.567, 1.856, 
#            -1.856, -5.567, -9.278, -12.989, -16.700, -20.411, -24.122, 
#            -27.833, -31.545, -35.256, -38.967, -42.678, -46.389, -50.099, 
#            -53.810, -57.521, -61.232, -64.942, -68.652, -72.362, -76.070, 
#            -79.777, -83.479, -87.159]

# lr_lons = [0.0, 3.75, 7.5, 11.25, 15.0, 18.75, 22.5, 26.25, 30.0, 33.75,
#            37.5, 41.25, 45.0, 48.75, 52.5, 56.25, 60.0, 63.75, 67.5, 71.25,
#            75.0, 78.75, 82.5, 86.25, 90.0, 93.75, 97.5, 101.25, 105.0, 108.75,
#            112.5, 116.25, 120.0, 123.75, 127.5, 131.25, 135.0, 138.75, 142.5, 146.25,
#            150.0, 153.75, 157.5, 161.25, 165.0, 168.75, 172.5, 176.25, 180.0, 183.75,
#            187.5, 191.25, 195.0, 198.75, 202.5, 206.25, 210.0, 213.75, 217.5, 221.25,
#            225.0, 228.75, 232.5, 236.25, 240.0, 243.75, 247.5, 251.25, 255.0, 258.75,
#            262.5, 266.25, 270.0, 273.75, 277.5, 281.25, 285.0, 288.75, 292.5, 296.25,
#            300.0, 303.75, 307.5, 311.25, 315.0, 318.75, 322.5, 326.25, 330.0, 333.75,
#            337.5, 341.25, 345.0, 348.75, 352.5, 356.25]

# # Create the dataset
# dataset = ClimateDataset(input_dir_lr, input_dir_hr, input_vars, output_vars, lr_lats, lr_lons, normalize=True, input_normalization_file=input_normalization_file, output_normalization_file=output_normalization_file) 

# # Access a data sample
# data_sample = dataset[500]
# print(data_sample)

