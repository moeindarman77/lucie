import glob
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# class LucieDataset(Dataset):
#     def __init__(self, input_dir_lr, input_dir_hr, input_vars, output_vars, lr_lats, lr_lons,
#                  year_range=None, normalize=False, input_normalization_file=None, 
#                  output_normalization_file=None, lucie_file_path=None, lucie_vars=None):
#         """
#         Initialize the ClimateDataset with integrated LucieDataset functionality.

#         Args:
#             input_dir_lr (str): Directory for LR files.
#             input_dir_hr (str): Directory for HR files.
#             input_vars (list): List of input variable names.
#             output_vars (list): List of output variable names.
#             lr_lats (list): List of LR latitudes.
#             lr_lons (list): List of LR longitudes.
#             year_range (tuple): Optional range of years to filter files.
#             normalize (bool): Whether to normalize the data.
#             input_normalization_file (str): Path to input normalization file.
#             output_normalization_file (str): Path to output normalization file.
#             lucie_file_path (str): Path to the Lucie dataset .npz file.
#             lucie_vars (list): List of variable names in the Lucie dataset.
#         """
#         self.input_dir_lr = input_dir_lr
#         self.input_dir_hr = input_dir_hr
#         self.input_vars = input_vars
#         self.output_vars = output_vars
#         self.lr_lats = lr_lats
#         self.lr_lons = lr_lons
#         self.normalize = normalize
#         self.year_range = year_range

#         # Get all files in directories and sort them chronologically
#         self.lr_files = sorted(glob.glob(os.path.join(self.input_dir_lr, "*.h5")))  # Hourly files
#         self.hr_files = sorted(glob.glob(os.path.join(self.input_dir_hr, "*.h5")))  # 6-hourly files

#         # Filter files based on the year range
#         if self.year_range:
#             start_year, end_year = self.year_range
#             self.lr_files = [f for f in self.lr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]
#             self.hr_files = [f for f in self.hr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]

#         # Adjust LR files to every 6th file to align with HR files
#         self.lr_files_six_hourly = self.lr_files[::6]

#         # Use limited data for debugging
#         self.hr_files = self.hr_files[:64]
#         self.lr_files_six_hourly = self.lr_files_six_hourly[:64]

#         assert len(self.hr_files) == len(self.lr_files_six_hourly), \
#             "Mismatch between high-resolution and low-resolution timesteps."

#         print(len(self.hr_files), len(self.lr_files_six_hourly))

#         # Pre-filter files to remove those with NaN values
#         valid_lr_files = []
#         valid_hr_files = []
#         for lr_file, hr_file in zip(self.lr_files_six_hourly, self.hr_files):
#             try:
#                 # Load HR data and check for NaNs
#                 with h5py.File(hr_file, 'r') as f:
#                     hr_data = {var: f[var][:] for var in self.output_vars}
#                     if any(np.isnan(hr_data[var]).any() for var in self.output_vars):
#                         continue  # Skip this file if NaNs are present

#                 # Load LR data and check for NaNs
#                 with h5py.File(lr_file, 'r') as f:
#                     lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}
#                     if any(np.isnan(lr_data[var]).any() for var in self.input_vars):
#                         continue  # Skip this file if NaNs are present

#                 # If no NaNs are found, add to valid files list
#                 valid_lr_files.append(lr_file)
#                 valid_hr_files.append(hr_file)

#             except Exception as e:
#                 print(f"Error reading files {lr_file} or {hr_file}: {e}")
        
#         self.lr_files_six_hourly = valid_lr_files
#         self.hr_files = valid_hr_files

#         print(f"Found {len(self.hr_files)} valid files.")
#         print(f"Found {len(self.lr_files_six_hourly)} valid files.")

#         # Load normalization parameters if normalization is enabled
#         if self.normalize:
#             # Load input normalization parameters
#             if input_normalization_file:
#                 input_norm_data = np.load(input_normalization_file, allow_pickle=True)
#                 self.input_mean_std = {var: (input_norm_data[var].item()['mean'], input_norm_data[var].item()['std']) 
#                                        for var in self.input_vars}
#             else:
#                 raise ValueError("Input normalization file path is required when normalization is enabled.")

#             # Load output normalization parameters
#             if output_normalization_file:
#                 output_norm_data = np.load(output_normalization_file, allow_pickle=True)
#                 self.output_mean_std = {var: (output_norm_data[var].item()['mean'], output_norm_data[var].item()['std']) 
#                                         for var in self.output_vars}
#             else:
#                 raise ValueError("Output normalization file path is required when normalization is enabled.")

#         # Load Lucie dataset if provided
#         self.lucie_data = None
#         if lucie_file_path and lucie_vars:
#             lucie_dataset = np.load(lucie_file_path)
#             self.lucie_data = lucie_dataset['data'][:, :, :, :len(lucie_vars)]  # Use only specified variables
#             self.lucie_vars = lucie_vars

#     def __len__(self):
#         return len(self.hr_files)

#     def __getitem__(self, idx):
#         # Load HR data for the current timestep
#         hr_file_path = self.hr_files[idx]
#         with h5py.File(hr_file_path, 'r') as f:
#             hr_data = {var: f[var][:] for var in self.output_vars}
#             hr_lats = f['Latitude'][:]
#             hr_lons = f['Longitude'][:]

#         if "tp6hr" in hr_data:
#             hr_data["tp6hr"] = np.log(hr_data["tp6hr"] + 1e-6)  # Adding a small constant to avoid log(0)

#         # Normalize HR data if needed
#         if self.normalize:
#             hr_data = {var: (hr_data[var] - self.output_mean_std[var][0]) / self.output_mean_std[var][1]
#                        for var in self.output_vars}

#         # Load corresponding LR data (every 6th LR file matches each HR timestep)
#         lr_file_path = self.lr_files_six_hourly[idx]
#         with h5py.File(lr_file_path, 'r') as f:
#             lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}

#         if "tp6hr" in lr_data:
#             lr_data["tp6hr"] = np.log(lr_data["tp6hr"] + 1e-6)  # Adding a small constant to avoid log(0)

#         # Normalize LR data if needed
#         if self.normalize:
#             lr_data = {var: (lr_data[var] - self.input_mean_std[var][0]) / self.input_mean_std[var][1]
#                        for var in self.input_vars}

#         # Prepare LR and HR data as tensors
#         input_tensor = torch.tensor(np.array([lr_data[var] for var in self.input_vars]), dtype=torch.float32)
#         output_tensor = torch.tensor(np.array([hr_data[var] for var in self.output_vars]), dtype=torch.float32)

#         # Add Lucie data for the same timestep (every 6th hour)
#         lucie_tensor = None
#         if self.lucie_data is not None:
#             lucie_sample = self.lucie_data[idx]
#             lucie_tensor = torch.tensor(lucie_sample, dtype=torch.float32).permute(2, 0, 1)

#         # Return combined data
#         data = {
#             'input': input_tensor,
#             'output': output_tensor,
#             'lucie': lucie_tensor,
#             'input_vars': self.input_vars,
#             'output_vars': self.output_vars,
#             'lucie_vars': self.lucie_vars if self.lucie_data is not None else None,
#             'lr_lats': self.lr_lats,
#             'lr_lons': self.lr_lons,
#             'hr_lats': hr_lats,
#             'hr_lons': hr_lons
#         }

#         return data

class LucieDataset(Dataset):
    def __init__(self, input_dir_lr, input_dir_hr, input_vars, output_vars, lr_lats, lr_lons,
                 year_range=None, normalize=False, input_normalization_file=None, 
                 output_normalization_file=None, lucie_file_path=None, lucie_vars=None):
        """
        Initialize the ClimateDataset with integrated LucieDataset functionality.

        Args:
            input_dir_lr (str): Directory for LR files.
            input_dir_hr (str): Directory for HR files.
            input_vars (list): List of input variable names.
            output_vars (list): List of output variable names.
            lr_lats (list): List of LR latitudes.
            lr_lons (list): List of LR longitudes.
            year_range (tuple): Optional range of years to filter files.
            normalize (bool): Whether to normalize the data.
            input_normalization_file (str): Path to input normalization file.
            output_normalization_file (str): Path to output normalization file.
            lucie_file_path (str): Path to the Lucie dataset .npz file.
            lucie_vars (list): List of variable names in the Lucie dataset.
        """
        self.input_dir_lr = input_dir_lr
        self.input_dir_hr = input_dir_hr
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.lr_lats = lr_lats
        self.lr_lons = lr_lons
        self.normalize = normalize
        self.year_range = year_range

        # Load and process LR and HR files
        self.lr_files = sorted(glob.glob(os.path.join(self.input_dir_lr, "*.h5")))
        self.hr_files = sorted(glob.glob(os.path.join(self.input_dir_hr, "*.h5")))
        if self.year_range:
            start_year, end_year = self.year_range
            self.lr_files = [f for f in self.lr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]
            self.hr_files = [f for f in self.hr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]
        self.lr_files_six_hourly = self.lr_files[::6]
        self.hr_files = self.hr_files[:64]
        self.lr_files_six_hourly = self.lr_files_six_hourly[:64]
        assert len(self.hr_files) == len(self.lr_files_six_hourly), \
            "Mismatch between high-resolution and low-resolution timesteps."

        # Pre-filter files to remove those with NaN values
        valid_lr_files = []
        valid_hr_files = []
        for lr_file, hr_file in zip(self.lr_files_six_hourly, self.hr_files):
            try:
                with h5py.File(hr_file, 'r') as f:
                    hr_data = {var: f[var][:] for var in self.output_vars}
                    if any(np.isnan(hr_data[var]).any() for var in self.output_vars):
                        continue
                with h5py.File(lr_file, 'r') as f:
                    lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}
                    if any(np.isnan(lr_data[var]).any() for var in self.input_vars):
                        continue
                valid_lr_files.append(lr_file)
                valid_hr_files.append(hr_file)
            except Exception as e:
                print(f"Error reading files {lr_file} or {hr_file}: {e}")
        
        self.lr_files_six_hourly = valid_lr_files
        self.hr_files = valid_hr_files

        # Load normalization parameters if needed
        if self.normalize:
            if input_normalization_file:
                input_norm_data = np.load(input_normalization_file, allow_pickle=True)
                self.input_mean_std = {var: (input_norm_data[var].item()['mean'], input_norm_data[var].item()['std']) 
                                       for var in self.input_vars}
            else:
                raise ValueError("Input normalization file path is required when normalization is enabled.")
            if output_normalization_file:
                output_norm_data = np.load(output_normalization_file, allow_pickle=True)
                self.output_mean_std = {var: (output_norm_data[var].item()['mean'], output_norm_data[var].item()['std']) 
                                        for var in self.output_vars}
            else:
                raise ValueError("Output normalization file path is required when normalization is enabled.")

        # Load Lucie dataset
        self.lucie_data = None
        if lucie_file_path and lucie_vars:
            lucie_dataset = np.load(lucie_file_path)
            self.lucie_data = lucie_dataset['data'][-1460:, :, :, :len(lucie_vars)]
            self.lucie_vars = lucie_vars

            # Normalize Lucie data if required
            if self.normalize:
                for i, var in enumerate(self.lucie_vars):
                    mean, std = self.input_mean_std[var]  # Use LR stats for normalization
                    self.lucie_data[:, :, :, i] = (self.lucie_data[:, :, :, i] - mean) / std

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # Load HR data for the current timestep
        hr_file_path = self.hr_files[idx]
        with h5py.File(hr_file_path, 'r') as f:
            hr_data = {var: f[var][:] for var in self.output_vars}
        if self.normalize:
            hr_data = {var: (hr_data[var] - self.output_mean_std[var][0]) / self.output_mean_std[var][1]
                       for var in self.output_vars}

        # Load LR data
        lr_file_path = self.lr_files_six_hourly[idx]
        with h5py.File(lr_file_path, 'r') as f:
            lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}
        if self.normalize:
            lr_data = {var: (lr_data[var] - self.input_mean_std[var][0]) / self.input_mean_std[var][1]
                       for var in self.input_vars}

        # Prepare Lucie data
        lucie_tensor = None
        if self.lucie_data is not None:
            lucie_sample = self.lucie_data[idx]
            lucie_tensor = torch.tensor(lucie_sample, dtype=torch.float32).permute(2, 0, 1)

        # Prepare tensors for LR and HR data
        input_tensor = torch.tensor(np.array([lr_data[var] for var in self.input_vars]), dtype=torch.float32)
        output_tensor = torch.tensor(np.array([hr_data[var] for var in self.output_vars]), dtype=torch.float32)

        # Combine and return data
        data = {
            'input': input_tensor,
            'output': output_tensor,
            'lucie': lucie_tensor,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'lucie_vars': self.lucie_vars if self.lucie_data is not None else None,
        }
        return data