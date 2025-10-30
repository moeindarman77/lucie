import glob
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class ClimateDataset_v2(Dataset):
    def __init__(self, input_dir_lr, input_dir_hr, input_vars, output_vars, lr_lats, lr_lons,
                 year_range=None, normalize=False, input_normalization_file=None, output_normalization_file=None,
                 cache_file="/glade/derecho/scratch/mdarman/lucie/valid_files.npz", force_recompute=False):
        self.input_dir_lr = input_dir_lr
        self.input_dir_hr = input_dir_hr
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.lr_lats = lr_lats
        self.lr_lons = lr_lons
        self.normalize = normalize
        self.year_range = year_range
        self.cache_file = cache_file  # Cache file for storing valid files
        self.force_recompute = force_recompute  # If True, force recalculation of valid files

        # Get all files in directories and sort them chronologically
        self.lr_files = sorted(glob.glob(os.path.join(self.input_dir_lr, "*.h5")))  # Hourly files
        self.hr_files = sorted(glob.glob(os.path.join(self.input_dir_hr, "*.h5")))  # 6-hourly files

        # Filter files based on the year range
        if self.year_range:
            start_year, end_year = self.year_range
            self.lr_files = [f for f in self.lr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]
            self.hr_files = [f for f in self.hr_files if start_year <= int(os.path.basename(f).split('_')[0]) <= end_year]

        # Adjust LR files to every 6th file to align with HR files
        self.lr_files_six_hourly = self.lr_files[::6]        

        # Load or compute valid file lists
        self.load_or_compute_valid_files()

        assert len(self.hr_files) == len(self.lr_files_six_hourly), \
            "Mismatch between high-resolution and low-resolution timesteps."
        print(f"Found {len(self.hr_files)} valid files.")
        print(f"Found {len(self.lr_files_six_hourly)} valid files.")

        # Load normalization parameters if normalization is enabled
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

    def load_or_compute_valid_files(self):
        """Loads valid file lists from cache or computes them if needed."""
        if os.path.exists(self.cache_file) and not self.force_recompute:
            print(f"Loading valid files from {self.cache_file}...")
            data = np.load(self.cache_file, allow_pickle=True)
            self.lr_files_six_hourly = list(data["valid_lr_files"])
            self.hr_files = list(data["valid_hr_files"])
        else:
            print("Computing valid files (this may take time)...")
            valid_lr_files = []
            valid_hr_files = []
            
            for lr_file, hr_file in zip(self.lr_files_six_hourly, self.hr_files):
                try:
                    # Check HR data for NaNs
                    with h5py.File(hr_file, 'r') as f:
                        hr_data = {var: f['input'][var][:] for var in self.output_vars}
                        if any(np.isnan(hr_data[var]).any() for var in self.output_vars):
                            continue  # Skip file if NaNs are present

                    # Check LR data for NaNs
                    with h5py.File(lr_file, 'r') as f:
                        lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}
                        if any(np.isnan(lr_data[var]).any() for var in self.input_vars):
                            continue  # Skip file if NaNs are present

                    # Add valid files
                    valid_lr_files.append(lr_file)
                    valid_hr_files.append(hr_file)

                except Exception as e:
                    print(f"Error reading {lr_file} or {hr_file}: {e}")

            # Save valid files
            np.savez(self.cache_file, valid_lr_files=valid_lr_files, valid_hr_files=valid_hr_files)
            print(f"Valid file list saved to {self.cache_file}")

            # Update instance variables
            self.lr_files_six_hourly = valid_lr_files
            self.hr_files = valid_hr_files

        # Use limited data for debugging
        # self.hr_files = self.hr_files[:256]
        # self.lr_files_six_hourly = self.lr_files_six_hourly[:256]

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        """Load and return a single data sample"""
        hr_file_path = self.hr_files[idx]
        with h5py.File(hr_file_path, 'r') as f:
            hr_data = {var: f['input'][var][:] for var in self.output_vars}

        if self.normalize:
            hr_data = {var: (hr_data[var] - self.output_mean_std[var][0]) / self.output_mean_std[var][1]
                       for var in self.output_vars}

        lr_file_path = self.lr_files_six_hourly[idx]
        with h5py.File(lr_file_path, 'r') as f:
            lr_data = {var: np.flipud(f['input'][var][:]) for var in self.input_vars}

        if self.normalize:
            lr_data = {var: (lr_data[var] - self.input_mean_std[var][0]) / self.input_mean_std[var][1]
                       for var in self.input_vars}

        return {
            'input': torch.tensor(np.array([lr_data[var] for var in self.input_vars])),
            'output': torch.tensor(np.array([hr_data[var] for var in self.output_vars])),
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'lr_lats': self.lr_lats,
            'lr_lons': self.lr_lons
        }
