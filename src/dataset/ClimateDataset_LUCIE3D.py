import os
import h5py
import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class ClimateDataset_LUCIE3D(Dataset):
    def __init__(self, lucie3d_file, lsm_path, oro_lr_path, oro_hr_path, input_vars, output_vars, 
                 lr_lats, lr_lons, hr_lats, hr_lons, normalize=False, 
                 input_normalization_file=None, output_normalization_file=None):
        """
        Dataset for Lucie-3D NetCDF files with LSM and orography from separate files.
        
        Args:
            lucie3d_file: Path to Lucie-3D NetCDF file
            lsm_path: Path to land-sea mask file
            oro_lr_path: Path to low-resolution orography file (for FNO input)
            oro_hr_path: Path to high-resolution orography file (for diffusion conditioning)
            input_vars: List of input variable names expected by the model
            output_vars: List of output variable names (for compatibility)
            lr_lats: Low-res latitude coordinates
            lr_lons: Low-res longitude coordinates 
            hr_lats: High-res latitude coordinates for output
            hr_lons: High-res longitude coordinates for output
            normalize: Whether to apply normalization
            input_normalization_file: Path to normalization stats
            output_normalization_file: Path to output normalization stats
        """
        self.lucie3d_file = lucie3d_file
        self.lsm_path = lsm_path
        self.oro_lr_path = oro_lr_path
        self.oro_hr_path = oro_hr_path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.lr_lats = lr_lats
        self.lr_lons = lr_lons
        self.hr_lats = hr_lats
        self.hr_lons = hr_lons
        self.normalize = normalize
        
        # Open NetCDF file
        self.nc_file = nc.Dataset(lucie3d_file, 'r')
        
        # Get time dimension
        self.times = self.nc_file.variables['time'][:]
        self.time_units = self.nc_file.variables['time'].units
        
        # Convert time to datetime objects for timestamps
        base_time = datetime(2000, 1, 1)  # Based on "hours since 2000-01-01 00:00:00"
        self.timestamps = [base_time + timedelta(hours=float(t)) for t in self.times]
        
        # Load static fields
        # Load land-sea mask with normalization stats
        if os.path.exists(lsm_path):
            lsm_data = np.load(lsm_path, allow_pickle=True)
            self.land_sea_mask = lsm_data['land_sea_mask']
            self.lsm_mean = float(lsm_data['land_sea_mask_mean'])
            self.lsm_std = float(lsm_data['land_sea_mask_std'])
        else:
            print(f"Warning: LSM file not found at {lsm_path}, using placeholder")
            self.land_sea_mask = np.ones((48, 96))
            self.lsm_mean = 0.0
            self.lsm_std = 1.0
        
        # Load low-resolution orography with normalization stats (for FNO input)
        if os.path.exists(oro_lr_path):
            oro_lr_data = np.load(oro_lr_path, allow_pickle=True)
            self.orography_lr = oro_lr_data['orography']
            self.oro_lr_mean = float(oro_lr_data['orography_mean'])
            self.oro_lr_std = float(oro_lr_data['orography_std'])
        else:
            print(f"Warning: LR orography file not found at {oro_lr_path}, using placeholder")
            self.orography_lr = np.zeros((48, 96))
            self.oro_lr_mean = 0.0
            self.oro_lr_std = 1.0
        
        # Load high-resolution orography (for diffusion conditioning)
        if os.path.exists(oro_hr_path):
            oro_hr_data = np.load(oro_hr_path, allow_pickle=True)
            self.orography_hr = oro_hr_data['orography']
            self.oro_hr_mean = float(oro_hr_data['orography_mean'])
            self.oro_hr_std = float(oro_hr_data['orography_std'])
        else:
            print(f"Warning: HR orography file not found at {oro_hr_path}, using placeholder")
            self.orography_hr = np.zeros((len(self.hr_lats), len(self.hr_lons)))
            self.oro_hr_mean = 0.0
            self.oro_hr_std = 1.0
        
        print(f"Loaded Lucie-3D dataset with {len(self.times)} time steps")
        print(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        # Load normalization parameters if normalization is enabled
        if self.normalize:
            if input_normalization_file:
                input_norm_data = np.load(input_normalization_file, allow_pickle=True)
                self.input_mean_std = {var: (input_norm_data[var].item()['mean'], input_norm_data[var].item()['std']) 
                                       for var in self.input_vars}
            else:
                raise ValueError("Input normalization file path is required when normalization is enabled.")
            
            # Load output normalization stats for denormalization during inference
            if output_normalization_file:
                output_norm_data = np.load(output_normalization_file, allow_pickle=True)
                self.output_mean_std = {var: (output_norm_data[var].item()['mean'], output_norm_data[var].item()['std']) 
                                        for var in ['2m_temperature', 'u_component_of_wind_83', 'v_component_of_wind_83', 'total_precipitation_6hr']}
            else:
                print("Warning: Output normalization file not provided - denormalization will not be available")
                self.output_mean_std = None

    def __len__(self):
        return len(self.times)
    
    def get_timestamp(self, idx):
        """Get timestamp for a given index"""
        return self.timestamps[idx]
    
    def get_output_normalization_stats(self):
        """Get output normalization stats for denormalization"""
        return self.output_mean_std
    
    def __getitem__(self, idx):
        """
        Load and return a single data sample following ClimateDataset_v2 conventions.
        
        Returns data in the same format as ClimateDataset_v2:
        - Variable order: ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'tp6hr', 'orography', 'land_sea_mask', 'logp']
        - Channels stacked in first dimension
        - Same normalization and scaling
        """
        
        # Load Lucie-3D variables for this timestep
        # Map Lucie-3D variables to expected input variables
        lucie_data = {}
        
        # Temperature at level 7 (index 7)
        lucie_data['Temperature_7'] = self.nc_file.variables['temperature'][idx, 7, :, :]
        
        # Specific humidity at level 7
        lucie_data['Specific_Humidity_7'] = self.nc_file.variables['specific_humidity'][idx, 7, :, :]
        
        # U-wind at level 3 (index 3)
        lucie_data['U-wind_3'] = self.nc_file.variables['u_component_of_wind'][idx, 3, :, :]
        
        # V-wind at level 3
        lucie_data['V-wind_3'] = self.nc_file.variables['v_component_of_wind'][idx, 3, :, :]
        
        # 6-hour precipitation
        lucie_data['tp6hr'] = self.nc_file.variables['tp6hr'][idx, :, :]
        
        # Surface pressure (already in log form)
        lucie_data['logp'] = self.nc_file.variables['surface_pressure'][idx, :, :]
        
        # Add static fields (use LR orography for FNO input)
        lucie_data['orography'] = self.orography_lr
        lucie_data['land_sea_mask'] = self.land_sea_mask
        
        # Apply normalization if enabled
        if self.normalize:
            for var in self.input_vars:
                if var in lucie_data:
                    if var == 'orography':
                        # Normalize orography with its own stats
                        lucie_data[var] = (lucie_data[var] - self.oro_lr_mean) / self.oro_lr_std
                    elif var == 'land_sea_mask':
                        # Normalize land-sea mask with its own stats
                        lucie_data[var] = (lucie_data[var] - self.lsm_mean) / self.lsm_std
                    elif var in self.input_mean_std:
                        # Normalize other variables with input stats
                        mean, std = self.input_mean_std[var]
                        lucie_data[var] = (lucie_data[var] - mean) / std
        
        # Stack variables in the order expected by the model
        input_tensor = torch.tensor(np.array([lucie_data[var] for var in self.input_vars]), dtype=torch.float32)
        
        # Flip latitude dimension to match ClimateDataset_v2 convention (np.flipud)
        input_tensor = torch.flip(input_tensor, dims=[1])
        
        return {
            'input': input_tensor,
            'input_vars': self.input_vars,
            'lr_lats': self.lr_lats,
            'lr_lons': self.lr_lons,
            'hr_lats': self.hr_lats,
            'hr_lons': self.hr_lons,
            'orography_hr': self.orography_hr,  # High-res orography for diffusion conditioning
            'timestamp': self.timestamps[idx],
            'time_index': idx
        }
    
    def __del__(self):
        """Close NetCDF file when dataset is destroyed"""
        if hasattr(self, 'nc_file'):
            self.nc_file.close()