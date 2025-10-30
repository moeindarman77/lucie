import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LucieLoader(Dataset):
    def __init__(self,
                 lucie_file_path,
                 normalization_file,
                 input_vars,
                 normalize=True):
        self.normalize = normalize
        self.input_vars = input_vars

        # Load the new LUCIE file format with separate arrays
        # New format has keys: 'temperature', 'humidity', 'u_wind', 'v_wind', 'pressure', 'precipitation'
        # We need to map input_vars to the file keys and stack them
        lucie_data = np.load(lucie_file_path)

        # Mapping from input_vars to LUCIE file keys
        var_mapping = {
            'Temperature_7': 'temperature',
            'Specific_Humidity_7': 'humidity',
            'U-wind_3': 'u_wind',
            'V-wind_3': 'v_wind',
            'logp': 'pressure',
            'tp6hr': 'precipitation'
        }

        # Stack the variables in the order specified by input_vars
        # Expected input_vars: ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
        data_list = []
        for var in input_vars:
            if var in var_mapping:
                data_list.append(lucie_data[var_mapping[var]])
            else:
                raise ValueError(f"Unknown variable: {var}")

        # Stack along last axis to create shape (time, H, W, C)
        self.data = np.stack(data_list, axis=-1)  # Shape: (14600, 48, 96, 6)

        # Apply log transform to pressure (logp is at index 4, which is -2 in the list)
        # Find the index of 'logp' in input_vars
        if 'logp' in input_vars:
            logp_idx = input_vars.index('logp')
            self.data[..., logp_idx] = np.log(self.data[..., logp_idx] * 1e-5)

        # Load normalization stats
        if self.normalize:
            norm_data = np.load(normalization_file, allow_pickle=True)
            self.input_mean_std = {
                var: (
                    norm_data[var].item()['mean'],
                    norm_data[var].item()['std']
                ) for var in self.input_vars
            }
            for i, var in enumerate(self.input_vars):
                mean, std = self.input_mean_std[var]
                self.data[:, :, :, i] = (self.data[:, :, :, i] - mean) / std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]  # shape: (H, W, C)
        tensor = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1)
        tensor = tensor.flip(1)  # flip vertically
        return tensor
