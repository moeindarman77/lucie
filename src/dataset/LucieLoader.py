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
        self.lucie_vars = input_vars  # Assuming we want to use the same vars
        self.data = np.load(lucie_file_path)['data'][:, :, :, :len(input_vars)]

        # Apply log transform only to inverse pressure (assumed second-to-last)
        self.data[..., -2] = np.log(self.data[..., -2] * 1e-5)

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

# # --- Usage Example ---
# lucie_file_path = "/glade/derecho/scratch/mdarman/lucie/LUCIE_inference_start2010.npz"
# normalization_file = "/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009_updated.npz"
# lucie_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']

# # Create dataset and dataloader
# dataset = LucieDataset(
#     lucie_file_path=lucie_file_path,
#     normalization_file=normalization_file,
#     input_vars=lucie_vars,
#     normalize=True
# )

# dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# # --- Iterate over DataLoader ---
# for batch in dataloader:
#     print(batch.shape)  # Expected shape: (batch_size, C, H, W)
#     break