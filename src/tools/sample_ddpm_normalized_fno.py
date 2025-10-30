import torch
import argparse
import yaml
import os
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.cosine_noise_scheduler import CosineNoiseScheduler
from torch.nn import DataParallel
# from models.fno_ddpm import FNO2d as FNO_DDPM
from models.fno import FNO2d
from models.simple_unet_final_v2 import SimpleUnet
from dataset.LucieLoader_10yr import LucieLoader
# from dataset.utilities import GlorysRomsDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
                diffusion_config, dataset_config, SR_model, cond_data_index, lucie_dataset,
                orography_lr, land_sea_mask, orography_hr, fno_mean, fno_std):


    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions

    This version processes ONLY LUCIE data (no ERA5 dataset needed)
    FNO outputs are NORMALIZED before being used as conditioning

    Args:
        orography_lr: Normalized low-res orography tensor (1, 1, 48, 96)
        land_sea_mask: Normalized low-res land_sea_mask tensor (1, 1, 48, 96)
        orography_hr: Normalized high-res orography tensor (1, 1, 721, 1440)
        fno_mean: Scalar mean for FNO outputs (1, 4, 1, 1)
        fno_std: Scalar std for FNO outputs (1, 4, 1, 1)
    """

    # Load LUCIE data
    lucie = lucie_dataset[cond_data_index].unsqueeze(0).to(device)

    # Step 2: Reorder lucie to match order (swap tp6hr and logp)
    # Original lucie_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
    lucie = torch.cat([
        lucie[:, :4, :, :],     # ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3']
        lucie[:, 5:6, :, :],    # 'tp6hr'
        lucie[:, 4:5, :, :]     # 'logp'
    ], dim=1)

    # Step 3: Insert orography and land_sea_mask after tp6hr (i.e., at index 5)
    lucie_aligned = torch.cat([
        lucie[:, :5, :, :],     # up to 'tp6hr'
        orography_lr,           # low-res orography
        land_sea_mask,          # low-res land_sea_mask
        lucie[:, 5:, :, :]      # 'logp'
    ], dim=1)

    # Process LUCIE data through FNO
    lucie_upsampled = F.interpolate(lucie_aligned, size=(721,1440), mode='bicubic', align_corners=True)
    fno_output_lucie = SR_model(lucie_upsampled)

    # ============ NORMALIZE FNO OUTPUT ============
    # Normalize FNO output to zero mean and unit std using SCALAR statistics
    # fno_output_lucie shape: (1, 4, 721, 1440)
    # fno_mean/std shape: (1, 4, 1, 1) - broadcasted across batch and spatial dims
    fno_output_normalized = (fno_output_lucie - fno_mean) / (fno_std + 1e-6)
    # ==============================================

    # Use normalized FNO output as conditioning (concatenate with high-res orography)
    # Use the pre-loaded high-res orography for diffusion model conditioning
    cond_input = torch.cat([fno_output_normalized, orography_hr], dim=1)

    # We want to reconstruct only: 2m temperature, u and v wind components and tp6hr
    # This matches the 4 output channels from FNO
    target_shape = fno_output_lucie.shape

    # Define the results directory
    results_dir = train_config['results_dir']
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Start with random noise
    xt = torch.randn_like(fno_output_lucie).to(device)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):

        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond=cond_input)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Check if xt is NaN
        if torch.isnan(xt).any():
            print(f"NaN detected at timestep {i}")
            break

        # Save final output at timestep 0
        if i == 0:
            ims = xt
            # Create output directory
            output_dir = os.path.join(task_dir, 'samples_normalized_fno')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Convert tensors to numpy arrays
            ims = ims.cpu().numpy()
            fno_output_lucie_numpy = fno_output_lucie.cpu().numpy()

            # Save only output and fno_output as requested
            save_path = os.path.join(output_dir, f'{cond_data_index+1}.npz')
            np.savez(save_path,
                    output=ims,
                    fno_output=fno_output_lucie_numpy
                    )
            print(f'Saved LUCIE sample {cond_data_index+1}')

def infer(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ddpm_fno_config = config['ddpm_fno_params']
    train_config = config['train_params']
    fno_config = config['fno_params']

    # Load FNO output normalization statistics (SCALAR)
    fno_stats_path = dataset_config.get('fno_normalization_file',
                                        '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz')
    print(f"Loading FNO normalization statistics from {fno_stats_path}")

    with np.load(fno_stats_path, allow_pickle=True) as stats:
        # Load SCALAR mean and std for each channel (matching original normalization format)
        fno_mean_temp = float(stats['temperature'].item()['mean'])
        fno_std_temp = float(stats['temperature'].item()['std'])
        fno_mean_uwind = float(stats['uwind'].item()['mean'])
        fno_std_uwind = float(stats['uwind'].item()['std'])
        fno_mean_vwind = float(stats['vwind'].item()['mean'])
        fno_std_vwind = float(stats['vwind'].item()['std'])
        fno_mean_precip = float(stats['precipitation'].item()['mean'])
        fno_std_precip = float(stats['precipitation'].item()['std'])

        # Create tensors for normalization (4,) - scalars
        fno_mean = torch.tensor([fno_mean_temp, fno_mean_uwind, fno_mean_vwind, fno_mean_precip], dtype=torch.float32)
        fno_std = torch.tensor([fno_std_temp, fno_std_uwind, fno_std_vwind, fno_std_precip], dtype=torch.float32)

    # Move to GPU and reshape for broadcasting: (1, 4, 1, 1)
    fno_mean = fno_mean.view(1, 4, 1, 1).to(device)
    fno_std = fno_std.view(1, 4, 1, 1).to(device)

    print(f"FNO normalization stats loaded (scalar). Values: mean={fno_mean.squeeze()}, std={fno_std.squeeze()}")

    # Define the results directory
    results_dir = train_config['results_dir']
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Create the diffusion model
    model = SimpleUnet()
    model = model.to(device)
    model.eval()

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters in diffusion model: {total_params:,}")
    print(f"Trainable parameters in diffusion model: {trainable_params:,}")

    # Load checkpoint from the normalized FNO task directory
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_ldm.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f'Loaded normalized FNO diffusion model checkpoint from epoch {checkpoint["epoch"]}')
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        raise Exception(f'Diffusion model checkpoint not found at {checkpoint_path}')

    # Create the SR model (FNO)
    SR_model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
        output_channels=dataset_config['output_channels'],
        model_config=fno_config,)
    SR_model = SR_model.to(device)
    SR_model.eval()

    # Count parameters in SR_model (FNO)
    fno_total_params = sum(p.numel() for p in SR_model.parameters())
    fno_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)
    print(f"Total parameters in SR model (FNO): {fno_total_params:,}")
    print(f"Trainable parameters in SR model (FNO): {fno_trainable_params:,}")

    SR_model_load_dir = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v1/checkpoints/best_fno.pth"
    if os.path.exists(SR_model_load_dir):
        checkpoint = torch.load(SR_model_load_dir)
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        SR_model.load_state_dict(new_state_dict)
    else:
        raise Exception('SR model checkpoint not found')

    # Freeze the parameters of the SR model
    for param in SR_model.parameters():
        param.requires_grad = False

    # Load LUCIE dataset - use direct numpy loading since file has 'data' key only
    print("Loading LUCIE data...")
    lucie_file = np.load(dataset_config['lucie_file_path'])
    lucie_data = lucie_file['data']  # Shape: (14600, 48, 96, 6)
    print(f"LUCIE data shape: {lucie_data.shape}")

    # Apply log transform to pressure (index 4 = logp)
    lucie_data[..., 4] = np.log(lucie_data[..., 4] * 1e-5)

    # Normalize using stats
    input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
    norm_data = np.load(dataset_config['input_normalization_dir'], allow_pickle=True)
    for i, var in enumerate(input_vars):
        mean = norm_data[var].item()['mean']
        std = norm_data[var].item()['std']
        lucie_data[:, :, :, i] = (lucie_data[:, :, :, i] - mean) / std

    # Convert to tensor dataset (flip vertically and permute to C, H, W)
    lucie_tensor = torch.tensor(lucie_data, dtype=torch.float32).permute(0, 3, 1, 2).flip(2)

    class SimpleTensorDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    lucie_dataset = SimpleTensorDataset(lucie_tensor)
    print(f"LUCIE dataset length: {len(lucie_dataset)}")

    # Load orography and land_sea_mask once (they're constant for all samples)
    print("\nLoading static fields (orography and land_sea_mask)...")
    oro_lr_data = np.load('/glade/derecho/scratch/mdarman/lucie/orography_lr.npz')
    oro_hr_data = np.load('/glade/derecho/scratch/mdarman/lucie/oro.npz')
    lsm_hr_data = np.load('/glade/derecho/scratch/mdarman/lucie/lsm.npz')

    # Load low-res orography (48, 96) for FNO input
    orography_raw = oro_lr_data['orography']
    orography_lr = torch.from_numpy(orography_raw).unsqueeze(0).unsqueeze(0).float().to(device)

    # Load high-res land_sea_mask and downsample to low-res (48, 96)
    lsm_hr = lsm_hr_data['land_sea_mask']
    lsm_hr_torch = torch.from_numpy(lsm_hr).unsqueeze(0).unsqueeze(0).float()
    land_sea_mask = F.interpolate(lsm_hr_torch, size=(48, 96), mode='bilinear', align_corners=True).to(device)

    # Normalize low-res fields using stats from normalization file
    norm_data = np.load(dataset_config['input_normalization_dir'], allow_pickle=True)
    oro_mean, oro_std = norm_data['orography'].item()['mean'], norm_data['orography'].item()['std']
    lsm_mean, lsm_std = norm_data['land_sea_mask'].item()['mean'], norm_data['land_sea_mask'].item()['std']

    orography_lr = (orography_lr - oro_mean) / oro_std
    land_sea_mask = (land_sea_mask - lsm_mean) / lsm_std

    # Load high-res orography (721, 1440) for diffusion conditioning
    # This is already normalized in the file
    orography_hr_raw = oro_hr_data['geopotential_normalized']
    orography_hr = torch.from_numpy(orography_hr_raw).unsqueeze(0).unsqueeze(0).float().to(device)

    print(f"  LR Orography shape: {orography_lr.shape}, mean: {orography_lr.mean():.4f}, std: {orography_lr.std():.4f}")
    print(f"  HR Orography shape: {orography_hr.shape}, mean: {orography_hr.mean():.4f}, std: {orography_hr.std():.4f}")
    print(f"  Land-sea mask shape: {land_sea_mask.shape}, mean: {land_sea_mask.mean():.4f}, std: {land_sea_mask.std():.4f}")

    with torch.no_grad():
        # Generate samples based on start_index and num_samples
        end_index = min(args.start_index + args.num_samples, len(lucie_dataset))
        for cond_data_index in range(args.start_index, end_index):
            sample(model, scheduler, train_config, ddpm_fno_config,
                 diffusion_config, dataset_config, SR_model, cond_data_index, lucie_dataset,
                 orography_lr, land_sea_mask, orography_hr, fno_mean, fno_std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for LUCIE sampling with normalized FNO')
    parser.add_argument('--config', dest='config_path',
                        default='config/ERA5_config_normalized_fno.yaml', type=str)
    parser.add_argument('--start', dest='start_index',
                        default=0, type=int,
                        help='Start index for sampling')
    parser.add_argument('--num_samples', dest='num_samples',
                        default=10, type=int,
                        help='Number of samples to generate')
    args = parser.parse_args()
    infer(args)
