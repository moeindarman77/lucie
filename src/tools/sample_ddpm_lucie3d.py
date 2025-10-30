"""
Sample DDPM for Lucie-3D downscaling

This script performs inference on Lucie-3D outputs to generate timestamped, 
downscaled NetCDF files using a trained diffusion model.

Required environment variables/paths:
- INPUT_NC: Path to Lucie-3D NetCDF file
- LSM_PATH: Path to land-sea mask file  
- ORO_LR_PATH: Path to low-resolution orography file (for FNO input)
- ORO_HR_PATH: Path to high-resolution orography file (for diffusion conditioning)
- OUT_NC: Output NetCDF file path
- MODEL_CKPT: Path to trained DDPM checkpoint

Example usage:
export INPUT_NC="/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_3D/LUCIE_co2_nosst_nomask_range_2000_2020.nc"
export LSM_PATH="/glade/derecho/scratch/mdarman/lucie/lsm_lr.npz"
export ORO_LR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_lr.npz"
export ORO_HR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_hr.npz"
export OUT_NC="/path/to/output_downscaled.nc"
export MODEL_CKPT="/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/checkpoints/best_ldm.pth"

python sample_ddpm_lucie3d.py --config config/ERA5_config_final_v2.yaml --start 0 --end 100
"""

import torch
import argparse
import yaml
import os
import netCDF4 as nc
import numpy as np
from tqdm import tqdm
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from models.fno import FNO2d
from models.simple_unet_final_v2 import SimpleUnet
from dataset.ClimateDataset_LUCIE3D import ClimateDataset_LUCIE3D
import torch.nn.functional as F
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample_and_save_netcdf(model, scheduler, SR_model, dataset, output_nc_path, 
                          start_idx=0, end_idx=None):
    """
    Sample from the model and save results to NetCDF format
    """
    if end_idx is None:
        end_idx = len(dataset)
    
    # Get output normalization stats for denormalization
    output_norm_stats = dataset.get_output_normalization_stats()
    
    # Initialize lists to store outputs
    all_outputs = []
    all_timestamps = []
    
    print(f"Processing {end_idx - start_idx} samples from index {start_idx} to {end_idx-1}")
    
    for idx in tqdm(range(start_idx, end_idx), desc="Sampling"):
        data = dataset[idx]
        lres = data['input'].unsqueeze(0).to(device)
        timestamp = data['timestamp']
        
        # Get high-resolution orography and normalize it
        hr_orography_raw = data['orography_hr']
        hr_orography_normalized = (hr_orography_raw - dataset.oro_hr_mean) / dataset.oro_hr_std
        hr_orography = torch.tensor(hr_orography_normalized).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, HR_H, HR_W]
        
        # Pre-process for VAE part - upsample to high-res
        lres_upsampled = F.interpolate(lres, size=(721, 1440), mode='bicubic', align_corners=True)
        
        # Get FNO output
        fno_output = SR_model(lres_upsampled)
        
        # Prepare conditioning input (FNO output + high-res orography)
        cond_input = torch.cat([fno_output, hr_orography], dim=1)
        
        # Target variables: [2m_temperature, u_wind, v_wind, tp6hr] 
        # Corresponding to indices [0, 2, 3, 4] from full output
        target_shape = (1, 4, 721, 1440)  # 4 variables we want to reconstruct
        
        # Start with random noise for diffusion sampling
        xt = torch.randn(target_shape).to(device)
        
        # Diffusion sampling loop
        for i in reversed(range(1000)):  # Assuming 1000 timesteps
            # Get noise prediction
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond=cond_input)
            
            # Use scheduler to get next sample
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
            # Check for NaN
            if torch.isnan(xt).any():
                print(f"NaN detected at timestep {i} for sample {idx}")
                break
        
        # Final output
        final_output = xt.cpu().numpy()  # Shape: [1, 4, 721, 1440]
        
        # Denormalize the output if normalization stats are available
        if output_norm_stats is not None:
            # Map model output channels to variable names
            output_var_names = ['2m_temperature', 'u_component_of_wind_83', 'v_component_of_wind_83', 'total_precipitation_6hr']
            
            for i, var_name in enumerate(output_var_names):
                if var_name in output_norm_stats:
                    mean, std = output_norm_stats[var_name]
                    final_output[0, i, :, :] = final_output[0, i, :, :] * std + mean
        
        all_outputs.append(final_output[0])  # Remove batch dimension
        all_timestamps.append(timestamp)
    
    # Stack all outputs
    all_outputs = np.stack(all_outputs, axis=0)  # Shape: [time, channels, lat, lon]
    
    # Save to NetCDF
    save_to_netcdf(all_outputs, all_timestamps, output_nc_path, dataset)
    
    return all_outputs


def save_to_netcdf(outputs, timestamps, output_path, dataset):
    """
    Save outputs to NetCDF format with proper coordinates and attributes
    """
    print(f"Saving results to {output_path}")
    
    # Create output NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        ncfile.createDimension('time', len(timestamps))
        ncfile.createDimension('latitude', len(dataset.hr_lats))
        ncfile.createDimension('longitude', len(dataset.hr_lons))
        
        # Create coordinate variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
        lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
        
        # Set coordinate values
        # Convert timestamps to hours since 2000-01-01
        base_time = datetime(2000, 1, 1)
        time_values = [(ts - base_time).total_seconds() / 3600.0 for ts in timestamps]
        time_var[:] = time_values
        lat_var[:] = dataset.hr_lats
        lon_var[:] = dataset.hr_lons
        
        # Set coordinate attributes
        time_var.units = 'hours since 2000-01-01 00:00:00'
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        
        lat_var.units = 'degrees_north'
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'latitude'
        
        lon_var.units = 'degrees_east'
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'longitude'
        
        # Create data variables
        # Map model outputs to Lucie-3D variable names
        var_names = ['2m_temperature', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'total_precipitation_6hr']
        var_long_names = ['2 metre temperature', '10 metre U wind component', '10 metre V wind component', '6-hour total precipitation']
        var_units = ['K', 'm s**-1', 'm s**-1', 'm']
        
        for i, (var_name, long_name, units) in enumerate(zip(var_names, var_long_names, var_units)):
            var = ncfile.createVariable(var_name, 'f4', ('time', 'latitude', 'longitude'), 
                                       zlib=True, complevel=4)
            var[:] = outputs[:, i, :, :]
            var.units = units
            var.long_name = long_name
            var.standard_name = var_name
        
        # Global attributes
        ncfile.description = 'Downscaled climate data from Lucie-3D using diffusion model'
        ncfile.history = f'Created on {datetime.now().isoformat()}'
        ncfile.source = 'LUCIE downscaling model with DDPM'
        ncfile.Conventions = 'CF-1.7'
    
    print(f"Successfully saved {len(timestamps)} timesteps to {output_path}")


def infer(args):
    # Read environment variables
    input_nc = os.getenv('INPUT_NC', '/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_3D/LUCIE_co2_nosst_nomask_range_2000_2020.nc')
    lsm_path = os.getenv('LSM_PATH', '/glade/derecho/scratch/mdarman/lucie/lsm_lr.npz')
    oro_lr_path = os.getenv('ORO_LR_PATH', '/glade/derecho/scratch/mdarman/lucie/orography_lr.npz')
    oro_hr_path = os.getenv('ORO_HR_PATH', '/glade/derecho/scratch/mdarman/lucie/orography_hr.npz')
    output_nc = os.getenv('OUT_NC', 'downscaled_lucie3d_output.nc')
    model_ckpt = os.getenv('MODEL_CKPT', '/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/checkpoints/best_ldm.pth')
    
    print(f"Input NC: {input_nc}")
    print(f"Output NC: {output_nc}")
    print(f"Model checkpoint: {model_ckpt}")
    
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params'] 
    train_config = config['train_params']
    fno_config = config['fno_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Create and load the diffusion model
    model = SimpleUnet()
    model = model.to(device)
    model.eval()
    
    if os.path.exists(model_ckpt):
        checkpoint = torch.load(model_ckpt)
        print('Loaded DDPM checkpoint')
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        raise Exception(f'DDPM checkpoint not found: {model_ckpt}')
    
    # Create and load the SR model (FNO)
    SR_model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
        output_channels=dataset_config['output_channels'],
        model_config=fno_config,
    )
    SR_model = SR_model.to(device)
    SR_model.eval()
    
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
    
    # Freeze SR model parameters
    for param in SR_model.parameters():
        param.requires_grad = False
    
    # Define coordinates
    lr_lats = [87.159, 83.479, 79.777, 76.070, 72.362, 68.652, 64.942, 61.232, 
               57.521, 53.810, 50.099, 46.389, 42.678, 38.967, 35.256, 31.545, 
               27.833, 24.122, 20.411, 16.700, 12.989, 9.278, 5.567, 1.856, 
               -1.856, -5.567, -9.278, -12.989, -16.700, -20.411, -24.122, 
               -27.833, -31.545, -35.256, -38.967, -42.678, -46.389, -50.099, 
               -53.810, -57.521, -61.232, -64.942, -68.652, -72.362, -76.070, 
               -79.777, -83.479, -87.159]
    
    lr_lons = [0.0, 3.75, 7.5, 11.25, 15.0, 18.75, 22.5, 26.25, 30.0, 33.75,
               37.5, 41.25, 45.0, 48.75, 52.5, 56.25, 60.0, 63.75, 67.5, 71.25,
               75.0, 78.75, 82.5, 86.25, 90.0, 93.75, 97.5, 101.25, 105.0, 108.75,
               112.5, 116.25, 120.0, 123.75, 127.5, 131.25, 135.0, 138.75, 142.5, 146.25,
               150.0, 153.75, 157.5, 161.25, 165.0, 168.75, 172.5, 176.25, 180.0, 183.75,
               187.5, 191.25, 195.0, 198.75, 202.5, 206.25, 210.0, 213.75, 217.5, 221.25,
               225.0, 228.75, 232.5, 236.25, 240.0, 243.75, 247.5, 251.25, 255.0, 258.75,
               262.5, 266.25, 270.0, 273.75, 277.5, 281.25, 285.0, 288.75, 292.5, 296.25,
               300.0, 303.75, 307.5, 311.25, 315.0, 318.75, 322.5, 326.25, 330.0, 333.75,
               337.5, 341.25, 345.0, 348.75, 352.5, 356.25]
    
    # High-resolution coordinates (ERA5 0.25 degree grid)
    hr_lats = np.arange(90, -90.25, -0.25)  # 90 to -90 with 0.25 degree spacing
    hr_lons = np.arange(0, 360, 0.25)       # 0 to 360 with 0.25 degree spacing
    
    # Define input variables in expected order
    input_vars = [
        'Temperature_7',
        'Specific_Humidity_7', 
        'U-wind_3', 
        'V-wind_3', 
        'tp6hr', 
        'orography',
        'land_sea_mask', 
        'logp', 
    ]
    
    # Create dataset
    dataset = ClimateDataset_LUCIE3D(
        lucie3d_file=input_nc,
        lsm_path=lsm_path,
        oro_lr_path=oro_lr_path,
        oro_hr_path=oro_hr_path,
        input_vars=input_vars,
        output_vars=None,  # Not needed for inference
        lr_lats=lr_lats,
        lr_lons=lr_lons,
        hr_lats=hr_lats,
        hr_lons=hr_lons,
        normalize=True,
        input_normalization_file=dataset_config['input_normalization_dir'],
        output_normalization_file=dataset_config['output_normalization_dir']
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Run inference
    with torch.no_grad():
        outputs = sample_and_save_netcdf(
            model=model,
            scheduler=scheduler, 
            SR_model=SR_model,
            dataset=dataset,
            output_nc_path=output_nc,
            start_idx=args.start_index,
            end_idx=args.end_index
        )
    
    print(f"Inference completed. Results saved to {output_nc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM Lucie-3D downscaling')
    parser.add_argument('--config', dest='config_path',
                        default='config/ERA5_config_final_v2.yaml', type=str)
    parser.add_argument('--start', dest='start_index',
                        default=0, type=int,
                        help='Start index for sampling')
    parser.add_argument('--end', dest='end_index',
                        default=None, type=int,
                        help='End index for sampling (None for all)')
    args = parser.parse_args()
    infer(args)