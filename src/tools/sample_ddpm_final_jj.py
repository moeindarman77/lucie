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
from models.simple_unet_final import SimpleUnet
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset
# from dataset.utilities import GlorysRomsDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
                diffusion_config, dataset_config, SR_model, cond_data_index, dataset):

    
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    
    data = dataset[cond_data_index]
    lres, hres = data['input'], data['output']
    lres, hres = lres.unsqueeze(0).to(device), hres.unsqueeze(0).to(device)

    # Pre-process for VAE part
    lres_upsampled = F.interpolate(lres, size=(721,1440), mode='bicubic', align_corners=True)
    lres_upsampled[:,5] = hres[:,5]
    fno_output = SR_model(lres_upsampled)
    fno_output[:, 5] = hres[:, 5]
    cond_input = fno_output

    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Start with the FNO output but this can be changed to pure noise
    xt = torch.randn_like(fno_output[:,:2]).to(device)
    
    save_count = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
    # for i in tqdm(reversed(range(T))):
        
        # Get prediction of noise
        # The differnece here is that I am using the fno output as the input to the model instead of pure noise (xt)
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond=cond_input) 
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Check if xt is NaN
        if torch.isnan(xt).any():
            print(f"NaN detected at timestep {i}")
            break

        # Save x0
        #ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            ims = xt
            # Save the samples
            if not os.path.exists(os.path.join(task_dir, 'samples_ddpm')):
                os.mkdir(os.path.join(task_dir, 'samples_ddpm'))
        
            # Convert tensors to numpy arrays
            lres_numpy = lres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            lres_interp_numpy = lres_upsampled.cpu().numpy()
            hres_numpy = hres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            ims = ims.cpu().numpy()   # Shape: [num_samples, batch_size, channels, height, width]
            fno_output = fno_output.cpu().numpy()

            # Save to an npz file named with the index of the data loader
            save_dir = os.path.join(task_dir,'samples_ddpm', f'{cond_data_index+1}.npz')
            np.savez(save_dir,
                    lres=lres_numpy,
                    lres_interp=lres_interp_numpy,
                    hres=hres_numpy, 
                    output=ims,
                    fno_output=fno_output
                    )
            print(f'Saved sample {cond_data_index+1}')
        else:
            ims = xt

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ddpm_fno_config = config['ddpm_fno_params']

    train_config = config['train_params']
    fno_config = config['fno_params']


    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)


    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    # scheduler = CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                    #  )
    
    # Create the FNO model for the noise prediction
    model = SimpleUnet()
    model= model.to(device)
    model.eval()
    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters in diffusion model: {total_params:,}")
    print(f"Trainable parameters in diffusion model: {trainable_params:,}")
    # checkpoint_path = '/glade/derecho/scratch/mdarman/lucie/results/unet_simple_v0/checkpoints/best_ldm.pth' # Best results so far
    checkpoint_path = '/glade/derecho/scratch/mdarman/lucie/results/unet_final_v4/checkpoints/best_ldm.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('Loaded FNO-DDPM checkpoint')
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k.replace("module.", "")  # Remove "module." from key names
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        raise Exception('FNO-DDPM checkpoint not found')
   
    # Create the SR model
    SR_model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
         output_channels=dataset_config['output_channels'],
           model_config=fno_config,)
    # move the model to the device
    SR_model = SR_model.to(device)
    # SR_model = create_model(SR_model, rank)
    SR_model.eval()
    # Count parameters in SR_model (FNO)
    fno_total_params = sum(p.numel() for p in SR_model.parameters())
    fno_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)
    print(f"Total parameters in SR model (FNO): {fno_total_params:,}")
    print(f"Trainable parameters in SR model (FNO): {fno_trainable_params:,}")
    SR_model_load_dir = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v0/checkpoints/best_fno.pth"

    if os.path.exists(SR_model_load_dir):
        checkpoint = torch.load(SR_model_load_dir)
        # Remove "module." from key names
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k.replace("module.", "")  # Remove "module." from key names
            new_state_dict[new_key] = v
        SR_model.load_state_dict(new_state_dict)
    else:
        raise Exception('SR model checkpoint not found')
    
    # Freeze the parameters of the SR model
    for param in SR_model.parameters():
        param.requires_grad = False
    
    # Define the input and output variables
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
    output_vars = [
    "2m_temperature",
    "specific_humidity_133",
    "u_component_of_wind_83",
    "v_component_of_wind_83",
    "total_precipitation_6hr",
    "geopotential_at_surface",
    ]
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
    dataset = ClimateDataset(
                            input_dir_lr=dataset_config['input_data_dir'], 
                            input_dir_hr=dataset_config['output_data_dir'], 
                            input_vars=input_vars, 
                            output_vars=output_vars, 
                            lr_lats=lr_lats, 
                            lr_lons=lr_lons,
                            year_range=(dataset_config['year_range_start'], dataset_config['year_range_end']),
                            normalize=True, 
                            input_normalization_file=dataset_config['input_normalization_dir'], 
                            output_normalization_file=dataset_config['output_normalization_dir'],
                            cache_file=dataset_config['cache_file'],
                            force_recompute=dataset_config['force_recompute']
                            )
    
    with torch.no_grad():
        # for cond_data_index in range(0, dataset_config['test_data_beg']+train_config['num_samples']):
        for cond_data_index in range(len(dataset)):
            sample(model, scheduler, train_config, ddpm_fno_config,
                 diffusion_config, dataset_config, SR_model, cond_data_index, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/glor_config.yaml', type=str)
    args = parser.parse_args()
    infer(args)
