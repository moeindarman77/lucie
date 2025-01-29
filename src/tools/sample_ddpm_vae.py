import torch
import argparse
import yaml
import os
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from torch.nn import DataParallel
from dataset.ClimateDataset import ClimateDataset
# from dataset.utilities import GlorysRomsDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, cond_data_index, dataset):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    # input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
    # output_vars = ['2m_temperature', 'tp6hr']
    # lr_lats = [87.159, 83.479, 79.777, 76.070, 72.362, 68.652, 64.942, 61.232, 
    #        57.521, 53.810, 50.099, 46.389, 42.678, 38.967, 35.256, 31.545, 
    #        27.833, 24.122, 20.411, 16.700, 12.989, 9.278, 5.567, 1.856, 
    #        -1.856, -5.567, -9.278, -12.989, -16.700, -20.411, -24.122, 
    #        -27.833, -31.545, -35.256, -38.967, -42.678, -46.389, -50.099, 
    #        -53.810, -57.521, -61.232, -64.942, -68.652, -72.362, -76.070, 
    #        -79.777, -83.479, -87.159]

    # lr_lons = [0.0, 3.75, 7.5, 11.25, 15.0, 18.75, 22.5, 26.25, 30.0, 33.75,
    #        37.5, 41.25, 45.0, 48.75, 52.5, 56.25, 60.0, 63.75, 67.5, 71.25,
    #        75.0, 78.75, 82.5, 86.25, 90.0, 93.75, 97.5, 101.25, 105.0, 108.75,
    #        112.5, 116.25, 120.0, 123.75, 127.5, 131.25, 135.0, 138.75, 142.5, 146.25,
    #        150.0, 153.75, 157.5, 161.25, 165.0, 168.75, 172.5, 176.25, 180.0, 183.75,
    #        187.5, 191.25, 195.0, 198.75, 202.5, 206.25, 210.0, 213.75, 217.5, 221.25,
    #        225.0, 228.75, 232.5, 236.25, 240.0, 243.75, 247.5, 251.25, 255.0, 258.75,
    #        262.5, 266.25, 270.0, 273.75, 277.5, 281.25, 285.0, 288.75, 292.5, 296.25,
    #        300.0, 303.75, 307.5, 311.25, 315.0, 318.75, 322.5, 326.25, 330.0, 333.75,
    #        337.5, 341.25, 345.0, 348.75, 352.5, 356.25]
    # dataset = ClimateDataset(input_dir_lr=dataset_config['input_data_dir'], 
    #                          input_dir_hr=dataset_config['output_data_dir'], 
    #                          input_vars=input_vars, 
    #                          output_vars=output_vars, 
    #                          lr_lats=lr_lats, 
    #                          lr_lons=lr_lons,
    #                          year_range=(dataset_config['year_range_start'], dataset_config['year_range_end']),
    #                          normalize=True, 
    #                          input_normalization_file=dataset_config['input_normalization_dir'], 
    #                          output_normalization_file=dataset_config['output_normalization_dir']) 
    if dataset_config['land_sea_mask'] == 1:
        lsm = torch.tensor(np.load(dataset_config['land_sea_mask_dir'])['land_sea_mask']).unsqueeze(0).unsqueeze(1)
    else:
        lsm = None
    
    # dataset = GlorysRomsDataset(steps=range(dataset_config['total_data']),
    #                              channels=["SSU", "SSV", "SSH",], 
    #                              added_channels=[], 
    #                              data_dir=dataset_config["data_dir"],
    #                              lat_lon_keep= tuple(dataset_config["lat_lon_keep"]), 
    #                              interpolator_use="scipy", )

    data = dataset[0]
    lres, hres = data['input'], data['output']
    lres = lres.unsqueeze(0).to(device)
    hres = hres.unsqueeze(0).to(device)

    # Pre-process for VAE part
    lres_upsampled = F.interpolate(lres, size=(721,1440), mode='bilinear', align_corners=True)
    lres_upsampled = F.pad(lres_upsampled, (0, 0, 0, 7), mode='constant', value=0)

    cond_input = lres_upsampled

    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    im_size_x = dataset_config['im_size_x'] // 2**sum(autoencoder_model_config['down_sample'])
    im_size_y = dataset_config['im_size_y'] // 2**sum(autoencoder_model_config['down_sample'])
    # Keep this for batch sampling (maybe later)
    # xt = torch.randn((train_config['num_samples'],
    #                   autoencoder_model_config['z_channels'],
    #                   im_size_x,
    #                   im_size_y)).to(device)

    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      91,
                      180)).to(device)
    
    xt = F.pad(xt, (0, (184 - 180), 0, (96 - 91)), mode='constant', value=0)
    save_count = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond_input=cond_input) # This should be hres according to how I trained the model which is stupid
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        #ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            _, _, encoded_features = vae.encode(lres_upsampled)
            xt = xt[..., :91, :180]
            ims = vae.decode(xt, encoded_features)
            ims = ims[:, :, :-7, :] 
            # Save the samples
            if not os.path.exists(os.path.join(task_dir, 'samples_ddpm')):
                os.mkdir(os.path.join(task_dir, 'samples_ddpm'))
        
            # Convert tensors to numpy arrays
            hres_numpy = hres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            ims = ims.cpu().numpy()   # Shape: [num_samples, batch_size, channels, height, width]

            # Save to an npz file named with the index of the data loader
            save_dir = os.path.join(task_dir,'samples_ddpm', f'ldm_{cond_data_index}.npz')
            np.savez(save_dir, hres=hres_numpy, output=ims)
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
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    

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
    
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(task_dir,
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(task_dir, 
                                                      train_config['ldm_ckpt_name']))['model_state_dict'])
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    vae = VAE(input_channels=dataset_config['input_channels'],
                output_channels=dataset_config['output_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(task_dir,
                                 train_config['vae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        # vae.load_state_dict(torch.load(os.path.join(task_dir,
        #                                             train_config['vae_autoencoder_ckpt_name']),
        #                                map_location=device)['model_state_dict'], strict=True)
        vae.load_state_dict(torch.load(os.path.join(task_dir,
                                                    train_config['vae_autoencoder_ckpt_name']),
                                       map_location=device)['model_state_dict'], strict=True)
    
    input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
    output_vars = ['2m_temperature', 'tp6hr']
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
    dataset = ClimateDataset(input_dir_lr=dataset_config['input_data_dir'], 
                             input_dir_hr=dataset_config['output_data_dir'], 
                             input_vars=input_vars, 
                             output_vars=output_vars, 
                             lr_lats=lr_lats, 
                             lr_lons=lr_lons,
                             year_range=(dataset_config['year_range_start'], dataset_config['year_range_end']),
                             normalize=True, 
                             input_normalization_file=dataset_config['input_normalization_dir'], 
                             output_normalization_file=dataset_config['output_normalization_dir']) 
    
    with torch.no_grad():
        # for cond_data_index in range(0, dataset_config['test_data_beg']+train_config['num_samples']):
        for cond_data_index in range(0, len(dataset)):
            sample(model, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae, cond_data_index, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/glor_config.yaml', type=str)
    args = parser.parse_args()
    infer(args)
