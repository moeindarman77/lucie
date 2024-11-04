import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.utilities import GlorysRomsDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, cond_data_index):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    dataset = GlorysRomsDataset(steps=range(dataset_config['total_data']),
                                 channels=["SSU", "SSV", "SSH",], 
                                 added_channels=[], 
                                 data_dir=dataset_config["data_dir"],
                                 lat_lon_keep= tuple(dataset_config["lat_lon_keep"]), 
                                 interpolator_use="scipy", )

    (lres, hres) = dataset[cond_data_index]
    lres = lres.unsqueeze(0).to(device)
    hres = hres.unsqueeze(0).to(device)

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
                      im_size_x,
                      im_size_y)).to(device)
    xt = F.pad(xt, (0, (80 - 75), 0, (48 - 43)), mode='constant', value=0)
    save_count = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond_input=lres) # This should be hres according to how I trained the model which is stupid
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        #ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            lres_upsampled = F.pad(lres, (0, 3, 1, 1), mode='constant', value=0) # lres_upsampled = F.interpolate(lres, size=(352, 608), mode='bilinear', align_corners=False)
            _, _, encoded_features = vae.encode(lres_upsampled)
            xt = xt[..., :43, :75]
            ims = vae.decode(xt, encoded_features)
            ims = ims[..., 1:-1, :-3]
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
    
    vae = VAE(im_channels=dataset_config['im_channels'],
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
    with torch.no_grad():
        for cond_data_index in range(dataset_config["test_data_beg"], dataset_config['test_data_beg']+train_config['num_samples']):
            sample(model, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae, cond_data_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/glor_config.yaml', type=str)
    args = parser.parse_args()
    infer(args)
