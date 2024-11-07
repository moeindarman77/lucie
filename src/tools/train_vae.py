import yaml
from torch.nn import DataParallel
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from models.vae import VAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from dataset.utilities import GlorysRomsDataset, spectral_sqr_abs2
from dataset.ClimateDataset import ClimateDataset
import torch.nn.functional as F
import logging

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):

    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading config file: {exc}")
            return
        
    # Extract the necessary configurations
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler(os.path.join(os.path.join(task_dir, 'training.log'))),  # Log to a file
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info(config)
    logging.info("Configuration loaded successfully.")

    # Save the config file in the task directory
    config_save_path = os.path.join(task_dir, 'config.yaml')
    with open(config_save_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    logging.info(f"Configuration saved to {config_save_path}")
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Seed set to {seed}")

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VAE(input_channels=dataset_config['input_channels'],
                output_channels=dataset_config['output_channels'],
                model_config=autoencoder_config)
    model = DataParallel(model)
    model = model.to(device)
    logging.info("Model instantiated.")

    # Create dataset instance    
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
    logging.info("Dataset loaded.")

    # Create DataLoader
    data_loader = DataLoader(dataset, 
                             batch_size=train_config['autoencoder_batch_size'], #GPU can't take more than 8
                             shuffle=True, 
                             num_workers=2) # 2 workers are enough as I incease them don't see any improvement
    logging.info("DataLoader created.")

    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    logging.info("Directories verified/created.")

    checkpoint_interval = train_config['checkpoint_interval']
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can be BCEWithLogitsLoss or MSELoss
    disc_criterion = torch.nn.MSELoss()
    
    # discriminator = Discriminator(im_channels=dataset_config['output_channels']).to(device)

    discriminator = Discriminator(im_channels=dataset_config['output_channels'])
    discriminator = DataParallel(discriminator)
    discriminator = discriminator.to(device)

    
    # After optimizer definitions
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    #Scheduler
    scheduler_step_size_g = train_config.get('scheduler_step_size_g', 10)
    scheduler_gamma_g = train_config.get('scheduler_gamma_g', 0.1)

    scheduler_step_size_d = train_config.get('scheduler_step_size_d', 10)
    scheduler_gamma_d = train_config.get('scheduler_gamma_d', 0.1)

    scheduler_g = StepLR(optimizer_g, step_size=scheduler_step_size_g, gamma=scheduler_gamma_g)
    scheduler_d = StepLR(optimizer_d, step_size=scheduler_step_size_d, gamma=scheduler_gamma_d)

    # Initialize flags
    optimizer_g_has_stepped = False
    optimizer_d_has_stepped = False

    # Load the model if load_from_checkpoint is True
    if train_config['load_from_checkpoint']:
        # Load the VAE autoencoder checkpoint
        vae_checkpoint_path = train_config["vae_autoencoder_resume_ckpt_path"]
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        
        # Load the VAE model state
        model.load_state_dict(vae_checkpoint['model_state_dict'])
        
        # Load the optimizer state for the VAE autoencoder (generator)
        optimizer_g.load_state_dict(vae_checkpoint['optimizer_state_dict'])
        logging.info(f"VAE autoencoder model and optimizer loaded from checkpoint: {vae_checkpoint_path}")

        # Load the discriminator checkpoint
        discriminator_checkpoint_path = train_config["vae_discriminator_resume_ckpt_path"]
        discriminator_checkpoint = torch.load(discriminator_checkpoint_path, map_location=device)
        
        # Load the discriminator model state
        discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
        
        # Load the optimizer state for the discriminator
        optimizer_d.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
        logging.info(f"Discriminator model and optimizer loaded from checkpoint: {discriminator_checkpoint_path}")
    
    disc_epoch_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    accumulation_steps = train_config['autoencoder_accumulation_steps']
    # Ensure it's at least 1
    kl_anneal_epochs = max(train_config['kl_anneal_epochs'], 1)  
    
    logging.info("Starting training loop.")

    for epoch_idx in range(num_epochs):
        recon_losses = []
        disc_losses = []
        gen_losses = []
        kl_losses =[]
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        accumulation_counter = 0
        # Calculate annealed beta for KL divergence
        current_beta = min(train_config['kl_end'], train_config['kl_start'] + (epoch_idx / kl_anneal_epochs) * (train_config['kl_end'] - train_config['kl_start']))

        for batch_idx, data in enumerate(tqdm(data_loader)):
            step_count += 1
            lres, hres = data['input'], data['output']
            lres = lres.float().to(device)
            hres = hres.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(lres)
            output, latent_distribution = model_output # For VAE,    
             # latent_distribution is in the shape of [batch_size, 2, padded_lres//2^(N_downsampling_layers), padded_lres//2^(N_downsampling_layers)]                            
            
            ######### Optimize Generator ##########
            # 'L2 Loss + spectral loss'
            recon_loss = spectral_sqr_abs2(output, hres[...,:-1,:]) # recon_loss = recon_criterion(output, hres) 
            # recon_loss = recon_criterion(output, hres[...,:-1,:])
            recon_losses.append(recon_loss.item())

            # recon_loss = recon_loss / accumulation_steps
            # g_loss = (recon_loss )

            # KL divergence loss
            mu, logvar = torch.chunk(latent_distribution, 2, dim=1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / lres.size(0)  # Normalize by batch size
            kl_losses.append(kl_loss.item())

            # Total generator loss
            # g_loss = (recon_loss + train_config['kl_weight'] * kl_loss) / accumulation_steps
            g_loss = (recon_loss + current_beta * kl_loss) / accumulation_steps

            # Adversarial loss only if disc_epoch_start steps passed
            if epoch_idx > disc_epoch_start:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / accumulation_steps

            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if epoch_idx > disc_epoch_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(hres)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / accumulation_steps
                disc_loss.backward()

            # Accumulate gradients
            accumulation_counter += 1

            # Optimizer steps
            if accumulation_counter % accumulation_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                optimizer_g_has_stepped = True
                if epoch_idx > disc_epoch_start:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                    optimizer_d_has_stepped = True

        # Handle remaining gradients at the end of the epoch
        if accumulation_counter % accumulation_steps != 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
            optimizer_g_has_stepped = True
            if epoch_idx > disc_epoch_start:
                optimizer_d.step()
                optimizer_d.zero_grad()
                optimizer_d_has_stepped = True

        # Logging
        if len(disc_losses) > 0:
            logging.info(
                f'Epoch {epoch_idx + 1} | Recon Loss: {np.mean(recon_losses):.4f} | '
                f'KL Loss: {np.mean(kl_losses):.4f} | Gen Loss: {np.mean(gen_losses):.4f} | '
                f'Disc Loss: {np.mean(disc_losses):.4f}'
            )
        else:
            logging.info(
                f'Epoch {epoch_idx + 1} | Recon Loss: {np.mean(recon_losses):.4f} | '
                f'KL Loss: {np.mean(kl_losses):.4f}'
            )
        current_lr_g = scheduler_g.get_last_lr()[0]
        current_lr_d = scheduler_d.get_last_lr()[0]
        logging.info(f"Epoch {epoch_idx + 1} | Generator LR: {current_lr_g:.4e} | Discriminator LR: {current_lr_d:.4e}")
        
        ##### Save checkpoints at defined intervals ########
        if (epoch_idx + 1) % checkpoint_interval == 0 or (epoch_idx + 1) == num_epochs:
            # Use the base checkpoint names from the config and append the epoch number
            base_autoencoder_ckpt_name = train_config['vae_autoencoder_ckpt_name']
            base_discriminator_ckpt_name = train_config['vae_discriminator_ckpt_name']
            
            # Construct filenames with epoch number for periodic checkpoint saving
            autoencoder_ckpt_name = f"{os.path.splitext(base_autoencoder_ckpt_name)[0]}_epoch_{epoch_idx + 1}.pth"
            discriminator_ckpt_name = f"{os.path.splitext(base_discriminator_ckpt_name)[0]}_epoch_{epoch_idx + 1}.pth"
            
            # Save the periodic model checkpoints with epoch number
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
            }, os.path.join(checkpoint_dir, autoencoder_ckpt_name))

            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_d.state_dict(),
            }, os.path.join(checkpoint_dir, discriminator_ckpt_name))

            logging.info(f"Checkpoint saved at interval epoch {epoch_idx + 1}")

        ###### Save the latest model checkpoint, overwriting each time ######
        latest_autoencoder_ckpt_name = os.path.join(checkpoint_dir, 'latest_autoencoder.pth')
        latest_discriminator_ckpt_name = os.path.join(checkpoint_dir, 'latest_discriminator.pth')

        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
        }, latest_autoencoder_ckpt_name)

        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
        }, latest_discriminator_ckpt_name)

        logging.info(f"Latest model checkpoint saved at epoch {epoch_idx + 1}")

        # Step the schedulers after optimizer steps, only if optimizer has stepped
        if optimizer_g_has_stepped:
            scheduler_g.step()
        if optimizer_d_has_stepped:
            scheduler_d.step()

    ###### After training is complete, save the final models using the base names #####
    torch.save({
        'epoch': epoch_idx + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_g.state_dict(),
    }, os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name']))

    torch.save({
        'epoch': epoch_idx + 1,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_d.state_dict(),
    }, os.path.join(task_dir, train_config['vae_discriminator_ckpt_name']))

    logging.info("Final model saved.")
    logging.info('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)