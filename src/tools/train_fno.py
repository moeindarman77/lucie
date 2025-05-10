import yaml
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from models.fno import FNO2d
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils.spectral_sqr_abs2 import spectral_sqr_abs2
from dataset.ClimateDataset import ClimateDataset
import torch.nn.functional as F
import logging

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_model(model, rank):
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model

def create_dataloader(dataset, rank, world_size, batch_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return dataloader

def train(rank, world_size, args):
    setup(rank, world_size)

    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading config file: {exc}")
            return
        
    # Extract the necessary configurations
    dataset_config = config['dataset_params']
    fno_config = config['fno_params']
    train_config = config['train_params']

    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if rank==0 and not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if rank==0 and not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Configure logging
    if rank == 0 and not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(task_dir, 'training.log')),
                logging.StreamHandler()
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
    model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
         output_channels=dataset_config['output_channels'],
           model_config=fno_config,)
    model = create_model(model, rank)
    # model = DistributedDataParallel(model)
    # model = DataParallel(model)
    # model = model.to(device)
    logging.info("Model instantiated.")
#     Input shape: torch.Size([2, 20, 10, 10])
# Weights shape: torch.Size([20, 20, 10, 10, 2])

    # Create dataset instance    
    input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr', 'land_sea_mask', 'orography']
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
    if dataset_config['land_sea_mask'] == 1:
        lsm = torch.tensor(np.load(dataset_config['land_sea_mask_dir'])['land_sea_mask']).unsqueeze(0).unsqueeze(1)
    else:
        lsm = None
    logging.info("Dataset loaded.")

    # Create DataLoaders
    data_loader = create_dataloader(dataset, rank, world_size, train_config['fno_batch_size'])

    # data_loader = DataLoader(dataset, 
    #                          batch_size=train_config['fno_batch_size'], #GPU can't take more than 8
    #                          shuffle=True, 
    #                          num_workers=2) # 2 workers are enough as I incease them don't see any improvement
    logging.info("DataLoader created.")

    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if rank==0 and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    logging.info("Directories verified/created.")

    checkpoint_interval = train_config['checkpoint_interval']
    num_epochs = train_config['fno_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    
    # After optimizer definitions
    optimizer_g = Adam(model.parameters(), lr=train_config['fno_lr'], betas=(0.5, 0.999))
    
    #Scheduler
    scheduler_step_size_g = train_config.get('scheduler_step_size_g', 10)
    scheduler_gamma_g = train_config.get('scheduler_gamma_g', 0.1)

    # scheduler_g = StepLR(optimizer_g, step_size=scheduler_step_size_g, gamma=scheduler_gamma_g)
    scheduler_g = ReduceLROnPlateau(optimizer_g, 'min', factor=0.1, min_lr=1e-5, threshold=1e-2, patience=5)
    
    # Initialize flags
    optimizer_g_has_stepped = False

    # Load the model if load_from_checkpoint is True
    if train_config['load_from_checkpoint']:
        # Load the FNO checkpoint
        fno_checkpoint_path = train_config["fno_resume_ckpt_path"]
        fno_checkpoint = torch.load(fno_checkpoint_path, map_location=device)
        
        # Load the fno model state
        model.load_state_dict(fno_checkpoint['model_state_dict'])
        
        # Load the optimizer state for the FNO model
        optimizer_g.load_state_dict(fno_checkpoint['optimizer_state_dict'])
        logging.info(f"FNO model and optimizer loaded from checkpoint: {fno_checkpoint}")
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    accumulation_steps = train_config['fno_accumulation_steps']
    
    logging.info("Starting training loop.")
    best_epoch_loss = float('inf')

    for epoch_idx in range(num_epochs):
        recon_losses = []
        losses = []
        accumulation_counter = 0
        optimizer_g.zero_grad()
        data_loader.sampler.set_epoch(epoch_idx)

        for batch_idx, data in enumerate(tqdm(data_loader, disable=(rank != 0))):
            lres, hres = data['input'], data['output']
            lres = lres.float().to(rank)
            hres = hres.float().to(rank)
            
            lres_upsampled = F.interpolate(lres, size=(721,1440), mode='bicubic', align_corners=True)
            lsm_expanded = None
            if lsm is not None:
                lsm = lsm.float().to(rank)
                lsm_expanded = lsm.expand(lres.shape[0], -1, -1, -1)  # Shape: (batch_size, 1, 721, 1440)
                lres_upsampled_cat = torch.cat([lres_upsampled, lsm_expanded], dim=1)  # Shape: (batch_size, input_channel+1, 721, 1440)

            output = model(x=lres_upsampled)
            
            # Adjusting output
            # output[:, 0] += lres_upsampled[:, 0] 
            # output[:, -1] += lres_upsampled[:, -3]
            
            ######### Optimize FNO ##########
            # 'L2 Loss + spectral loss'
            recon_loss = spectral_sqr_abs2(output, hres, lambda_fft=train_config['lambda_fft']) # recon_loss = recon_criterion(output, hres) 
            recon_losses.append(recon_loss.item())

            recon_loss = recon_loss / accumulation_steps
            g_loss = (recon_loss )

            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
        
            # Accumulate gradients
            accumulation_counter += 1

            # Optimizer steps
            if accumulation_counter % accumulation_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                optimizer_g_has_stepped = True

        # Handle remaining gradients at the end of the epoch
        if accumulation_counter % accumulation_steps != 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
            optimizer_g_has_stepped = True

        # Logging
        current_lr_g = scheduler_g.get_last_lr()[0]
        epoch_loss = torch.tensor(np.mean(recon_losses), device=rank)  # Convert loss to tensor
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)  # Synchronize across GPUs
        epoch_loss = epoch_loss.item()  # Convert back to scalar
        logging.info(
                f'Epoch {epoch_idx + 1} | Recon Loss: {epoch_loss:.4f}| Learning Rate: {current_lr_g:.4e}' 
            )
        
        ##### Save checkpoints at defined intervals ########
        if (epoch_idx + 1) % checkpoint_interval == 0 or (epoch_idx + 1) == num_epochs:
            # Use the base checkpoint names from the config and append the epoch number
            base_fno_ckpt_name = train_config['fno_ckpt_name']
            
            # Construct filenames with epoch number for periodic checkpoint saving
            fno_ckpt_name = f"{os.path.splitext(base_fno_ckpt_name)[0]}_epoch_{epoch_idx + 1}.pth"
            
            # Save the periodic model checkpoints with epoch number
            if rank==0:
                torch.save({
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                }, os.path.join(checkpoint_dir, fno_ckpt_name))

                logging.info(f"Checkpoint saved at interval epoch {epoch_idx + 1}")

        ###### Save the latest model checkpoint, overwriting each time ######
        latest_fno_ckpt_name = os.path.join(checkpoint_dir, 'latest_fno.pth')

        if rank==0:
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
            }, latest_fno_ckpt_name)

        logging.info(f"Latest model checkpoint saved at epoch {epoch_idx + 1}")

        # Save the model that has the best recon loss among all epochs:



        if rank == 0 and epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_fno_ckpt_name = os.path.join(checkpoint_dir, 'best_fno.pth')
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
            }, best_fno_ckpt_name)
            logging.info(f"Best model saved at epoch {epoch_idx + 1} with loss {best_epoch_loss:.4f}")
                
        # Step the schedulers after optimizer steps, only if optimizer has stepped
        if optimizer_g_has_stepped:
            scheduler_g.step(epoch_loss)
        
    ###### After training is complete, save the final models using the base names #####
    if rank==0:
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
        }, os.path.join(task_dir, train_config['fno_ckpt_name']))

        logging.info("Final model saved.")
        logging.info('Done Training...')

if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Arguments for FNO training')
        parser.add_argument('--config', dest='config_path', default='config/mnist.yaml', type=str)
        args = parser.parse_args()
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
        # train(args)