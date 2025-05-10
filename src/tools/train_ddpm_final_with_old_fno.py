import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset
from torch.utils.data import DataLoader
from models.fno import FNO2d
from models.simple_unet import SimpleUnet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.cosine_noise_scheduler import CosineNoiseScheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from utils.config_utils import *
from utils.diffusion_utils import *
import torch.nn.functional as F
import logging

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
    # ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    return ddp_model

def create_dataloader(dataset, rank, world_size, batch_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return dataloader

def train(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    ########################
    
    dataset_config = config['dataset_params']
    diffusion_config = config['diffusion_params']
    fno_config = config['fno_params']
    ddpm_fno_config = config['ddpm_fno_params']
    train_config = config['train_params']
    
    # Define the results directory #
    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if rank==0 and not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if rank==0 and not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if rank==0 and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # Configure logging
    if rank == 0 and not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            handlers=[
                logging.FileHandler(os.path.join(os.path.join(task_dir, 'training_ldm.log'))),  # Log to a file
                logging.StreamHandler()  # Log to console
            ]
        )
    logging.info(config)
    logging.info("Configuration loaded successfully.")

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    # scheduler = CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                    #  )
    ###############################################
    # Create dataset instance    
    input_vars = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr', 'land_sea_mask', 'orography']
    output_vars = ['2m_temperature', 'total_precipitation_6hr', 'geopotential_at_surface']
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
    logging.info("Dataset loaded.")

    # Create DataLoader
    data_loader = create_dataloader(dataset, rank, world_size, train_config['ldm_batch_size'])

    # data_loader = DataLoader(dataset, 
    #                          batch_size=train_config['ldm_batch_size'], #GPU can't take more than 8
    #                          shuffle=True, 
    #                          num_workers=2) # 2 workers are enough as I incease them don't see any improvement
    logging.info("DataLoader created.")

    # Create the SR model
    SR_model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
        output_channels=dataset_config['output_channels'],
        model_config=fno_config,)
    
    # # move the model to the device
    SR_model = create_model(SR_model, rank)
    SR_model.eval()
    SR_model_load_dir = "/glade/derecho/scratch/mdarman/lucie/results/fno_v2/checkpoints/best_fno.pth"
    # checkpoint = torch.load(SR_model_load_dir)
    checkpoint = torch.load(SR_model_load_dir, map_location=lambda storage, loc: storage.cuda(rank), weights_only=True)
    SR_model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Instantiate the unet model
    model = SimpleUnet()
    model = create_model(model, rank)
    # model = DataParallel(model)
    # model = model.to(device)
    # checkpoint = torch.load("/glade/derecho/scratch/mdarman/lucie/results/vae_concat_v6/checkpoints/best_ldm_bb.pth")
    # model.module.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    scheduler_optimizer = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
    
    criterion = torch.nn.MSELoss()
  
    # Define the path to your checkpoint file
    # resume_checkpoint = "/glade/derecho/scratch/mdarman/lucie/results/unet_final_v0/checkpoints/ckpt.pth"
    resume_checkpoint = ''
    start_epoch = 0  # Default starting epoch

    if os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=lambda storage, loc: storage.cuda(rank), weights_only=True)
        
        # For a DistributedDataParallel model, load state dict into model.module
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore the optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Set the starting epoch to resume training from
        start_epoch = checkpoint['epoch']
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, starting training from scratch.")

    optimizer_has_stepped = False

    best_epoch_loss = float('inf')
    
    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []

        for batch_idx, data in enumerate(tqdm(data_loader, disable=(rank != 0))):

            # Extract inputs and move to device
            lres, hres = data['input'], data['output']
            lres, hres = lres.float().to(rank), hres.float().to(rank)

            # Upsample low-resolution input
            lres_upsampled = F.interpolate(lres, size=(721,1440), mode='bicubic', align_corners=True)

            fno_output = SR_model(lres_upsampled) # Using old FNO with the correctd input
            cond_input = fno_output  
            cond_input = torch.cat([cond_input, hres[:,2:3]], dim=1)  # Concate the orography to the condition just like old setup.
            hres = hres[:,0:2]  # Only keep the first two channels of hres

            # Zero grad
            optimizer.zero_grad()
            
            ##########################
            # Sample random noiseg
            noise = torch.randn_like(hres).to(rank)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (hres.shape[0],)).to(rank)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(hres, noise, t)

            # Forward pass
            noise_pred = model(noisy_im, t, cond=cond_input)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # scheduler_optimizer.step()

            optimizer_has_stepped = True

        current_lr = scheduler_optimizer.get_last_lr()[0]
        epoch_loss = torch.tensor(np.mean(losses), device=rank)  # Convert loss to tensor
        # dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)  #no Synchronize across GPUs
        dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.AVG)
        
        epoch_loss = epoch_loss.item()  # Convert back to scalar
        logging.info(f'Epoch [{epoch_idx + 1}/{num_epochs}], Learning Rate: {current_lr:.4e} Loss: {epoch_loss:.4f}')
        if (epoch_idx + 1) % train_config['ldm_save_interval'] == 0:
            checkpoint_dir = os.path.join(task_dir, 'checkpoints')
            # Use the base checkpoint names from the config and append the epoch number
            base_ldm_ckpt_name = train_config['ldm_ckpt_name']
            
            # Construct filenames with epoch number for periodic checkpoint saving
            ldm_ckpt_name = f"{os.path.splitext(base_ldm_ckpt_name)[0]}_epoch_{epoch_idx + 1}.pth"
            
            # Save the periodic model checkpoints with epoch number
            if rank==0:
                torch.save({
                    'epoch': epoch_idx + 1,
                    # 'model_state_dict': model.state_dict(),
                    'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(checkpoint_dir, ldm_ckpt_name))

        save_path = os.path.join(checkpoint_dir, train_config['ldm_ckpt_name'])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if rank==0:
            torch.save({
                'epoch': epoch_idx + 1,
                # 'model_state_dict': model.state_dict(),
                'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)

        # epoch_loss = np.mean(losses)

        if rank==0 and epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_ldm_ckpt_name = os.path.join(checkpoint_dir, 'best_ldm.pth')
            torch.save({
                'epoch': epoch_idx + 1,
                # 'model_state_dict': model.module.state_dict(),
                'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_ldm_ckpt_name)
            logging.info(f"Best model saved at epoch {epoch_idx + 1} with loss {best_epoch_loss:.4f}")

        if optimizer_has_stepped:
            scheduler_optimizer.step(epoch_loss)
            # scheduler_optimizer.step()

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/ERA5_config_lsm.yaml', type=str)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    # train(args)
