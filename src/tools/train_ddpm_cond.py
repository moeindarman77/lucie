import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.utilities import load_glorys_roms_whole, zero_mask
from torch.utils.data import DataLoader
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
from dataset.utilities import GlorysRomsDataset
import torch.nn.functional as F
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
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

    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    # # Instantiate Condition related components
    # condition_types = []
    # condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    # if condition_config is not None:
    #     assert 'condition_types' in condition_config, \
    #         "condition type missing in conditioning config"
    #     condition_types = condition_config['condition_types']
    #     if 'text' in condition_types:
    #         validate_text_config(condition_config)
    #         with torch.no_grad():
    #             # Load tokenizer and text model based on config
    #             # Also get empty text representation
    #             text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
    #                                                                  ['text_embed_model'], device=device)
    #             empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)


    # Create dataset instance
    dataset = GlorysRomsDataset(steps=range(dataset_config['num_train_data']),
                                 channels=["SSU", "SSV", "SSH",], 
                                 added_channels=[], 
                                 data_dir=dataset_config["data_dir"],
                                 lat_lon_keep= tuple(dataset_config["lat_lon_keep"]), 
                                 interpolator_use="scipy", )
    logging.info("Dataset loaded.")

    # Create DataLoader
    data_loader = DataLoader(dataset, 
                             batch_size=train_config['ldm_batch_size'], 
                             shuffle=True, 
                             num_workers=4) #GPU can't take more than 8
    logging.info("DataLoader created.")

    # Instantiate the unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    if not train_config['load_latents']:
        print('Loading VAE model as latents not present')
        vae = VAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        vae_load_dir = os.path.join(train_config['results_dir'], train_config['task_name'])
        if os.path.exists(vae_load_dir):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(vae_load_dir,
                                                        train_config['vae_autoencoder_ckpt_name']),
                                           map_location=device)['model_state_dict'])
            # vae.load_state_dict(torch.load(os.path.join(vae_load_dir,
            #                                             train_config['vae_autoencoder_ckpt_name']),
            #                                map_location=device))
        else:
            raise Exception('VAE checkpoint not found and use_latents was disabled')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Load vae and freeze parameters ONLY if latents already not saved
    if not train_config['load_latents']:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for batch_idx, data in enumerate(tqdm(data_loader)):

            cond_input = None
            lres, hres = data
            lres = lres.float().to(device)
            hres = hres.float().to(device)

            # hres = F.pad(hres, (3, 4, 2, 2), mode='constant', value=0)
            # lres = F.pad(lres, (2, 2, 3, 4), mode='constant', value=0)
            
            # lres_upsampled = F.interpolate(lres, size=(352, 608), mode='bilinear', align_corners=False)
            lres_padded = F.pad(lres, (0, 3, 1, 1), mode='constant', value=0)
            cond_input = lres_padded
            # Fetch autoencoders output(reconstructions)
            with torch.no_grad():        
                _, latent_distribution, _ = vae.encode(lres_padded)
                latent_mean, _ = torch.chunk(latent_distribution, 2, dim=1)

            # Padd zeros to im
            # Here, we pad with 2 zeros on top and bottom, 3 zeros on left and right (padding_left, padding_right, padding_top, padding_bottom).
            # im = F.pad(im, (3, 4, 2, 2), mode='constant', value=0)
            # cond_input  = F.pad(cond_input, (2, 2, 3, 4), mode='constant', value=0)
            
            # if condition_config is not None:
            #     cond_input, im = data
            # else:
            #     im = data
            optimizer.zero_grad()
            # im = im.float().to(device) 
            # if not im_dataset.use_latents:
                # with torch.no_grad():
                #     im, _ = vae.encode(im)
                    
            ########### Handling Conditional Input ###########
            # if 'text' in condition_types:
            #     with torch.no_grad():
            #         assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
            #         validate_text_config(condition_config)
            #         text_condition = get_text_representation(cond_input['text'],
            #                                                      text_tokenizer,
            #                                                      text_model,
            #                                                      device)
            #         text_drop_prob = get_config_value(condition_config['text_condition_config'],
            #                                           'cond_drop_prob', 0.)
            #         text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
            #         cond_input['text'] = text_condition
            # if 'image' in condition_types:
            #     assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
            #     validate_image_config(condition_config)
            #     cond_input_image = cond_input['image'].to(device)
            #     # Drop condition
            #     im_drop_prob = get_config_value(condition_config['image_condition_config'],
            #                                           'cond_drop_prob', 0.)
            #     cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            # if 'class' in condition_types:
            #     assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
            #     validate_class_config(condition_config)
            #     class_condition = torch.nn.functional.one_hot(
            #         cond_input['class'],
            #         condition_config['class_condition_config']['num_classes']).to(device)
            #     class_drop_prob = get_config_value(condition_config['class_condition_config'],
            #                                        'cond_drop_prob', 0.)
            #     # Drop condition
            #     cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)
            ################################################
            
            # Sample random noise
            noise = torch.randn_like(latent_mean).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (latent_mean.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(latent_mean, noise, t)

            # Apply padding to be divisible by 8
            noisy_im_padded = F.pad(noisy_im, (0, (80 - 75), 0, (48 - 43)), mode='constant', value=0)

            
            noise_pred = model(noisy_im_padded, t, cond_input=cond_input)
            # Unpadd the noise_pred
            noise_pred = noise_pred[..., :43, :75]

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        if (epoch_idx + 1) % train_config['ldm_save_interval'] == 0:
            checkpoint_dir = os.path.join(task_dir, 'checkpoints')
            # Use the base checkpoint names from the config and append the epoch number
            base_ldm_ckpt_name = train_config['ldm_ckpt_name']
            
            # Construct filenames with epoch number for periodic checkpoint saving
            ldm_ckpt_name = f"{os.path.splitext(base_ldm_ckpt_name)[0]}_epoch_{epoch_idx + 1}.pth"
            
            
            # Save the periodic model checkpoints with epoch number
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, ldm_ckpt_name))

        save_path = os.path.join(task_dir, train_config['ldm_ckpt_name'])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/glor_config.yaml', type=str)
    args = parser.parse_args()
    train(args)
