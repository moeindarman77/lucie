import argparse
import glob
import os
import pickle
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from models.vqvae import VQVAE
from models.vae import VAE
from dataset.utilities import GlorysRomsDataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    results_dir = train_config['load_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    steps = range(9130, 10226)  # Some example steps 
    data_dir = "/home/exouser/qgm1/fujitsu_data"

    # Create dataset instance
    dataset = GlorysRomsDataset(steps=range(dataset_config['num_train_data']),
                                 channels=["SSU", "SSV", "SSH",], 
                                 added_channels=[], 
                                 data_dir=dataset_config["data_dir"],
                                 lat_lon_keep= tuple(dataset_config["lat_lon_keep"]), 
                                 interpolator_use="scipy", )

    # Create DataLoader
    data_loader = DataLoader(dataset, 
                             batch_size=train_config['autoencoder_batch_size'], 
                             shuffle=False, 
                             num_workers=2)

    model = VAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    # load_dir = os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name'])
    load_dir = '/media/volume/moein-storage-1/ddpm_ocean/results/spectral_loss_v3/vae_autoencoder_ckpt.pth'
    model.load_state_dict(torch.load(load_dir, 
                                    map_location=device)['model_state_dict'])
    model.eval()
    step = 0
    with torch.no_grad():
        for data in data_loader:
            step += 1
            lres, hres = data
            lres = lres.float().to(device)
            hres = hres.float().to(device)

            # hres = F.pad(hres, (3, 4, 2, 2), mode='constant', value=0)
            # lres_upsampled = F.interpolate(lres, size=(352, 608), mode='bilinear', align_corners=False)
        
            # encoded_output, _, out_for_encoder = model.encode(lres)
            # decoded_output = model.decode(encoded_output, out_for_encoder)
            
            decoded_output, _ = model(lres)

            # Convert tensors to numpy arrays
            hres_numpy = hres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            decoded_outputs_numpy = decoded_output.cpu().numpy()   # Shape: [num_samples, batch_size, channels, height, width]

            # Save to an npz file named with the index of the data loader
            save_dir = os.path.join(task_dir,'samples', f'{step}.npz')
            np.savez(save_dir, hres=hres_numpy, decoded_outputs=decoded_outputs_numpy)
            print(f'Saved {step}.npz')
        # encoded_output = torch.clamp(encoded_output, -1., 1.)
        # encoded_output = (encoded_output + 1) / 2
        # decoded_output = torch.clamp(decoded_output, -1., 1.)
        # decoded_output = (decoded_output + 1) / 2
        # ims = (ims + 1) / 2

        # encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
        # decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        # input_grid = make_grid(ims.cpu(), nrow=ngrid)
        # encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        # decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        # input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        # input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        # encoder_grid.save(os.path.join(train_config['task_name'], 'encoded_samples.png'))
        # decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))

    # 
    # num_images = train_config['num_samples']
    # ngrid = train_config['num_grid_rows']
    
    # idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    # ims = torch.cat([dataset[idx][None, :] for idx in idxs]).float()
    # ims = ims.to(device)
  
        if train_config['save_latents']:
            # save Latents (but in a very unoptimized way)
            latent_path = os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'],
                                                   '*.pkl'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))
            
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, im in enumerate(tqdm(data_loader)):
                encoded_output, _ = model.encode(im.float().to(device))
                fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                if (count+1) % 1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    '{}.pkl'.format(part_count)), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                   '{}.pkl'.format(part_count)), 'wb'))
            print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
