import argparse
import glob
import os
import pickle
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models.fno import FNO2d
# from dataset.LucieDataset import LucieDataset
from dataset.ClimateDataset_v2 import ClimateDataset_v2 as ClimateDataset
import torch.nn.functional as F
from torch.nn import DataParallel


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
    fno_config = config['fno_params']
    train_config = config['train_params']

    results_dir = train_config['results_dir']  # Define the results directory
    # Ensure the 'results' directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'])
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    
    save_dir = os.path.join(task_dir,'samples')
    os.makedirs(save_dir, exist_ok=True)

    # Create dataset instance
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
    # dataset = LucieDataset(input_dir_lr=dataset_config['input_data_dir'], 
    #                          input_dir_hr=dataset_config['output_data_dir'], 
    #                          input_vars=input_vars, 
    #                          output_vars=output_vars, 
    #                          lr_lats=lr_lats, 
    #                          lr_lons=lr_lons,
    #                          year_range=(dataset_config['year_range_start'], dataset_config['year_range_end']),
    #                          normalize=True, 
    #                          input_normalization_file=dataset_config['input_normalization_dir'], 
    #                          output_normalization_file=dataset_config['output_normalization_dir'],
    #                          lucie_file_path=dataset_config['lucie_file_path'],
    #                          lucie_vars=input_vars) 
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
    

    if dataset_config['land_sea_mask'] == 1:
        lsm = torch.tensor(np.load(dataset_config['land_sea_mask_dir'])['land_sea_mask']).unsqueeze(0).unsqueeze(1)
    else:
        lsm = None
    # Create DataLoader
    data_loader = DataLoader(dataset, 
                             batch_size=train_config['fno_batch_size'], 
                             shuffle=False, 
                             num_workers=2)

    model = FNO2d(
        input_channels=dataset_config['input_channels']+dataset_config['land_sea_mask'],
         output_channels=dataset_config['output_channels'],
           model_config=fno_config,)
    # model = DataParallel(model)
    model = model.to(device)

    # load_dir = os.path.join(task_dir, train_config['vae_autoencoder_ckpt_name'])
    load_dir = "/glade/derecho/scratch/mdarman/lucie/results/fno_final_v2/checkpoints/best_fno.pth"
    model.load_state_dict(torch.load(load_dir, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    step = 0
    with torch.no_grad():
        for data in data_loader:
            step += 1
            # lucie, lres, hres = data['lucie'], data['input'], data['output']
            lres, hres = data['input'], data['output']
            # lucie = lucie.float().to(device)
            lres = lres.float().to(device)
            hres = hres.float().to(device)

            lres_upsampled = F.interpolate(lres, size=(721,1440), mode='bicubic', align_corners=True)
            # lucie_upsampled = F.interpolate(lucie, size=(721,1440), mode='bicubic', align_corners=True)

            lsm_expanded = None
            if lsm is not None:
                lsm = lsm.float().to(device)
                lsm_expanded = lsm.expand(lres.shape[0], -1, -1, -1)  # Shape: (batch_size, 1, 721, 1440)
                lres_upsampled_cat = torch.cat((lres_upsampled, lsm_expanded), dim=1)
                lucie_upsampled_cat = torch.cat((lucie_upsampled, lsm_expanded), dim=1)

            lres_upsampled[:,5] = hres[:,5]
            hres = hres[:, [0, 2, 3, 4]] # ONLY want to reconstruct the 2m temperature, u and v wind components and tp6hr
            decoded_output = model(x=lres_upsampled)
            
            # decoded_output[:, 0] += lres_upsampled[:, 0] 
            # decoded_output[:, -1] += lres_upsampled[:, -1] 

            lucie_zero_shot = model(x=lres_upsampled)
            # lucie_zero_shot[:, 0] += lucie_upsampled[:, 0] 
            # lucie_zero_shot[:, -1] += lucie_upsampled[:, -1]

            # Convert tensors to numpy arrays
            lres_interp_numpy = lres_upsampled.cpu().numpy()
            # lucie_interp_numpy = lucie_upsampled.cpu().numpy()
            # lucie_numpy = lucie.cpu().numpy()
            lres_numpy = lres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            hres_numpy = hres.cpu().numpy()            # Shape: [batch_size, channels, height, width]
            lucie_zero_shot = lucie_zero_shot.cpu().numpy()
            decoded_outputs_numpy = decoded_output.cpu().numpy()   # Shape: [num_samples, batch_size, channels, height, width]

            # Save to an npz file named with the index of the data loader
            save_path = os.path.join(save_dir, f'{step}.npz')
            np.savez(save_path, 
                     lres=lres_numpy, 
                     lres_interp=lres_interp_numpy, 
                     hres=hres_numpy, 
                     output=decoded_outputs_numpy, 
                    #  lucie=lucie_numpy,
                    #  lucie_interp=lucie_interp_numpy,
                     lucie_zero_shot=lucie_zero_shot,
                     )
            print(f'Saved {step}.npz')
  
        # if train_config['save_latents']:
        #     # save Latents (but in a very unoptimized way)
        #     latent_path = os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'])
        #     latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'],
        #                                            '*.pkl'))
        #     assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
        #     if not os.path.exists(latent_path):
        #         os.mkdir(latent_path)
        #     print('Saving Latents for {}'.format(dataset_config['name']))
            
        #     fname_latent_map = {}
        #     part_count = 0
        #     count = 0
        #     for idx, im in enumerate(tqdm(data_loader)):
        #         encoded_output, _ = model.encode(im.float().to(device))
        #         fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
        #         # Save latents every 1000 images
        #         if (count+1) % 1000 == 0:
        #             pickle.dump(fname_latent_map, open(os.path.join(latent_path,
        #                                                             '{}.pkl'.format(part_count)), 'wb'))
        #             part_count += 1
        #             fname_latent_map = {}
        #         count += 1
        #     if len(fname_latent_map) > 0:
        #         pickle.dump(fname_latent_map, open(os.path.join(latent_path,
        #                                            '{}.pkl'.format(part_count)), 'wb'))
        #     print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
