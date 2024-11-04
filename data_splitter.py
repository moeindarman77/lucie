import netCDF4 as nc
import h5py
import numpy as np
import os

# Define directories
output_dir = '/glade/derecho/scratch/asheshc/ERA5_lr'  # Directory to save .h5 files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to save combined data to HDF5 format
def save_to_h5(year, timestep, temperature, tp6, hr_lats, hr_lons, output_dir):
    file_name = f"{year}_{timestep:04d}.h5"  # Naming convention: year_timestep.h5
    file_path = os.path.join(output_dir, file_name)
    
    with h5py.File(file_path, 'w') as f:
        # Create datasets for the variables
        f.create_dataset('2m_temperature', data=temperature)
        f.create_dataset('tp6hr', data=tp6)
        f.create_dataset('Latitude', data=hr_lats)
        f.create_dataset('Longitude', data=hr_lons)
        
    print(f"Saved: {file_path}")

# Loop over the years and timesteps
years = range(1979, 2022)  # For years from 1979 to 2021

for year in years:
    # Open the temperature and precipitation files for the given year
    temp_file = f"/glade/derecho/scratch/asheshc/ERA5_highres/2m_temperature/{year}.nc"  # Assuming the files are named as 'year.nc'
    tp6_file = f"/glade/derecho/scratch/asheshc/ERA5_highres/total_precipitation_6hr/{year}.nc"  # Assuming both variables are in the same file
    
    with nc.Dataset(temp_file, 'r') as temp_data, nc.Dataset(tp6_file, 'r') as tp6_data:
        # Read the variables
        temperature = temp_data.variables['2m_temperature'][:]
        tp6 = tp6_data.variables['total_precipitation_6hr'][:]
        hr_lats = temp_data.variables['latitude'][:]
        hr_lons = temp_data.variables['longitude'][:]
        
        # Ensure temperature and tp6 have the same number of timesteps
        assert temperature.shape[0] == tp6.shape[0], f"Number of timesteps mismatch for {year}: Temperature has {temperature.shape[0]}, tp6 has {tp6.shape[0]}"

        num_timesteps = temperature.shape[0]
        
        # Loop over each timestep and save to .h5
        for timestep in range(num_timesteps):
            save_to_h5(year, timestep, temperature[timestep], tp6[timestep], hr_lats, hr_lons, output_dir)