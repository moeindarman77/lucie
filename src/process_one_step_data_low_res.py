import os
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
from datetime import datetime, timedelta

variable_list = ['Temperature',
    'U-wind',
    'V-wind',
    'Specific_Humidity',
    'logp',
    # 'q_con'
    ]

# SINGLE_LEVEL_VARS = ['logp','sst','precip','tisr']
SINGLE_LEVEL_VARS = ['logp','sst','precip']
PRESSURE_LEVEL_VARS = ['Temperature','U-wind','V-wind','Specific_Humidity','q_con']
CONSTANT_VARS = ['orography']
ACCUMULATE_VARS = ['tp6hr']

DEFAULT_PRESSURE_LEVELS = np.arange(0,8)

file_dict = {'main':'era_5_yYYYY_regridded_mpi_fixed_var_gcc.nc',
            'sst': 'era_5_yYYYY_sst_regridded_mpi_fixed_var.nc',
            'tp':'era_5_yYYYY_precip_regridded_mpi.nc',
            # 'tisr':'toa_incident_solar_radiation_YYYY_regridded_classic4.nc'
            }

def make_list_vars(variables3d,sigma_levels,variables2d):
    var_list = []
    for var in variables3d:
        for sigma in sigma_levels:
            var_list.append(f'{var.lower()}_level_{sigma}')

    for var in variables2d:
        var_list.append(var)

    return var_list

def make_total_precip_accumulated(startdate,enddate,timestep,rolling_mean_step):
    start_year = startdate.year - 1
    end_year = enddate.year

    currentdate = datetime(start_year,1,1,0) #startdate
    counter = 0
    while currentdate.year <= enddate.year:
        # ds_era = xr.open_dataset(f'/eagle/MDClimSim/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_precip_regridded_mpi.nc')
        ds_era = xr.open_dataset(f'/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/t30_era5_2010-2017/era_5_y{currentdate.year}_precip_regridded_mpi.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:
           ds_merged = xr.merge([ds_merged,ds_era])

        currentdate = currentdate + timedelta(hours=ds_era.sizes['Timestep'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    print('time_slice',time_slice)
    print('ds_merged',ds_merged)
    ds = ds_merged.sel(Timestep=time_slice)
    print('ds',ds)
    da = ds['tp']
    da = da.rolling(Timestep=rolling_mean_step).sum()
    print(da)
    return da.values

def create_one_step_dataset(root_dir, save_dir, split, years, list_vars, chunk_size=None):
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    
    list_pressure_vars = [v for v in list_vars if v in PRESSURE_LEVEL_VARS]
    list_single_vars = [v for v in list_vars if v in SINGLE_LEVEL_VARS]
    list_constant_vars = [v for v in list_vars if v in CONSTANT_VARS]
    list_accumulate_vars = [v for v in list_vars if v in ACCUMULATE_VARS]
    

    startdate = datetime(years[0],1,1,0)
    enddate = datetime(years[-1],12,31,23)
    tp6hr = make_total_precip_accumulated(startdate,enddate,1,6)
    print('tp6hr shape first',np.shape(tp6hr))
    # if train set, init normalization stats
    if split == 'train':
        normalize_mean = {}
        normalize_std = {}
        for var in list_constant_vars:
            # constant_path = '/eagle/MDClimSim/troyarcomano/ERA_5/orography_data/regridded_era_orography.nc' #os.path.join(root_dir, f'{var}.nc')
            constant_path = '/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/data/regridded_era_orography.nc' #os.path.join(root_dir, f'{var}.nc')
            constant_field = xr.open_dataset(constant_path)[var].to_numpy()
            normalize_mean[var] = constant_field.mean().reshape([1])
            normalize_std[var] = constant_field.std().reshape([1])

        for var in list_single_vars:
            normalize_mean[var] = []
            normalize_std[var] = []

        for var in list_accumulate_vars:
            normalize_mean[var] = []
            normalize_std[var] = []

        for var in list_pressure_vars:
            for level in DEFAULT_PRESSURE_LEVELS:
                normalize_mean[f'{var}_{level}'] = []
                normalize_std[f'{var}_{level}'] = []

    accumulate_counter = 0 
    for year in tqdm(years, desc='years', position=0):
        print('test',file_dict['main'].replace("YYYY",f'{year}'))
        # ds_sample = xr.open_dataset(os.path.join(root_dir, f'{year}', file_dict['main'].replace("YYYY",f'{year}')))
        ds_sample = xr.open_dataset(os.path.join(root_dir, file_dict['main'].replace("YYYY",f'{year}')))
        print('ds_sample',ds_sample)
        if chunk_size is not None:
            n_chunks = len(ds_sample.Timestep) // chunk_size + 1
            chunk_size_temp = chunk_size
        else:
            n_chunks = 1
            chunk_size_temp = len(ds_sample.Timestep)
        
        idx_in_year = 0

        for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
            dict_np = {}
            list_time_stamps = None
            if chunk_id > 0:
                print('huge problem',chunk_id,year)
            ### convert ds to numpy
            for var in (list_single_vars + list_pressure_vars):
                if var in variable_list:
                    # ds_path = os.path.join(root_dir, f'{year}', file_dict['main'].replace("YYYY",f'{year}'))
                    ds_path = os.path.join(root_dir, file_dict['main'].replace("YYYY",f'{year}'))
                else:
                    #  ds_path = os.path.join(root_dir,f'{year}',file_dict[var].replace("YYYY",f'{year}'))
                     ds_path = os.path.join(root_dir, file_dict[var].replace("YYYY",f'{year}'))
                print('ds_path',ds_path)
                ds = xr.open_dataset(ds_path)
                ds = ds.isel(Timestep=slice(chunk_id*chunk_size_temp, (chunk_id+1)*chunk_size_temp))
                #ds = ds.fillna(value=270.0)
                print('here')

                if list_time_stamps is None:
                    list_time_stamps = ds.Timestep.values
                if var in list_single_vars:
                    print('var',var)
                    dict_np[var] = ds[var].values
                    
                else:
                    available_levels = ds.Sigma_Level.values
                    ds_np = ds[var].values
                    for i, level in enumerate(available_levels):
                        if level in DEFAULT_PRESSURE_LEVELS:
                            print(f'{var}_{level}')
                            dict_np[f'{var}_{level}'] = ds_np[:, i]
           
            for var in list_accumulate_vars:
                if var == 'tp6hr':
                    dict_np['tp6hr'] = tp6hr[accumulate_counter:accumulate_counter + chunk_size_temp,:]
                    print('accumulate_counter:accumulate_counter + chunk_size',accumulate_counter,accumulate_counter + chunk_size_temp)
                    accumulate_counter += chunk_size_temp
            # compute mean and std of each variable of this year
            if split == 'train':
                for k in dict_np.keys():
                    normalize_mean[k].append(dict_np[k].mean())
                    normalize_std[k].append(dict_np[k].std())
                    
                # for var in list_single_vars:
                #     normalize_mean[var].append(dict_np[var].mean())
                #     normalize_std[var].append(dict_np[var].std())
                # for var in list_pressure_vars:
                #     for level in DEFAULT_PRESSURE_LEVELS:
                        # normalize_mean[f'{var}_{level}'].append(dict_np[f'{var}_{level}'].mean())
                        # normalize_std[f'{var}_{level}'].append(dict_np[f'{var}_{level}'].std())
                    
            for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                data_dict = {
                    'input': {'time': str(list_time_stamps[i])},
                }
                for var in dict_np.keys():
                    data_dict['input'][var] = dict_np[var][i]
                for var in list_constant_vars:
                    # constant_path = '/eagle/MDClimSim/troyarcomano/ERA_5/orography_data/regridded_era_orography.nc'#os.path.join(root_dir, f'{var}.nc')
                    constant_path = '/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/regridded_era_orography.nc'#os.path.join(root_dir, f'{var}.nc')
                    constant_field = xr.open_dataset(constant_path)[var].to_numpy()
                    constant_field = constant_field.reshape(constant_field.shape[-2:])
                    data_dict['input'][var] = constant_field
                    
                with h5py.File(os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict.items():
                        # Create a group for the main key (e.g., 'input' or 'output')l;
                        group = f.create_group(main_key)
                        
                        # Now, save each array in the sub-dictionary to this group
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                group.create_dataset(sub_key, data=array, compression=None)
                
                idx_in_year += 1
    
    if split == 'train':
        for var in normalize_mean.keys():
            if var not in list_constant_vars:
                mean_over_years, std_over_years = np.array(normalize_mean[var]), np.array(normalize_std[var])
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std_over_years**2).mean() + (mean_over_years**2).mean() - mean_over_years.mean()**2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean_over_years.mean()
                normalize_mean[var] = mean.reshape([1])
                normalize_std[var] = std.reshape([1])
            
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)
        
full_list = ['Temperature',
    'U-wind',
    'V-wind',
    'Specific_Humidity',
    'logp',
    # 'q_con',
    'sst',
    'tisr',
    'tp',
    'tp6hr',
    'orography',
]

sigma_levels = np.arange(0,8)
print('full_list',full_list)

# create_one_step_dataset(
#     root_dir='/eagle/MDClimSim/troyarcomano/ERA_5/',
#     save_dir='/eagle/MDClimSim/troyarcomano/data/ERA5_SPEEDY_GRID/1_step_1hr_h5df_test/',
#     split='train',
#     years=np.arange(2004,2010),
#     list_vars=full_list,
#     )

# create_one_step_dataset(
#     root_dir='/eagle/MDClimSim/troyarcomano/ERA_5/',
#     save_dir='/eagle/MDClimSim/troyarcomano/data/ERA5_SPEEDY_GRID/1_step_1hr_h5df_test/',
#     split='val',
#     years=[2010],
#     list_vars=full_list,
# )
create_one_step_dataset(
    root_dir='/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/t30_era5_2010-2017/',
    save_dir='/glade/derecho/scratch/mdarman/ERA5_t30/test',
    split='test',
    years=list(range(2010, 2019)),
    list_vars=full_list,
)

