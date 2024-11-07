import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import seaborn as sn
from numpy import sin, cos, arctan2, arcsin, arccos
import torch
from torch.nn.functional import interpolate
import xarray as xr
import gc
from importlib import reload
import os
from scipy import fft  
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde, spearmanr
from scipy.signal import convolve2d
from torch.utils.data import Dataset, DataLoader
from dataset.data_locs import data_dir # location of the data
import pickle

torch.manual_seed(0)
np.random.seed(0)

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

## time dict, to arange the times in a straightforward way to pull from the correct dataset without having to save


def roms_glorys_daily_data():
    num_days_glorys_roms = 0
    time_dict_glorys_roms = {}
    t = 0
    for year in range(1993,2021):
        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
        romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_{year}_5dmean_gom.nc")
        times1 = gloryshr["time"] ## once a year
        times2 = romshr["ocean_time"]
        assert len(times1) == len(times2)

        for day in range(len(times1)):
            time_dict_glorys_roms[t] = (year,day)
            t+=1
        num_days_glorys_roms+=len(times1)
    
    rev_time_dict_glorys_roms = {v:k for [k,v] in time_dict_glorys_roms.items()} 
    
    ## num_days_glorys_roms = 10227
    return time_dict_glorys_roms, rev_time_dict_glorys_roms, num_days_glorys_roms

def roms_glorys_daily_data2():
    num_days_glorys_roms = 0
    time_dict_glorys_roms = {}
    t = 0
    for year in range(1993,2021):
        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
        romshr = nc.Dataset(f"{data_dir}/ENKF_GoM/EnKF_surface_{year}_5dmean_wholegom.nc")
        times1 = gloryshr["time"] ## once a year
        times2 = romshr["ocean_time"]
        assert len(times1) == len(times2)

        for day in range(len(times1)):
            time_dict_glorys_roms[t] = (year,day)
            t+=1
        num_days_glorys_roms+=len(times1)
    
    rev_time_dict_glorys_roms = {v:k for [k,v] in time_dict_glorys_roms.items()} 
    
    return time_dict_glorys_roms, rev_time_dict_glorys_roms, num_days_glorys_roms

def waverys_3hour_data():
    num_steps_waverys = 0
    time_dict_waverys = {}
    t = 0
    for year in range(1994,2022):
        waverys = nc.Dataset(f"{data_dir}/WAVERYS_GoM_YearWise/cmems_mod_glo_wav_my_0.2deg_PT3H-i_{year}.nc")
        times1 = waverys["time"] ## once every 3 hours, 8 times a day

        for step in range(len(times1)):
            time_dict_waverys[t] = (year,step/8)
            t+=1
            
        num_steps_waverys+=len(times1)

    rev_time_dict_waverys = {v:k for [k,v] in time_dict_waverys.items()} 
    
    return time_dict_waverys, rev_time_dict_waverys, num_steps_waverys
    
time_dict_glorys_roms_old, rev_time_dict_glorys_roms_old, num_days_glorys_roms_old = roms_glorys_daily_data()
time_dict_glorys_roms, rev_time_dict_glorys_roms, num_days_glorys_roms = roms_glorys_daily_data2()
time_dict_waverys, rev_time_dict_waverys, num_steps_waverys = waverys_3hour_data()

## num_days_glorys_roms = 10227
##num_steps_waverys = 81816

def load_days_glorys_roms_noCoord(beg, end, time_dict = time_dict_glorys_roms):
    lrv_use = np.empty((end-beg,169,300,3))
    hrv_use = np.empty((end-beg,314,425,3))
    times = []
    
    for it, t in enumerate(range(beg,end),0):
        (year,day) = time_dict[t]
        times.append((year,day))

        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
        tlen = len(gloryshr["time"])
        lrv_use[it,:,:,0] = gloryshr["uo"][day,0,...]
        lrv_use[it,:,:,1] = gloryshr["vo"][day,0,...]
        lrv_use[it,:,:,2] = gloryshr["zos"][day]

        romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_{year}_5dmean_gom.nc")
        hrv_use[it,:,:,0] = romshr["SSU"][day]
        hrv_use[it,:,:,1] = romshr["SSV"][day]
        hrv_use[it,:,:,2] = romshr["SSH"][day]

        gloryshr.close()
        romshr.close()
        
    return lrv_use, hrv_use, times

def load_glorys_roms_2(steps, time_dict = time_dict_glorys_roms):
    g = np.empty((len(steps),169,300,3))
    
    lrv_use = np.empty((len(steps),169,300,3))
    hrv_use = np.empty((len(steps),314,425,3))
    times = []
    
    for it, t in enumerate(steps,0):
        (year,day) = time_dict[t]
        times.append((year,day))

        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
            
        tlen = len(gloryshr["time"])
        lrv_use[it,:,:,0] = gloryshr["uo"][day,0,...]
        lrv_use[it,:,:,1] = gloryshr["vo"][day,0,...]
        lrv_use[it,:,:,2] = gloryshr["zos"][day]

        romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_{year}_5dmean_gom.nc")
        hrv_use[it,:,:,0] = romshr["SSU"][day]
        hrv_use[it,:,:,1] = romshr["SSV"][day]
        hrv_use[it,:,:,2] = romshr["SSH"][day]
        
        gloryshr.close()
        romshr.close()
        
    ## masks for computing
    gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
    romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_1993_5dmean_gom.nc")
    
    lrcoordmeshX, lrcoordmeshY = np.meshgrid(gloryshr["longitude"][:], gloryshr["latitude"][:])
    lrcoordmeshX, lrcoordmeshY = lrcoordmeshX.data, lrcoordmeshY.data
    hrcoordmeshX, hrcoordmeshY = romshr["lon_rho"][:], romshr["lat_rho"][:]
    hrcoordmeshX, hrcoordmeshY = hrcoordmeshX.data, hrcoordmeshY.data
    
    lrloncoords = lrcoordmeshX[0,:]
    lrlatcoords = lrcoordmeshY[:,0]
    hrloncoords = hrcoordmeshX[0,:]
    hrlatcoords = hrcoordmeshY[:,0]
    
    gloryshr.close()
    romshr.close()
    
    return lrv_use, hrv_use, lrloncoords, lrlatcoords, hrloncoords, hrlatcoords, times

def load_glorys_roms_interpolate_2(steps, time_dict = time_dict_glorys_roms, lat_lon_keep = (18, 28,-92, -77)):
    lrv_use = np.empty((len(steps),169,300,3))
    hrv_use = np.empty((len(steps),314,425,3))
    times = []
    
    for it, t in enumerate(steps,0):
        (year,day) = time_dict[t]
        times.append((year,day))

        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
            
        tlen = len(gloryshr["time"])
        lrv_use[it,:,:,0] = gloryshr["uo"][day,0,...]
        lrv_use[it,:,:,1] = gloryshr["vo"][day,0,...]
        lrv_use[it,:,:,2] = gloryshr["zos"][day]

        romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_{year}_5dmean_gom.nc")
        hrv_use[it,:,:,0] = romshr["SSU"][day]
        hrv_use[it,:,:,1] = romshr["SSV"][day]
        hrv_use[it,:,:,2] = romshr["SSH"][day]
        
        gloryshr.close()
        romshr.close()
    
    ## masks for computing
    gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
    romshr = nc.Dataset(f"{data_dir}/ROMS_GoM_YearWise/EnKF_surface_1993_5dmean_gom.nc")
    
    lrcoordmeshX, lrcoordmeshY = np.meshgrid(gloryshr["longitude"][:], gloryshr["latitude"][:])
    lrcoordmeshX, lrcoordmeshY = lrcoordmeshX.data, lrcoordmeshY.data
    hrcoordmeshX, hrcoordmeshY = romshr["lon_rho"][:], romshr["lat_rho"][:]
    hrcoordmeshX, hrcoordmeshY = hrcoordmeshX.data, hrcoordmeshY.data

    # for comparing same regions between lr and hr
    (latmin, latmax, lonmin, lonmax) = lat_lon_keep

    lrmask = ((latmin < lrcoordmeshY) * (lrcoordmeshY < latmax) * (lonmin < lrcoordmeshX) * (lrcoordmeshX < lonmax))
    hrmask = ((latmin < hrcoordmeshY) * (hrcoordmeshY < latmax) * (lonmin < hrcoordmeshX) * (hrcoordmeshX < lonmax))
    
    lrmask_lonKeep = lrmask.any(axis = 0)
    lrmask_latKeep = lrmask.any(axis = 1)
    hrmask_lonKeep = hrmask.any(axis = 0)
    hrmask_latKeep = hrmask.any(axis = 1)
    
    lrv_use_crop = lrv_use[:, lrmask_latKeep, :, :]
    lrv_use_crop = lrv_use_crop[:, :, lrmask_lonKeep, :]

    hrv_use_reshape = hrv_use[:]
    hrv_use_reshape = hrv_use_reshape[:, hrmask_latKeep, :, :]
    hrv_use_reshape = hrv_use_reshape[:, :, hrmask_lonKeep, :]
    
    ## bicubic interpolation
    lrv_use_reshape = interpolate(torch.from_numpy(lrv_use_crop).permute(0,3,1,2), hrv_use_reshape.shape[1:3], mode = "bicubic").permute(0,2,3,1).numpy()
    # lrv_use_reshape2 = interpolate(torch.from_numpy(lrv_use_reshape).permute(0,3,1,2), hrv_use_reshape.shape[1:3]).permute(0,2,3,1).numpy()
    
    hrloncoords = hrcoordmeshX[0,hrmask_lonKeep]
    hrlatcoords = hrcoordmeshY[hrmask_latKeep,0]
    
    return lrv_use, hrv_use, lrv_use_reshape, hrv_use_reshape, hrloncoords, hrlatcoords, times

# class GlorysRomsDataset(Dataset):
#     def __init__(self, steps, time_dict=time_dict_glorys_roms, channels=["SSU", "SSV", "SSH"], added_channels=["SSKE", "SSH_curve"], data_dir='path_to_data'):
#         self.steps = steps
#         self.time_dict = time_dict
#         self.channels = channels
#         self.added_channels = added_channels
#         self.channels_all = channels + added_channels
#         self.numchannels = len(self.channels_all)
#         self.data_dir = data_dir
        
#         # Load necessary data for coordinates
#         gloryshr_temp = nc.Dataset(f"{self.data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
#         romshr_temp = nc.Dataset(f"{self.data_dir}/ENKF_GoM/EnKF_surface_1993_5dmean_wholegom.nc")
#         self.lrcoordmeshX, self.lrcoordmeshY = np.meshgrid(gloryshr_temp["longitude"][:], gloryshr_temp["latitude"][:])
#         self.hrcoordmeshX, self.hrcoordmeshY = romshr_temp["lon_rho"][:], romshr_temp["lat_rho"][:]
#         self.lrloncoords = self.lrcoordmeshX[0,:]
#         self.lrlatcoords = self.lrcoordmeshY[:,0]
#         self.hrloncoords = self.hrcoordmeshX[0,:]
#         self.hrlatcoords = self.hrcoordmeshY[:,0]
        
#         self.lonspace = np.abs(self.lrloncoords[1] - self.lrloncoords[0])
#         self.latspace = np.abs(self.lrlatcoords[1] - self.lrlatcoords[0])
#         gloryshr_temp.close()
#         romshr_temp.close()

#         # Define convolution kernel
#         self.convkernel = np.array([[-1.000, -1.414, -1.000],
#                                     [-1.414, 9.657, -1.414],
#                                     [-1.000, -1.414, -1.000]]) / (9.657 * self.lonspace * self.latspace)

#     def __len__(self):
#         return len(self.steps)

#     def __getitem__(self, idx):
#         step = self.steps[idx]
#         year, day = self.time_dict[step]
        
#         # Load data
#         gloryshr = nc.Dataset(f"{self.data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
#         romshr = nc.Dataset(f"{self.data_dir}/ENKF_GoM/EnKF_surface_{year}_5dmean_wholegom.nc")

#         lrv = np.empty((gloryshr["zos"][0,...].shape[0], gloryshr["zos"][0,...].shape[1], self.numchannels))
#         hrv = np.empty((romshr["SSH"][0,...].shape[0], romshr["SSH"][0,...].shape[1], self.numchannels))

#         for ich, ch in enumerate(self.channels_all):
#             if ch == "SSU":
#                 lrv[:,:,ich] = gloryshr["uo"][day,0,...]
#                 hrv[:,:,ich] = romshr["SSU"][day]
#             elif ch == "SSV":
#                 lrv[:,:,ich] = gloryshr["vo"][day,0,...]
#                 hrv[:,:,ich] = romshr["SSV"][day]
#             elif ch == "SSH":
#                 lrv[:,:,ich] = gloryshr["zos"][day]
#                 hrv[:,:,ich] = romshr["SSH"][day]
#             elif ch == "SSKE":
#                 uind = self.channels_all.index("SSU")
#                 vind = self.channels_all.index("SSV")
#                 lrv[:,:,ich] = lrv[:,:,uind]**2 + lrv[:,:,vind]**2
#                 hrv[:,:,ich] = hrv[:,:,uind]**2 + hrv[:,:,vind]**2
#             elif ch == "SSH_curve":
#                 hind = self.channels_all.index("SSH")
#                 lrv[:,:,ich] = convolve2d(lrv[:,:,hind], self.convkernel, mode="same", boundary="symm")

#         gloryshr.close()
#         romshr.close()
        
#         [lrv_ma] = zero_mask([lrv])
#         [hrv_ma] = zero_mask([hrv])
#         # Convert to torch tensors
#         lrv_tensor = torch.tensor(lrv_ma.data, dtype=torch.float32).permute(2,0,1)
#         hrv_tensor = torch.tensor(hrv_ma.data, dtype=torch.float32).permute(2,0,1)
        
#         return lrv_tensor, hrv_tensor
    
class GlorysRomsDataset(Dataset):
    def __init__(self, steps, time_dict=time_dict_glorys_roms, channels=["SSU", "SSV", "SSH"], added_channels=["SSKE", "SSH_curve"], data_dir='path_to_data', lat_lon_keep=None, interpolator_use=None, kernel_scale=1):
        self.steps = steps
        self.time_dict = time_dict
        self.channels = channels
        self.added_channels = added_channels
        self.channels_all = channels + added_channels
        self.numchannels = len(self.channels_all)
        self.data_dir = data_dir
        self.lat_lon_keep = lat_lon_keep
        self.interpolator_use = interpolator_use
        self.kernel_scale = kernel_scale
        
        # Load necessary data for coordinates
        gloryshr_temp = nc.Dataset(f"{self.data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
        romshr_temp = nc.Dataset(f"{self.data_dir}/ENKF_GoM/EnKF_surface_1993_5dmean_wholegom.nc")
        self.lrcoordmeshX, self.lrcoordmeshY = np.meshgrid(gloryshr_temp["longitude"][:], gloryshr_temp["latitude"][:])
        self.hrcoordmeshX, self.hrcoordmeshY = romshr_temp["lon_rho"][:], romshr_temp["lat_rho"][:]
        self.lrloncoords_og = self.lrcoordmeshX[0, :]
        self.lrlatcoords_og = self.lrcoordmeshY[:, 0]
        self.hrloncoords_og = self.hrcoordmeshX[0, :]
        self.hrlatcoords_og = self.hrcoordmeshY[:, 0]
        
        self.lonspace = np.abs(self.lrloncoords_og[1] - self.lrloncoords_og[0])
        self.latspace = np.abs(self.lrlatcoords_og[1] - self.lrlatcoords_og[0])
        gloryshr_temp.close()
        romshr_temp.close()

        # Define convolution kernel
        self.convkernel = np.array([[-1.000, -1.414, -1.000],
                                    [-1.414, 9.657, -1.414],
                                    [-1.000, -1.414, -1.000]]) / (9.657 * self.lonspace * self.latspace)
        
        # Compute area elements if needed (I shoud get the function from Lenny.)
        if "SSH_kernel1" in self.channels_all:
            self.area_elements_lr = self.area_elements_lats_lons(self.lrlatcoords_og, self.lrloncoords_og)
            self.area_elements_hr = self.area_elements_lats_lons(self.hrlatcoords_og, self.hrloncoords_og)
    

    def area_elements_lats_lons(self, lats, lons):
        pass

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        step = self.steps[idx]
        year, day = self.time_dict[step]
        
        # Load data
        gloryshr = nc.Dataset(f"{self.data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
        romshr = nc.Dataset(f"{self.data_dir}/ENKF_GoM/EnKF_surface_{year}_5dmean_wholegom.nc")

        lrv = np.empty((gloryshr["zos"][0, ...].shape[0], gloryshr["zos"][0, ...].shape[1], self.numchannels))
        hrv = np.empty((romshr["SSH"][0, ...].shape[0], romshr["SSH"][0, ...].shape[1], self.numchannels))

        for ich, ch in enumerate(self.channels_all):
            if ch == "SSU":
                lrv[:, :, ich] = gloryshr["uo"][day, 0, ...]
                hrv[:, :, ich] = romshr["SSU"][day]
            elif ch == "SSV":
                lrv[:, :, ich] = gloryshr["vo"][day, 0, ...]
                hrv[:, :, ich] = romshr["SSV"][day]
            elif ch == "SSH":
                lrv[:, :, ich] = gloryshr["zos"][day]
                hrv[:, :, ich] = romshr["SSH"][day]
            elif ch == "SSKE":
                uind = self.channels_all.index("SSU")
                vind = self.channels_all.index("SSV")
                lrv[:, :, ich] = lrv[:, :, uind] ** 2 + lrv[:, :, vind] ** 2
                hrv[:, :, ich] = hrv[:, :, uind] ** 2 + hrv[:, :, vind] ** 2
            elif ch == "SSH_curve":
                hind = self.channels_all.index("SSH")
                lrv[:, :, ich] = convolve2d(lrv[:, :, hind], self.convkernel, mode="same", boundary="symm")
                hrv[:, :, ich] = convolve2d(hrv[:, :, hind], self.convkernel, mode="same", boundary="symm")
            elif ch == "SSH_kernel1":
                hind = self.channels_all.index("SSH")
                lrv[:, :, ich] = convolve2d(lrv[:, :, hind], self.convkernel, mode="same", boundary="symm") / self.area_elements_lr * self.kernel_scale
                hrv[:, :, ich] = convolve2d(hrv[:, :, hind], self.convkernel, mode="same", boundary="symm") / self.area_elements_hr * self.kernel_scale
            elif ch == "SSH_curve_lat":
                hind = self.channels_all.index("SSH")
                lrv[:, :, ich] = np.gradient(lrv[:, :, hind], axis=0)
                hrv[:, :, ich] = np.gradient(hrv[:, :, hind], axis=0)
            elif ch == "SSH_curve_lon":
                hind = self.channels_all.index("SSH")
                lrv[:, :, ich] = np.gradient(lrv[:, :, hind], axis=1)
                hrv[:, :, ich] = np.gradient(hrv[:, :, hind], axis=1)

        gloryshr.close()
        romshr.close()
        
        # Apply geographical mask if specified
        if self.lat_lon_keep is not None:
            latmin, latmax, lonmin, lonmax = self.lat_lon_keep
            
            lrmask = ((latmin < self.lrcoordmeshY) & (self.lrcoordmeshY < latmax) & (lonmin < self.lrcoordmeshX) & (self.lrcoordmeshX < lonmax))
            hrmask = ((latmin < self.hrcoordmeshY) & (self.hrcoordmeshY < latmax) & (lonmin < self.hrcoordmeshX) & (self.hrcoordmeshX < lonmax))
            
            lrmask_lonKeep = lrmask.any(axis=0)
            lrmask_latKeep = lrmask.any(axis=1)
            hrmask_lonKeep = hrmask.any(axis=0)
            hrmask_latKeep = hrmask.any(axis=1)
            
            lrv = lrv[lrmask_latKeep, :, :]
            lrv = lrv[:, lrmask_lonKeep, :]

            hrv = hrv[hrmask_latKeep, :, :]
            hrv = hrv[:, hrmask_lonKeep, :]

            lrloncoords = self.lrcoordmeshX[0, lrmask_lonKeep]
            lrlatcoords = self.lrcoordmeshY[lrmask_latKeep, 0]
            hrloncoords = self.hrcoordmeshX[0, hrmask_lonKeep]
            hrlatcoords = self.hrcoordmeshY[hrmask_latKeep, 0]
        else:
            lrloncoords = self.lrloncoords_og
            lrlatcoords = self.lrlatcoords_og
            hrloncoords = self.hrloncoords_og
            hrlatcoords = self.hrlatcoords_og
        
        # Perform interpolation if specified
        if self.interpolator_use == "scipy":
            # Prepare grids
            X_hr, Y_hr = np.meshgrid(hrloncoords, hrlatcoords)
            
            # Initialize interpolated LR data
            lrv_interpolated = np.empty_like(hrv)
            
            for ich in range(self.numchannels):
                # Handle NaNs if necessary
                lrv_channel = lrv[:, :, ich]
                mask_nan = np.isnan(lrv_channel)
                lrv_channel[mask_nan] = 0  # Replace NaNs with zeros or handle appropriately
                
                # Create interpolator
                interpolator = RegularGridInterpolator((lrlatcoords, lrloncoords), lrv_channel, method='linear', bounds_error=False, fill_value=np.nan)
                
                # Interpolate onto HR grid
                lrv_interpolated[:, :, ich] = interpolator((Y_hr, X_hr))
                
            lrv = lrv_interpolated
        
        elif self.interpolator_use == "torch":
            # Convert LR data to torch tensor
            lrv_tensor = torch.from_numpy(lrv).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
            # Interpolate using torch
            lrv_interpolated = torch.nn.functional.interpolate(lrv_tensor, size=hrv.shape[:2], mode='bicubic', align_corners=False)
            lrv = lrv_interpolated.squeeze(0).permute(1, 2, 0).numpy()
        
        # Apply zero mask if necessary
        [lrv_ma] = zero_mask([lrv])
        [hrv_ma] = zero_mask([hrv])
        
        # Convert to torch tensors
        lrv_tensor = torch.tensor(lrv_ma.data, dtype=torch.float32).permute(2, 0, 1)
        hrv_tensor = torch.tensor(hrv_ma.data, dtype=torch.float32).permute(2, 0, 1)
        
        return lrv_tensor, hrv_tensor

class FTDataset(Dataset):
    # Dataset for fine-tuning the model
    def __init__(self, data_path):
        # Load data from pickle
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Determine the number of data points
        self.num_samples = self.data['fcpreds_interp_ma'].shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Fetch data for the specified index
        lrv = self.data['fcpreds_interp_ma'][idx, :, :, :]
        hrv = self.data['hr_ma'][idx, :, :, :]
        
        # Convert to torch tensors
        lrv_tensor = torch.tensor(lrv, dtype=torch.float32).permute(2, 0, 1)
        hrv_tensor = torch.tensor(hrv, dtype=torch.float32).permute(2, 0, 1)
        
        return lrv_tensor, hrv_tensor

def load_glorys_roms_whole(steps, 
                           time_dict = time_dict_glorys_roms, 
                           channels = ["SSU","SSV","SSH"],
                           added_channels = ["SSKE", "SSH_curve"]):
    
    channels_all = channels + added_channels
    numchannels = len(channels_all)
    
    gloryshr_temp = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
    romshr_temp = nc.Dataset(f"{data_dir}/ENKF_GoM/EnKF_surface_1993_5dmean_wholegom.nc")
    
    lrcoordmeshX, lrcoordmeshY = np.meshgrid(gloryshr_temp["longitude"][:], gloryshr_temp["latitude"][:])
    lrcoordmeshX, lrcoordmeshY = lrcoordmeshX.data, lrcoordmeshY.data
    hrcoordmeshX, hrcoordmeshY = romshr_temp["lon_rho"][:], romshr_temp["lat_rho"][:]
    hrcoordmeshX, hrcoordmeshY = hrcoordmeshX.data, hrcoordmeshY.data
    
    lrloncoords = lrcoordmeshX[0,:]
    lrlatcoords = lrcoordmeshY[:,0]
    hrloncoords = hrcoordmeshX[0,:]
    hrlatcoords = hrcoordmeshY[:,0]
    
    lonspace = np.abs(lrloncoords[1]-lrloncoords[0])
    latspace = np.abs(lrlatcoords[1]-lrlatcoords[0])
    
    g1 = gloryshr_temp["zos"][0,...]
    r1 = romshr_temp["SSH"][0,...]
    
    lrv_use = np.empty((len(steps),g1.shape[0],g1.shape[1],numchannels))
    hrv_use = np.empty((len(steps),r1.shape[0],r1.shape[1],numchannels))
    
    gloryshr_temp.close()
    romshr_temp.close()
    
    times = []
    convkernel = np.array([[-1.000,-1.414,-1.000],
                           [-1.414,+9.657,-1.414],
                           [-1.000,-1.414,-1.000]])/(9.657*lonspace*latspace) # semi-curvature kernel
                           
    for it, t in enumerate(steps,0):
        (year,day) = time_dict[t]
        times.append((year,day))

        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
        romshr = nc.Dataset(f"{data_dir}/ENKF_GoM/EnKF_surface_{year}_5dmean_wholegom.nc")
        
        for ich, ch in enumerate(channels_all,0):
            if ch == "SSU":
                lrv_use[it,:,:,ich] = gloryshr["uo"][day,0,...]
                hrv_use[it,:,:,ich] = romshr["SSU"][day]
            if ch == "SSV":
                lrv_use[it,:,:,ich] = gloryshr["vo"][day,0,...]
                hrv_use[it,:,:,ich] = romshr["SSV"][day]
            if ch == "SSH":
                lrv_use[it,:,:,ich] = gloryshr["zos"][day]
                hrv_use[it,:,:,ich] = romshr["SSH"][day]
            ## diagnostic channels
            if ch == "SSKE":
                uind = channels_all.index("SSU")
                vind = channels_all.index("SSV")
                lrv_use[it,:,:,ich] = lrv_use[it,:,:,uind]**2+lrv_use[it,:,:,vind]**2
                hrv_use[it,:,:,ich] = hrv_use[it,:,:,uind]**2+hrv_use[it,:,:,vind]**2
            if ch == "SSH_curve": # this does not include future info
                hind = channels_all.index("SSH")
                # convkernel = np.array([[1]])
                lrv_use[it,:,:,ich] = convolve2d(lrv_use[it,:,:,hind], convkernel, mode ="same", boundary="symm")
        
        gloryshr.close()
        romshr.close()
    
    return lrv_use, hrv_use, lrloncoords, lrlatcoords, hrloncoords, hrlatcoords, times
     
def load_glorys_roms_interpolate_whole(steps, 
                                       time_dict = time_dict_glorys_roms, 
                                       lat_lon_keep = (17.1, 30.9, -98.0, -74.1), 
                                       numchannels = 3, 
                                       interpolator_use = "scipy"):

    # print("Loading interpolated data...")
    
    gloryshr_temp = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_1993.nc")
    romshr_temp = nc.Dataset(f"{data_dir}/ENKF_GoM/EnKF_surface_1993_5dmean_wholegom.nc")
    
    g1 = gloryshr_temp["zos"][0,...]
    r1 = romshr_temp["SSH"][0,...]
    
    lrcoordmeshX, lrcoordmeshY = np.meshgrid(gloryshr_temp["longitude"][:], gloryshr_temp["latitude"][:])
    lrcoordmeshX, lrcoordmeshY = lrcoordmeshX.data, lrcoordmeshY.data
    hrcoordmeshX, hrcoordmeshY = romshr_temp["lon_rho"][:], romshr_temp["lat_rho"][:]
    hrcoordmeshX, hrcoordmeshY = hrcoordmeshX.data, hrcoordmeshY.data
    
    lrloncoords_og = lrcoordmeshX[0,:]
    lrlatcoords_og = lrcoordmeshY[:,0]
    
    lr = np.empty((len(steps),g1.shape[0],g1.shape[1],numchannels))
    hr = np.empty((len(steps),r1.shape[0],r1.shape[1],numchannels))
    
    gloryshr_temp.close()
    romshr_temp.close()
    
    times = []
    for it, t in enumerate(steps,0):
        (year,day) = time_dict[t]
        times.append((year,day))

        gloryshr = nc.Dataset(f"{data_dir}/GLORYS_GoM_YearWise_V2/cmems_mod_glo_phy_my_0.083deg_P1D-m_{year}.nc")
            
        tlen = len(gloryshr["time"])
        lr[it,:,:,0] = gloryshr["uo"][day,0,...]
        lr[it,:,:,1] = gloryshr["vo"][day,0,...]
        lr[it,:,:,2] = gloryshr["zos"][day]

        romshr = nc.Dataset(f"{data_dir}/ENKF_GoM/EnKF_surface_{year}_5dmean_wholegom.nc")
        hr[it,:,:,0] = romshr["SSU"][day]
        hr[it,:,:,1] = romshr["SSV"][day]
        hr[it,:,:,2] = romshr["SSH"][day]
        
        gloryshr.close()
        romshr.close()
        
        # if it %1000 == 0:
            # print(f" {it} : time step {t}")
    


    # for comparing same regions between lr and hr
    (latmin, latmax, lonmin, lonmax) = lat_lon_keep

    lrmask = ((latmin < lrcoordmeshY) * (lrcoordmeshY < latmax) * (lonmin < lrcoordmeshX) * (lrcoordmeshX < lonmax))
    hrmask = ((latmin < hrcoordmeshY) * (hrcoordmeshY < latmax) * (lonmin < hrcoordmeshX) * (hrcoordmeshX < lonmax))
    
    lrmask_lonKeep = lrmask.any(axis = 0)
    lrmask_latKeep = lrmask.any(axis = 1)
    hrmask_lonKeep = hrmask.any(axis = 0)
    hrmask_latKeep = hrmask.any(axis = 1)
    
    lr_reshape = lr[:, lrmask_latKeep, :, :]
    lr_reshape = lr_reshape[:, :, lrmask_lonKeep, :]

    hr_crop = hr[:, hrmask_latKeep, :, :]
    hr_crop = hr_crop[:, :, hrmask_lonKeep, :]
    
    lrloncoords = lrcoordmeshX[0,lrmask_lonKeep]
    lrlatcoords = lrcoordmeshY[lrmask_latKeep,0]
    loncoords = hrcoordmeshX[0,hrmask_lonKeep]
    latcoords = hrcoordmeshY[hrmask_latKeep,0]
    
    ## bicubic interpolation, torch
    if interpolator_use == "torch":
        lr_crop = interpolate(torch.from_numpy(lr_reshape).permute(0,3,1,2), hr_crop.shape[1:3], mode = "bicubic").permute(0,2,3,1).numpy()
    
    ## step x lat x lon x channel
    elif interpolator_use == "scipy":
        X, Y = np.meshgrid(loncoords, latcoords)
        lr_crop = np.tile(np.nan,hr_crop.shape)
        # print(lr_crop.shape)
        # print(Y.shape, X.shape)
        ## each time step and channel needs to be separated, so cant do each interpolation at once.
        for istep in range(hr_crop.shape[0]):
            for ivar in range(hr_crop.shape[3]):                                     
                interpolator = RegularGridInterpolator((lrlatcoords_og, lrloncoords_og), lr[istep,:,:,ivar], method = "nearest")

                lr_crop[istep,:,:,ivar] = interpolator((Y,X))
    else:
        raise Exception(f"Invalid interpolator: Valid options: torch, scipy.")
    
    return lr, hr, lr_crop, hr_crop, loncoords, latcoords, times

def load_times_waverys2(beg, end, time_dict = time_dict_waverys):
    wave_use = np.empty((end-beg,70,125,2))
    times = []
    
    for it, t in enumerate(range(beg,end),0):
        (year,step) = time_dict[t]
        times.append((year,step))

        waverys = nc.Dataset(f"{data_dir}/WAVERYS_GoM_YearWise/cmems_mod_glo_wav_my_0.2deg_PT3H-i_{year}.nc")
        # print(np.mean(waverys["VHM0"][int(step*8)]))
        tlen = len(waverys["time"])
        wave_use[it,:,:,0] = waverys["VHM0"][int(step*8)]
        wave_use[it,:,:,1] = waverys["VMDR"][int(step*8)]
  
        waverys.close()
    
    return wave_use, times

def load_times_waverys2_unitCircle(steps, time_dict = time_dict_waverys):
    wave_use = np.empty((len(steps),70,125,3))
    times = []
    channels = ["VHM0", "VMDR_x", "VMDR_y"]
    waverys = nc.Dataset(f"{data_dir}/WAVERYS_GoM_YearWise/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1994.nc")
    lons, lats = waverys["longitude"][:], waverys["latitude"][:]
    waverys.close()

    for it, t in enumerate(steps,0):
        (year,step) = time_dict[t]
        times.append((year,step))

        waverys = nc.Dataset(f"{data_dir}/WAVERYS_GoM_YearWise/cmems_mod_glo_wav_my_0.2deg_PT3H-i_{year}.nc")
        # print(np.mean(waverys["VHM0"][int(step*8)]))
        tlen = len(waverys["time"])
        wave_use[it,:,:,0] = waverys["VHM0"][int(step*8)]
        angle = waverys["VMDR"][int(step*8)]
        unitc_x = np.cos(angle*np.pi/180)
        unitc_y = np.sin(angle*np.pi/180)
        # wave_use[it,:,:,1] = unitc_x
        wave_use[it,:,:,1] = unitc_x
        wave_use[it,:,:,2] = unitc_y
  
        waverys.close()
    
    return wave_use, lons.data, lats.data, times, channels

# def load_times_waverys(beg, end, time_dict = time_dict_waverys):
    # wave_use = np.empty((end-beg,70,125,1))
    # times = []
    
    # for it, t in enumerate(range(beg,end),0):
        # (year,step) = time_dict[t]
        # times.append((year,step))

        # waverys = nc.Dataset(f"{data_dir}/WAVERYS_GoM_YearWise/cmems_mod_glo_wav_my_0.2deg_PT3H-i_{year}.nc")
        # # print(np.mean(waverys["VHM0"][int(step*8)]))
        # tlen = len(waverys["time"])
        # wave_use[it,:,:,0] = waverys["VHM0"][int(step*8)]

        # waverys.close()
    
    # return wave_use, times

def get_accs(preds, truths):
    ## steps x lat x lon x channels
    d2c = truths.mean(axis = 0)
    d1t = preds
    d2t = truths
    num = np.nansum((d1t - d2c)*(d2t - d2c), axis = (1,2))
    den = np.sqrt(np.nansum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.nansum((d2t - d2c)**2, axis = (1,2)))
    accs = num/den
    
    return accs

def get_rmse(preds, truths):
    return np.sqrt(np.nanmean((preds-truths)**2,axis=(1,2)))

def update_projection(ax, axi, projection='3d', fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = fig.add_subplot(rows, cols, start+1, projection=projection)

def zero_mask(arrs):
    ## Givin list of arrays with nan values representing masked regions
    ##  finds the overlapping masks, and returns masked numpy arrays with the same dimension and masks
    
    shared_mask = np.sum([np.isnan(arr) for arr in arrs], axis = 0)
    arrs_ma = [np.ma.masked_array(np.where(shared_mask, 0, arr), shared_mask) for arr in arrs]
    
    return arrs_ma

def plot_metrics_forecast(preds, 
                          truths, 
                          times, 
                          latcoords, 
                          loncoords, 
                          net_dir, 
                          nn_name, 
                          channels=["SSU", "SSV", "SSH"], 
                          channel_lims=[[-2,2],[-2,2],[-1,1]],
                          steps_plot = 200,
                          time_unit = "day",
                          time_per_step = 1.0,
                          do_animation = True):
    
    plt.rcParams.update({'font.size': 12})

    ## general forecast
    forecast_dir = f"{net_dir}/forecasts"
    if not os.path.exists(forecast_dir):
        os.makedirs(forecast_dir)
    print(f"Running for ''{nn_name}'':")
    

    time_steps = np.arange(steps_plot)*time_per_step

    ## rmse
    print("RMSE...")
    rmse = get_rmse(preds, truths)
    rmse_min = rmse.min(axis = 1)

    fig, axs = plt.subplots(len(channels),1,figsize = (8,3*len(channels)))
    for ivar, var in enumerate(channels,0):
        axs[ivar].plot(time_steps, rmse[:steps_plot,ivar])
        axs[ivar].set_ylim(-0.5, 3.0)
        axs[ivar].set_title(var)
        axs[ivar].grid()
        
    plt.suptitle("RMSE\n%s"%nn_name)
    axs[len(channels)-1].set_xlabel(time_unit)
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_RMSE_steps-{steps_plot}.png")
    plt.close()

    ## ACC
    print("ACC...")
    accs = get_accs(preds, truths)
    accs_min = accs.min(axis = 1)
    
    fig, axs = plt.subplots(len(channels),1,figsize = (8,3*len(channels)))
    for ivar, var in enumerate(channels,0):
      axs[ivar].plot(time_steps, accs[:steps_plot,ivar])
      axs[ivar].set_ylim(-0.5, 1.0)
      axs[ivar].set_title(var)
      axs[ivar].grid()

    plt.suptitle("ACC\n%s"%nn_name)
    axs[len(channels)-1].set_xlabel(time_unit)
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_ACC_steps-{steps_plot}.png")
    plt.close()

    ## Spectrums
    print("Spectrums...")
    spectrum_pred = np.abs(fft.rfft(preds[:,:,:,:], axis = 2)).mean(axis = 1)
    spectrum_actual_mean = np.abs(fft.rfft(truths[:,:,:,:], axis = 2)).mean(axis = (0,1))
    ks = np.arange(1, spectrum_pred.shape[1])

    steps = [b for b in [0,1,2,5,10,20,50,100,200,500] if b < steps_plot]
    colors = cm.rainbow(np.arange(len(steps))/(len(steps)-1))

    fig, axs = plt.subplots(len(channels),1,figsize = (8,3*len(channels)))
    for istep, step in enumerate(steps,0):
        for ivar, var in enumerate(channels,0):
            axs[ivar].plot(ks, 
                         spectrum_pred[step, ks, ivar],
                         color = colors[istep])
            
    for istep, step in enumerate(steps,0):
        axs[0].plot([],[],
                         color = colors[istep],label=f"{time_unit}: {np.around(step*time_per_step,3)}")

    for ivar, var in enumerate(channels,0):
        axs[ivar].plot(ks, 
                         spectrum_actual_mean[ks,ivar],
                         color = "black",
                         linestyle = "--")
        axs[ivar].grid(alpha = .8)
        axs[ivar].set_yscale("log")
        axs[ivar].set_ylabel(f"{var}")

    axs[0].plot([], 
                         [],
                         color = "black",
                         linestyle = "--",
                         label = "truth")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_spectrum_plots.png")
    plt.close()

    print("Animation...")
    ## animation predictions, actual, difference
    
    # projection = ccrs.Robinson()
    projection = ccrs.PlateCarree()
    for istep, time_step in enumerate(time_steps,0):
        fig, axs = plt.subplots(3,len(channels),figsize = (4*len(channels),7), subplot_kw={'projection': projection})
        
        for ivar, (var,lims) in enumerate(zip(channels,channel_lims),0):         
            im1 = axs[0,ivar].contourf(loncoords, latcoords, preds[istep,:,:,ivar], levels = np.linspace(lims[0], lims[1], 200), vmin = lims[0], vmax = lims[1], transform = projection, cmap="coolwarm", extend='both')
            im2 = axs[1,ivar].contourf(loncoords, latcoords, truths[istep,:,:,ivar], levels = np.linspace(lims[0], lims[1], 200), vmin = lims[0], vmax = lims[1], transform = projection, cmap="coolwarm", extend='both')
            im3 = axs[2,ivar].contourf(loncoords, latcoords, preds[istep,:,:,ivar]-truths[istep,:,:,ivar], levels = np.linspace( -1, 1, 200), vmin = -1, vmax=1, transform = projection, cmap="twilight", extend='both')
            
            axs[0,ivar].coastlines()
            axs[1,ivar].coastlines()
            axs[2,ivar].coastlines()
            
            cbar1 = fig.colorbar(im1, orientation='vertical', ticks = [lims[0], 0, lims[1]], fraction=0.046, pad=0.04)
            cbar2 = fig.colorbar(im2, orientation='vertical', ticks = [lims[0], 0, lims[1]], fraction=0.046, pad=0.04)
            cbar3 = fig.colorbar(im3, orientation='vertical', ticks = [-1.0, 0,1.0], fraction=0.046, pad=0.04)
            
            for ax_rnum in range(3):
                gl = axs[ax_rnum,ivar].gridlines(draw_labels=True, alpha = .3)
                
                gl.top_labels = False
                gl.right_labels = False
                
                if ivar != 0 or ax_rnum != 0:
                    gl.bottom_labels = False
                    gl.left_labels = False
            
            axs[0,ivar].set_title(f"{var}")
         
        axs[0,0].text(-0.2, 0.55, "Prediction", va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axs[0,0].transAxes)
        
        axs[1,0].text(-0.2, 0.55, "Truth", va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axs[1,0].transAxes)
        
        axs[2,0].text(-0.2, 0.55, r"$\Delta$", va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axs[2,0].transAxes)
        
        plt.suptitle(f"(year,day): {times[istep]}, pred step: {istep}\nmin ACC: {np.around(accs_min[istep],3)}, RMSE: {np.around(rmse_min[istep],3)}")
        str_step = "0"*(6-len(str(istep)))+str(istep)
        plt.savefig(f"{forecast_dir}/{nn_name}_pred_actual_diff_{str_step}.png")
        plt.tight_layout()
        plt.close()

    os.system(f'ffmpeg -y -r 10 -f image2 -s 1920x1080 -i {forecast_dir}/{nn_name}_pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_steps-{steps_plot}.mp4')

def plot_metrics_downscale(lowres, 
                           preds, 
                           truths,
                           times, 
                           latcoords,
                           loncoords,
                           net_dir, 
                           nn_name, 
                           channels=["SSU", "SSV", "SSH"], 
                           days = 200, 
                           do_animation = True):
    
    print(f"Running for ''{nn_name}'':")
    
    comparisons_dir = f"{net_dir}/comparisons_raw"
    if not os.path.exists(comparisons_dir):
        os.makedirs(comparisons_dir)
        
    print(f"Intensity scatter...")
    ## intensity scatter subsets the data to use for the pred/actual scatter plot
    numuse = 1
    x1, x2, y = preds.reshape(-1, preds.shape[-1]), lowres.reshape(-1, preds.shape[-1]), truths.reshape(-1, preds.shape[-1])
    x1, x2, y = x1[~x1.mask[:,0],:], x2[~x2.mask[:,0],:], y[~y.mask[:,0],:]

    ishuff = np.arange(x1.shape[0])
    np.random.shuffle(ishuff)
    ishuff = ishuff[:20000]
    x1, x2, y = x1[ishuff], x2[ishuff], y[ishuff]
    print(f" intensity scatter shapes: {x1.shape}, {x2.shape}, {y.shape}")
    
    fig, axs = plt.subplots(2,3,figsize = (10,6))
    
    for ich, ch in enumerate(channels,0):
        xy1 = np.vstack([x1[:,0],y[:,0]])
        z1 = gaussian_kde(xy1)(xy1)
        xy2 = np.vstack([x2[:,0],y[:,0]])
        z2 = gaussian_kde(xy2)(xy2)
        axs[0,ich].scatter(x1[:,0], y[:,0],c=z1,cmap="rainbow",s=2)
        axs[0,ich].plot([-2,2],[-2,2],color = "black",linestyle = "--", alpha = .4)
        xy1corr = spearmanr(x1[:,0],y[:,0]).statistic
        axs[0,ich].set_title(f"{ch} \n correlation: {np.around(xy1corr,3)}")
        axs[0,ich].set_xlim(-2,2)
        axs[0,ich].set_ylim(-2,2)
        axs[0,ich].grid()
        
        xy2corr = spearmanr(x2[:,0],y[:,0]).statistic
        axs[1,ich].scatter(x2[:,0], y[:,0],c=z2,cmap="rainbow",s=2)
        axs[1,ich].plot([-2,2],[-2,2],color = "black",linestyle = "--", alpha = .4)
        axs[1,ich].set_title(f"correlation: {np.around(xy2corr,3)}")
        axs[1,ich].set_xlim(-2,2)
        axs[1,ich].set_ylim(-2,2)
        axs[1,ich].grid()
        
    axs[0,0].set_xlabel("GLORYS FNO Downscaling")
    axs[0,0].set_ylabel("ROMS")
    axs[1,0].set_xlabel("Low Resolution Interpolation")
    axs[1,0].set_ylabel("ROMS")
        
    
    plt.suptitle("Value Comparison Plots, Pred vs Truth")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_valueComparison.png", dpi = 300)
    plt.close()


    print("Abs diff Image...")
    
    # fig, axs = plt.subplots(2,3,figsize = (10,6))
    fig, axs = plt.subplots(2,len(channels),figsize = (4*len(channels),7), subplot_kw={'projection': projection})
    
    pred_error = preds[:,::-1,:,:]-truths[:,::-1,:,:]
    lr_error = lowres[:,::-1,:,:]-truths[:,::-1,:,:]
    
    pred_error_mean = np.mean(pred_error, axis = 0)
    lr_error_mean = np.mean(lr_error, axis = 0)
    
    im1 = axs[0,ivar].contourf(loncoords, latcoords, pred_error_mean[...,0], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[0,0].set_title("SSU")
    im2 = axs[0,ivar].contourf(loncoords, latcoords, pred_error_mean[...,1], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[0,1].set_title("SSV")
    im3 = axs[0,ivar].contourf(loncoords, latcoords, pred_error_mean[...,2], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[0,2].set_title("SSH")
    axs[0,0].set_ylabel("GLORYS Downscaling")
    cbar1 = fig.colorbar(im1, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    cbar2 = fig.colorbar(im2, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    cbar3 = fig.colorbar(im3, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    
    
    im1 = axs[0,ivar].contourf(loncoords, latcoords, lr_error_mean[...,0], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[1,0].set_title("SSU")
    im2 = axs[0,ivar].contourf(loncoords, latcoords, lr_error_mean[...,1], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[1,1].set_title("SSV")
    im3 = axs[0,ivar].contourf(loncoords, latcoords, lr_error_mean[...,2], levels = np.linspace(-1, 1, 200), vmin = -1, vmax = 1, transform = projection, cmap="twilight", extend='both')
    axs[1,2].set_title("SSH")
    axs[1,0].set_ylabel("Low Resolution Interpolation")
    cbar1 = fig.colorbar(im1, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    cbar2 = fig.colorbar(im2, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    cbar3 = fig.colorbar(im3, orientation='vertical', ticks = [-1, 0, 1], fraction=0.046, pad=0.04)
    
    for ax_rnum in range(2):
        for ich, ch in enumerate(channels,0):
            gl = axs[ax_rnum,ich].gridlines(draw_labels=True, alpha = .3)
            
            gl.top_labels = False
            gl.right_labels = False
            
            if ich != 0 or ax_rnum != 0:
                gl.bottom_labels = False
                gl.left_labels = False
    
    divider1 = make_axes_locatable(axs[0,0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    
    plt.suptitle("Error Interpolation and Downscaling, Time Averaged")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_absError.png", dpi = 300)
    plt.close()
    
    if do_animation:
        comparisons_dir = f"{net_dir}/comparisons_raw"
        if not os.path.exists(comparisons_dir):
            os.makedirs(comparisons_dir)
    
        print("Animation Raw...")
        ## ds vs roms, actual, difference
        for step in range(days):
            fig, axs = plt.subplots(4,3,figsize = (13,9))
            for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[-2,2],[-2,2],[-1,1]]),0):
                im1 = axs[0,ivar].imshow(lowres[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im2 = axs[1,ivar].imshow(preds[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im3 = axs[2,ivar].imshow(truths[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im4 = axs[3,ivar].imshow(np.abs(truths[step,::-1,:,ivar]-preds[step,::-1,:,ivar]),vmin=0,vmax=2,cmap="inferno")
                
                divider1 = make_axes_locatable(axs[0,ivar])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                divider2 = make_axes_locatable(axs[1,ivar])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                divider3 = make_axes_locatable(axs[2,ivar])
                cax3 = divider3.append_axes('right', size='5%', pad=0.05)
                divider4 = make_axes_locatable(axs[3,ivar])
                cax4 = divider4.append_axes('right', size='5%', pad=0.05)
                
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                fig.colorbar(im3, cax=cax3, orientation='vertical')
                fig.colorbar(im4, cax=cax4, orientation='vertical')
                axs[0,ivar].set_title(f"{var}")

            axs[0,0].set_ylabel("Interpolated GLORYS")
            axs[1,0].set_ylabel("SR.IN. GLORYS")
            axs[2,0].set_ylabel("ROMS")
            axs[3,0].set_ylabel(r"$|\Delta|$\nROMS - SR INTERP GLORYS")

            plt.suptitle(f"(year,day): {times[step]}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plt.savefig(f"{comparisons_dir}/{nn_name}_pred_actual_diff_{str_step}.png")
            plt.tight_layout()
            plt.close()

        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {comparisons_dir}/{nn_name}_pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_days-{days}.mp4')
        
        
        comparisons_dir = f"{net_dir}/comparisons_diff"
        if not os.path.exists(comparisons_dir):
            os.makedirs(comparisons_dir)
    
        print("Animation Diff...")
        ## ds vs roms, actual, difference
        for step in range(days):
            fig, axs = plt.subplots(3,3,figsize = (13,7))
            for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[0,2],[0,2],[0,1]]),0):
                im1 = axs[0,ivar].imshow(np.abs(preds[step,::-1,:,ivar]-truths[step,::-1,:,ivar]),vmin=lims[0],vmax=lims[1], cmap = "inferno")
                im2 = axs[1,ivar].imshow(np.abs(lowres[step,::-1,:,ivar]-truths[step,::-1,:,ivar]),vmin=lims[0],vmax=lims[1], cmap = "inferno")
                im3 = axs[2,ivar].imshow(preds[step,::-1,:,ivar]-lowres[step,::-1,:,ivar],vmin=-1,vmax=1,cmap="coolwarm")
                
                divider1 = make_axes_locatable(axs[0,ivar])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                divider2 = make_axes_locatable(axs[1,ivar])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                divider3 = make_axes_locatable(axs[2,ivar])
                cax3 = divider3.append_axes('right', size='5%', pad=0.05)
                
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                fig.colorbar(im3, cax=cax3, orientation='vertical')
                axs[0,ivar].set_title(f"{var}")

            axs[0,0].set_ylabel("ABS(SR.IN. GLORYS - ROMS)")
            axs[1,0].set_ylabel("ABS(IN. GLORYS - ROMS)")
            axs[2,0].set_ylabel(r"Adjustment: SR - IN. GLORYS")

            plt.suptitle(f"(year,day): {times[step]}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plt.savefig(f"{comparisons_dir}/diff_{str_step}.png")
            plt.tight_layout()
            plt.close()

        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {comparisons_dir}/diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred-truth_interp-truth_diff_days-{days}.mp4')



def plot_metrics_glorys_forecast(preds, truths, times, net_dir, nn_name, days, do_animation = True): 
    
    forecast_dir = f"{net_dir}/forecasts"

    if not os.path.exists(forecast_dir):
        os.makedirs(forecast_dir)
    print(f"Running for  ''{nn_name}'':")
    ## rmse
    print("RMSE...")
    rmse = np.sqrt(np.mean((preds-truths)**2,axis=(1,2)))
    days = 200

    fig, axs = plt.subplots(3,1,figsize = (8,8))
    axs[0].plot(np.arange(days), rmse[:days,0])
    axs[0].set_title("SSU")
    axs[0].grid()
    axs[1].plot(np.arange(days), rmse[:days,1])
    axs[1].set_title("SSV")
    axs[1].grid()
    axs[2].plot(np.arange(days), rmse[:days,2])
    axs[2].set_title("SSH")
    axs[2].grid()
    axs[2].set_xlabel("days")
    plt.suptitle("RMSE")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_RMSE_days-{days}.png")
    plt.close()

    ## ACC
    print("ACC...")
    d2c = truths.mean(axis = 0)
    d1t = preds
    d2t = truths
    num = np.nansum((d1t - d2c)*(d2t - d2c), axis = (1,2))
    den = np.sqrt(np.nansum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.nansum((d2t - d2c)**2, axis = (1,2)))
    accs = num/den

    fig, axs = plt.subplots(3,1,figsize = (8,8))
    axs[0].plot(np.arange(days), accs[:days,0])
    axs[0].set_title("SSU")
    axs[0].grid()
    axs[0].set_ylim(-.5, 1.0)
    axs[1].plot(np.arange(days), accs[:days,1])
    axs[1].set_title("SSV")
    axs[1].grid()
    axs[1].set_ylim(-.5, 1.0)
    axs[2].plot(np.arange(days), accs[:days,2])
    axs[2].set_title("SSH")
    axs[2].grid()
    axs[2].set_ylim(-.5, 1.0)
    axs[2].set_xlabel("days")
    plt.suptitle("ACC")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_ACC_days-{days}.png")
    plt.close()
    
    ## Spectrums
    print("Spectrums...")
    spectrum_pred = np.abs(fft.rfft(preds[:,:,:,:], axis = 2)).mean(axis = 1)
    spectrum_actual_mean = np.abs(fft.rfft(truths[:,:,:,:], axis = 2)).mean(axis = (0,1))
    ks = np.arange(1, spectrum_pred.shape[1])

    steps = [b for b in [0,1,2,5,10,20,50,100,200,500] if b <= days]
    colors = cm.rainbow(np.arange(len(steps))/(len(steps)-1))

    fig, axs = plt.subplots(3,1,figsize = (8,8))
    for istep, step in enumerate(steps,0):
        for ivar, var in enumerate(["SSU","SSV","SSH"],0):
            axs[ivar].plot(ks, 
                         spectrum_pred[step, ks, ivar],
                         color = colors[istep])
            
    for istep, step in enumerate(steps,0):
        axs[0].plot([],[],
                         color = colors[istep],label=f"pred step: {step}")

    for ivar, var in enumerate(["SSU","SSV","SSH"],0):
        axs[ivar].plot(ks, 
                         spectrum_actual_mean[ks,ivar],
                         color = "black",
                         linestyle = "--")
        axs[ivar].grid(alpha = .8)
        axs[ivar].set_yscale("log")
        axs[ivar].set_ylabel(f"{var}")

    axs[0].plot([], 
                         [],
                         color = "black",
                         linestyle = "--",
                         label = "truth")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_spectrum_plots.png")
    plt.close()

    if do_animation:
        print("Animation...")
        ## animation predictions, actual, difference
        for step in range(days):
            fig, axs = plt.subplots(3,3,figsize = (13,7))
            for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[-2,2],[-2,2],[-1,1]]),0):
                im1 = axs[0,ivar].imshow(preds[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im2 = axs[1,ivar].imshow(truths[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im3 = axs[2,ivar].imshow(np.abs(truths[step,::-1,:,ivar]-preds[step,::-1,:,ivar]),vmin=0,vmax=2,cmap="inferno")
                
                divider1 = make_axes_locatable(axs[0,ivar])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                divider2 = make_axes_locatable(axs[1,ivar])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                divider3 = make_axes_locatable(axs[2,ivar])
                cax3 = divider3.append_axes('right', size='5%', pad=0.05)
                
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                fig.colorbar(im3, cax=cax3, orientation='vertical')
                axs[0,ivar].set_title(f"{var}")

            axs[0,0].set_ylabel("Prediction")
            axs[1,0].set_ylabel("Actual")
            axs[2,0].set_ylabel(r"$\Delta$")

            plt.suptitle(f"(year,day): {times[step]}, pred step: {step}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plt.savefig(f"{forecast_dir}/pred_actual_diff_{str_step}.png")
            plt.tight_layout()
            plt.close()

        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {forecast_dir}/pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_days-{days}.mp4')
        
    return accs, rmse

def plot_metrics_waverys_forecast_1ch(preds, truths, times, net_dir, nn_name, days):
    
    forecast_dir = f"{net_dir}/forecasts"

    if not os.path.exists(forecast_dir):
        os.makedirs(forecast_dir)
    print(f"Running for  ''{nn_name}'':")
    ## rmse
    print("RMSE...")
    ave_mse = np.sqrt(np.mean((preds-truths)**2,axis=(1,2)))
    days = 200

    fig, axs = plt.subplots(1,1,figsize = (8,4))
    axs.plot(np.arange(days), ave_mse[:days,0])
    axs.set_title("VHM0")
    axs.grid()
    axs.set_xlabel("days")
    plt.suptitle("RMSE")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_RMSE_days-{days}.png")
    plt.close()

    ## ACC
    print("ACC...")
    d2c = truths.mean(axis = 0)
    d1t = preds
    d2t = truths
    num = np.nansum((d1t - d2c)*(d2t - d2c), axis = (1,2))
    den = np.sqrt(np.nansum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.nansum((d2t - d2c)**2, axis = (1,2)))
    accs = num/den

    fig, axs = plt.subplots(1,1,figsize = (8,4))
    axs.plot(np.arange(days), accs[:days,0])
    axs.set_title("VHM0")
    axs.grid()
    axs.set_xlabel("days")
    plt.suptitle("ACC")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_ACC_days-{days}.png")
    plt.close()

    ## Spectrums
    print("Spectrums...")
    spectrum_pred = np.abs(fft.rfft(preds[:,:,:,:], axis = 2)).mean(axis = 1)
    spectrum_actual_mean = np.abs(fft.rfft(truths[:,:,:,:], axis = 2)).mean(axis = (0,1))
    ks = np.arange(1, spectrum_pred.shape[1])

    steps = [b for b in [0,1,2,5,10,20,50,100,200,500] if b <= days]
    colors = cm.rainbow(np.arange(len(steps))/(len(steps)-1))

    fig, axs = plt.subplots(1,1,figsize = (8,4))
    for istep, step in enumerate(steps,0):
        axs.plot(ks, 
                     spectrum_pred[step, ks, 0],
                     color = colors[istep])
            
    for istep, step in enumerate(steps,0):
        axs.plot([],[],
                         color = colors[istep],label=f"pred step: {step}")

    axs.plot(ks, 
                     spectrum_actual_mean[ks,0],
                     color = "black",
                     linestyle = "--")
    axs.grid(alpha = .8)
    axs.set_yscale("log")
    axs.set_ylabel(f"VHM0")

    axs.plot([], 
                         [],
                         color = "black",
                         linestyle = "--",
                         label = "truth")

    axs.legend()
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_spectrum_plots.png")
    plt.close()

    print("Animation...")
    ## animation predictions, actual, difference
    for step in range(days):
        fig, axs = plt.subplots(3,1,figsize = (6,7))
        for ivar, (var,lims) in enumerate(zip(["VHM0"],[[-2,2]]),0):
            im1 = axs[0].imshow(preds[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
            im2 = axs[1].imshow(truths[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
            im3 = axs[2].imshow(np.abs(truths[step,::-1,:,ivar]-preds[step,::-1,:,ivar]),vmin=0,vmax=2,cmap="inferno")
            
            divider1 = make_axes_locatable(axs[0])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            divider2 = make_axes_locatable(axs[1])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            divider3 = make_axes_locatable(axs[2])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            
            fig.colorbar(im1, cax=cax1, orientation='vertical')
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            fig.colorbar(im3, cax=cax3, orientation='vertical')
            axs[0].set_title(f"{var}")

        axs[0].set_ylabel("Prediction")
        axs[1].set_ylabel("Actual")
        axs[2].set_ylabel(r"$\Delta$")

        plt.suptitle(f"(year,day): ({times[step][0]}, {format(times[step][1], '.3f')}), pred step: {step}")
        str_step = "0"*(6-len(str(step)))+str(step)
        plt.savefig(f"{forecast_dir}/pred_actual_diff_{str_step}.png")
        plt.tight_layout()
        plt.close()

    os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {forecast_dir}/pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_days-{days}.mp4')

def angle_diff_deg(theta1, theta2):
    dtheta = abs(theta1-theta2)%360
    if dtheta <= 180:
        return dtheta
    else:
        return 360-dtheta

def plot_metrics_forecast_waverys_2ch_angle(preds, truths, times, net_dir, nn_name, days, channels=["VHM0","VMDR"]):
    
    ## general forecast
    forecast_dir = f"{net_dir}/forecasts"

    if not os.path.exists(forecast_dir):
        os.makedirs(forecast_dir)
    print(f"Running for  ''{nn_name}'':")
    ## rmse
    print("RMSE...")
    ave_mse_h = np.sqrt(np.mean((preds[...,0]-preds[...,0])**2,axis=(1,2)))
    days = 200

    fig, axs = plt.subplots(len(channels),1,figsize = (8,4*len(channels)))
    axs[0].plot(np.arange(days), ave_mse[:days])
    axs[0].set_title(var)
    axs[0].grid()
    
    plt.suptitle("RMSE")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_RMSE_days-{days}.png")
    plt.close()

    ## ACC
    print("ACC...")
    d2c = truths.mean(axis = 0)
    d1t = preds
    d2t = truths
    num = np.nansum((d1t - d2c)*(d2t - d2c), axis = (1,2))
    den = np.sqrt(np.nansum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.nansum((d2t - d2c)**2, axis = (1,2)))
    accs = num/den

    fig, axs = plt.subplots(len(channels),1,figsize = (8,4*len(channels)))
    for ivar, var in enumerate(channels,0):
      axs[ivar].plot(np.arange(days), accs[:days,ivar])
      axs[ivar].set_title(var)
      axs[ivar].grid()

    plt.suptitle("ACC")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_ACC_days-{days}.png")
    plt.close()

    ## Spectrums
    print("Spectrums...")
    spectrum_pred = np.abs(fft.rfft(preds[:,:,:,:], axis = 2)).mean(axis = 1)
    spectrum_actual_mean = np.abs(fft.rfft(truths[:,:,:,:], axis = 2)).mean(axis = (0,1))
    ks = np.arange(1, spectrum_pred.shape[1])

    steps = [b for b in [0,1,2,5,10,20,50,100,200,500] if b <= days]
    colors = cm.rainbow(np.arange(len(steps))/(len(steps)-1))

    fig, axs = plt.subplots(len(channels),1,figsize = (8,4*len(channels)))
    for istep, step in enumerate(steps,0):
        for ivar, var in enumerate(channels,0):
            axs[ivar].plot(ks, 
                         spectrum_pred[step, ks, ivar],
                         color = colors[istep])
            
    for istep, step in enumerate(steps,0):
        axs[0].plot([],[],
                         color = colors[istep],label=f"pred step: {step}")

    for ivar, var in enumerate(channels,0):
        axs[ivar].plot(ks, 
                         spectrum_actual_mean[ks,ivar],
                         color = "black",
                         linestyle = "--")
        axs[ivar].grid(alpha = .8)
        axs[ivar].set_yscale("log")
        axs[ivar].set_ylabel(f"{var}")

    axs[0].plot([], 
                         [],
                         color = "black",
                         linestyle = "--",
                         label = "truth")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_spectrum_plots.png")
    plt.close()

    print("Animation...")
    ## animation predictions, actual, difference
    for step in range(days):
        fig, axs = plt.subplots(3,len(channels),figsize = (4*len(channels),7))
        for ivar, (var,lims,cmapuse) in enumerate(zip(channels,[[-2,2],[0,360]]),0):
            im1 = axs[0,ivar].imshow(preds[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
            im2 = axs[1,ivar].imshow(truths[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
            im3 = axs[2,ivar].imshow(np.abs(truths[step,::-1,:,ivar]-preds[step,::-1,:,ivar]),vmin=0,vmax=2,cmap="inferno")
            
            divider1 = make_axes_locatable(axs[0,ivar])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            divider2 = make_axes_locatable(axs[1,ivar])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            divider3 = make_axes_locatable(axs[2,ivar])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            
            fig.colorbar(im1, cax=cax1, orientation='vertical')
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            fig.colorbar(im3, cax=cax3, orientation='vertical')
            axs[0,ivar].set_title(f"{var}")

        axs[0,0].set_ylabel("Prediction")
        axs[1,0].set_ylabel("Actual")
        axs[2,0].set_ylabel(r"$\Delta$")

        plt.suptitle(f"(year,day): {times[step]}, pred step: {step}")
        str_step = "0"*(6-len(str(step)))+str(step)
        plt.savefig(f"{forecast_dir}/pred_actual_diff_{str_step}.png")
        plt.tight_layout()
        plt.close()

    os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {forecast_dir}/pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_days-{days}.mp4')

def plot_metrics_glorys_roms_ds(lowres, preds, truths, times, net_dir, nn_name, days, do_animation = True):
    
    print(f"Running for ''{nn_name}'':")
    
    numuse = 1
    x1, x2, y = preds.reshape(-1, preds.shape[-1]), lowres.reshape(-1, preds.shape[-1]), truths.reshape(-1, preds.shape[-1])
    x1, x2, y = x1[~x1.mask[:,0],:], x2[~x2.mask[:,0],:], y[~y.mask[:,0],:]
    print(x1.shape)
    ishuff = np.arange(x1.shape[0])
    np.random.shuffle(ishuff)
    ishuff = ishuff[:10000]
    x1, x2, y = x1[ishuff], x2[ishuff], y[ishuff]

    print(f"Intensity scatter: {x1.shape},{x2.shape},{y.shape}...")
    fig, axs = plt.subplots(2,3,figsize = (10,6))
    
    xy1 = np.vstack([x1[:,0],y[:,0]])
    z1 = gaussian_kde(xy1)(xy1)
    xy2 = np.vstack([x2[:,0],y[:,0]])
    z2 = gaussian_kde(xy2)(xy2)
    axs[0,0].scatter(x1[:,0], y[:,0],c=z1,cmap="rainbow",s=2)
    axs[0,0].set_title(f"SSU \n correlation: {np.around(spearmanr(x1[:,0],y[:,0]).statistic,3)}")
    axs[0,0].set_xlim(-2,2)
    axs[0,0].set_ylim(-2,2)
    axs[0,0].grid()
    axs[1,0].scatter(x2[:,0], y[:,0],c=z2,cmap="rainbow",s=2)
    axs[1,0].set_title(f"correlation: {np.around(spearmanr(x2[:,0],y[:,0]).statistic,3)}")
    axs[1,0].set_xlim(-2,2)
    axs[1,0].set_ylim(-2,2)
    axs[1,0].grid()
    
    axs[0,0].set_xlabel("GLORYS FNO Downscaling")
    axs[0,0].set_ylabel("ROMS")
    axs[1,0].set_xlabel("Low Resolution Interpolation")
    axs[1,0].set_ylabel("ROMS")
    
    xy1 = np.vstack([x1[:,1],y[:,1]])
    z1 = gaussian_kde(xy1)(xy1)
    xy2 = np.vstack([x2[:,1],y[:,1]])
    z2 = gaussian_kde(xy2)(xy2)
    axs[0,1].scatter(x1[:,1], y[:,1],c=z1,cmap="rainbow",s=2)
    axs[0,1].set_title(f"SSV\n correlation: {np.around(spearmanr(x1[:,1],y[:,1]).statistic,3)}")
    axs[0,1].grid()
    axs[0,1].set_xlim(-2,2)
    axs[0,1].set_ylim(-2,2)
    axs[1,1].scatter(x2[:,1], y[:,1],c=z2,cmap="rainbow",s=2)
    axs[1,1].set_title(f"correlation: {np.around(spearmanr(x2[:,1],y[:,1]).statistic,3)}")
    axs[1,1].grid()
    axs[1,1].set_xlim(-2,2)
    axs[1,1].set_ylim(-2,2)
    
    xy1 = np.vstack([x1[:,2],y[:,2]])
    z1 = gaussian_kde(xy1)(xy1)
    xy2 = np.vstack([x2[:,2],y[:,2]])
    z2 = gaussian_kde(xy2)(xy2)
    axs[0,2].scatter(x1[:,2], y[:,2],c=z1,cmap="rainbow",s=2)
    axs[0,2].set_title(f"SSH\n correlation: {np.around(spearmanr(x1[:,2],y[:,2]).statistic,3)}")
    axs[0,2].grid()
    axs[0,2].set_xlim(-2,2)
    axs[0,2].set_ylim(-2,2)
    axs[1,2].scatter(x2[:,2], y[:,2],c=z2,cmap="rainbow",s=2)
    axs[1,2].set_title(f"correlation: {np.around(spearmanr(x2[:,2],y[:,2]).statistic,3)}")
    axs[1,2].grid()
    axs[1,2].set_xlim(-2,2)
    axs[1,2].set_ylim(-2,2)
    
    plt.suptitle("Value Comparison Plots")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_valueComparison.png", dpi = 300)
    plt.close()

    print("Abs diff Image...")
    sqrerror_pred = np.abs(preds[:,::-1,:,:]-truths[:,::-1,:,:])
    sqrerror_lr = np.abs(lowres[:,::-1,:,:]-truths[:,::-1,:,:])
    
    sqrerror_pred_mean = np.mean(sqrerror_pred, axis = 0)
    sqrerror_lr_mean = np.mean(sqrerror_lr, axis = 0)
    days = 200
    fig, axs = plt.subplots(2,3,figsize = (10,6))
    im1 = axs[0,0].imshow(sqrerror_pred_mean[...,0], vmin=0, vmax = 1, cmap="inferno")
    axs[0,0].set_title("SSU")
    axs[0,1].imshow(sqrerror_pred_mean[...,1], vmin=0, vmax = 1, cmap="inferno")
    axs[0,1].set_title("SSV")
    axs[0,2].imshow(sqrerror_pred_mean[...,2], vmin=0, vmax = 1, cmap="inferno")
    axs[0,2].set_title("SSH")
    axs[0,0].set_ylabel("GLORYS FNO Downscaling")
    
    axs[1,0].imshow(sqrerror_lr_mean[...,0], vmin=0, vmax = 1, cmap="inferno")
    axs[1,0].set_title("SSU")
    axs[1,1].imshow(sqrerror_lr_mean[...,1], vmin=0, vmax = 1, cmap="inferno")
    axs[1,1].set_title("SSV")
    axs[1,2].imshow(sqrerror_lr_mean[...,2], vmin=0, vmax = 1, cmap="inferno")
    axs[1,2].set_title("SSH")
    axs[1,0].set_ylabel("Low Resolution Interpolation")
    
    divider1 = make_axes_locatable(axs[0,0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    
    plt.suptitle("Absolute Error Grid, Time Averaged")
    plt.tight_layout()
    plt.savefig(f"{net_dir}/{nn_name}_absError.png", dpi = 300)
    plt.close()
    

    if do_animation:
        comparisons_dir = f"{net_dir}/comparisons_raw"
        if not os.path.exists(comparisons_dir):
            os.makedirs(comparisons_dir)
    
        print("Animation Raw...")
        ## ds vs roms, actual, difference
        for step in range(days):
            fig, axs = plt.subplots(4,3,figsize = (13,9))
            for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[-2,2],[-2,2],[-1,1]]),0):
                im1 = axs[0,ivar].imshow(lowres[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im2 = axs[1,ivar].imshow(preds[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im3 = axs[2,ivar].imshow(truths[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
                im4 = axs[3,ivar].imshow(np.abs(truths[step,::-1,:,ivar]-preds[step,::-1,:,ivar]),vmin=0,vmax=2,cmap="inferno")
                
                divider1 = make_axes_locatable(axs[0,ivar])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                divider2 = make_axes_locatable(axs[1,ivar])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                divider3 = make_axes_locatable(axs[2,ivar])
                cax3 = divider3.append_axes('right', size='5%', pad=0.05)
                divider4 = make_axes_locatable(axs[3,ivar])
                cax4 = divider4.append_axes('right', size='5%', pad=0.05)
                
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                fig.colorbar(im3, cax=cax3, orientation='vertical')
                fig.colorbar(im4, cax=cax4, orientation='vertical')
                axs[0,ivar].set_title(f"{var}")

            axs[0,0].set_ylabel("Interpolated GLORYS")
            axs[1,0].set_ylabel("SR.IN. GLORYS")
            axs[2,0].set_ylabel("ROMS")
            axs[3,0].set_ylabel(r"$|\Delta|$\nROMS - SR INTERP GLORYS")

            plt.suptitle(f"(year,day): {times[step]}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plt.savefig(f"{comparisons_dir}/pred_actual_diff_{str_step}.png")
            plt.tight_layout()
            plt.close()

        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {comparisons_dir}/pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred_actual_diff_days-{days}.mp4')
        
        
        comparisons_dir = f"{net_dir}/comparisons_diff"
        if not os.path.exists(comparisons_dir):
            os.makedirs(comparisons_dir)
    
        print("Animation Diff...")
        ## ds vs roms, actual, difference
        for step in range(days):
            fig, axs = plt.subplots(3,3,figsize = (13,7))
            for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[0,2],[0,2],[0,1]]),0):
                im1 = axs[0,ivar].imshow(np.abs(preds[step,::-1,:,ivar]-truths[step,::-1,:,ivar]),vmin=lims[0],vmax=lims[1], cmap = "inferno")
                im2 = axs[1,ivar].imshow(np.abs(lowres[step,::-1,:,ivar]-truths[step,::-1,:,ivar]),vmin=lims[0],vmax=lims[1], cmap = "inferno")
                im3 = axs[2,ivar].imshow(preds[step,::-1,:,ivar]-lowres[step,::-1,:,ivar],vmin=-1,vmax=1,cmap="coolwarm")
                
                divider1 = make_axes_locatable(axs[0,ivar])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                divider2 = make_axes_locatable(axs[1,ivar])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                divider3 = make_axes_locatable(axs[2,ivar])
                cax3 = divider3.append_axes('right', size='5%', pad=0.05)
                
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                fig.colorbar(im3, cax=cax3, orientation='vertical')
                axs[0,ivar].set_title(f"{var}")

            axs[0,0].set_ylabel("ABS(SR.IN. GLORYS - ROMS)")
            axs[1,0].set_ylabel("ABS(IN. GLORYS - ROMS)")
            axs[2,0].set_ylabel(r"Adjustment: SR - IN. GLORYS")

            plt.suptitle(f"(year,day): {times[step]}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plt.savefig(f"{comparisons_dir}/diff_{str_step}.png")
            plt.tight_layout()
            plt.close()

        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {comparisons_dir}/diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {net_dir}/{nn_name}_pred-truth_interp-truth_diff_days-{days}.mp4')

def plot_testLoss():
    fig, axs = plt.subplots(2,3,figsize=(10,6))

    lrvtest_temp, hrv_use, lrloncoords, lrlatcoords, hrloncoords, hrlatcoords, times = load_glorys_roms_2(np.arange(0,30))
    
    lrvtest_temp = lrvtest_temp[:,::-1,:,:]
    
    lrvmask = np.isnan(lrvtest_temp)
    lrvtest_temp = np.where(np.isnan(lrvtest_temp), 0, lrvtest_temp)
    lrvtest_temp = np.ma.masked_array(lrvtest_temp, lrvmask)

    pred1 = lrvtest_temp[0,:,:,0]
    test1 = lrvtest_temp[20,:,:,0]
    pred1spec = fft.rfft(pred1)
    test1spec = fft.rfft(test1)

    axs[0,0].imshow(pred1, cmap = "coolwarm")
    axs[1,0].imshow(np.abs(pred1spec), cmap = "coolwarm")
    axs[0,1].imshow(test1, cmap = "coolwarm")
    axs[1,1].imshow(np.abs(test1spec), cmap = "coolwarm")
    axs[0,2].imshow((pred1-test1)**2, cmap = "inferno",vmin=0)
    axs[1,2].imshow(np.abs(pred1spec-test1spec)**2, cmap = "inferno",vmin=0)
    axs[0,0].set_title("Prediction")
    axs[0,1].set_title("Actual")
    axs[0,2].set_title("Squared Difference")
    
    axs[0,0].set_ylabel("Grid Space")
    axs[1,0].set_ylabel("Spectral (lon) Space")
    
    plt.show()

def plot_variance():
    pass

# ## test plot of WAVERLYS data 
# fig, axs = plt.subplots(2,1,figsize=(8,8))
# waves2, times = load_times_waverys2(0,10)
# im1 = axs[0].imshow(waves2[0,:,:,0])
# im2 = axs[1].imshow(waves2[0,:,:,1], cmap="hsv")

# divider1 = make_axes_locatable(axs[0])
# cax1 = divider1.append_axes('right', size='5%', pad=0.05)
# divider2 = make_axes_locatable(axs[1])
# cax2 = divider2.append_axes('right', size='5%', pad=0.05)

# axs[0].set_title("Wave Significant Height")
# axs[1].set_title("Mean Wave Direction")

# fig.colorbar(im1, cax=cax1, orientation='vertical')
# fig.colorbar(im2, cax=cax2, orientation='vertical')

# plt.show()

## nicer plots using cartopy
# fig = plt.figure()
 
# # Set the axes using the specified map projection
# ax=plt.axes(projection=ccrs.PlateCarree())
 
# # Make a filled contour plot
# im = ax.contourf(lon_GLORYS, lat_GLORYS, zos[0], 60, vmin = np.nanmin(ssh[0]), vmax = np.nanmax(ssh[0]),
            # transform = ccrs.PlateCarree(), cmap='coolwarm')
 
# # Add coastlines
# ax.coastlines()
# gl = ax.gridlines(draw_labels=True)
# gl.top_labels = False
# gl.right_labels = False
# clb = plt.colorbar(im, fraction=0.025, pad=0.04)
# clb.ax.set_ylabel('SSH')
# plt.title('GLORYS_SSH')
# plt.show()
 
 
# fig = plt.figure()
 
# # Set the axes using the specified map projection
# ax=plt.axes(projection=ccrs.PlateCarree())
 
# # Make a filled contour plot
# im = ax.contourf(lon_ROMS[0,:], lat_ROMS[:,0], ssh[0], 60, vmin = np.nanmin(ssh[0]), vmax = np.nanmax(ssh[0]),
            # transform = ccrs.PlateCarree(), cmap='coolwarm')
 
# # Add coastlines
# ax.coastlines()
# gl = ax.gridlines(draw_labels=True)
# gl.top_labels = False
# gl.right_labels = False
# clb = plt.colorbar(im, fraction=0.025, pad=0.04)
# clb.ax.set_ylabel('SSH')
# plt.title('ROMS_SSH')
 
# plt.show()


# def plot_extreme_events(lr, times, outdir, days, do_animation = True): 
    
    # plot_dir = f"{outdir}/extremes"

    # if not os.path.exists(plot_dir):
        # os.makedirs(plot_dir)
    
    # lr_mean = lr.mean(axis = 0)
    # lr_std = lr.std(axis = 0)
    
    # print("Extremes Animation...")
    # ## animation predictions, actual, difference
    # for step in range(days):
        # fig, axs = plt.subplots(3,4,figsize = (13,7))
        # for ivar, (var,lims) in enumerate(zip(["SSU","SSV","SSH"],[[-2,2],[-2,2],[-1,1]]),0):
            # im1 = axs[ivar, 0].imshow(lr[step,::-1,:,ivar],vmin=lims[0],vmax=lims[1], cmap = "coolwarm")
            # im3 = axs[ivar, 2].imshow(,vmin=0,vmax=2,cmap="inferno")
            
            # divider1 = make_axes_locatable(axs[0,ivar])
            # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            # divider2 = make_axes_locatable(axs[1,ivar])
            # cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            # divider3 = make_axes_locatable(axs[2,ivar])
            # cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            
            # fig.colorbar(im1, cax=cax1, orientation='vertical')
            # fig.colorbar(im2, cax=cax2, orientation='vertical')
            # fig.colorbar(im3, cax=cax3, orientation='vertical')
            # axs[ivar,0].set_title(f"{var}")

        # axs[0,0].set_ylabel("Prediction")
        # axs[0,1].set_ylabel("Actual")
        # axs[0,2].set_ylabel(r"$\Delta$")

        # plt.suptitle(f"(year,day): {times[step]}, pred step: {step}")
        # str_step = "0"*(6-len(str(step)))+str(step)
        # plt.savefig(f"{forecast_dir}/pred_actual_diff_{str_step}.png")
        # plt.tight_layout()
        # plt.close()

    # os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {plot_dir}/pred_actual_diff_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {outdir}/extreme_event.mp4')
        
    # return accs, rmse


def correlation_metrics_glorys_roms_interp1(dir, 
                                             name = None, 
                                             channels=["SSU", "SSV", "SSH"], 
                                             time_dict = time_dict_glorys_roms,
                                             stepnum = 10227,
                                             do_animation = True):
    
    print(f"Intensity scatter...")

    comparisons_dir = f"{dir}/comparisons"
    if not os.path.exists(comparisons_dir):
        os.makedirs(comparisons_dir)
    
    
    stepbin = 50
    steps = np.arange(0, stepnum, stepbin)
    
    for istep, step in enumerate(steps,0):
        _, _, data1, data2, loncoords, latcoords, times = load_glorys_roms_interpolate_whole(np.arange(step, step+stepbin), interpolator_use = "scipy")
        
        [data1, data2] = zero_mask([data1,data2])
        ## intensity scatter subsets the data to use for the pred/actual scatter plot
        x1, y1 = data1.reshape(-1, data1.shape[-1]), data2.reshape(-1, data2.shape[-1])
        x1, y1 = x1[~x1.mask[:,0],:], y1[~y1.mask[:,0],:]
        
        ishuff = np.arange(x1.shape[0])
        np.random.shuffle(ishuff)
        ishuff = ishuff[:10000]
        x1, y1 = x1[ishuff], y1[ishuff]
        
        # print(f" intensity scatter shapes: {x1.shape}, {x2.shape}, {y.shape}")
        
        fig, axs = plt.subplots(1,3,figsize = (10,4))
        
        for ich, ch in enumerate(channels,0):
            xy1 = np.vstack([x1[:,0],y1[:,0]])
            z1 = gaussian_kde(xy1)(xy1)
            axs[ich].scatter(x1[:,0], y1[:,0],c=z1,cmap="rainbow",s=2)
            axs[ich].plot([-2,2],[-2,2],color = "black",linestyle = "--", alpha = .4)
            xy1corr = spearmanr(x1[:,0],y1[:,0]).statistic
            axs[ich].set_title(f"{ch} \n correlation: {np.around(xy1corr,3)}")
            axs[ich].set_xlim(-2,2)
            axs[ich].set_ylim(-2,2)
            axs[ich].grid()
            
            # xy2corr = spearmanr(x2[:,0],y[:,0]).statistic
            # axs[1,ich].scatter(x2[:,0], y[:,0],c=z2,cmap="rainbow",s=2)
            # axs[1,ich].plot([-2,2],[-2,2],color = "black",linestyle = "--", alpha = .4)
            # axs[1,ich].set_title(f"correlation: {np.around(xy2corr,3)}")
            # axs[1,ich].set_xlim(-2,2)
            # axs[1,ich].set_ylim(-2,2)
            # axs[1,ich].grid()
            
        axs[0].set_xlabel("GLORYS FNO Downscaling")
        axs[0].set_ylabel("ROMS")
        axs[1].set_xlabel("Low Resolution Interpolation")
        axs[1].set_ylabel("ROMS")
            
        
        plt.suptitle(f"Value Comparison Plots, Pred vs Truth\n date range: {time_dict[step]} to {time_dict[step+stepbin]}")
        plt.tight_layout()
        str_step = "0"*(6-len(str(step)))+str(step)
        plt.savefig(f"{comparisons_dir}/valueComparison_{str_step}.png", dpi = 300)
        plt.close()
        
        if istep % 200 == 0:
            print(f" {istep}: data {time_dict[step]}")
            
    # for istep, step in enumerate(steps[::-1],0):
        # str_step = "0"*(6-len(str(step)))+str(step)
        # str_step2 = "0"*(6-len(str(istep)))+str(istep)
        # os.system(f"mv {comparisons_dir}/valueComparison_{str_step}.png {comparisons_dir}/valueComparison_{str_step}.png")
        
    os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {comparisons_dir}/valueComparison_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {dir}/valueComparison.mp4')
    
## testing interpolation comparisons
def test_interpolation_projection():
    import time
    
    # stime = time.time()
    # lrv_use, hrv_use, lrv_use_reshape, hrv_use_reshape, hrloncoords, hrlatcoords, times = load_glorys_roms_interpolate_whole(np.arange(0,1000), interpolator_use = "scipy")
    # lrv_use_reshape_ma, hrv_use_reshape_ma = zero_mask([lrv_use_reshape, hrv_use_reshape])
    # print(f"scipy: {time.time() - stime}")
    
    
    correlation_metrics_glorys_roms_interp1("/media/volume/qgm1/fujitsu_outputs/temp_glorys_roms_comparisons/", 
                                     name = None, 
                                     channels=["SSU", "SSV", "SSH"], 
                                     time_dict = time_dict_glorys_roms,
                                     do_animation = True)
    
    
    # plot_metrics_downscale(lrv_use_reshape_ma[:10], 
                          # lrv_use_reshape_ma[:10], 
                          # hrv_use_reshape_ma[:10], 
                          # times, 
                          # hrlatcoords, 
                          # hrloncoords, 
                          # "/media/volume/qgm1/fujitsu_outputs/temp_downscaleTest/", 
                          # "lr_hr_test_downscale", 
                          # channels=["SSU", "SSV", "SSH"], 
                          # # channel_lims=[[-2,2],[-2,2],[-1,1]], 
                          # days = 10, 
                          # do_animation = True)
    
    # lrv_use2, hrv_use2, lrv_use_reshape2, hrv_use_reshape2, hrloncoords2, hrlatcoords2, times2 = load_glorys_roms_interpolate_whole(np.arange(500,520), interpolator_use = "scipy")
    
    # lrv_use_reshape_ma2, hrv_use_reshape_ma2 = zero_mask([lrv_use_reshape2, hrv_use_reshape2])
    
    # plot_metrics_forecast(lrv_use_reshape_ma2[:10], 
                          # hrv_use_reshape_ma2[:10], 
                          # times2, 
                          # hrlatcoords2, 
                          # hrloncoords2, 
                          # "/media/volume/qgm1/fujitsu_outputs/temp_scipyInterp/", 
                          # "lr_hr_diff_interpolation", 
                          # channels=["SSU", "SSV", "SSH"], 
                          # channel_lims=[[-2,2],[-2,2],[-1,1]], 
                          # days = 10, 
                          # do_animation = True)
    
    # hrv_torch, hrv_scipy = zero_mask([hrv_use_reshape, hrv_use_reshape2])
    # lrv_torch, lrv_scipy = zero_mask([lrv_use_reshape, lrv_use_reshape2])
    
    # plot_metrics_forecast(lrv_torch[:10], 
                          # lrv_scipy[:10], 
                          # times, 
                          # hrlatcoords, 
                          # hrloncoords, 
                          # "/media/volume/qgm1/fujitsu_outputs/temp_interp_compare/", 
                          # "lr_hr_diff_interpolation", 
                          # channels=["SSU", "SSV", "SSH"], 
                          # channel_lims=[[-2,2],[-2,2],[-1,1]], 
                          # days = 10, 
                          # do_animation = True)
    
    ## neares low resolution interpolation is better for the valid overlapping ocean mask regions
    
def plot_truth_glorys_cnaps(channels=["SSU", "SSV", "SSH"], 
                            channel_lims=[[-2,2],[-2,2],[-1,1]],
                            day = 4648):
    
    lrv_use, hrv_use, lrloncoords, lrlatcoords, hrloncoords, hrlatcoords, times = load_glorys_roms_whole([day])
    
    projection = ccrs.PlateCarree()
    fig, axs = plt.subplots(2,len(channels),figsize = (4*len(channels),4), subplot_kw={'projection': projection})
        
    for ivar, (var,lims) in enumerate(zip(channels,channel_lims),0):         
        im1 = axs[0,ivar].contourf(lrloncoords, lrlatcoords, lrv_use[0,:,:,ivar], levels = np.linspace(lims[0], lims[1], 200), vmin = lims[0], vmax = lims[1], transform = projection, cmap="coolwarm", extend='both')
        im2 = axs[1,ivar].contourf(hrloncoords, hrlatcoords, hrv_use[0,:,:,ivar], levels = np.linspace(lims[0], lims[1], 200), vmin = lims[0], vmax = lims[1], transform = projection, cmap="coolwarm", extend='both')
        # im3 = axs[2,ivar].contourf(loncoords, latcoords, preds[step,:,:,ivar]-truths[step,:,:,ivar], levels = np.linspace( -1, 1, 200), vmin = -1, vmax=1, transform = projection, cmap="twilight", extend='both')
        
        axs[0,ivar].coastlines()
        axs[1,ivar].coastlines()
        # axs[2,ivar].coastlines()
        
        cbar1 = fig.colorbar(im1, orientation='vertical', ticks = [lims[0], 0, lims[1]], fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(im2, orientation='vertical', ticks = [lims[0], 0, lims[1]], fraction=0.046, pad=0.04)
        # cbar3 = fig.colorbar(im3, orientation='vertical', ticks = [-1.0, 0,1.0], fraction=0.046, pad=0.04)
        
        for ax_rnum in range(2):
            gl = axs[ax_rnum,ivar].gridlines(draw_labels=True, alpha = .3)
            
            gl.top_labels = False
            gl.right_labels = False
            
            if ivar != 0 or ax_rnum != 0:
                gl.bottom_labels = False
                gl.left_labels = False
        
        axs[0,ivar].set_title(f"{var}")
     
    axs[0,0].text(-0.2, 0.55, "GLORYS LR", va='bottom', ha='center',
    rotation='vertical', rotation_mode='anchor',
    transform=axs[0,0].transAxes)
    
    axs[1,0].text(-0.2, 0.55, "CNAPS HR", va='bottom', ha='center',
    rotation='vertical', rotation_mode='anchor',
    transform=axs[1,0].transAxes)
    
    # axs[2,0].text(-0.2, 0.55, r"$\Delta$", va='bottom', ha='center',
    # rotation='vertical', rotation_mode='anchor',
    # transform=axs[2,0].transAxes)
    
    plt.suptitle(f"(year,day): {times[0]}")
    plt.savefig(f"/media/volume/qgm1/fujitsu_outputs/temp/glorys_roms_day-{day}.png")
    plt.tight_layout()
    plt.close()
    


def spectral_sqr_abs2(
    output, 
    target, 
    grid_valid_size=None,
    wavenum_init_lon=1, 
    wavenum_init_lat=1, 
    lambda_fft=0.5,
    lat_lon_bal=0.5,
    channels="all",
    fft_loss_scale=1./110.
):
    """
    Grid and spectral losses, both with MSE.
    Modified to accommodate data shape [batch_size, num_channels, latitude_size, longitude_size].
    """

    # Ensure that output and target have the same shape
    assert output.shape == target.shape, "Output and target must have the same shape."

    # Number of channels (second dimension)
    num_channels = output.shape[1]

    # Calculate grid_valid_size if not provided
    if grid_valid_size is None: 
        grid_valid_size = output.numel()  # Total number of elements

    # Compute grid space loss (MSE)
    loss_grid = torch.sum((output - target) ** 2) / (grid_valid_size * num_channels)
    
    # Initialize spectral loss accumulator
    run_loss_run = torch.zeros(1, device=output.device, dtype=output.dtype)
    
    # Define channels and their weights
    if channels == "all":
        num_spectral_chs = num_channels
        channels = [["_", i, 1. / num_spectral_chs] for i in range(num_spectral_chs)]
    
    totcw = 0  # Total channel weight

    # Prepare data for periodic FFT along latitude and longitude
    # Concatenate along latitude (dimension 2)
    output2lat = torch.cat([output, torch.flip(output, [2])], dim=2)
    target2lat = torch.cat([target, torch.flip(target, [2])], dim=2)
    
    # Concatenate along longitude (dimension 3)
    output2lon = torch.cat([output, torch.flip(output, [3])], dim=3)
    target2lon = torch.cat([target, torch.flip(target, [3])], dim=3)
    
    # Loop over channels
    for [cname, c, cw] in channels:
        if cw != 0:
            # Select the c-th channel
            output_c = output[:, c, :, :]        # Shape: [batch_size, latitude_size, longitude_size]
            target_c = target[:, c, :, :]
            output2lat_c = output2lat[:, c, :, :]
            target2lat_c = target2lat[:, c, :, :]
            output2lon_c = output2lon[:, c, :, :]
            target2lon_c = target2lon[:, c, :, :]

            # Compute FFT along latitude (dimension 1 after channel selection)
            # out_fft_lat = torch.abs(torch.fft.rfft(output2lat_c, dim=1))[:, wavenum_init_lat:, :]
            # target_fft_lat = torch.abs(torch.fft.rfft(target2lat_c, dim=1))[:, wavenum_init_lat:, :]
            out_fft_lat = torch.abs(torch.fft.rfft(output_c, dim=1))[:, wavenum_init_lat:, :]
            target_fft_lat = torch.abs(torch.fft.rfft(target_c, dim=1))[:, wavenum_init_lat:, :]
            loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat) ** 2)

            # Compute FFT along longitude (dimension 2 after channel selection)
            # out_fft_lon = torch.abs(torch.fft.rfft(output2lon_c, dim=2))[:, :, wavenum_init_lon:]
            # target_fft_lon = torch.abs(torch.fft.rfft(target2lon_c, dim=2))[:, :, wavenum_init_lon:]
            out_fft_lon = torch.abs(torch.fft.rfft(output_c, dim=2))[:, :, wavenum_init_lon:]
            target_fft_lon = torch.abs(torch.fft.rfft(target_c, dim=2))[:, :, wavenum_init_lon:]
            loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon) ** 2)

            # Accumulate weighted spectral loss
            run_loss_run += ((1 - lat_lon_bal) * loss_fft_lon + lat_lon_bal * loss_fft_lat) * cw
            totcw += cw

    # Normalize and scale spectral loss
    loss_fft = run_loss_run / totcw * fft_loss_scale

    # Combine grid and spectral losses
    loss = (1 - lambda_fft) * loss_grid + lambda_fft * loss_fft

    return loss