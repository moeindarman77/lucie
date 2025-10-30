#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
from datetime import datetime

# =============================================================================
# 1) USER PARAMETERS
# =============================================================================
data_dir       = "/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/SFNO_half_unet_LUCIE_SR_2010-2020"
file_pattern   = "SFNO_half_unet_SR_LUCIE_2010-2020_{}.npz"
n_files        = 10
steps_per_file = 1460        # 6 h × 1,460 = 365 days
total_steps    = n_files * steps_per_file
startdate      = datetime(2000, 1, 1, 0)
freq_hours     = 6
varname        = "temperature"  # change to 'u_wind', 'v_wind', 'precipitation', etc.

# =============================================================================
# 2) BUILD FULL TIME INDEX & FIND SEASONAL INDICES
# =============================================================================
time_index = pd.date_range(
    start=startdate,
    periods=total_steps,
    freq=f"{freq_hours}h"
)

# SAM = DJF, NAM = JJA
sam_pos = np.where(time_index.month.isin([12, 1, 2]))[0]
nam_pos = np.where(time_index.month.isin([6, 7, 8]))[0]

# =============================================================================
# 3) MEMORY-EFFICIENT SEASONAL LOADER
# =============================================================================
def load_seasonal(varname, season_idx):
    """
    Load only the time slices in season_idx from each .npz chunk,
    concatenate along time, and return a NumPy array of shape
    (ntime_season, 720, 1440).
    """
    parts = []
    for i in range(n_files):
        path = os.path.join(data_dir, file_pattern.format(i))
        with np.load(path) as d:
            arr = d[varname]          # (1460, 720, 1440)
        start = i * steps_per_file
        end   = start + steps_per_file
        # select only the in-season indices within this block
        local = season_idx[(season_idx >= start) & (season_idx < end)] - start
        if local.size:
            parts.append(arr[local])
        del arr
    return np.concatenate(parts, axis=0)

# =============================================================================
# 4) DEFINE SPATIAL GRID
# =============================================================================
# 720 lat points from -90 to +90, 1440 lon points from 0 to 360°
lats = np.linspace(-90, 90, 720)
lons = np.linspace(0, 360, 1440, endpoint=False)

# =============================================================================
# 5) XARRAY DATASET BUILDER
# =============================================================================
def build_ds(data3d, name):
    """
    Wrap a (ntime, 720, 1440) array into an xarray.Dataset with
    a proper time coordinate at freq_hours intervals.
    """
    times = pd.date_range(
        start=startdate,
        periods=data3d.shape[0],
        freq=f"{freq_hours}h"
    )
    da = xr.DataArray(
        data3d,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
        name=name
    )
    return da.to_dataset()

# =============================================================================
# 6) SAM (DJF, Southern Hemisphere EOF)
# =============================================================================
# Load & anomaly-correct
sam_raw  = load_seasonal(varname, sam_pos)                  # (ntime_sam,720,1440)
sam_anom = sam_raw - sam_raw.mean(axis=0)[None, ...]        # remove time-mean

# Build dataset and select lat < 0
ds_sam = build_ds(sam_anom, varname).sel(lat=slice(-90, 0))

# 1-D latitude weights → xarray will broadcast over lon
lat_weights_sam = xr.DataArray(
    np.sqrt(np.cos(np.deg2rad(ds_sam.lat))),
    dims=["lat"],
    coords={"lat": ds_sam.lat}
)

# EOF analysis
solver_sam = Eof(ds_sam[varname], weights=lat_weights_sam)
eof1_sam   = solver_sam.eofsAsCovariance(neofs=1)  # leading EOF pattern
pc1_sam    = solver_sam.pcs(npcs=3, pcscaling=0)   # leading PCs
varfrac_s  = solver_sam.varianceFraction()         # variance fractions

# =============================================================================
# 7) NAM (JJA, Northern Hemisphere EOF)
# =============================================================================
nam_raw  = load_seasonal(varname, nam_pos)
nam_anom = nam_raw - nam_raw.mean(axis=0)[None, ...]
ds_nam   = build_ds(nam_anom, varname).sel(lat=slice(0, 90))

lat_weights_nam = xr.DataArray(
    np.sqrt(np.cos(np.deg2rad(ds_nam.lat))),
    dims=["lat"],
    coords={"lat": ds_nam.lat}
)

solver_nam = Eof(ds_nam[varname], weights=lat_weights_nam)
eof1_nam   = solver_nam.eofsAsCovariance(neofs=1)
pc1_nam    = solver_nam.pcs(npcs=3, pcscaling=0)
varfrac_n  = solver_nam.varianceFraction()

# =============================================================================
# 8) RESULTS SUMMARY
# =============================================================================
print(f"SAM EOF1 shape: {eof1_sam.shape}, explains {varfrac_s[0]*100:.2f}% of variance")
print(f"NAM EOF1 shape: {eof1_nam.shape}, explains {varfrac_n[0]*100:.2f}% of variance")