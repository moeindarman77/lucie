import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import Union

def build_lucie_vars(ds: xr.Dataset,
                     level_dim: str = "level",
                     lat_dim: str = "latitude",
                     lon_dim: str = "longitude") -> xr.DataArray:
    # Fixed-level selects (drop level coord to match 2D vars)
    T7  = ds["temperature"].isel({level_dim: 7}, drop=True)
    SH7 = ds["specific_humidity"].isel({level_dim: 7}, drop=True)
    U3  = ds["u_component_of_wind"].isel({level_dim: 3}, drop=True)
    V3  = ds["v_component_of_wind"].isel({level_dim: 3}, drop=True)
    SP  = ds["surface_pressure"]
    TP  = ds["tp6hr"]

    # Align on common grid
    T7, SH7, U3, V3, SP, TP = xr.align(T7, SH7, U3, V3, SP, TP, join="inner")

    # lucie 3D already outputs logp; keep as-is
    logp = SP

    # Consistent dim order and dtype
    def order3(da): return da.transpose("time", lat_dim, lon_dim, ...)
    T7, SH7, U3, V3, logp, TP = map(order3, (T7, SH7, U3, V3, logp, TP))
    T7, SH7, U3, V3, logp, TP = (da.astype(np.float32) for da in (T7, SH7, U3, V3, logp, TP))

    # Concatenate in your exact feature order
    features = ["Temperature_7", "Specific_Humidity_7", "U-wind_3", "V-wind_3", "logp", "tp6hr"]
    stacked = xr.concat([T7, SH7, U3, V3, logp, TP],
                        dim="feature", coords="minimal", compat="override", join="inner")
    stacked = stacked.assign_coords(feature=("feature", features)) \
                     .transpose("time", lat_dim, lon_dim, "feature")
    return stacked


class LucieXRTimeSliceDataset(Dataset):
    """
    Build the stacked (time, lat, lon, feature) once in __init__,
    then __getitem__(t) returns a torch.float32 tensor of shape (C, H, W).
    """

    def __init__(self,
                 ds_or_path: Union[str, xr.Dataset],
                 *,
                 level_dim: str = "level",
                 lat_dim: str = "latitude",
                 lon_dim: str = "longitude",
                 preload_to_mem: bool = False):
        self.lat_dim, self.lon_dim = lat_dim, lon_dim

        ds = xr.open_dataset(ds_or_path) if isinstance(ds_or_path, str) else ds_or_path
        self.stacked = build_lucie_vars(ds, level_dim=level_dim,
                                        lat_dim=lat_dim, lon_dim=lon_dim)  # (time, lat, lon, feature)

        # Optional: fully materialize into a numpy array for faster indexing
        self.array = None
        if preload_to_mem:
            # (time, feature, lat, lon) to match (T, C, H, W)
            self.array = self.stacked.transpose("time", "feature", lat_dim, lon_dim).values  # float32

    def __len__(self):
        return self.stacked.sizes["time"]

    def __getitem__(self, t: int):
        if self.array is not None:
            # (C, H, W) from preloaded numpy
            return torch.from_numpy(self.array[t])  # float32

        # Lazy path: slice from xarray each time
        da_t = self.stacked.isel(time=t)  # (lat, lon, feature)
        chw  = da_t.transpose("feature", self.lat_dim, self.lon_dim).values  # (C, H, W), float32
        # Flipud
        chw = torch.from_numpy(chw).flip(1)
        return chw
