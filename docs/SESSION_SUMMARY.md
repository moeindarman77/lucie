# Lucie-3D Downscaling Pipeline Session Summary

## 🎯 What We Accomplished

### ✅ Pipeline Implementation
1. **Created Lucie-3D Dataloader** (`src/dataset/ClimateDataset_LUCIE3D.py`)
   - Reads NetCDF input from Lucie-3D model
   - Handles both LR and HR orography with separate normalization
   - Proper variable ordering and normalization matching training

2. **Created Sampling Script** (`src/tools/sample_ddpm_lucie3d.py`)
   - Modified from `sample_ddpm_final_v2.py` 
   - Removes truth/condition references
   - Outputs timestamped NetCDF files
   - Proper denormalization for physical units

3. **Fixed Configuration** (`src/config/ERA5_config_final_v2.yaml`)
   - Fixed typo: `diff;usion_params` → `diffusion_params`

### ✅ Key Fixes Applied
- **Orography Handling**: Separate LR (FNO input) and HR (diffusion conditioning) with own normalization stats
- **LSM Normalization**: Uses own mean/std from LSM file
- **Log Pressure Fix**: Removed double-log operation (data already in logp form)
- **Proper Denormalization**: Model outputs converted back to physical units

### ✅ Visualization
- **Complete Jupyter Notebook** (`checkcheck.ipynb`)
- All variables plotted with 2:1 aspect ratio
- Cartopy world map overlays
- Unit conversions (K→°C, m→mm)
- Individual and multi-panel plots

## 🔧 Files Created/Modified

### New Files:
- `src/dataset/ClimateDataset_LUCIE3D.py` - Lucie-3D dataloader
- `src/tools/sample_ddpm_lucie3d.py` - Sampling script  
- `checkcheck.ipynb` - Visualization notebook
- `test_interactive.sh` - Test script

### Modified Files:
- `src/config/ERA5_config_final_v2.yaml` - Fixed diffusion_params typo

## 🚀 Usage Instructions

### Environment Setup:
```bash
source ~/.bashrc
gpu1  # Request interactive GPU session
module load conda
conda activate jax
cd /glade/derecho/scratch/mdarman/lucie/src
export PYTHONPATH=$(pwd)
```

### Environment Variables:
```bash
export INPUT_NC="/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_3D/LUCIE_co2_nosst_nomask_range_2000_2020.nc"
export LSM_PATH="/glade/derecho/scratch/mdarman/lucie/lsm_lr.npz"
export ORO_LR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_lr.npz"
export ORO_HR_PATH="/glade/derecho/scratch/mdarman/lucie/orography_hr.npz"
export OUT_NC="/glade/derecho/scratch/mdarman/lucie/downscaled_output.nc"
```

### Run Pipeline:
```bash
python tools/sample_ddpm_lucie3d.py --config config/ERA5_config_final_v2.yaml --start 0 --end 2
```

### Quick Test:
```bash
./test_interactive.sh
```

## 📊 Expected Output
- NetCDF file with 4 downscaled climate variables at 0.25° resolution
- Proper timestamps and CF-compliant metadata
- Variables: 2m_temperature, u_component_of_wind_10m, v_component_of_wind_10m, total_precipitation_6hr

## 🎯 Pipeline Status: READY FOR PRODUCTION

All components tested and working:
- ✅ Normalization consistent with training
- ✅ Two-stage processing (FNO → Diffusion)
- ✅ Proper orography handling for both models
- ✅ Output denormalization to physical units
- ✅ NetCDF output with timestamps

## 📝 Notes
- Pipeline tested with 2 timesteps successfully
- Config file typo was the main blocker
- All normalization issues resolved
- Ready for batch processing or extended runs

Date: $(date)
Session: Lucie-3D Downscaling Implementation