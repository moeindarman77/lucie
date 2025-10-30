# LUCIE 10yr Sampling Setup

## Summary
Created new sampling pipeline for LUCIE data from `/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_2010ini_10yr.npz`

## Key Changes
1. **New LUCIE data file**: Changed from `LUCIE_inference_start2010.npz` to `LUCIE_2010ini_10yr.npz`
2. **LUCIE-only processing**: Only processes LUCIE data (not ERA5), avoiding length mismatch issues
3. **Reduced output**: Saves only `output` (diffusion result) and `fno_output` (FNO result)
4. **Same output directory**: Uses `unet_final_v10` but with new subfolder `samples_lucie_10yr`

## Files Created

### 1. Dataset Loader
- **File**: [src/dataset/LucieLoader_10yr.py](src/dataset/LucieLoader_10yr.py)
- **Description**: Copy of original LucieLoader, ready for new data file

### 2. Configuration
- **File**: [src/config/ERA5_config_lucie_10yr.yaml](src/config/ERA5_config_lucie_10yr.yaml)
- **Key change**: Updated `lucie_file_path` to new location

### 3. Sampling Script
- **File**: [src/tools/sample_ddpm_lucie_10yr.py](src/tools/sample_ddpm_lucie_10yr.py)
- **Changes**:
  - Removed ERA5 dataset dependency
  - Only processes LUCIE data
  - Saves only `output` and `fno_output` (not lres, hres, lres_interp)
  - Output directory: `results/unet_final_v10/samples_lucie_10yr/`
  - Added `--num_samples` argument for better control

### 4. Job Scripts

#### Test Job (10 samples)
- **File**: [job_test_lucie_10yr.slurm](job_test_lucie_10yr.slurm)
- **Command**: `qsub job_test_lucie_10yr.slurm`
- **Walltime**: 1 hour
- **Samples**: 10 samples (indices 0-9)

#### Production Job Array
- **File**: [job_array_lucie_10yr.slurm](job_array_lucie_10yr.slurm)
- **Array size**: 0-35 (adjust based on total samples)
- **Samples per job**: 1000
- **Note**: You may need to adjust the array size and samples_per_job based on actual dataset length

### 5. Verification Script
- **File**: [check_lucie_10yr_setup.py](check_lucie_10yr_setup.py)
- **Purpose**: Verify all required files exist before running

### 6. Visualization
- **File**: [visualize_lucie_10yr_test.ipynb](visualize_lucie_10yr_test.ipynb)
- **Purpose**: Visualize test results and verify outputs look correct

## How to Run

### Step 1: Request GPU Node
```bash
# Use the alias from your .bashrc
gpu1
# OR manually:
# qinteractive -A UCSC0009 --ngpus 1
```

### Step 2: Setup Environment
```bash
# Use the prep alias
prep
# This executes:
# module load conda && conda activate jax && cd "/glade/derecho/scratch/mdarman/lucie/src" && export PYTHONPATH=$(pwd)
```

### Step 3: Verify Setup
```bash
cd /glade/derecho/scratch/mdarman/lucie
python check_lucie_10yr_setup.py
```

### Step 4: Run Test Job (10 samples)
```bash
cd /glade/derecho/scratch/mdarman/lucie
qsub job_test_lucie_10yr.slurm
```

### Step 5: Monitor Test Job
```bash
# Check job status
q  # alias for qstat -u mdarman

# Check output file when done
ls -lh lucie_10yr_test.o*
cat lucie_10yr_test.o*
```

### Step 6: Verify Test Results
Open and run the visualization notebook:
```bash
jupyter notebook visualize_lucie_10yr_test.ipynb
```

Check:
- No NaN values in outputs
- FNO and diffusion outputs have reasonable values
- Spatial patterns look correct
- All 10 samples generated successfully

### Step 7: Run Production Job (after verification)
```bash
# IMPORTANT: First check the actual dataset length and adjust job array parameters
# Update job_array_lucie_10yr.slurm if needed:
#   - PBS -J 0-XX  (adjust based on dataset_length / samples_per_job)
#   - samples_per_job variable

qsub job_array_lucie_10yr.slurm
```

## Important Notes

### Orography and Land-Sea Mask
The sampling script loads orography and land_sea_mask from:
```
/glade/derecho/scratch/mdarman/lucie/lsm.npz
```

**Expected keys in lsm.npz**: `'orography'` and `'lsm'`

If the keys are different, you'll need to update lines 47-48 in `sample_ddpm_lucie_10yr.py`.

### Output Structure
Each sample is saved as: `results/unet_final_v10/samples_lucie_10yr/{index}.npz`

Contents:
- `output`: Diffusion model output [1, 4, 721, 1440]
- `fno_output`: FNO model output [1, 4, 721, 1440]

4 channels correspond to:
1. 2m Temperature
2. U-wind component
3. V-wind component
4. Total Precipitation 6hr

### Array Job Parameters
Current settings in `job_array_lucie_10yr.slurm`:
- Array jobs: 0-35 (36 total jobs)
- Samples per job: 1000
- Total samples: 36,000

**TO DO**: Verify the actual LUCIE dataset length and adjust accordingly!

## Troubleshooting

### If lsm.npz keys are wrong:
Check the actual keys:
```python
import numpy as np
lsm = np.load('/glade/derecho/scratch/mdarman/lucie/lsm.npz')
print(list(lsm.keys()))
```

Then update `sample_ddpm_lucie_10yr.py` lines 47-48 accordingly.

### If samples look wrong:
1. Check for NaN values in visualization notebook
2. Compare with previous HRES samples
3. Verify normalization is being applied correctly
4. Check that the LUCIE data ordering matches expectations

### If job crashes:
1. Check error output file: `cat lucie_10yr_*.o*`
2. Verify all checkpoint files exist
3. Check GPU memory usage
4. Try reducing batch size if needed (currently 1)

## Questions Before Running

**IMPORTANT**: I need you to verify one thing before running:

The script assumes your `lsm.npz` file has keys `'orography'` and `'lsm'`.
Please check this on a GPU node and let me know if the keys are different!

```bash
gpu1
prep
python -c "import numpy as np; d=np.load('lsm.npz'); print(list(d.keys())); print('oro shape:', d[list(d.keys())[0]].shape if len(d.keys())>0 else 'N/A')"
```
