# Normalized FNO Conditioning Solution

## Problem Summary

The v-wind climatology from retrained diffusion models showed systematic bias (~-6.2 m/s instead of ~-0.03 m/s). Root cause analysis revealed:

1. **FNO uses unequal loss weights** in spectral_sqr_abs2.py:
   - Temperature: 10%, U-wind: 40%, V-wind: 40%
   - This creates variable-specific biases in FNO outputs

2. **Diffusion model learns corrections** based on biased FNO conditioning:
   - Both U-wind and V-wind have ~-6.2 m/s bias from diffusion
   - U-wind: 11.86 → 5.66 m/s (52% error, noticeable)
   - V-wind: -0.03 → -6.23 m/s (infinite% error, catastrophic)

3. **Relative impact differs** due to different base means:
   - Absolute bias is same, but relative error destroys v-wind climatology

## Solution: Normalize FNO Outputs Before Conditioning

Instead of retraining FNO (expensive), normalize FNO outputs to have zero mean and unit std before using as diffusion conditioning.

### Key Insight: Use SCALAR Statistics (Not Spatial Maps)

The original data normalization uses **scalar mean/std values** per variable:
```python
# From ClimateDataset_v2.py
hr_data = (hr_data[var] - scalar_mean) / scalar_std
```

Therefore, FNO output normalization should also use **scalar statistics** to match this format.

## Implementation

### 1. Compute FNO Output Statistics (SCALAR)

**Script**: `compute_fno_output_stats_scalar.py`
- Processes all 14,611 training samples
- Computes global mean and std for each of 4 FNO output channels
- Saves in NPZ format matching original stats structure
- **Job submitted**: 3472200

**Output file**: `fno_output_stats_scalar.npz`
Structure:
```python
{
  'temperature': {'mean': scalar, 'std': scalar},
  'uwind': {'mean': scalar, 'std': scalar},
  'vwind': {'mean': scalar, 'std': scalar},
  'precipitation': {'mean': scalar, 'std': scalar}
}
```

### 2. Modified Training Script

**File**: `src/tools/train_ddpm_normalized_fno.py`

Key changes:
```python
# Load scalar FNO stats
with np.load(fno_stats_path, allow_pickle=True) as stats:
    fno_mean_temp = float(stats['temperature'].item()['mean'])
    fno_std_temp = float(stats['temperature'].item()['std'])
    # ... repeat for other variables

    fno_mean = torch.tensor([...], dtype=torch.float32)
    fno_std = torch.tensor([...], dtype=torch.float32)

# Reshape for broadcasting: (1, 4, 1, 1)
fno_mean = fno_mean.view(1, 4, 1, 1).to(rank)
fno_std = fno_std.view(1, 4, 1, 1).to(rank)

# During training loop:
fno_output = SR_model(lres_upsampled)
fno_output_normalized = (fno_output - fno_mean) / (fno_std + 1e-6)
cond_input = torch.cat([fno_output_normalized, hres[:, 5:6]], dim=1)
```

### 3. Modified Sampling Script

**File**: `src/tools/sample_ddpm_normalized_fno.py`

Same normalization applied during inference:
```python
fno_output_lucie = SR_model(lucie_upsampled)
fno_output_normalized = (fno_output_lucie - fno_mean) / (fno_std + 1e-6)
cond_input = torch.cat([fno_output_normalized, orography_hr], dim=1)
```

### 4. Configuration File

**File**: `src/config/ERA5_config_normalized_fno.yaml`

New parameters:
```yaml
dataset_params:
  fno_normalization_file: '/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz'
  # ... other params

train_params:
  task_name: 'unet_normalized_fno'
  description: 'Training with normalized FNO outputs to fix v-wind bias issue'
```

## Workflow

### Current Status (2025-10-28)

1. ✅ Identified root cause of v-wind bias
2. ✅ Created scalar FNO stats computation script
3. ✅ Modified training script with normalized conditioning
4. ✅ Modified sampling script with normalized conditioning
5. ✅ Created config file for new task
6. 🔄 **RUNNING**: FNO stats computation (Job 3472200)

### Next Steps

1. **Wait for stats job to complete** (~2-4 hours for 14,611 samples)
   ```bash
   qstat | grep 3472200
   # When done, verify output:
   ls -lh fno_output_stats_scalar.npz
   ```

2. **Verify stats file structure**:
   ```python
   import numpy as np
   stats = np.load('fno_output_stats_scalar.npz', allow_pickle=True)
   for key in stats.keys():
       data = stats[key].item()
       print(f"{key}: mean={data['mean']}, std={data['std']}")
   ```

3. **Train new diffusion model** with normalized FNO:
   ```bash
   cd /glade/derecho/scratch/mdarman/lucie
   python src/tools/train_ddpm_normalized_fno.py --config src/config/ERA5_config_normalized_fno.yaml
   ```

4. **Sample from trained model**:
   ```bash
   python src/tools/sample_ddpm_normalized_fno.py --config src/config/ERA5_config_normalized_fno.yaml --start 0 --num_samples 14600
   ```

5. **Verify v-wind climatology** matches expected mean (~-0.03 m/s)

## Expected Outcome

By normalizing FNO outputs before conditioning:
- FNO biases are removed from conditioning signal
- Diffusion model sees zero-mean, unit-std conditioning (matching ground truth format)
- Diffusion should **not** learn systematic corrections
- V-wind climatology should match ground truth: **mean ≈ -0.03 m/s**

## Files Created/Modified

### New Files:
- `compute_fno_output_stats_scalar.py` - Compute scalar FNO statistics
- `job_compute_fno_stats_scalar.slurm` - SLURM job for stats computation
- `src/tools/train_ddpm_normalized_fno.py` - Training with normalized FNO
- `src/tools/sample_ddpm_normalized_fno.py` - Sampling with normalized FNO
- `src/config/ERA5_config_normalized_fno.yaml` - Config for normalized FNO task
- `NORMALIZED_FNO_SOLUTION.md` - This document

### Obsolete Files (spatial stats - incorrect approach):
- ~~`compute_fno_output_stats_spatial.py`~~
- ~~`job_compute_fno_stats_spatial.slurm`~~

## Technical Notes

### Why Scalar Stats Match Original Format

From `ClimateDataset_v2.py` (lines 48-49, 55-56):
```python
self.input_mean_std = {var: (input_norm_data[var].item()['mean'],
                              input_norm_data[var].item()['std'])
                       for var in self.input_vars}
```

The `.item()['mean']` returns a **scalar** (numpy.float32 with shape `()`), not a spatial array.

### Broadcasting Behavior

With shape `(1, 4, 1, 1)`:
```python
fno_output: (batch, 4, 721, 1440)
fno_mean:   (1, 4, 1, 1)
# Broadcasting automatically expands to (batch, 4, 721, 1440)
normalized = (fno_output - fno_mean) / fno_std
```

This applies the **same** scalar mean/std to all spatial locations per channel, matching the original normalization approach.

## References

- Root cause analysis: `VWIND_BIAS_ROOT_CAUSE.md`
- Normalization data flow: `NORMALIZATION_DATA_FLOW.md`
- FNO loss weights: `src/utils/spectral_sqr_abs2.py` (lines 11-16)
- Dataset normalization: `src/dataset/ClimateDataset_v2.py` (lines 114-116)
