# Normalization/Denormalization Data Flow in LUCIE Training

## Overview
This document traces the complete data flow for normalization and denormalization in the LUCIE diffusion model training pipeline.

---

## 🔴 **THE PROBLEM**
**V-wind has a systematic bias of ~-6.2 m/s in diffusion outputs, while FNO outputs are correct.**

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA LOADING (ClimateDataset_v2.py, line 112-113)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HR Data (High-Resolution, from ERA5_hr_haiwen/data):                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  Variables:                                                             │
│    - 2m_temperature                    (index 0)                        │
│    - specific_humidity_133             (index 1)                        │
│    - u_component_of_wind_83            (index 2)                        │
│    - v_component_of_wind_83            (index 3) ← **PROBLEM VARIABLE** │
│    - total_precipitation_6hr           (index 4)                        │
│    - geopotential_at_surface           (index 5)                        │
│                                                                         │
│  LR Data (Low-Resolution, from ERA5_t30/train):                        │
│  ─────────────────────────────────────────────────────────────────────  │
│  Variables:                                                             │
│    - Temperature_7                     (index 0)                        │
│    - Specific_Humidity_7               (index 1)                        │
│    - U-wind_3                          (index 2)                        │
│    - V-wind_3                          (index 3)                        │
│    - tp6hr                             (index 4)                        │
│    - orography                         (index 5)                        │
│    - land_sea_mask                     (index 6)                        │
│    - logp                              (index 7)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. NORMALIZATION (ClimateDataset_v2.py, lines 114-124)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Using: stats_hr_2000_2009_updated.npz                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  V-wind HR normalization:                                              │
│    Mean: -0.0338262580 m/s                                             │
│    Std:  13.3956926014 m/s                                             │
│                                                                         │
│  Formula: normalized = (raw - mean) / std                              │
│                                                                         │
│  HR DATA NORMALIZED:                                                   │
│    hr_data[var] = (hr_data[var] - mean) / std                         │
│                                                                         │
│  LR DATA NORMALIZED:                                                   │
│    lr_data[var] = (lr_data[var] - mean) / std                         │
│                                                                         │
│  Expected normalized distribution:                                     │
│    Mean ≈ 0.0                                                          │
│    Std ≈ 1.0                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TRAINING FORWARD PASS (train_ddpm_final_v2_retrain.py)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 3a: Load batch (line 245-246)                                   │
│  ───────────────────────────────────────────────────────────────────    │
│    lres, hres = data['input'], data['output']                         │
│                                                                         │
│    lres shape: (batch, 8, 48, 96)   - NORMALIZED LR data              │
│    hres shape: (batch, 6, 721, 1440) - NORMALIZED HR data             │
│                                                                         │
│  Step 3b: Bicubic upsampling (line 252)                               │
│  ───────────────────────────────────────────────────────────────────    │
│    lres_upsampled = F.interpolate(lres, size=(721,1440),              │
│                                   mode='bicubic')                      │
│                                                                         │
│    lres_upsampled shape: (batch, 8, 721, 1440) - STILL NORMALIZED     │
│                                                                         │
│  Step 3c: FNO processing (line 255)                                   │
│  ───────────────────────────────────────────────────────────────────    │
│    fno_output = SR_model(lres_upsampled)                              │
│                                                                         │
│    FNO input:  NORMALIZED LR data (upsampled)                         │
│    FNO output: NORMALIZED HR prediction                               │
│                  Variables: [temp, u-wind, v-wind, precip]            │
│                  Shape: (batch, 4, 721, 1440)                         │
│                                                                         │
│  Step 3d: Prepare conditioning (line 257)                             │
│  ───────────────────────────────────────────────────────────────────    │
│    cond_input = torch.cat([fno_output, hres[:, 5:6]], dim=1)         │
│                                                                         │
│    Concatenates:                                                       │
│      - fno_output (4 channels): NORMALIZED predictions                │
│      - hres[:, 5:6] (1 channel): NORMALIZED geopotential             │
│    Result: (batch, 5, 721, 1440)                                      │
│                                                                         │
│  Step 3e: Prepare ground truth (line 259)                             │
│  ───────────────────────────────────────────────────────────────────    │
│    hres = hres[:, [0, 2, 3, 4]]  # Select only target variables      │
│                                                                         │
│    Ground truth (NORMALIZED):                                          │
│      Index 0: 2m_temperature                                           │
│      Index 1: u_component_of_wind_83                                  │
│      Index 2: v_component_of_wind_83  ← **PROBLEM: Training target** │
│      Index 3: total_precipitation_6hr                                  │
│                                                                         │
│  Step 3f: Add noise to ground truth (lines 266-272)                   │
│  ───────────────────────────────────────────────────────────────────    │
│    noise = torch.randn_like(hres)                                     │
│    t = torch.randint(0, num_timesteps, (batch_size,))                │
│    noisy_im = scheduler.add_noise(hres, noise, t)                    │
│                                                                         │
│    The diffusion model learns to predict the noise added to          │
│    NORMALIZED ground truth.                                            │
│                                                                         │
│  Step 3g: Diffusion model prediction (line 275)                       │
│  ───────────────────────────────────────────────────────────────────    │
│    noise_pred = model(noisy_im, t, cond=cond_input)                  │
│                                                                         │
│    Model is conditioned on:                                            │
│      - fno_output (NORMALIZED FNO predictions)                        │
│      - geopotential (NORMALIZED)                                       │
│                                                                         │
│  Step 3h: Loss computation (line 277)                                 │
│  ───────────────────────────────────────────────────────────────────    │
│    loss = MSELoss(noise_pred, noise)                                  │
│                                                                         │
│    Training objective: Predict noise that was added to NORMALIZED     │
│    ground truth data.                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. SAMPLING/INFERENCE (sample_ddpm_lucie_10yr_retrain_best.py)        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 4a: Load LUCIE data (lines 41-50, NORMALIZED by LucieLoader)   │
│  ───────────────────────────────────────────────────────────────────    │
│    lucie = lucie_dataset[index]  # NORMALIZED LR data                │
│                                                                         │
│  Step 4b: Bicubic upsampling (line 61)                               │
│  ───────────────────────────────────────────────────────────────────    │
│    lucie_upsampled = F.interpolate(lucie_aligned,                    │
│                                     size=(721,1440), mode='bicubic')  │
│                                                                         │
│  Step 4c: FNO inference (line 62)                                     │
│  ───────────────────────────────────────────────────────────────────    │
│    fno_output_lucie = SR_model(lucie_upsampled)                      │
│                                                                         │
│    FNO input:  NORMALIZED LR data                                     │
│    FNO output: NORMALIZED HR prediction                               │
│                  [temp, u-wind, v-wind, precip]                       │
│                                                                         │
│  Step 4d: Prepare conditioning (line 66)                              │
│  ───────────────────────────────────────────────────────────────────    │
│    cond_input = torch.cat([fno_output_lucie, orography_hr], dim=1)   │
│                                                                         │
│    Both inputs are NORMALIZED                                          │
│                                                                         │
│  Step 4e: Diffusion denoising loop (lines 84-95)                      │
│  ───────────────────────────────────────────────────────────────────    │
│    xt = torch.randn_like(fno_output_lucie)  # Start with noise       │
│                                                                         │
│    for t in reversed(range(num_timesteps)):                           │
│        noise_pred = model(xt, t, cond=cond_input)                    │
│        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred)  │
│                                                                         │
│    Final output (xt): NORMALIZED diffusion predictions               │
│                                                                         │
│  Step 4f: Save outputs (lines 111-114)                                │
│  ───────────────────────────────────────────────────────────────────    │
│    np.savez(save_path,                                                │
│             output=ims,              # NORMALIZED diffusion output    │
│             fno_output=fno_output_lucie_numpy)  # NORMALIZED FNO      │
│                                                                         │
│    📌 KEY POINT: Outputs are saved in NORMALIZED form!                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING / ANALYSIS                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  For visualization and metrics, data must be DENORMALIZED:            │
│  ───────────────────────────────────────────────────────────────────    │
│                                                                         │
│  Formula: denormalized = (normalized * std) + mean                    │
│                                                                         │
│  V-wind denormalization:                                               │
│    denormalized = (normalized * 13.3956926014) - 0.0338262580         │
│                                                                         │
│  📊 OBSERVED RESULTS (from verify_vwind_normalization.py):            │
│  ───────────────────────────────────────────────────────────────────    │
│                                                                         │
│  FNO V-wind (100 samples):                                            │
│    Normalized:  Mean = -0.0090, Std = 0.9391                         │
│    Denormalized: Mean = -0.1549 m/s ✅ (close to expected -0.034)    │
│                                                                         │
│  Diffusion V-wind (100 samples):                                      │
│    Normalized:  Mean = -0.4636, Std = 0.8751                         │
│    Denormalized: Mean = -6.2441 m/s ❌ (WRONG! Expected -0.034)      │
│                                                                         │
│  Expected HR V-wind mean: -0.0338 m/s                                 │
│                                                                         │
│  ⚠️ BIAS IN NORMALIZED SPACE: -0.4636                                 │
│  ⚠️ BIAS IN PHYSICAL SPACE: -6.21 m/s                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

---

## 🔍 **DIAGNOSIS**

### What's Working:
1. ✅ Normalization statistics are correct
2. ✅ FNO model produces correct v-wind climatology
3. ✅ Denormalization formula is correct
4. ✅ Other variables (temp, u-wind, precip) are fine

### What's NOT Working:
1. ❌ Diffusion model has learned a systematic bias in v-wind
2. ❌ The bias is **-0.464 in normalized space** (σ units)
3. ❌ This translates to **-6.2 m/s in physical space**
4. ❌ Retraining didn't fix the issue

---

## 🤔 **POSSIBLE ROOT CAUSES**

### 1. **Training Data Issue**
- The ground truth HR v-wind data (index 3 in hres) might have incorrect values
- Possible sign error or channel mismatch during data preparation

### 2. **Channel Indexing Problem**
- Training script line 259: `hres = hres[:, [0, 2, 3, 4]]`
- This selects indices: 0 (temp), 2 (u-wind), 3 (v-wind), 4 (precip)
- If the original hres has different ordering, this could grab the wrong variable

### 3. **FNO Conditioning Problem**
- The diffusion model is conditioned on FNO output
- If FNO v-wind is correct but diffusion learns to "correct" it wrongly, this creates bias
- The model might be learning residuals incorrectly

### 4. **Loss Function Asymmetry**
- MSE loss treats positive and negative errors equally
- But the model might have learned asymmetric corrections

---

## 🔬 **RECOMMENDED VERIFICATION STEPS**

### Step 1: Verify Training Data
```python
# Check what values are actually in the training ground truth
import h5py
hr_file = "/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/data/2000010100.h5"
with h5py.File(hr_file, 'r') as f:
    vwind = f['input']['v_component_of_wind_83'][:]
    print(f"Raw HR v-wind: mean={vwind.mean():.4f}, std={vwind.std():.4f}")
```

### Step 2: Check FNO Training
```python
# Verify FNO was trained correctly on v-wind
# Load FNO checkpoint and check if it preserves v-wind mean
```

### Step 3: Verify Channel Ordering
```python
# Print the actual variable order in dataset
dataset = ClimateDataset(...)
sample = dataset[0]
print("Output vars:", sample['output_vars'])
print("Output shape:", sample['output'].shape)
```

### Step 4: Check Diffusion Model Outputs During Training
```python
# Add logging to training loop to monitor v-wind predictions
# Check if bias develops over epochs or exists from start
```

---

## 📝 **NOTES**

- All data remains in **NORMALIZED** form throughout training and sampling
- Denormalization only happens during post-processing/visualization
- The diffusion model learns to predict **noise** added to normalized data
- The conditioning includes FNO output which is also normalized
- V-wind is the **ONLY** variable showing this bias issue
