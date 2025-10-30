# Recommendations to Fix Systematic Bias in Normalized FNO Model

## 🔍 ROOT CAUSE IDENTIFIED

The systematic wind biases (-3.78 m/s U-wind, -4.56 m/s V-wind) are caused by **incorrect FNO output normalization statistics**.

### Current Problem:
- FNO normalization stats have means ≈ 0 and std ≈ 1
- These were computed on ALREADY NORMALIZED data (not raw FNO outputs)
- During training: `normalized = (fno_output - 0.003) / 0.94` ≈ `fno_output / 0.94`
- This creates a scale mismatch and systematic bias

### Evidence:
```
FNO Stats (WRONG):              Ground Truth Stats (CORRECT):
U-Wind: mean=0.003, std=0.941   U-Wind: mean=11.86, std=17.28
V-Wind: mean=0.003, std=0.919   V-Wind: mean=-0.03, std=13.40
```

---

## 🎯 RECOMMENDED SOLUTIONS (Ranked by Priority)

### **OPTION 1: Recompute Correct FNO Normalization Stats** ⭐ BEST SOLUTION

**What to do:**
1. Generate FNO outputs on training data WITHOUT any normalization
2. Compute statistics on these RAW FNO outputs (not normalized ones)
3. The stats should have means similar to ground truth HR data

**Implementation:**
```bash
# Create script to compute correct FNO stats
python compute_correct_fno_stats.py --output fno_output_stats_corrected.npz
```

**Expected results:**
```
U-Wind: mean ≈ 11.86 m/s (matching HR data)
V-Wind: mean ≈ -0.03 m/s (matching HR data)
```

**Then:**
- Update config to use corrected stats file
- Retrain diffusion model from the pretrained checkpoint
- Expected bias reduction: ~90%

**Pros:**
- Fixes root cause
- Maintains model architecture
- Clean, principled solution

**Cons:**
- Requires recomputing FNO outputs (computational cost)
- Needs retraining (~40-100 more epochs)

**Estimated improvement:**
- U-Wind bias: -3.78 → <0.5 m/s
- V-Wind bias: -4.56 → <0.5 m/s

---

### **OPTION 2: Remove FNO Normalization Entirely** ⭐ QUICK FIX

**What to do:**
Simply DON'T normalize FNO outputs - use them as-is (they're already in the right scale).

**Implementation:**
In `train_ddpm_normalized_fno.py` (line 287):
```python
# OLD (WRONG):
fno_output_normalized = (fno_output - fno_mean) / (fno_std + 1e-6)

# NEW (CORRECT):
# FNO outputs are already properly scaled - use them directly!
fno_output_normalized = fno_output  # No normalization needed!
```

**Then:**
- Continue training from current checkpoint
- The model will gradually adapt

**Pros:**
- Simplest fix - 1 line change
- No recomputation needed
- Can continue from current checkpoint

**Cons:**
- Model was trained with wrong normalization for 451 epochs
- May take many epochs to fully adapt
- Not as clean as Option 1

**Estimated improvement:**
- Gradual bias reduction over 50-100 epochs
- Final bias: <1 m/s (with enough training)

---

### **OPTION 3: Post-Processing Bias Correction** ⚠️ WORKAROUND

**What to do:**
Apply systematic bias correction during sampling/inference.

**Implementation:**
```python
# After diffusion sampling, apply correction:
bias_corrections = {
    'uwind': 3.78,   # m/s
    'vwind': 4.56,   # m/s
    'temp': -1.05,   # K
}

# In sampling code:
output_corrected = output.copy()
output_corrected[:, 1] += bias_corrections['uwind']  # U-wind channel
output_corrected[:, 2] += bias_corrections['vwind']  # V-wind channel
output_corrected[:, 0] -= bias_corrections['temp']   # Temperature channel
```

**Pros:**
- Immediate fix - no retraining
- Works with current model
- Can be applied retroactively to existing samples

**Cons:**
- **Doesn't fix underlying issue**
- Constant bias correction (not spatially varying)
- Unprincipled workaround

**Use case:**
- Quick evaluation while retraining with Option 1
- Emergency fix for urgent analysis

---

### **OPTION 4: Train Bias Correction Layer** 🔬 ADVANCED

**What to do:**
Add a learnable bias/scale layer after FNO normalization.

**Implementation:**
```python
class FNOBiasCorrection(nn.Module):
    def __init__(self, num_channels=4):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias

# Add to model:
self.fno_correction = FNOBiasCorrection(num_channels=4)
fno_corrected = self.fno_correction(fno_output_normalized)
```

**Then:**
- Freeze diffusion model weights
- Train only bias correction layer for 10-20 epochs
- Unfreeze and fine-tune all together

**Pros:**
- Learns optimal correction
- Can handle spatially-varying biases
- Relatively quick to train

**Cons:**
- Adds model complexity
- Still doesn't fix root normalization issue
- Requires code modification

---

## 📋 DETAILED ACTION PLAN

### **Recommended Approach: Option 1 (Best) + Option 3 (Temporary)**

#### **Phase 1: Immediate Workaround (Today)**
1. Apply post-processing bias correction (Option 3)
2. Generate corrected samples for urgent analysis
3. Document the correction in results

#### **Phase 2: Compute Correct Stats (1-2 days)**
1. Create script to compute FNO stats correctly:
   ```python
   # compute_correct_fno_stats.py
   # - Load training data
   # - Run FNO on low-res inputs
   # - Compute mean/std on RAW FNO outputs (before any normalization)
   # - Save to fno_output_stats_corrected.npz
   ```

2. Verify stats match ground truth distributions:
   ```python
   # Check: FNO mean should ≈ HR mean for each variable
   assert abs(fno_uwind_mean - hr_uwind_mean) < 1.0  # Within 1 m/s
   assert abs(fno_vwind_mean - hr_vwind_mean) < 1.0  # Within 1 m/s
   ```

#### **Phase 3: Retrain with Correct Normalization (1-2 days)**
1. Update config file:
   ```yaml
   fno_normalization_file: '/path/to/fno_output_stats_corrected.npz'
   ```

2. Start fresh training OR continue from pretrained checkpoint:
   ```bash
   # Option A: Start fresh (recommended)
   rm results/unet_normalized_fno/checkpoints/*
   ./submit_normalized_fno_chain.sh

   # Option B: Continue training (faster, but may retain some bias)
   # Just update config and submit
   ./submit_normalized_fno_chain.sh
   ```

3. Monitor bias reduction during training:
   - Sample every 50 epochs
   - Compute bias metrics
   - Training is complete when bias < 0.5 m/s for all variables

#### **Phase 4: Validation (1 day)**
1. Generate 10-20 samples with corrected model
2. Compute comprehensive statistics
3. Verify:
   - All biases < 0.5 m/s (or < 0.5 K for temperature)
   - No NaN values
   - Reasonable RMSE (temperature <3K, winds <5 m/s)

---

## 🔧 Additional Improvements (Secondary Priority)

### **A. Learning Rate Adjustment**
The current LR is very low (1e-6). Consider:
```yaml
ldm_lr: 0.00001  # Change from 1e-6 to 1e-5
```
This will help the model adapt faster to corrected normalization.

### **B. Monitor Per-Variable Losses**
Add separate loss tracking for each variable during training:
```python
loss_per_var = {
    'temp': F.mse_loss(noise_pred[:, 0], noise[:, 0]),
    'uwind': F.mse_loss(noise_pred[:, 1], noise[:, 1]),
    'vwind': F.mse_loss(noise_pred[:, 2], noise[:, 2]),
    'precip': F.mse_loss(noise_pred[:, 3], noise[:, 3]),
}
```
This helps identify if bias is specific to certain variables.

### **C. Increase Batch Size (if memory allows)**
Current: 2 per GPU
Suggested: 4 per GPU
Benefits: More stable gradients, potentially faster convergence

### **D. Add Validation Metrics**
During training, periodically:
1. Sample from validation data
2. Compute bias, RMSE per variable
3. Log to tensorboard/wandb
4. Early stopping if bias increases

---

## 📊 Expected Outcomes

### **After Option 1 (Correct normalization + retrain):**
```
Variable      Current Bias    Expected Bias    Improvement
Temperature   +1.05 K        <0.3 K           ~70%
U-Wind        -3.78 m/s      <0.5 m/s         ~87%
V-Wind        -4.56 m/s      <0.5 m/s         ~89%
Precipitation +0.00006 m     <0.0001 m        Maintains
```

### **Training Time Estimate:**
- FNO stats computation: 2-4 hours (single job)
- Retraining: 40-100 epochs × 12h/job ÷ 40 epochs/job = 1-3 chain jobs = 12-36 hours
- Total: ~2-3 days

---

## 🎬 START HERE - Script Creation

I can create these scripts for you right now:

1. **compute_correct_fno_stats.py** - Compute proper FNO normalization stats
2. **apply_bias_correction.py** - Post-process existing samples (temporary fix)
3. **monitor_training_bias.py** - Track bias during retraining
4. **train_ddpm_normalized_fno_v2.py** - Updated training script with fixes

Would you like me to create these scripts?

---

## 📝 Summary

**ROOT CAUSE:** FNO normalization stats computed on already-normalized data (mean≈0, std≈1)

**BEST FIX:** Recompute correct FNO stats from raw outputs, then retrain (Option 1)

**QUICK FIX:** Remove FNO normalization entirely (Option 2)

**TEMPORARY:** Post-processing bias correction (Option 3)

**RECOMMENDED TIMELINE:**
- Today: Apply temporary bias correction (Option 3)
- Days 1-2: Compute correct stats and start retraining (Option 1)
- Days 3-4: Validate and compare results
- Total: ~4 days to fully corrected model

**EXPECTED RESULT:**
- Wind biases reduced from ~4 m/s to <0.5 m/s (>87% improvement)
- Temperature bias reduced from 1.05 K to <0.3 K (~70% improvement)
- Maintain model stability (no NaN values)
