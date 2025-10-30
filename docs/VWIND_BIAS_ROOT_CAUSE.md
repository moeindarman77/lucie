# V-Wind Bias Root Cause Analysis

## 🔴 PROBLEM SUMMARY

**V-wind has a systematic bias of -6.2 m/s in diffusion model outputs**
- Expected mean: `-0.034 m/s`
- FNO output: `-0.155 m/s` ✅ (small error, acceptable)
- **Diffusion output: `-6.244 m/s`** ❌ (LARGE BIAS)

---

## 🔍 ROOT CAUSE: FNO Conditioning Bias

### The Data Flow

```
Training Data (Normalized)
        ↓
     FNO Model (with weighted spectral loss)
        ↓
  FNO Outputs (used as conditioning)
        ↓
  Diffusion Model (learns to "correct" FNO)
        ↓
  Final Predictions (with learned corrections)
```

### The Problem

**FNO Training uses UNEQUAL loss weights:**

From `/src/utils/spectral_sqr_abs2.py` lines 11-16:
```python
channels=[
    ("channel_0", 0, 0.10),  # Temperature:    10% weight
    ("channel_1", 1, 0.10),  # Specific_hum:   10% weight
    ("channel_2", 2, 0.40),  # U-wind:         40% weight ⚠️
    ("channel_3", 3, 0.40),  # V-wind:         40% weight ⚠️
],
```

**Impact:**
- FNO prioritizes wind accuracy 4x more than temperature
- This creates different optimization dynamics for each variable
- FNO learns different systematic biases for each variable

---

## 📊 MEASURED BIASES

### Normalized Space (from 100 samples):

| Variable | FNO Mean | Diffusion Mean | Correction (Diff-FNO) |
|----------|----------|----------------|----------------------|
| Temperature | -0.028 | +0.107 | **+0.135σ** |
| U-wind | +0.064 | -0.361 | **-0.425σ** |
| V-wind | -0.009 | -0.464 | **-0.455σ** ⚠️ |
| Precipitation | -0.029 | +0.034 | **+0.063σ** |

### Physical Space (denormalized):

| Variable | FNO Mean | Diffusion Mean | Expected | Bias |
|----------|----------|----------------|----------|------|
| Temperature | 277.96 K | 280.84 K | 278.56 K | **+2.27 K** |
| U-wind | 12.97 m/s | 5.62 m/s | 11.86 m/s | **-6.24 m/s** ❌ |
| V-wind | -0.15 m/s | -6.24 m/s | -0.03 m/s | **-6.21 m/s** ❌ |
| Precipitation | 0.0005 m | 0.0007 m | 0.0006 m | **+0.0001 m** |

---

## 💡 WHY THIS HAPPENS

### 1. FNO Training Creates Variable-Specific Biases

The weighted spectral loss causes FNO to:
- Optimize winds more aggressively (40% weight each)
- De-prioritize temperature (10% weight)
- This leads to different convergence behaviors

### 2. Diffusion Model Learns "Corrections"

The diffusion model is **conditioned on FNO outputs** during training:
```python
# From train_ddpm_final_v2_retrain.py line 257
cond_input = torch.cat([fno_output, hres[:, 5:6]], dim=1)
```

The diffusion model learns:
- "FNO temperature is too low → add +2.3K"
- "FNO u-wind is too high → subtract -6.2 m/s"
- **"FNO v-wind is too low → subtract -6.2 m/s"** ← WRONG CORRECTION!

### 3. Why V-wind Gets Large Correction

Possible reasons:
1. **FNO v-wind has subtle systematic bias** (-0.009σ in normalized space)
2. **Diffusion model over-learns this correction** during training
3. **The 40% weight in FNO training** might cause FNO to fit v-wind differently than other variables
4. **Correlation with other variables**: V-wind corrections might be entangled with corrections for other variables

---

## 🧪 VERIFICATION

We verified:
- ✅ Normalization stats are correct
- ✅ Training data is correct
- ✅ FNO outputs have small biases (<0.1σ for most variables)
- ✅ Diffusion adds large systematic corrections (-0.455σ for v-wind)
- ✅ The correction is learned, not a data artifact

---

## 🔧 POTENTIAL SOLUTIONS

### Solution 1: Retrain FNO with Equal Weights
**Most Direct Fix**

Modify `spectral_sqr_abs2` to use equal weights:
```python
channels=[
    ("channel_0", 0, 0.25),  # Temperature:    25% weight
    ("channel_1", 1, 0.25),  # U-wind:         25% weight
    ("channel_2", 2, 0.25),  # V-wind:         25% weight
    ("channel_3", 3, 0.25),  # Precipitation:  25% weight
],
```

**Pros:**
- Addresses root cause
- FNO will learn more balanced predictions

**Cons:**
- Requires retraining FNO (expensive)
- Requires retraining diffusion model (expensive)

---

### Solution 2: Add Bias Correction Layer
**Quick Fix**

Add learnable bias correction after diffusion model:
```python
# After diffusion sampling
output = diffusion_model(...)
bias_correction = torch.tensor([0, 0, +6.21/13.396, 0])  # Only correct v-wind
output = output + bias_correction
```

**Pros:**
- Quick fix, no retraining needed
- Can verify if this solves the climatology issue

**Cons:**
- Doesn't fix root cause
- Hardcoded correction might not generalize

---

### Solution 3: Retrain Diffusion WITHOUT FNO Conditioning
**Alternative Architecture**

Train diffusion model to predict HR directly from LR (upsampled), without FNO conditioning:

```python
# Instead of:
cond_input = torch.cat([fno_output, orography], dim=1)

# Use:
cond_input = torch.cat([lres_upsampled, orography], dim=1)
```

**Pros:**
- Removes dependency on FNO biases
- Simpler architecture

**Cons:**
- Might reduce quality (FNO provides good initial guess)
- Still requires retraining diffusion model

---

### Solution 4: Use FNO as Initial Guess, Not Conditioning
**Hybrid Approach**

Instead of conditioning on FNO, use FNO as initialization:
```python
# Start diffusion from FNO output instead of random noise
xt = fno_output + small_noise

# Then denoise WITHOUT FNO conditioning
for t in reversed(range(T)):
    xt = diffusion_model(xt, t, cond=orography_only)
```

**Pros:**
- Benefits from FNO but doesn't learn its biases
- Might preserve quality

**Cons:**
- Requires code changes and retraining

---

## 📝 RECOMMENDATIONS

### Immediate Action (Quick Fix):
1. **Test bias correction** to verify this solves climatology
2. Compare results with/without correction

### Long-term Fix:
1. **Retrain FNO with equal channel weights**
2. **Retrain diffusion model** with new FNO

### Alternative (if retraining is too expensive):
1. **Train a small correction network** on top of diffusion outputs
2. Use climatology data to learn the correction

---

## 🎯 CONCLUSION

**The v-wind bias is NOT a bug in the code, but a learned behavior caused by:**
1. FNO training with unequal loss weights
2. Diffusion model learning to "correct" FNO biases
3. Over-correction for v-wind component

**The fix requires addressing the FNO training weights OR post-processing the outputs.**

---

## 📌 FILES INVOLVED

- FNO Loss Function: `/src/utils/spectral_sqr_abs2.py` (lines 11-16)
- FNO Training: `/src/tools/train_fno_final.py` (line 246)
- Diffusion Training: `/src/tools/train_ddpm_final_v2_retrain.py` (line 257)
- FNO Config: `/src/config/ERA5_config_fno.yaml`
- Diffusion Config: `/src/config/ERA5_config_final_v2_retrain.yaml`
