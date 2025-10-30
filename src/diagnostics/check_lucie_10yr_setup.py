"""
Quick script to verify the LUCIE 10yr setup before running the full sampling
"""
import numpy as np
import os

print("="*80)
print("LUCIE 10yr Setup Verification")
print("="*80)

# 1. Check LUCIE data file
lucie_file = '/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_2010ini_10yr.npz'
print(f"\n1. Checking LUCIE data file: {lucie_file}")
if os.path.exists(lucie_file):
    data = np.load(lucie_file)
    print(f"   ✓ File exists")
    print(f"   Keys: {list(data.keys())}")
    if 'data' in data.keys():
        print(f"   Data shape: {data['data'].shape}")
        print(f"   Number of samples: {data['data'].shape[0]}")
    data.close()
else:
    print(f"   ✗ File NOT found")

# 2. Check LSM file
lsm_file = '/glade/derecho/scratch/mdarman/lucie/lsm.npz'
print(f"\n2. Checking Land-Sea Mask file: {lsm_file}")
if os.path.exists(lsm_file):
    lsm_data = np.load(lsm_file)
    print(f"   ✓ File exists")
    print(f"   Keys: {list(lsm_data.keys())}")
    for key in lsm_data.keys():
        print(f"   {key} shape: {lsm_data[key].shape}")
    lsm_data.close()
else:
    print(f"   ✗ File NOT found")

# 3. Check normalization files
norm_file_lr = '/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009_updated.npz'
print(f"\n3. Checking LR normalization file: {norm_file_lr}")
if os.path.exists(norm_file_lr):
    norm_data = np.load(norm_file_lr, allow_pickle=True)
    print(f"   ✓ File exists")
    print(f"   Variables: {list(norm_data.keys())}")
    norm_data.close()
else:
    print(f"   ✗ File NOT found")

norm_file_hr = '/glade/derecho/scratch/mdarman/lucie/stats_hr_2000_2009_updated.npz'
print(f"\n4. Checking HR normalization file: {norm_file_hr}")
if os.path.exists(norm_file_hr):
    norm_data = np.load(norm_file_hr, allow_pickle=True)
    print(f"   ✓ File exists")
    print(f"   Variables: {list(norm_data.keys())}")
    norm_data.close()
else:
    print(f"   ✗ File NOT found")

# 5. Check model checkpoints
checkpoint_diffusion = '/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/checkpoints/best_ldm.pth'
print(f"\n5. Checking diffusion model checkpoint: {checkpoint_diffusion}")
print(f"   {'✓' if os.path.exists(checkpoint_diffusion) else '✗'} File {'exists' if os.path.exists(checkpoint_diffusion) else 'NOT found'}")

checkpoint_fno = '/glade/derecho/scratch/mdarman/lucie/results/fno_final_v1/checkpoints/best_fno.pth'
print(f"\n6. Checking FNO model checkpoint: {checkpoint_fno}")
print(f"   {'✓' if os.path.exists(checkpoint_fno) else '✗'} File {'exists' if os.path.exists(checkpoint_fno) else 'NOT found'}")

print("\n" + "="*80)
print("Verification complete!")
print("="*80)
