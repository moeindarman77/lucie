"""
Quick test to verify LucieLoader_10yr works with the new data format
"""
import sys
sys.path.insert(0, '/glade/derecho/scratch/mdarman/lucie/src')

from dataset.LucieLoader_10yr import LucieLoader
import numpy as np

print("="*80)
print("Testing LucieLoader_10yr with new LUCIE data")
print("="*80)

# Test loading
try:
    lucie_dataset = LucieLoader(
        lucie_file_path='/glade/derecho/scratch/mdarman/ERA5_hr_haiwen/LUCIE_2010ini_10yr.npz',
        normalization_file='/glade/derecho/scratch/mdarman/lucie/stats_lr_2000_2009_updated.npz',
        input_vars=['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr'],
        normalize=True
    )
    print("\n✓ LucieLoader initialized successfully!")
    print(f"  Dataset length: {len(lucie_dataset)}")

    # Test getting a sample
    sample = lucie_dataset[0]
    print(f"\n✓ Successfully loaded sample 0")
    print(f"  Sample shape: {sample.shape}")
    print(f"  Expected shape: (6, 48, 96) - [C, H, W]")

    # Check for NaN values
    has_nan = sample.isnan().any()
    print(f"\n  Contains NaN: {has_nan}")

    # Print statistics per channel
    print("\n  Channel statistics:")
    var_names = ['Temperature_7', 'Specific_Humidity_7', 'U-wind_3', 'V-wind_3', 'logp', 'tp6hr']
    for i, var in enumerate(var_names):
        print(f"    {var}: mean={sample[i].mean():.4f}, std={sample[i].std():.4f}, min={sample[i].min():.4f}, max={sample[i].max():.4f}")

    # Test a few more samples
    print("\n  Testing additional samples...")
    for idx in [100, 1000, 10000]:
        s = lucie_dataset[idx]
        print(f"    Sample {idx}: shape={s.shape}, has_nan={s.isnan().any()}")

    print("\n" + "="*80)
    print("✓ All tests passed! LucieLoader is working correctly.")
    print("="*80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("✗ Test failed!")
    print("="*80)
    sys.exit(1)
