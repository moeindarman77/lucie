#!/bin/bash
# Script to verify setup and submit test job for LUCIE 10yr sampling
# Run this on a GPU node after doing: gpu1 && prep

echo "=========================================="
echo "LUCIE 10yr Sampling - Test Run"
echo "=========================================="
echo ""

# Check current directory
echo "Current directory: $(pwd)"
echo ""

# Verify environment
echo "1. Checking Python environment..."
which python
python --version
echo ""

# Run verification script
echo "2. Verifying files..."
python check_lucie_10yr_setup.py
echo ""

# Ask user to confirm
echo "=========================================="
echo "Verification complete!"
echo ""
echo "If everything looks good, submit the test job with:"
echo "  cd /glade/derecho/scratch/mdarman/lucie"
echo "  qsub job_test_lucie_10yr.slurm"
echo ""
echo "Then monitor with:"
echo "  q  # check job status"
echo "  cat lucie_10yr_test.o*  # check output when done"
echo "=========================================="
