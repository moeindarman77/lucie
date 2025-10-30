#!/bin/bash
# Check status of climatology computation jobs and results

echo "============================================================================"
echo "Climatology Computation Status Check"
echo "============================================================================"
echo ""

echo "Job Status:"
echo "-----------"
qstat 3444402 3444403 2>/dev/null || echo "Jobs completed or not found"
echo ""

echo "Checking for output files..."
echo "-----------------------------"
echo ""

# Check retrain_best
echo "Retrain Best:"
BEST_DIR="/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr_retrain_best"
if [ -f "$BEST_DIR/climatology_fno.npz" ]; then
    echo "  ✓ climatology_fno.npz exists ($(du -h $BEST_DIR/climatology_fno.npz | cut -f1))"
else
    echo "  ✗ climatology_fno.npz not found"
fi

if [ -f "$BEST_DIR/climatology_diffusion.npz" ]; then
    echo "  ✓ climatology_diffusion.npz exists ($(du -h $BEST_DIR/climatology_diffusion.npz | cut -f1))"
else
    echo "  ✗ climatology_diffusion.npz not found"
fi
echo ""

# Check retrain_latest
echo "Retrain Latest:"
LATEST_DIR="/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr_retrain_latest"
if [ -f "$LATEST_DIR/climatology_fno.npz" ]; then
    echo "  ✓ climatology_fno.npz exists ($(du -h $LATEST_DIR/climatology_fno.npz | cut -f1))"
else
    echo "  ✗ climatology_fno.npz not found"
fi

if [ -f "$LATEST_DIR/climatology_diffusion.npz" ]; then
    echo "  ✓ climatology_diffusion.npz exists ($(du -h $LATEST_DIR/climatology_diffusion.npz | cut -f1))"
else
    echo "  ✗ climatology_diffusion.npz not found"
fi
echo ""

echo "Log Files:"
echo "----------"
if ls compute_clim_best.o* 2>/dev/null; then
    echo "Retrain Best log:"
    ls -lh compute_clim_best.o*
fi

if ls compute_clim_latest.o* 2>/dev/null; then
    echo "Retrain Latest log:"
    ls -lh compute_clim_latest.o*
fi

echo ""
echo "============================================================================"
