#!/bin/bash
# Helper script to check if FNO stats computation is done and submit training job

STATS_FILE="/glade/derecho/scratch/mdarman/lucie/fno_output_stats_scalar.npz"
STATS_JOB_ID="3472212"

echo "============================================================================"
echo "Auto-Submit Training After FNO Stats Computation"
echo "============================================================================"
echo ""

# Check if stats job is still running
JOB_STATUS=$(qstat -x $STATS_JOB_ID 2>/dev/null | grep $STATS_JOB_ID | awk '{print $5}')

if [ ! -z "$JOB_STATUS" ]; then
    if [[ "$JOB_STATUS" == "R" ]] || [[ "$JOB_STATUS" == "Q" ]]; then
        echo "⏳ Stats computation job ($STATS_JOB_ID) is still $JOB_STATUS"
        echo "   Waiting for completion..."
        echo ""
        echo "   Check progress with:"
        echo "   tail -f compute_fno_stats.o$STATS_JOB_ID"
        echo ""
        echo "   Run this script again later, or wait for job to finish."
        exit 0
    fi
fi

# Check if stats file exists
if [ ! -f "$STATS_FILE" ]; then
    echo "❌ ERROR: Stats file not found at $STATS_FILE"
    echo ""
    echo "   Stats job may have failed. Check output:"
    echo "   less compute_fno_stats.o$STATS_JOB_ID"
    exit 1
fi

echo "✓ Stats file exists: $STATS_FILE"
echo ""

# Verify stats file
echo "Verifying stats file..."
python verify_fno_stats.py

VERIFY_STATUS=$?
if [ $VERIFY_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Stats file verification failed!"
    echo "   Please check the output above and fix any issues."
    exit 1
fi

echo ""
echo "============================================================================"
echo "Ready to Submit Training Job"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Training script: src/tools/train_ddpm_normalized_fno.py"
echo "  Config file:     src/config/ERA5_config_normalized_fno.yaml"
echo "  Task name:       unet_normalized_fno"
echo "  Stats file:      $STATS_FILE"
echo ""
echo "Job will use:"
echo "  - 4 GPUs"
echo "  - 235GB RAM"
echo "  - 12 hour walltime"
echo "  - main queue"
echo ""

read -p "Submit training job now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Submitting job..."
    JOB_ID=$(qsub job_train_normalized_fno.slurm)
    echo "✓ Training job submitted: $JOB_ID"
    echo ""
    echo "Monitor progress with:"
    echo "  qstat -u \$USER"
    echo "  tail -f train_normalized_fno.o${JOB_ID#*.}"
    echo ""
    echo "Training logs will be saved to:"
    echo "  results/unet_normalized_fno/training_ldm.log"
else
    echo ""
    echo "Job not submitted. To submit manually:"
    echo "  qsub job_train_normalized_fno.slurm"
fi

echo ""
echo "Done!"
