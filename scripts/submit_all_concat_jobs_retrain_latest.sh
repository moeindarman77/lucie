#!/bin/bash
# Submit all concatenation jobs for Latest Retrained Model

echo "Submitting all concatenation jobs for Latest Retrained Model (epoch 415)..."
echo ""

qsub job_concat_temperature_retrain_latest.slurm
echo "✓ Submitted temperature concatenation job"

qsub job_concat_uwind_retrain_latest.slurm
echo "✓ Submitted u-wind concatenation job"

qsub job_concat_vwind_retrain_latest.slurm
echo "✓ Submitted v-wind concatenation job"

qsub job_concat_precipitation_retrain_latest.slurm
echo "✓ Submitted precipitation concatenation job"

echo ""
echo "All 4 concatenation jobs submitted for Latest Retrained Model!"
echo "Check status with: qstat -u $USER"
echo "Check logs with: ls -lh concat_*_latest.o*"
