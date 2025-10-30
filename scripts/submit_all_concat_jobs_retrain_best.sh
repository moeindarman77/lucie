#!/bin/bash
# Submit all concatenation jobs for Best Retrained Model

echo "Submitting all concatenation jobs for Best Retrained Model (epoch 404)..."
echo ""

qsub job_concat_temperature_retrain_best.slurm
echo "✓ Submitted temperature concatenation job"

qsub job_concat_uwind_retrain_best.slurm
echo "✓ Submitted u-wind concatenation job"

qsub job_concat_vwind_retrain_best.slurm
echo "✓ Submitted v-wind concatenation job"

qsub job_concat_precipitation_retrain_best.slurm
echo "✓ Submitted precipitation concatenation job"

echo ""
echo "All 4 concatenation jobs submitted for Best Retrained Model!"
echo "Check status with: qstat -u $USER"
echo "Check logs with: ls -lh concat_*_best.o*"
