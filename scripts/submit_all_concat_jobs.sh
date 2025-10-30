#!/bin/bash
# Submit all concatenation jobs at once

echo "Submitting all concatenation jobs..."
echo ""

qsub job_concat_temperature.slurm
echo "✓ Submitted temperature concatenation job"

qsub job_concat_uwind.slurm
echo "✓ Submitted u-wind concatenation job"

qsub job_concat_vwind.slurm
echo "✓ Submitted v-wind concatenation job"

qsub job_concat_precipitation.slurm
echo "✓ Submitted precipitation concatenation job"

echo ""
echo "All 4 concatenation jobs submitted!"
echo "Check status with: q"
echo "Check logs with: ls -lh concat_*.o*"
