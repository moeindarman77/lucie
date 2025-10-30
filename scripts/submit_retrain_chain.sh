#!/bin/bash

# Submit the first job and capture its job ID
echo "Submitting job 1..."
job1=$(qsub job_retrain_chain.slurm)
echo "Job 1 submitted: $job1"

# Submit job 2 that depends on job 1 (afterany = run after job ends, regardless of exit status)
echo "Submitting job 2 (depends on job 1)..."
job2=$(qsub -W depend=afterany:$job1 job_retrain_chain.slurm)
echo "Job 2 submitted: $job2"

# Submit job 3 that depends on job 2
echo "Submitting job 3 (depends on job 2)..."
job3=$(qsub -W depend=afterany:$job2 job_retrain_chain.slurm)
echo "Job 3 submitted: $job3"

# Submit job 4 that depends on job 3
echo "Submitting job 4 (depends on job 3)..."
job4=$(qsub -W depend=afterany:$job3 job_retrain_chain.slurm)
echo "Job 4 submitted: $job4"

echo ""
echo "All jobs submitted successfully!"
echo "Job chain: $job1 -> $job2 -> $job3 -> $job4"
echo ""
echo "Monitor progress with: qstat -u mdarman"
echo "Check logs in: ERA5_DDPM_retrain.o*"
echo "Training will run for up to 48 hours total (4 x 12 hours)"
echo "Results will be saved to: /glade/derecho/scratch/mdarman/lucie/results/unet_final_v10_retrain/"
