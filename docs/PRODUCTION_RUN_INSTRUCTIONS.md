# LUCIE 10yr Production Sampling Instructions

## Summary

**Configuration**: 25 concurrent GPU jobs to process 14,600 samples in ~9.7 hours

## Job Array Configuration

- **Total samples**: 14,600
- **Array jobs**: 25 (indices 0-24)
- **Samples per job**: 584
- **Time per job**: ~9.7 hours (within 12h limit)
- **Coverage**: 25 × 584 = 14,600 ✓ (verified)

## How to Submit

### 1. Verify Configuration (Optional)
```bash
cd /glade/derecho/scratch/mdarman/lucie
python verify_job_coverage.py
```

### 2. Submit Production Jobs
```bash
cd /glade/derecho/scratch/mdarman/lucie
qsub job_array_lucie_10yr.slurm
```

This will submit 25 jobs that will run in parallel on 25 GPUs.

### 3. Monitor Progress

**Check queue status:**
```bash
q  # alias for qstat -u mdarman
```

**Check sampling progress:**
```bash
python check_sampling_progress.py
```

**Check output files:**
```bash
watch -n 60 'ls results/unet_final_v10/samples_lucie_10yr/*.npz | wc -l'
```

**Check logs:**
```bash
ls -lh lucie_10yr_sampling.o*
tail -f lucie_10yr_sampling.o*.0  # Follow job 0 log
```

## Job Distribution

| Job Index | Start Index | End Index | Samples |
|-----------|-------------|-----------|---------|
| 0         | 0           | 583       | 584     |
| 1         | 584         | 1167      | 584     |
| 2         | 1168        | 1751      | 584     |
| ...       | ...         | ...       | 584     |
| 24        | 14016       | 14599     | 584     |

(See `verify_job_coverage.py` output for complete list)

## Output Location

Samples will be saved to:
```
/glade/derecho/scratch/mdarman/lucie/results/unet_final_v10/samples_lucie_10yr/
```

Each file: `{index}.npz` (1-indexed)
- Contains: `output` (diffusion) and `fno_output` arrays
- Shape: [1, 4, 721, 1440]
- 4 variables: 2m temp, u-wind, v-wind, precipitation

## Expected Timeline

- **Start**: When you submit
- **Duration**: ~9.7 hours (all jobs run in parallel)
- **Completion**: Check with `python check_sampling_progress.py`

## Troubleshooting

### If jobs fail:
1. Check error logs: `cat lucie_10yr_sampling.o*`
2. Identify missing samples: `python check_sampling_progress.py`
3. Re-run specific jobs if needed (modify `#PBS -J` line)

### If you need to cancel:
```bash
qdel <job_id>  # Cancel all array jobs
```

## Files Created/Modified

- `job_array_lucie_10yr.slurm` - Updated production job script
- `verify_job_coverage.py` - Verify job parameters
- `check_sampling_progress.py` - Monitor progress

## Next Steps After Completion

1. Verify all 14,600 samples generated: `python check_sampling_progress.py`
2. Run visualization notebook: `visualize_lucie_10yr_test.ipynb`
3. Compute full statistics using your analysis pipeline

---

**Ready to submit!** Run `qsub job_array_lucie_10yr.slurm`
