# LUCIE-DS - Downscaling Project

A PyTorch-based research project for climate data super-resolution using diffusion models and Fourier Neural Operators (FNOs). LUCIE-DS implements DDPM for climate data downscaling and super-resolution, working with ERA5 climate datasets.

## Overview

LUCIE-DS combines two powerful approaches for climate modeling:
- **Fourier Neural Operators (FNO)**: For initial super-resolution of climate variables
- **Denoising Diffusion Probabilistic Models (DDPM)**: For refinement

The project focuses on downscaling low-resolution climate data to high-resolution predictions for multiple atmospheric variables including temperature, precipitation, and wind components.

## Features

-  Multi-variable climate modeling (temperature, precipitation, u-wind, v-wind)
-  Two-stage pipeline: FNO + Diffusion models
-  Distributed training on HPC systems with SLURM
-  Comprehensive diagnostic and visualization tools
-  YAML-based configuration system
-  Support for model retraining and ensemble generation

## Repository Structure

```
LUCIE-DS/
├── src/
│   ├── models/          # Model architectures (FNO, U-Net, VAE, etc.)
│   ├── dataset/         # Dataset loaders for ERA5 and LUCIE-DS data
│   ├── tools/           # Training and sampling scripts
│   ├── config/          # YAML configuration files
│   ├── utils/           # Utility functions and helpers
│   ├── scheduler/       # Learning rate schedulers
│   ├── data_processing/ # Data concatenation and preprocessing
│   ├── diagnostics/     # Diagnostic and verification scripts
│   ├── computation/     # Statistics and climatology computation
│   ├── visualization/   # Visualization and comparison tools
│   └── tests/           # Test scripts
├── jobs/               # SLURM job scripts
│   ├── training/       # Training job scripts
│   ├── testing/        # Testing and sampling jobs
│   ├── concat/         # Data concatenation jobs
│   ├── retrain/        # Retraining jobs
│   ├── climatology/    # Climatology computation jobs
│   ├── diagnostics/    # Diagnostic jobs
│   └── fno/            # FNO-specific jobs
├── scripts/            # Shell scripts for job submission
├── docs/               # Documentation
└── outputs/            # Job output logs
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU support)
- Access to NCAR's Derecho HPC system (for full workflow)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/moeindarman77/lucie.git
cd lucie

# Create conda environment
conda create -n lucie python=3.9
conda activate lucie

# Install dependencies
pip install -r src/requirements.txt

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/lucie/src"
```

### On NCAR Derecho

```bash
# Load required modules
module load conda
conda activate jax

# Set Python path
export PYTHONPATH="/glade/derecho/scratch/mdarman/lucie/src"
```

## Quick Start

### 1. Training Models

#### Train Fourier Neural Operator (FNO)
```bash
cd src
python tools/train_fno_final.py --config config/ERA5_config_fno.yaml
```

#### Train Diffusion Model
```bash
python tools/train_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml
```

#### Using SLURM
```bash
# Submit training job
qsub jobs/training/job.slurm

# Submit array job for multiple experiments
qsub jobs/training/job_array.slurm
```

### 2. Sampling/Inference

```bash
# Sample from trained model
python tools/sample_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml

# Sample for 10-year LUCIE-DS evaluation
python tools/sample_ddpm_LUCIE-DS_10yr.py --config config/ERA5_config_LUCIE-DS_10yr.yaml
```

### 3. Data Processing

```bash
# Concatenate output files
./scripts/submit_all_concat_jobs.sh

# Compute climatology
qsub jobs/climatology/job_compute_climatology.slurm
```

## Configuration

All experiments are configured via YAML files in `src/config/`. Key configuration files:

- `ERA5_config_final_v2.yaml` - Main diffusion model configuration
- `ERA5_config_fno.yaml` - FNO model configuration
- `ERA5_config_LUCIE-DS_10yr.yaml` - 10-year sampling configuration
- `ERA5_config_normalized_fno.yaml` - Normalized FNO setup

### Key Configuration Parameters

```yaml
dataset_params:
  data_path: "/path/to/data"
  normalization_stats: "/path/to/stats.npz"

fno_params:
  modes: 12
  width: 64

train_params:
  learning_rate: 0.0001
  batch_size: 8
  epochs: 100

diffusion_params:
  timesteps: 1000
  beta_schedule: "linear"
```

## Model Pipeline

### Two-Stage Training

1. **Stage 1: FNO Training**
   - Train FNO for initial super-resolution
   - Generates coarse high-resolution predictions

2. **Stage 2: Diffusion Model Training**
   - Uses FNO outputs as conditioning
   - Refines predictions with latent diffusion

### Data Flow

```
Low-Res ERA5 → FNO → Coarse HR → Diffusion → Fine HR Predictions
     ↓                                ↓
Land-Sea Mask ────────────────────────┘
```

## Diagnostic Tools

The project includes comprehensive diagnostic utilities:

```bash
# Verify concatenated files
python src/diagnostics/verify_concatenated_files.py

# Check FNO statistics
python src/diagnostics/verify_fno_stats.py

# Diagnose wind component issues
python src/diagnostics/diagnose_vwind.py

# Check sampling progress
python src/diagnostics/check_sampling_progress.py
```

## Visualization

```bash
# Visualize normalized FNO samples
python src/visualization/visualize_normalized_fno_samples.py

# Compare FNO vs Diffusion outputs
python src/visualization/compare_fno_vs_diffusion.py
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Production Run Instructions](docs/PRODUCTION_RUN_INSTRUCTIONS.md)
- [Normalization Data Flow](docs/NORMALIZATION_DATA_FLOW.md)
- [10-Year Sampling README](docs/LUCIE-DS_10YR_SAMPLING_README.md)
- [VWind Bias Root Cause Analysis](docs/VWIND_BIAS_ROOT_CAUSE.md)

## Scripts

Convenience scripts for common workflows:

```bash
# Auto-submit training jobs
./scripts/auto_submit_training.sh

# Submit all concatenation jobs
./scripts/submit_all_concat_jobs.sh

# Check climatology computation status
./scripts/check_climatology_status.sh

# Test normalized FNO stability
./scripts/test_normalized_fno_stability.sh
```

## Data Requirements

- ERA5 climate reanalysis data in HDF5 format
- Normalization statistics files (.npz)
- Land-sea mask files for conditioning
- Organized by year and temporal resolution

## Output Structure

Results are saved in `results/` directory:
```
results/
├── checkpoints/       # Model checkpoints
├── samples/          # Generated samples
├── logs/             # Training logs
└── config.yaml       # Copy of configuration
```

## Citation

If you use LUCIE-DS in your research, please cite:

```bibtex
@software{LUCIE-DS2025,
  title={LUCIE-DS: Learning with Unified Climate Intelligence Engine},
  author={Darman, Moein},
  year={2025},
  url={https://github.com/moeindarman77/LUCIE-DS}
}
```

## License

[Add your license here]

## Contact

For questions and support, please open an issue on GitHub or contact:
- Moein Darman - [GitHub](https://github.com/moeindarman77)

