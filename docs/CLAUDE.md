# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
LUCIE (Learning with Unified Climate Intelligence Engine) is a PyTorch-based research project focused on climate modeling and super-resolution using diffusion models and Fourier Neural Operators (FNOs). The project specifically implements stable diffusion models for climate data downscaling and super-resolution, working with ERA5 climate datasets.

## Running Environment
This code is designed to run on NCAR's Derecho HPC system with SLURM job scheduling. The project uses:
- SLURM job scripts (`.slurm` files) for distributed training
- Conda environment `jax` with CUDA support
- PBS/SLURM resource allocation with GPUs

## Common Commands

### Training Models
```bash
# Set PYTHONPATH to src directory
cd src
export PYTHONPATH=$(pwd)

# Train VAE/Autoencoder
python tools/train_vae.py --config config/ERA5_config_lsm.yaml
python tools/train_ae.py --config config/ERA5_config_lsm_ae.yaml

# Train FNO (Fourier Neural Operator)
python tools/train_fno_final.py --config config/ERA5_config_fno.yaml

# Train diffusion models
python tools/train_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml
python tools/train_lazy_diff.py --config config/ERA5_config_lazy_diff.yaml
```

### Sampling/Inference
```bash
# Sample from trained models
python tools/sample_ddpm_final_v2.py --config config/ERA5_config_final_v2.yaml
python tools/sample_fno_final.py --config config/ERA5_config_fno.yaml
python tools/sample_lazy_diff.py --config config/ERA5_config_lazy_diff.yaml

# Sample for LUCIE evaluation
python tools/sample_vae_4LUCIE.py --config config/ERA5_config_lsm.yaml
python tools/sample_ae_4LUCIE.py --config config/ERA5_config_lsm_ae.yaml
```

### Running on SLURM/PBS
```bash
# Submit jobs using SLURM
qsub job.slurm          # Main training job
qsub job_array.slurm    # Array jobs for multiple experiments
qsub job_cpu.slurm      # CPU-only jobs
```

### Environment Setup
```bash
# Load required modules
module load conda
conda activate jax

# Set Python path for imports
export PYTHONPATH="/glade/derecho/scratch/mdarman/lucie/src"
```

## Architecture Overview

### Core Components
1. **Models** (`src/models/`):
   - `fno.py`: Fourier Neural Operator for climate super-resolution
   - `simple_unet_final_v2.py`: U-Net architecture for diffusion models
   - `vae.py`, `ae.py`: Variational and standard autoencoders
   - `fno_ddpm_v2.py`: Combined FNO-DDPM models

2. **Dataset Loaders** (`src/dataset/`):
   - `ClimateDataset_v2.py`: Main dataset class for ERA5 climate data
   - Handles high-res/low-res climate data pairs
   - Supports HDF5 format with temporal alignment

3. **Training Scripts** (`src/tools/`):
   - Two-stage training: FNO first, then diffusion model
   - Distributed training support with DDP
   - Separate scripts for different model variants

4. **Configuration** (`src/config/`):
   - YAML-based configuration system
   - Separate configs for different experiments and model types
   - Key configs: `ERA5_config_final_v2.yaml`, `ERA5_config_lazy_diff.yaml`

### Data Flow
1. **Input**: Low-resolution ERA5 climate data (multiple variables)
2. **FNO Stage**: Super-resolution using Fourier Neural Operator
3. **Diffusion Stage**: Further refinement using latent diffusion models
4. **Output**: High-resolution climate predictions

### Model Training Pipeline
1. **Stage 1**: Train FNO model for initial super-resolution
2. **Stage 2**: Train diffusion model (DDPM) using FNO outputs
3. **Evaluation**: Generate samples and evaluate against ground truth

## Configuration System
- All experiments configured via YAML files in `src/config/`
- Key parameters:
  - `dataset_params`: Data paths, normalization settings
  - `fno_params`: FNO architecture (modes, width)
  - `train_params`: Learning rates, batch sizes, epochs
  - `diffusion_params`: Noise scheduling parameters

## Data Requirements
- ERA5 climate data in HDF5 format
- Normalization statistics files (`.npz` format)
- Land-sea mask files for conditioning
- File structure organized by year and time resolution

## Output Structure
Results saved in `results/` directory with:
- Model checkpoints in `checkpoints/` subdirectory
- Training logs (`training_ldm.log`, `training.log`)
- Sample outputs and evaluation metrics
- Configuration copies for reproducibility

## Important Notes
- Uses distributed training across multiple GPUs
- Memory-intensive due to high-resolution climate data
- Requires significant storage for datasets and checkpoints
- Some training scripts have commented-out variants for different experiments
- File paths are hardcoded for Derecho HPC system