#!/bin/bash
#PBS -N diagnose_vwind
#PBS -A UCSC0009
#PBS -l walltime=00:30:00
#PBS -q main
#PBS -l select=1:ncpus=1:mem=200GB:ngpus=1
#PBS -o diagnose_vwind.o
#PBS -e diagnose_vwind.e

module load conda
conda activate jax

cd /glade/derecho/scratch/mdarman/lucie

python -u diagnose_vwind_chunked.py
