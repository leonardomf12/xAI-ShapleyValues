#!/bin/bash
#
#SBATCH --partition=cpu_7cores    # Partition
#SBATCH --job-name=full_csv                # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR
#SBATCH --qos=cpu_7cores


python3 "/nas-ctm01/homes/mlsampaio/classifier/dataloaders.py"