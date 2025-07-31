#!/bin/bash
#
#SBATCH --partition=gpu_min32GB       # Partition
#SBATCH --job-name=stats_dataset                 # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR
#SBATCH --qos=gpu_min32GB


python3 "/nas-ctm01/homes/mlsampaio/classifier/dataset_stats.py"