#!/bin/bash
#
#SBATCH --partition=gpu_min8gb       # Partition
#SBATCH --job-name=classlr           # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR
#SBATCH --qos=gpu_min8gb


python3 "/nas-ctm01/homes/mlsampaio/classifier/train_eval.py"