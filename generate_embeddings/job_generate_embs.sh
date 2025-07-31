#!/bin/bash
#
#SBATCH --partition=gpu_min11gb    # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=generate_embs    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Activate environment
#conda activate embedding-env

# Loop through all models
for model in clip resnet101 facenet; do
    echo "Running model: $model"
    python generate_embeddings/generate_embs.py \
        --model $model \
        --batch_size 200 \
        --workers 2
done

echo "All models finished."

