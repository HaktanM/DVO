#!/bin/bash

#SBATCH -A romer
#SBATCH --output=/arf/scratch/hyalcin/slurm_outs/%A.out
#SBATCH --error=/arf/scratch/hyalcin/slurm_outs/%A.err
#SBATCH --time=3-00:00:00
#SBATCH --job-name=train

#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=deneme
#SBATCH --gres=gpu:1
#SBATCH -C H100

export EVO_ASSETS_PATH=/data

apptainer exec --nv --no-home \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/container \
    bash -c '
        source /env/bin/activate && \
        cd /DVO && \
        export EVO_ASSETS_PATH=/data && \
        export HOME=/DVO && \
        python3 train_h5.py --P 3 --R 3--name DC3X3
    '
exit