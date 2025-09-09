#!/bin/bash

#SBATCH -A romer
#SBATCH --output=/arf/scratch/hyalcin/slurm_outs/%A.out
#SBATCH --error=/arf/scratch/hyalcin/slurm_outs/%A.err
#SBATCH --time=00:10:00
#SBATCH --job-name=train

#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=printGPU
#SBATCH --gres=gpu:1
#SBATCH -C H100

bash -c "which nvcc"

apptainer exec --nv --no-home  \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/ubuntu2404 \
    bash -c "
        cd /DVO && \
        which nvcc && \
        source venv/bin/activate && \
        which nvcc
    "


exit

