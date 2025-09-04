#!/bin/bash

#SBATCH -A romer
#SBATCH --output=/arf/scratch/hyalcin/slurm_outs/%A.out
#SBATCH --error=/arf/scratch/hyalcin/slurm_outs/%A.err
#SBATCH --time=00:20:00
#SBATCH --job-name=train

#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=printGPU
#SBATCH --gres=gpu:1
#SBATCH -C H100

apptainer exec --nv --no-home \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/container \
    bash -c '
        cd /DVO && \
        source venv/bin/activate && \
        export HOME=/DVO && \
        nvidia-smi && nvcc --version && \
        python3 -c "import torch; print(torch.__version__)" && \
        pip install . --no-build-isolation --no-cache && \
        python3 train_h5.py
    '
exit

