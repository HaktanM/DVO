#!/bin/bash

#SBATCH --account=hyalcin
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --time=00:25:00
#SBATCH --job-name=train

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=tr_layers

#SBATCH --gres=gpu:1

apptainer exec --no-home \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/miniconda3-hyalcin \
    bash -c "
        cd /DVO && \
        source .venv/bin/activate && \
        python3 train_h5.py
    "
exit

