#!/bin/bash

#SBATCH --account=hyalcin
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --time=00:15:00
#SBATCH --job-name=train

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=build_dvo
#SBATCH --gres=gpu:1

apptainer exec --nv --no-home --writable --fakeroot   \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/container_dvo \
    bash -c "
	cd /DVO && \
        source .venv/bin/activate && \
        pip install . --no-build-isolation
    "


exit

