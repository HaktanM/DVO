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

apptainer exec --writable --fakeroot --no-home  \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/container \
    bash -c "
        export TORCH_CUDA_ARCH_LIST=9.0
        source /env/bin/activate && \
        cd /DVO && 
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --no-cache && \
        pip install torch_scatter -f https://pytorch-geometric.com/whl/torch-2.7.0+cu126.html --no-cache && \
        pip install -r requirements.txt --no-build-isolation --no-cache && \
        pip install . --no-build-isolation --no-cache
    "


exit

