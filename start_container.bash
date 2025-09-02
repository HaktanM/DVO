apptainer shell --no-home --writable --fakeroot \
    --bind /arf/scratch/hyalcin/datasets/h5:/data \
    --bind /arf/scratch/hyalcin/DVO:/DVO \
    /arf/scratch/hyalcin/miniconda3-hyalcin
