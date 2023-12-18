#!/bin/bash

#SBATCH --job-name=fstupvis_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --output=visfastall_5.out
#SBATCH --partition=v100,rtx8000

# job info
shot=$1     # 1 | 5
split=$2    # 0 | 1 | 2 | 3

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate DIaM
cd /scratch/$USER/DIaM
python -m src.test_21way_check --config configs/pascal.yaml \
    --opts split ${split} shot ${shot}
"