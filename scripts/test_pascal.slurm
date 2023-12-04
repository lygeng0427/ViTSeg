#!/bin/bash

#SBATCH --job-name=gfss-pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
pi=$1       # uniform | self | imglvl | upperbound
shot=$2     # 1 | 5
split=$3    # 0 | 1 | 2 | 3

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/DIaM
python -m src.test --config configs/pascal.yaml \
    --opts split ${split} shot ${shot} \
    pi_estimation_strategy ${pi} \
    > logs/pascal_${pi}_${shot}shot_${split}.log 2>&1
"