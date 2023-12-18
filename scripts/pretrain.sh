#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jx2314@nyu.edu
#SBATCH --output=pretrain.out
#SBATCH --gres=gpu # How much gpu need, n is the number
#SBATCH --partition=v100

module purge

DATA=$1
SPLIT=$2


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

echo "start"
singularity exec --nv \
            --overlay ${ext3_path}:ro \
            ${sif_path} /bin/bash -c " 
            source /ext3/env.sh;
            python -m src.pretrain --config configs/${DATA}_pretrain.yaml \
					 --opts split ${SPLIT}"

echo "finish"


#GREENE GREENE_GPU_MPS=yes