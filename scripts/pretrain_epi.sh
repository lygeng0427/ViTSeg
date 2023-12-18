#!/bin/bash

#SBATCH --job-name=pres50UP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jx2314@nyu.edu
#SBATCH --output=preres50UP_pretrain_epi_0.out
#SBATCH --gres=gpu # How much gpu need, n is the number
#SBATCH --partition=v100,rtx8000

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
            conda activate DIaM
            python -m src.pretrain_GFSSval --config configs/${DATA}_pretrain_epi.yaml \
					 --opts split ${SPLIT}"

echo "finish"


#GREENE GREENE_GPU_MPS=yes