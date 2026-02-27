#!/bin/bash
#SBATCH --partition=gpu-ada
#SBATCH --gpus=1
#SBATCH --job-name=genesis
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --mail-user=<oxfd3513@visiting.ox.ac.uk>
#SBATCH --time=00-23:59:59
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
module load python-cbrg
python training/Train_VAE_UNET.py