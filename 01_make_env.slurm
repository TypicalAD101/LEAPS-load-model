#!/bin/bash
#SBATCH -J probe-gpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 0-00:02:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/%u/ts-train/logs/%x-%j.out
#SBATCH -e /scratch/%u/ts-train/logs/%x-%j.err
#SBATCH --export=NONE
##SBATCH --mail-type=ALL
##SBATCH --mail-user=%u@asu.edu

set -euo pipefail
# cuda for gpu
module load cuda-12.6.1-gcc-12.1.0 
# Load mamba environment manager
module load mamba/latest


mamba create -y -p /scratch/$USER/ts-train/env/tf_venv -c conda-forge python=3.12

# Activate using 'source activate' (do not use 'mamba activate') [4]
source activate /scratch/$USER/ts-train/env/tf_venv

mamba list

pip install tensorflow[and-cuda]
pip install pandas scikit-learn matplotlib scikit-learn h5py keras-tuner