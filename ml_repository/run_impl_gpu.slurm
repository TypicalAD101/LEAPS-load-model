#!/bin/bash
#SBATCH -J ts-impl
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -C a100_80
#SBATCH -o /scratch/%u/ts-train/logs/%x-%j.out
#SBATCH -e /scratch/%u/ts-train/logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE

set -euo pipefail

module purge
module load mamba/latest
source activate /scratch/$USER/ts-train/env/tf_venv

cd /home/$USER/projects/ts-train/ml_repository

# sanity
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# run
python run_impl.py