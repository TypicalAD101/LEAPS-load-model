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

echo "script start"
echo $USER

# Basic driver info
nvidia-smi || true

# Module visibility for CUDA/cudnn (informational)
module avail cuda || true
module avail cudnn || true
