#!/bin/bash
#SBATCH -J load-tuner-a100
#SBATCH -N 1
#SBATCH -G a100:2
#SBATCH --cpus-per-task=16
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH --exclude=sfpga01i,sfpga01a,sfpga01n
#SBATCH -C a100_80
#SBATCH -o /scratch/%u/ts-train/logs/%x-%j.out
#SBATCH -e /scratch/%u/ts-train/logs/%x-%j.err
#SBATCH --export=NONE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
set -euo pipefail

# =============================================================================
# SLURM SCRIPT FOR NEW LOAD MODEL — KERAS TUNER HYPERPARAMETER SEARCH
#
# Uses train_load.py (no differencing, 8-channel input with time features).
# Runs Keras Tuner RandomSearch to find the best hyperparameters.
# =============================================================================

# --- Environment Setup ---
module purge
module load mamba/latest
source activate /scratch/$USER/ts-train/env/tf_venv

# --- GPU Configuration ---
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# --- TensorFlow Performance ---
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_USE_CUDNN=1

# --- Directory Definitions ---
DATA_DIR=/scratch/$USER/ts-train/data/LOADMODEL
OUTPUT_DIR=/scratch/$USER/ts-train/outputs/NEW_LOAD_MODEL
LOGS_DIR=/scratch/$USER/ts-train/logs/NEW_LOAD_MODEL
VENV_PYTHON="/scratch/$USER/ts-train/env/tf_venv/bin/python"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "======================================================"
echo "NEW LOAD MODEL — Keras Tuner Hyperparameter Search"
echo "  - No differencing"
echo "  - 8-channel input (load + 7 time features)"
echo "  - Seq2Seq many-to-many (96 → 96)"
echo "======================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
scontrol show job $SLURM_JOB_ID | egrep -i "mem|tres|cpu|gres|qos|partition"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus=2 \
  "$VENV_PYTHON" $HOME/projects/ts-train/NEW_LOAD_MODEL/train_load.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_kind electric \
  --logs_dir "$LOGS_DIR" \
  --use_tuner \
  --max_trials 5 \
  --epochs_per_trial 5 \
  --batch_size 128

echo "Hyperparameter search completed!"
echo "Check $OUTPUT_DIR for best model and hyperparameters."
