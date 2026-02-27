#!/bin/bash
#SBATCH -J ts-train-best-A
#SBATCH -N 1
#SBATCH -G a100:2
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -t 7-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --exclude=sfpga01i,sfpga01a,sfpga01n
#SBATCH -C a100_80
#SBATCH -o /scratch/%u/ts-train/logs/%x-%j.out
#SBATCH -e /scratch/%u/ts-train/logs/%x-%j.err
#SBATCH --export=NONE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
set -euo pipefail

# =============================================================================
# SLURM SCRIPT FOR FINAL MODEL TRAINING
#
# Description:
# This script trains the final LSTM model using the best hyperparameters
# identified by the Keras Tuner. It runs for an extended number of epochs
# with early stopping to ensure the model is fully converged.
#
# Pre-requisite:
# The best hyperparameters MUST be hardcoded into the build_model()
# function in the train.py script before running this.
# =============================================================================

# --- Environment Setup ---
module purge
module load mamba/latest
source activate /scratch/$USER/ts-train/env/tf_venv

# --- GPU & NCCL Stability Settings ---
export CUDA_VISIBLE_DEVICES=0,1

# Memory optimization for Sol A100 80GB nodes
# Choose ONE approach - either memory growth OR memory limits, not both
export TF_FORCE_GPU_ALLOW_GROWTH=true
# export TF_GPU_MEMORY_LIMIT=75000  # Commented out to avoid conflicts
# export TF_MEMORY_ALLOCATION_GROWTH=true  # Commented out to avoid conflicts

# Sol-specific optimizations for A100 nodes
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# =============================================================================
# NCCL Configuration for Multi-GPU Training
# =============================================================================
# For modern HPC clusters like ASU Sol, these settings are typically NOT needed
# and can actually hurt performance. Only uncomment if you experience:
# - Multi-GPU communication failures
# - Training hangs during distributed training
# - NCCL connection errors
# =============================================================================

# Basic NCCL debugging (uncomment only for troubleshooting)
# export NCCL_DEBUG=INFO

# Advanced NCCL settings (uncomment only if needed for compatibility)
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_IB_DISABLE=1          # Disables InfiniBand (slower but more compatible)
# export NCCL_P2P_DISABLE=1         # Disables P2P (slower but more compatible)
# export NCCL_SOCKET_IFNAME=lo      # Forces loopback interface

# =============================================================================
# RECOMMENDATION: Start with NO NCCL variables set for optimal performance
# Only add these if you encounter specific communication issues
# =============================================================================

# --- TensorFlow Performance Settings ---
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_USE_CUDNN=1

# --- Directory and Path Definitions ---
DATA_DIR="/scratch/$USER/ts-train/data/for04slurmfileLOADMODEL"
# IMPORTANT: Saving to a new directory to keep results clean
OUTPUT_DIR="/scratch/$USER/ts-train/outputs/LOADMODEL/final_model_run"
LOGS_DIR="/scratch/$USER/ts-train/logs"
VENV_PYTHON="/scratch/$USER/ts-train/env/tf_venv/bin/python"

# Create the new output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "Starting FINAL training with best hyperparameters..."
echo "======================================================"
echo "Epochs set to 200 with early stopping."
echo "Results will be saved in: $OUTPUT_DIR"
echo "======================================================"

srun "$VENV_PYTHON" $HOME/projects/ts-train/trainCPU.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --logs_dir "$LOGS_DIR" \
  --dataset_kind electric \
  --epochs 200 \
  --batch_size 32

echo "======================================================"
echo "Final training completed!"
echo "Check the output directory for the final model and performance metrics."
echo "======================================================"