#!/bin/bash
#SBATCH -J load-final-train
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
# SLURM SCRIPT FOR NEW LOAD MODEL — FINAL TRAINING
#
# Uses train_load.py (no differencing, 8-channel input with time features).
# Trains for 200 epochs with early stopping using the best hyperparameters
# that have been hardcoded into train_load.py's build_model() after tuning.
#
# Pre-requisite:
#   The best hyperparameters from the tuner run (03_train_load_tuner.slurm)
#   MUST be hardcoded into build_model() in train_load.py before running this.
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
DATA_DIR="/scratch/$USER/ts-train/data/for04slurmfileLOADMODEL"
OUTPUT_DIR="/scratch/$USER/ts-train/outputs/NEW_LOAD_MODEL/final_model_run"
LOGS_DIR="/scratch/$USER/ts-train/logs"
VENV_PYTHON="/scratch/$USER/ts-train/env/tf_venv/bin/python"

mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "NEW LOAD MODEL — Final Training (Best Hyperparameters)"
echo "  - No differencing"
echo "  - 8-channel input (load + 7 time features)"
echo "  - Seq2Seq many-to-many (96 → 96)"
echo "  - 200 epochs with early stopping"
echo "======================================================"
echo "Results will be saved in: $OUTPUT_DIR"
echo "======================================================"

srun "$VENV_PYTHON" $HOME/projects/ts-train/NEW_LOAD_MODEL/train_load.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --logs_dir "$LOGS_DIR" \
  --dataset_kind electric \
  --epochs 200 \
  --batch_size 32

echo "======================================================"
echo "Final training completed!"
echo "Check $OUTPUT_DIR for the final model and performance metrics."
echo "======================================================"
