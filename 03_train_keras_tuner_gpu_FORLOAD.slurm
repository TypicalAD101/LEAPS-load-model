#!/bin/bash
#!/bin/bash
#SBATCH -J ts-train-a100-2x80
#SBATCH -N 1
#SBATCH -G a100:2

# changing sbatch -c 16 to sbatch --cpus-per-task=16 to request more cpu explicitly

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

module purge
module load mamba/latest
source activate /scratch/$USER/ts-train/env/tf_venv

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

# TensorFlow optimizations for multi-GPU
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_USE_CUDNN=1

DATA_DIR=/scratch/$USER/ts-train/data/LOADMODEL
OUTPUT_DIR=/scratch/$USER/ts-train/outputs/LOADMODEL
LOGS_DIR=/scratch/$USER/ts-train/logs/LOADMODEL
VENV_PYTHON="/scratch/$USER/ts-train/env/tf_venv/bin/python"


# echo "Starting FAST sanity run (no tuner)..."
echo "Single fixed model build for pipeline validation."

echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
scontrol show job $SLURM_JOB_ID | egrep -i "mem|tres|cpu|gres|qos|partition"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus=2 \
  "$VENV_PYTHON" $HOME/projects/ts-train/train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_kind electric \
  --logs_dir "$LOGS_DIR" \
  --use_tuner \
  --max_trials 5 \
  --epochs_per_trial 5 \
  --batch_size 128

echo "Hyperparameter optimization completed!"
echo "Check the output directory for the best model and hyperparameters." 