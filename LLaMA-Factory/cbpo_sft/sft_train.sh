#!/bin/bash

#================== Basic Configuration ==================#
export CUDA_VISIBLE_DEVICES=4,5  # List of visible GPUs
export PYTHONPATH="$(cd .. && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# Weights & Biases
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="CBPO-SFT"
export WANDB_NAME="Qwen3-1.7B-CBPO-SFT"

#================== Training Parameter Configuration ==================#
# Distributed training configuration
NNODES=1                 # Total number of nodes
NODE_RANK=0              # Rank of the current node
PROC_PER_NODE=2          # Number of processes per node
MASTER_ADDR="127.0.0.1"  # Address of the master node
MASTER_PORT=29500        # Port of the master node

# Output directory
OUTPUT_DIR="/path/to/workspace/models/Qwen3-1.7B-CBPO-SFT"
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Path to the training script
TRAIN_SCRIPT="../src/llamafactory/launcher.py"

# Path to the training argument configuration file
TRAIN_ARGS="yaml/qwen.yaml"

# Command to launch training
torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} \
         ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log

# Optionally enable logging redirection
# exec >> ${OUTPUT_DIR}/training.log 2>&1
