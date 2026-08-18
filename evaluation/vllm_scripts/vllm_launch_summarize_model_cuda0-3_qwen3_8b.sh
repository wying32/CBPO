#!/bin/bash

# Activate the Conda environment
CONDA_BASE="/path/to/miniconda3"
CONDA_ENV_NAME="cbpo_vllm"
source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV_NAME}"

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path
MODEL_PATH="/path/to/workspace/models/Qwen3-8B"
MODEL_NAME="Qwen3-8B"

# Launch summarization instance - using GPU 4 and 5
echo "Starting summarization instance on GPU 4,5"
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8004 > logs/summarize_model.log 2>&1 &
SUMMARIZE_PID=$!
echo "Summarization instance deployed on port 8004 using GPU 4,5"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $SUMMARIZE_PID" SIGTERM
wait $SUMMARIZE_PID
