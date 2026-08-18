#!/bin/bash

use_qwen3=true

# Activate the Conda environment
CONDA_BASE="/path/to/miniconda3"
CONDA_ENV_NAME="cbpo_vllm"
source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV_NAME}"

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path
MODEL_PATH="/path/to/workspace/models/Qwen3-8B-CBPO-DeepSearch"
MODEL_NAME="Qwen3-8B-CBPO-DeepSearch"

# Launch reasoning instance - using GPU 6 and 7
echo "Starting reasoning instance on GPU 6,7"
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8002 > logs/reasoning_model.log 2>&1 &
REASONING_PID=$!
echo "Reasoning instance deployed on port 8002 using GPU 6,7"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $REASONING_PID" SIGTERM
wait $REASONING_PID
