#!/bin/bash

# Activate the Conda environment
CONDA_BASE="/path/to/miniconda3"
CONDA_ENV_NAME="cbpo_eval"
source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV_NAME}"

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Set Python environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# datasets
data_names=(
    "aime24"
    "aime25"
    "math500"
    "gsm8k"
    # "math"
    # "webwalker"
    # "hotpotqa"
    # "2wiki"
    # "bamboogle"
    # "musique"
    # "hle"
    # "gaia"
)
DATASET_NAME=$(echo "${data_names[@]}" | tr '\n' ' ')

# Reasoning model endpoints
infer_endpoints=(
    "http://localhost:8002/v1"
)  
ENDPOINTS=$(echo "${infer_endpoints[@]}" | tr '\n' ' ')

SAMPLE_TIMEOUT=900  # Timeout for one sample

EXP_NAME="qwen3_1.7b_base_tir"  # TIR prompt (math) + Python tool
MODEL_PATH="/path/to/workspace/models/Qwen3-1.7B"
OUTPUT_PATH="/path/to/workspace/training/outputs/eval/${EXP_NAME}"
DATA_PATH="data"                       
TURNS="1"  # Inference turns

with_tools=true
if [ "$with_tools" = true ]; then
    # Math evaluation uses Python only; search is neither prompted nor executed.
    PROMPT_TYPE="math"                 # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=5                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=0                 # Disable search for math problems
else
    PROMPT_TYPE="base"                 # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=0                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=0                 # Max search tool invocation times
fi

# Inference Mode
INFER_MODE=completion   # `completion_sds` (with web browser), 'default' (without web browser)


# VLLM config
echo "Inference endpoints: $ENDPOINTS"
API_KEYS=""                     # API keys list, corresponds to endpoints; empty means default "EMPTY"
DEFAULT_MODEL="Qwen3-1.7B"  # Default model name, must match vllm --served-model-name

# Generation parameters
TEMPERATURE=0.0                      # Temperature parameter (0.0 for deterministic eval)
MAX_TOKENS=2048                     # Max tokens per generation chunk (no-tool CoT needs headroom)
TOP_P=1.0                           # Top-p (disabled when temperature=0.0)
TOP_K=-1                            # Top-k (disabled when temperature=0.0)
MIN_P=0.0                           # Minimum probability threshold
REPETITION_PENALTY=1.0             # Repetition penalty factor (1.0 = disabled)
INCLUDE_STOP_STR=true              # Whether to include stop string in output

# Inference configuration
BATCH_SIZE=8                       # Batch size
MAX_CONCURRENT=32                  # Max concurrent requests
COUNTS=500                        # Number of samples to process

# Tool configurations
CONDA_PATH="/path/to/miniconda3/"   # Conda installation path
CONDA_ENV="cbpo_eval"                                # Conda environment name
PYTHON_MAX_CONCURRENT=32                        # Max concurrent Python executor
BING_API_KEY="${BING_API_KEY:-}"                 # Bing Search API key (export before running)
BING_ZONE="serp_wy"                        # Bing search zone
SEARCH_MAX_RESULTS=10                            # Max number of search results
SEARCH_RESULT_LENGTH=1000                        # Max length per search result
BING_REQUESTS_PER_SECOND=32.0                    # Max Bing search requests per second
BING_MAX_RETRIES=3                              # Max Bing search retries
BING_RETRY_DELAY=1.0                            # Bing search retry delay (seconds)

# Simple deep search config
SUMM_MODEL_URLS="http://localhost:8004/v1"
SUMM_MODEL_NAME="Qwen3-8B"
SUMM_MODEL_PATH="/path/to/workspace/models/Qwen3-8B"
SEARCH_CACHE_FILE="search_cache.db"
URL_CACHE_FILE="search_url_cache.db"

# Build command line arguments
CMD="python -u infer.py"
CMD+=" --endpoints $ENDPOINTS"
CMD+=" --model_path $MODEL_PATH"
CMD+=" --default_model $DEFAULT_MODEL"
CMD+=" --infer_mode $INFER_MODE"

# If API_KEYS is not empty, add the parameter
if [ ! -z "$API_KEYS" ]; then
    CMD+=" --api_keys $API_KEYS"
fi

# Add generation parameters
CMD+=" --temperature $TEMPERATURE"
CMD+=" --max_tokens $MAX_TOKENS"
CMD+=" --top_p $TOP_P"
CMD+=" --top_k $TOP_K"
CMD+=" --min_p $MIN_P"
CMD+=" --repetition_penalty $REPETITION_PENALTY"
CMD+=" --include_stop_str_in_output $INCLUDE_STOP_STR"

# Add inference config parameters
CMD+=" --max_concurrent_requests $MAX_CONCURRENT"
CMD+=" --dataset_name $DATASET_NAME"
CMD+=" --output_path $OUTPUT_PATH"
CMD+=" --prompt_type $PROMPT_TYPE"
CMD+=" --counts $COUNTS"
CMD+=" --max_python_times $MAX_PYTHON_TIMES"
CMD+=" --max_search_times $MAX_SEARCH_TIMES"
CMD+=" --sample_timeout $SAMPLE_TIMEOUT"

# If DATA_PATH is not empty, add the parameter
if [ ! -z "$DATA_PATH" ]; then
    CMD+=" --data_path $DATA_PATH"
fi

# Add tool config parameters
CMD+=" --conda_path $CONDA_PATH"
CMD+=" --conda_env $CONDA_ENV"
CMD+=" --python_max_concurrent $PYTHON_MAX_CONCURRENT"
CMD+=" --bing_api_key $BING_API_KEY"
CMD+=" --bing_zone $BING_ZONE"
CMD+=" --search_max_results $SEARCH_MAX_RESULTS"
CMD+=" --search_result_length $SEARCH_RESULT_LENGTH"
CMD+=" --bing_requests_per_second $BING_REQUESTS_PER_SECOND"
CMD+=" --bing_max_retries $BING_MAX_RETRIES"
CMD+=" --bing_retry_delay $BING_RETRY_DELAY"

# Additional parameters for search tool
CMD+=" --summ_model_urls $SUMM_MODEL_URLS"
CMD+=" --summ_model_name $SUMM_MODEL_NAME"
CMD+=" --summ_model_path $SUMM_MODEL_PATH"
CMD+=" --search_cache_file $SEARCH_CACHE_FILE"
CMD+=" --url_cache_file $URL_CACHE_FILE"

CMD+=" --turns $TURNS"

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

echo $CMD

# Execute command
eval $CMD | tee logs/infer.log 2>&1
