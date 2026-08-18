SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CBPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CBPO_ROOT"
echo "Switched to CBPO root directory: $CBPO_ROOT"


# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=INFO
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export CUDA_VISIBLE_DEVICES=4,5

# Set Python path
export PYTHONPATH="${CBPO_ROOT}/verl_cbpo:$PYTHONPATH"

# ============================ Basic Configuration ============================
# Unified algorithm entry. Supported values: grpo, cbpo, cbpo, reinforce_plus_plus, dapo, gspo, gigpo.
# Example: RL_ALGORITHM=grpo bash scripts/CBPO_1.7B_Reasoning_1node.sh
RL_ALGORITHM="${RL_ALGORITHM:-cbpo}"
# Experiment name and project. Separate default output directories prevent
# checkpoints and logs from different algorithms from being mixed together.
PROJECT_NAME="${PROJECT_NAME:-CBPO-RL}" # Modify experiment group
EXPERIMENT_NAME="${EXPERIMENT_NAME:-CBPO-1.7B-${RL_ALGORITHM}}" # Modify experiment name
# Real candidate/selection events used by the CBPO paper analysis. The export
# is active only when TEB is enabled; override with an absolute server path.
CBPO_BRANCH_ANALYSIS_DIR="${CBPO_BRANCH_ANALYSIS_DIR:-${CBPO_ROOT}/outputs/cbpo_branch_analysis/${EXPERIMENT_NAME}}"

# Configuration file path
CONFIG_PATH="${CBPO_ROOT}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

# Distributed training settings
NNODES=1                            
N_GPUS_PER_NODE=2

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                 # Prompt field name
TRAIN_BATCH_SIZE=64                 # Training batch size
PPO_MINI_BATCH_SIZE=64              # Fewer optimizer updates per rollout batch
MAX_PROMPT_LENGTH=1536              # Maximum prompt length
MAX_RESPONSE_LENGTH=8192            # Maximum response length（对齐 SFT cutoff_len=8192，消除截断压力）

# Data file paths
TRAIN_FILES="${CBPO_ROOT}/rl_datasets/train_10k_math_only_fixed.parquet" # sanitized math-only subset
VALID_FILES="${CBPO_ROOT}/rl_datasets/valid_math.parquet"

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/path/to/workspace/models/Qwen3-1.7B-CBPO-SFT}" # external model path

# ============================ LoRA Configuration ============================
# LoRA fine-tuning. Set LORA_RANK=0 to fall back to full fine-tuning.
LORA_RANK=0                        # 0 = full fine-tuning (recommended for 1.7B RL)
LORA_ALPHA=32                      # unused when LORA_RANK=0
ACTOR_LR=1e-6                      # Full-FT RL LR; raise back to ~1e-5 only for LoRA
ACTOR_KL_LOSS_COEF=0.001           # Keep the policy close to the reference model
LR_WARMUP_STEPS_RATIO=0.03         # Warm up over the first 3% of training steps

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool"       # Synchronous mode with tool support
ROLLOUT_N=16                         # Number of responses generated per sample
CBPO_INITIAL_ROLLOUTS=8              # Paper setting: N=8 initial + M-N=8 partial rollouts
CBPO_INITIAL_ROLLOUTS=6              # CBPO first-stage rollout number; branches fill the remainder
BEAM_SIZE=1                        # 保持1：关闭 legacy 在线 beam 分支，TEB 两段式接管分支
ENABLE_MULTI_TURN=False

# ============================ BC-CBPO Configuration =========================
# ---- CBV 优势修正（分支后训练信号的放大/缩小）----
# 公式: A_final = A + η·sign(A)·clip(Z_cbv, -|A|/φ, +|A|/φ)
BC_CBPO_ALPHA=0.2                   # η: CBV 修正强度（注意: 不是下面的分支概率 α）
BC_CBPO_PHI=2.0                     # φ: 修正量上限 = |A|/φ，防止盖过基础优势
BC_CBPO_CBV_MODE="std"              # CBV 计算方式: std=组内奖励标准差（定稿方案）
BC_CBPO_REWARD_STD_THRESHOLD=1.0e-6 # batch 内 CBV 差异小于此值时全部置 0（数值兜底）
REINFORCE_PP_GAMMA=1.0              # Discount factor used by REINFORCE++ token returns
DAPO_CLIP_RATIO_LOW=0.2             # DAPO Clip-Higher lower bound
DAPO_CLIP_RATIO_HIGH=0.28           # DAPO Clip-Higher upper bound
DAPO_CLIP_RATIO_C=10.0              # Dual-clip lower bound
DAPO_FILTER_GROUPS_METRIC="acc"      # Remove all-equal reward groups during dynamic sampling
DAPO_MAX_NUM_GEN_BATCHES=10         # Avoid unbounded resampling when rewards have no variance
DAPO_OVERLONG_BUFFER_LEN=$((MAX_RESPONSE_LENGTH / 5))
DAPO_OVERLONG_PENALTY_FACTOR=1.0
GSPO_CLIP_RATIO=0.0003              # Sequence-level clipping recommended for GSPO
GIGPO_GAMMA=0.95                    # Paper setting for discounted step returns
GIGPO_OMEGA=1.0                     # Weight of the step-relative advantage
GIGPO_NORMALIZATION="std"           # std | none (Fnorm in the paper)
GIGPO_MIN_ANCHOR_GROUP_SIZE=2
GIGPO_STATE_MATCH_MODE="auto"       # Similarity matching for search states, exact for Python
GIGPO_STATE_SIMILARITY_THRESHOLD=0.9
# ---- Paper CBPO adaptive rollout ----
# P = alpha + beta * delta_H; branch Z continuations when P > tau.
CBPO_BASE_PROBABILITY=0.5            # alpha in the paper
CBPO_ENTROPY_SCALE=0.2               # beta / entropy weight in the paper
CBPO_BRANCH_THRESHOLD=0.5            # tau in the paper
CBPO_BRANCHES_PER_EVENT=1            # Z; configurable because the paper does not fix one universal value
# ---- 熵平衡分支（在哪里分支）----
# 公式: P = clip(α + γ·ΔH, 0, 1) / [(1+L_path)^ρp(1+l_node)^ρn]
EB_BASE_PROBABILITY=0.2             # α: 基础分支概率
EB_ENTROPY_SCALE=2.0                # γ: 熵变化 ΔH 的放大系数（熵已除 log|V| 归一化，ΔH 量级仅 ~0.05，
                                    #    γ=0.5 时需 ΔH>0.2 永达不到→全程零分支；γ=2 时 ΔH>0.05 即可过门槛）
EB_BRANCH_THRESHOLD=0.25            # τ: 分支门槛，必须 > α；当前对应首次分支 ΔH>0.025
EB_ENTROPY_WINDOW=20                # token 熵滑窗宽度
EB_ENTROPY_TOP_K=10                 # vLLM top-k logprobs，用于近似 token entropy
EB_MAX_WINDOW_CANDIDATES=3          # 每个 generation chunk 最多保留的分离熵峰数
EB_MAX_NODES_PER_PATH=3             # 每条 initial trajectory 最多选择的分支节点数
EB_MIN_NODE_DISTANCE=64             # 同一路径候选节点的最小 token 间距
EB_PATH_DECAY_RHO=0.2               # ρp: 较弱路径衰减，允许同一轨迹多个节点入选
EB_NODE_DECAY_RHO=0.2               # ρn: 较弱节点衰减，允许关键节点多个 continuation
EB_MAX_BRANCHES_PER_NODE=3          # 单节点 continuation 数硬上限
EB_MAX_BRANCHES_PER_PATH=4          # 单条 initial trajectory 总分支数硬上限

# ============================ Algorithm Selection ============================
# Keep all algorithm-specific switches here. To add another algorithm later,
# add one case without rewriting the common training command below.
TRAINER_MODULE="verl.trainer.main_ppo"
REWARD_MANAGER="naive"
USE_KL_IN_REWARD=False
KL_CTRL_COEF=0.0
ACTOR_USE_KL_LOSS=True
SELECTED_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF}
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
CLIP_RATIO_C=3.0
POLICY_LOSS_TYPE="ppo"
LOSS_AGG_MODE="token-mean"
FILTER_GROUPS_ENABLED=False
FILTER_GROUPS_METRIC="acc"
MAX_NUM_GEN_BATCHES=0
GEN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
OVERLONG_BUFFER_ENABLED=False
OVERLONG_BUFFER_LEN=0
OVERLONG_PENALTY_FACTOR=0.0
GIGPO_ENABLED=False
TEB_SELECTION_MODE="balanced"
TEB_ADVANTAGE_MODE="local"
TEB_EVENT_MODE="grid"
TEB_BRANCHES_PER_EVENT=1
SELECTED_EB_BASE_PROBABILITY=${EB_BASE_PROBABILITY}
SELECTED_EB_ENTROPY_SCALE=${EB_ENTROPY_SCALE}
SELECTED_EB_BRANCH_THRESHOLD=${EB_BRANCH_THRESHOLD}

case "${RL_ALGORITHM}" in
    grpo|GRPO)
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=1.0
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        TEB_ENABLED=False
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${ROLLOUT_N}
        ;;
    cbpo|CBPO)
        # Agentic Reinforced Policy Optimization (paper): post-tool entropy
        # adaptive partial rollouts + soft GRPO advantage attribution.
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=1.0
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        ACTOR_USE_KL_LOSS=False
        SELECTED_KL_LOSS_COEF=0.0
        TEB_ENABLED=True
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${CBPO_INITIAL_ROLLOUTS}
        TEB_SELECTION_MODE="cbpo"
        TEB_ADVANTAGE_MODE="soft_grpo"
        TEB_EVENT_MODE="post_tool"
        TEB_BRANCHES_PER_EVENT=${CBPO_BRANCHES_PER_EVENT}
        SELECTED_EB_BASE_PROBABILITY=${CBPO_BASE_PROBABILITY}
        SELECTED_EB_ENTROPY_SCALE=${CBPO_ENTROPY_SCALE}
        SELECTED_EB_BRANCH_THRESHOLD=${CBPO_BRANCH_THRESHOLD}
        ;;
    cbpo|CBPO)
        # CBPO is currently implemented by TEB branching + BC-CBPO/CBV credit.
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=1.0
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        TEB_ENABLED=True
        BC_CBPO_ENABLED=True
        INITIAL_ROLLOUTS=${CBPO_INITIAL_ROLLOUTS}
        ;;
    reinforce_plus_plus|REINFORCE_PLUS_PLUS|reinforce++)
        ADV_ESTIMATOR="reinforce_plus_plus"
        ALGORITHM_GAMMA=${REINFORCE_PP_GAMMA}
        ACTOR_KL_LOSS_TYPE="mse"
        TEB_ENABLED=False
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${ROLLOUT_N}
        ;;
    dapo|DAPO)
        # Full DAPO recipe: GRPO advantages + Clip-Higher + dynamic sampling
        # + token-level loss + soft overlong punishment, without KL loss.
        TRAINER_MODULE="recipe.dapo.main_dapo"
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=1.0
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        ACTOR_USE_KL_LOSS=False
        SELECTED_KL_LOSS_COEF=0.0
        CLIP_RATIO_LOW=${DAPO_CLIP_RATIO_LOW}
        CLIP_RATIO_HIGH=${DAPO_CLIP_RATIO_HIGH}
        CLIP_RATIO_C=${DAPO_CLIP_RATIO_C}
        FILTER_GROUPS_ENABLED=True
        FILTER_GROUPS_METRIC=${DAPO_FILTER_GROUPS_METRIC}
        MAX_NUM_GEN_BATCHES=${DAPO_MAX_NUM_GEN_BATCHES}
        OVERLONG_BUFFER_ENABLED=True
        OVERLONG_BUFFER_LEN=${DAPO_OVERLONG_BUFFER_LEN}
        OVERLONG_PENALTY_FACTOR=${DAPO_OVERLONG_PENALTY_FACTOR}
        REWARD_MANAGER="dapo"
        TEB_ENABLED=False
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${ROLLOUT_N}
        ;;
    gspo|GSPO)
        # GSPO keeps GRPO group advantages but applies a geometric-mean
        # sequence importance ratio and sequence-level clipping in the actor.
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=1.0
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        POLICY_LOSS_TYPE="gspo"
        CLIP_RATIO_LOW=${GSPO_CLIP_RATIO}
        CLIP_RATIO_HIGH=${GSPO_CLIP_RATIO}
        TEB_ENABLED=False
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${ROLLOUT_N}
        ;;
    gigpo|GIGPO|GiGPO)
        # GiGPO = episode-level GRPO advantage + anchor-state step advantage.
        ADV_ESTIMATOR="grpo"
        ALGORITHM_GAMMA=${GIGPO_GAMMA}
        ACTOR_KL_LOSS_TYPE="low_var_kl"
        GIGPO_ENABLED=True
        TEB_ENABLED=False
        BC_CBPO_ENABLED=False
        INITIAL_ROLLOUTS=${ROLLOUT_N}
        ;;
    *)
        echo "Unsupported RL_ALGORITHM='${RL_ALGORITHM}'. Supported values: grpo, cbpo, cbpo, reinforce_plus_plus, dapo, gspo, gigpo." >&2
        exit 2
        ;;
esac

echo "RL algorithm: ${RL_ALGORITHM} (trainer=${TRAINER_MODULE}, adv_estimator=${ADV_ESTIMATOR}, policy_loss=${POLICY_LOSS_TYPE}, gamma=${ALGORITHM_GAMMA}, kl_loss=${ACTOR_USE_KL_LOSS}/${ACTOR_KL_LOSS_TYPE}, TEB=${TEB_ENABLED}/${TEB_SELECTION_MODE}/${TEB_ADVANTAGE_MODE}, CBV=${BC_CBPO_ENABLED}, GiGPO=${GIGPO_ENABLED}, dynamic_filter=${FILTER_GROUPS_ENABLED}, initial_rollouts=${INITIAL_ROLLOUTS})"

# ============================ Reward Model Configuration ==========================
# Reward model settings
CUSTOM_REWARD_FUNCTION_PATH="${CBPO_ROOT}/verl_cbpo/verl/utils/reward_score/deep_research_math.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"  # Total training epochs
# null means use epochs; set TOTAL_TRAINING_STEPS=20 for a short validation run.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-10}"       # Save frequency
TEST_FREQ="${TEST_FREQ:-10}"       # Test frequency
MAX_ACTOR_CKPT_TO_KEEP=1           # Rotate checkpoints and retain only the latest one
MAX_CRITIC_CKPT_TO_KEEP=1          # Kept consistent if a critic is enabled later

# ============================ Path Configuration ============================
# Save path
SAVE_PATH="/path/to/workspace/CBPO/outputs/models/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
# WandB settings
WANDB_API_KEY="${WANDB_API_KEY:-}" # set this in the environment if wandb logging is needed
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training ============================
python3 -m ${TRAINER_MODULE} \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.gamma=${ALGORITHM_GAMMA} \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
    algorithm.kl_ctrl.kl_coef=${KL_CTRL_COEF} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${SELECTED_KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${ACTOR_KL_LOSS_TYPE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.policy_loss_type=${POLICY_LOSS_TYPE} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.teb.enabled=${TEB_ENABLED} \
    actor_rollout_ref.rollout.teb.selection_mode=${TEB_SELECTION_MODE} \
    actor_rollout_ref.rollout.teb.advantage_mode=${TEB_ADVANTAGE_MODE} \
    actor_rollout_ref.rollout.teb.event_mode=${TEB_EVENT_MODE} \
    actor_rollout_ref.rollout.teb.branches_per_event=${TEB_BRANCHES_PER_EVENT} \
    algorithm.bc_cbpo.enabled=${BC_CBPO_ENABLED} \
    actor_rollout_ref.rollout.teb.base_probability=${SELECTED_EB_BASE_PROBABILITY} \
    actor_rollout_ref.rollout.teb.entropy_scale=${SELECTED_EB_ENTROPY_SCALE} \
    actor_rollout_ref.rollout.teb.branch_threshold=${SELECTED_EB_BRANCH_THRESHOLD} \
    actor_rollout_ref.rollout.teb.entropy_window=${EB_ENTROPY_WINDOW} \
    actor_rollout_ref.rollout.teb.entropy_top_k=${EB_ENTROPY_TOP_K} \
    actor_rollout_ref.rollout.teb.max_window_candidates_per_chunk=${EB_MAX_WINDOW_CANDIDATES} \
    actor_rollout_ref.rollout.teb.max_branch_nodes_per_path=${EB_MAX_NODES_PER_PATH} \
    actor_rollout_ref.rollout.teb.min_branch_node_distance=${EB_MIN_NODE_DISTANCE} \
    actor_rollout_ref.rollout.teb.path_power_decay_rho=${EB_PATH_DECAY_RHO} \
    actor_rollout_ref.rollout.teb.node_power_decay_rho=${EB_NODE_DECAY_RHO} \
    actor_rollout_ref.rollout.teb.max_branches_per_node=${EB_MAX_BRANCHES_PER_NODE} \
    actor_rollout_ref.rollout.teb.max_branches_per_path=${EB_MAX_BRANCHES_PER_PATH} \
    actor_rollout_ref.rollout.teb.analysis_export_dir=${CBPO_BRANCH_ANALYSIS_DIR} \
    algorithm.bc_cbpo.alpha=${BC_CBPO_ALPHA} \
    algorithm.bc_cbpo.phi=${BC_CBPO_PHI} \
    algorithm.bc_cbpo.cbv_mode=${BC_CBPO_CBV_MODE} \
    algorithm.bc_cbpo.reward_std_threshold=${BC_CBPO_REWARD_STD_THRESHOLD} \
    algorithm.gigpo.enabled=${GIGPO_ENABLED} \
    algorithm.gigpo.gamma=${GIGPO_GAMMA} \
    algorithm.gigpo.omega=${GIGPO_OMEGA} \
    algorithm.gigpo.normalization=${GIGPO_NORMALIZATION} \
    algorithm.gigpo.min_anchor_group_size=${GIGPO_MIN_ANCHOR_GROUP_SIZE} \
    algorithm.gigpo.state_match_mode=${GIGPO_STATE_MATCH_MODE} \
    algorithm.gigpo.similarity_threshold=${GIGPO_STATE_SIMILARITY_THRESHOLD} \
    algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLED} \
    algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.tools.max_workers=8 \
    actor_rollout_ref.rollout.tools.max_rollout_rounds=32 \
    +actor_rollout_ref.rollout.tools.tool_instances.python.params.result_length=800 \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.overlong_buffer.enable=${OVERLONG_BUFFER_ENABLED} \
    reward_model.overlong_buffer.len=${OVERLONG_BUFFER_LEN} \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_PENALTY_FACTOR} \
    reward_model.overlong_buffer.log=False \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log
