<div align="center">

---

## 简介

在基于可验证奖励的强化学习中，稀疏的结果奖励通常被广播到整条交互轨迹，难以辨别哪些中间决策真正影响了最终结果。分支采样可以比较共享前缀下的不同续写，但仍需回答两个问题：**有限的分支预算该投向哪里**，以及**分支结果如何转化为准确的局部信用**。

**CBPO** 将这两个问题明确解耦：

1. **全响应候选发现 + 双重衰减预算分配**
   沿完整响应扫描局部熵，把候选分支位置从工具调用边界扩展到一般推理过程；再以路径级和节点级双重衰减平衡分支质量与探索覆盖，避免预算集中在少数轨迹或相邻节点。
2. **对比分支价值（CBV）信用分配**
   将共享**精确 token 前缀**的父轨迹与分支轨迹组成精确前缀组，依据组内结果差异计算对比分支价值（Contrastive Branch Value, CBV），并据此调节分支后缀的优势。

该方法**仅依赖结果奖励**，无需额外的过程标注。

## 主要结果

在 Qwen3-1.7B 与 Qwen3-4B 上，覆盖五个数学工具推理基准与五个知识密集型搜索基准：

| 模型规模   | 数学宏平均 Pass@1 |       搜索平均 |
| ---------- | ----------------: | -------------: |
| Qwen3-1.7B |   **66.0%** | **53.2** |
| Qwen3-4B   |   **69.3%** | **57.5** |

消融实验表明：移除双重衰减时两种规模的宏平均均下降 2.2 个百分点；移除 CBV 时分别下降 3.0 和 2.9 个百分点 —— 覆盖均衡的局部探索与结果敏感的信用分配具有互补作用。

## 仓库结构

```
CBPO/
├── training/                  # 强化学习训练
│   ├── verl_cbpo/             # CBPO 算法实现（基于 verl，见下方致谢）
│   │   └── verl/trainer/ppo/
│   │       ├── tao_branching.py   # 熵候选扫描与双重衰减预算分配
│   │       ├── core_algos.py      # CBV 计算与 token 级优势整形
│   │       └── ray_trainer.py     # 分支 rollout 编排与训练主循环
│   ├── scripts/               # 训练启动脚本与配置
│   ├── rl_datasets/           # RL 训练/验证数据
│   └── merge_ckpt/            # 检查点转换为 HuggingFace 格式
├── LLaMA-Factory/             # SFT 冷启动（基于 LLaMA-Factory，见下方致谢）
│   └── cbpo_sft/              # SFT 数据处理与训练入口
└── evaluation/                # 评测与分析
    ├── data/                  # 14 个基准数据集
    └── src/                   # 推理与判分逻辑
```

## 快速开始

### 1. 环境准备

```bash
# RL 训练环境
pip install -r training/requirements.txt

# 评测环境
pip install -r evaluation/requirement.txt
```

### 2. 配置密钥（必须）

本仓库不含任何密钥。`WANDB_API_KEY` 通过环境变量注入：

```bash
export WANDB_API_KEY="<your_wandb_key>"    # 训练日志（可选）
```

搜索工具的密钥目前需直接填入配置文件（搜索类任务必需），把
`training/scripts/config/ppo_trainer_dr.yaml` 中 `search.params` 下的
`<your_api_key>` 与 `<your_zone>` 替换为自己的 Brightdata 凭据。

### 3. SFT 冷启动（可选）

```bash
cd LLaMA-Factory
bash cbpo_sft/sft_train.sh
```

### 4. CBPO 强化学习训练

```bash
bash training/scripts/CBPO_1.7B_Reasoning_1node.sh
```

### 5. 评测

```bash
# 检查点转为 HuggingFace 格式
bash training/merge_ckpt/convert_checkpoint_from_verl_to_hf_qwen3.sh

# 启动推理服务并评测
bash evaluation/vllm_scripts/vllm_launch_reasoning_model_cuda4-7.sh
bash evaluation/infer_local_sds.sh
bash evaluation/evaluate.sh
```

## 致谢与许可

本项目的强化学习训练框架基于 [**verl**](https://github.com/volcengine/verl)（Apache-2.0，Copyright 2024 Bytedance Ltd.）构建，SFT 流程基于 [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory)（Apache-2.0）。相应的 `LICENSE` 与 `Notice.txt` 已随代码保留于各自目录下。

SFT 冷启动阶段复用了 [**ARPO**](https://github.com/dongguanting/ARPO) 项目公开的 [ARPO-SFT-54K](https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K) 数据集（MIT）。

CBPO 的算法贡献主要位于 `training/verl_cbpo/verl/trainer/ppo/` 下的 `tao_branching.py`、`core_algos.py` 与 `ray_trainer.py`，以及 rollout 侧的全轨迹熵采集与精确前缀恢复逻辑。

评测中作为对比基线复现的既有方法（如 ARPO、GIGPO、Tree-GRPO 等）在代码中均保留其原始名称与文献引用。
