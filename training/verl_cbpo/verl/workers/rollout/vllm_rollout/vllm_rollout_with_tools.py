# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Counter

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
import math
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        
        tool_class = getattr(module, class_name)
        
        tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        
        tool_instance = tool_class(**tool_params)
        
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        # 从配置中获取beam search相关参数
        self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])
        self.beam_size = self.config.get("beam_size", 1)
        self.branch_probability = self.config.get("branch_probability", 0.5)
        self.entropy_weight = self.config.get("entropy_weight", 0.5)
        # Backward-compatible controls for the legacy one-pass online beamer.
        # The proposed method uses the entropy-balanced TEB configuration below.
        self.branch_penalty_enabled = self.config.get("branch_penalty_enabled", False)
        self.branch_penalty_slope = self.config.get("branch_penalty_slope", 0.0)
        self.branch_penalty_on_skip = self.config.get("branch_penalty_on_skip", False)
        self.teb_config = self.config.get("teb", OmegaConf.create({"enabled": False}))
        self.teb_enabled = bool(self.teb_config.get("enabled", False))
        self.teb_selection_mode = str(self.teb_config.get("selection_mode", "balanced"))
        self.teb_advantage_mode = str(self.teb_config.get("advantage_mode", "local"))
        self.teb_event_mode = str(self.teb_config.get("event_mode", "grid"))
        self.entropy_window = int(self.teb_config.get("entropy_window", 20))
        self.entropy_top_k = int(self.teb_config.get("entropy_top_k", 10))
        self.max_window_candidates_per_chunk = int(
            self.teb_config.get("max_window_candidates_per_chunk", 3)
        )
        self.min_branch_node_distance = int(self.teb_config.get("min_branch_node_distance", 64))
        if self.min_branch_node_distance < 1:
            raise ValueError("teb.min_branch_node_distance must be positive")
        
        # 从配置中获取工具设置
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # 获取工具通用配置
        self.tool_call_limit = tools_config.get("call_limit", 8)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)
        self.max_rollout_rounds = tools_config.get("max_rollout_rounds", 20)
        self.generation_chunk_tokens = tools_config.get("generation_chunk_tokens", 512)

        # 其他可能的工具通用配置
        self.tool_retry_count = tools_config.get("retry_count", 3)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)

        
        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = self.entropy_top_k

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str, allow_missing_start: bool = False) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            try:
                start_pos = text.rindex(start_tag, 0, end_pos)
            except ValueError:
                if allow_missing_start:
                    return text[:end_pos].strip()
                raise
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content):
        retry_count = 0
        start_time = time.time()
        is_failure_result = getattr(tool, "is_failure_result", None)
        
        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text:
                    execution_time = time.time() - start_time
                    success = not (callable(is_failure_result) and is_failure_result(result_text))
                    return {
                        "success": success,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text
                    }
                else:
                    logger.warning(f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}")
                    retry_count += 1
            except Exception as e:
                logger.error(f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                retry_count += 1
        
        execution_time = time.time() - start_time
        logger.warning(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": ""
        }

    def _format_tool_result(self, result_text: str) -> str:
        """将工具结果包装成与 SFT 训练一致的格式（qwen/qwen3 模板 format_observation）：
        关闭当前 assistant 轮 -> 以 <tool_response> user 轮投喂结果 -> 重开 assistant 轮。

        这样让 RL rollout 的结果投喂与 SFT 模型训练时看到的 token 流对齐，修复
        "工具结果后提前 EOS、不写 <answer>" 的问题。已验证(temp=1.0)：内联投喂
        </answer> 写出率约 19%，改为此 SFT 格式投喂 + 真实工具结果后约 87%，且提前
        EOS 归零。注意：注入串中含 <|im_end|>(=eos)，因此响应 attention mask 不能再
        用 get_response_mask(按首个 eos 截断)，须改为按真实长度构造（见 generate_sequences）。
        """
        return (
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
            f"<result>\n{result_text}\n</result>\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )

    def _calc_entropy(self, logprobs):
            if not logprobs:
                return 0.0
            p_list = [math.exp(l) for l in logprobs]
            entropy = -sum(p * l for p, l in zip(p_list, logprobs))
            return entropy

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)
        
        # 初始化工具调用统计信息
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
            "legacy_online_beam/created": 0,
            "legacy_online_beam/skipped_by_entropy": 0,
            "legacy_online_beam/skipped_by_penalty": 0,
            "legacy_online_beam/penalty_mean": 0.0,
        }
        penalty_values = []
        
        # 每个工具的统计信息
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        is_teb_branch = self.teb_enabled and "tao_prefix_token_ids" in prompts.non_tensor_batch
        
        # 更新采样参数设置
        beam_size = self.beam_size
        if not do_sample:
            kwargs.update({
                'best_of': 1, 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'temperature': 0, 'n': 1
            })
            beam_size = 1
        elif is_validate:
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1  # 验证模式下使用单个样本
            })
            beam_size = 1
        
        # fix oov error
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n

            prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids]

            # State for each sample in the batch
            # 为每个样本创建初始rollout，数量由initial_rollouts控制
            initial_rollouts = self.initial_rollouts
            initial_rollouts = min(initial_rollouts, num_samples)  # 但不超过num_samples
            if self.teb_enabled and do_sample and not is_validate:
                # TEB is a two-stage rollout.  The trainer asks for initial
                # trajectories first, then sends selected exact prefixes back
                # after outcome rewards are available.
                num_samples = 1 if is_teb_branch else initial_rollouts
                initial_rollouts = num_samples

            curr_inputs = []
            init_inputs = []
            result_masks = []
            call_counters = []
            consecutive_branch_counts = []
            entropy_events = []
            initial_entropy_dict = {}
            entropy_tool_call_dict = {}
            active_indices = []
            
            branch_prefixes = prompts.non_tensor_batch.get("tao_prefix_token_ids")
            branch_prefix_masks = prompts.non_tensor_batch.get("tao_prefix_model_mask")

            # 创建初始样本，或从 trainer 选中的 exact prefix 恢复分支状态。
            for i, ids in enumerate(prompt_token_ids_list):
                for _ in range(initial_rollouts):
                    prefix = [int(token) for token in branch_prefixes[i]] if is_teb_branch else []
                    prefix_mask = [int(value) for value in branch_prefix_masks[i]] if is_teb_branch else []
                    if len(prefix) != len(prefix_mask):
                        raise ValueError("TAO branch prefix ids and model mask must have equal length")
                    curr_inputs.append(ids.copy() + prefix)
                    init_inputs.append(ids.copy())
                    result_masks.append(prefix_mask)
                    prefix_text = self.tokenizer.decode(prefix) if prefix else ""
                    completed_tool_calls = sum(prefix_text.count(f"</{tag}>") for tag in self.tools)
                    call_counters.append(completed_tool_calls)
                    entropy_tool_call_dict[len(curr_inputs) - 1] = completed_tool_calls
                    consecutive_branch_counts.append(0)
                    entropy_events.append([])
                    active_indices.append(len(curr_inputs) - 1)
            
            # Track rollouts per original sample
            rollouts_per_sample = [initial_rollouts] * batch_size  # 每个样本初始有initial_rollouts个rollout
            # 初始时每个样本有多个索引
            sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}

            max_len = self.config.response_length
            rollout_round = 0

            while active_indices:
                rollout_round += 1
                if rollout_round > self.max_rollout_rounds:
                    logger.warning(
                        f"Rollout exceeded max_rollout_rounds={self.max_rollout_rounds}; "
                        f"forcing {len(active_indices)} active rollouts to finish."
                    )
                    for idx in active_indices:
                        curr_inputs[idx].append(eos_token_id)
                        result_masks[idx].append(1)
                    active_indices = []
                    break

                logger.info(
                    f"Rollout round {rollout_round}: active={len(active_indices)}, "
                    f"finished={len(curr_inputs) - len(active_indices)}, total={len(curr_inputs)}"
                )
                active_prompts = [curr_inputs[i] for i in active_indices]
                logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                logger.debug(f"active_indices: {active_indices}")
                logger.debug(f"active_prompts: {active_prompts}")

                # Update max_tokens for each active sample
                remaining_tokens = max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices)))
                round_max_tokens = (
                    min(remaining_tokens, int(self.generation_chunk_tokens))
                    if self.generation_chunk_tokens and self.generation_chunk_tokens > 0
                    else remaining_tokens
                )
                generate_start_time = time.time()
                logger.info(
                    f"Starting vLLM generate for rollout round {rollout_round}: "
                    f"active={len(active_indices)}, max_tokens={round_max_tokens}, remaining_max={remaining_tokens}"
                )
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=round_max_tokens,
                    detokenize=True,
                    logprobs = self.logprobs
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                logger.info(
                    f"Finished vLLM generate for rollout round {rollout_round} "
                    f"in {time.time() - generate_start_time:.2f}s"
                )
                # ========== Entropy Variation Monitoring ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                entropy_window = self.entropy_window
                current_entropy_dict = {}
                round_start_positions = {
                    out_idx: len(curr_inputs[out_idx]) - len(init_inputs[out_idx])
                    for out_idx in active_indices
                }
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    tokens = output.outputs[0].token_ids
                    # 对本轮全部 token 计算归一化熵（而非仅前 20 个），
                    # 使分支候选点可以落在轨迹任意位置，而不只是轮次边界。
                    token_entropies = []
                    for j in range(len(tokens)):
                        try:
                            logprob_info = output.outputs[0].logprobs[j]
                        except Exception:
                            # 保持 token_entropies 与 token 偏移对齐
                            token_entropies.append(0.0)
                            continue
                        token_list = list(logprob_info.values())
                        token_logprobs = [token.logprob for token in token_list]
                        if token_logprobs:
                            token_entropies.append(self._calc_entropy(token_logprobs) / entropy_norm_factor)
                        else:
                            token_entropies.append(0.0)
                    head_entropies = token_entropies[:entropy_window]
                    if head_entropies:
                        entropy = float(sum(head_entropies) / len(head_entropies))
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    if out_idx not in initial_entropy_dict:
                        initial_entropy_dict[out_idx] = entropy
                    root_entropy = initial_entropy_dict[out_idx]
                    if self.teb_event_mode == "post_tool":
                        # Paper ARPO measures the first k generated tokens
                        # immediately after each newly observed tool result.
                        completed_tool_calls = int(call_counters[out_idx])
                        measured_tool_calls = int(entropy_tool_call_dict.get(out_idx, 0))
                        if completed_tool_calls > measured_tool_calls and head_entropies:
                            # The k generated tokens estimate uncertainty, but
                            # continuations branch from the post-tool state
                            # before those tokens, as in ARPO Figure 4.
                            absolute_boundary = int(round_start_positions[out_idx])
                            entropy_events[out_idx].append(
                                {
                                    "boundary_response_idx": absolute_boundary,
                                    "entropy": entropy,
                                    "root_entropy": float(root_entropy),
                                    "entropy_delta": float(entropy - root_entropy),
                                    "tool_call_index": completed_tool_calls,
                                }
                            )
                            entropy_tool_call_dict[out_idx] = completed_tool_calls
                    elif len(token_entropies) > entropy_window:
                        # Balanced TEB keeps its trajectory-wide fixed grid.
                        prefix_sums = [0.0]
                        for value in token_entropies:
                            prefix_sums.append(prefix_sums[-1] + value)
                        window_candidates = []
                        round_start = int(round_start_positions[out_idx])
                        first_boundary = max(
                            self.min_branch_node_distance,
                            (
                                (round_start + self.min_branch_node_distance - 1)
                                // self.min_branch_node_distance
                            )
                            * self.min_branch_node_distance,
                        )
                        first_local_start = first_boundary - round_start
                        for start in range(
                            first_local_start,
                            len(token_entropies) - entropy_window + 1,
                            self.min_branch_node_distance,
                        ):
                            window_mean = (prefix_sums[start + entropy_window] - prefix_sums[start]) / entropy_window
                            window_candidates.append((float(window_mean), start))
                        for window_mean, start in sorted(window_candidates, reverse=True)[
                            : self.max_window_candidates_per_chunk
                        ]:
                            absolute_boundary = int(round_start + start)
                            entropy_events[out_idx].append(
                                {
                                    "boundary_response_idx": absolute_boundary,
                                    "entropy": window_mean,
                                    "root_entropy": float(root_entropy),
                                    "entropy_delta": float(window_mean - root_entropy),
                                }
                            )
                # ============================

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids
                    



                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason



                    is_tool_call = finish_reason == 'stop' and stop_reason in self.stop_sequences
                    
                    # Debug information
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.debug(f"  Sample {out_idx} output:")
                    logger.debug(f"  Token IDs: {generated_tokens}")
                    logger.debug(f"  Text: {decoded_text}")
                    logger.debug(f"  Finish reason: {finish_reason}")
                    logger.debug(f"  Stop reason: {stop_reason}")
                    logger.debug(f"  Is tool call: {is_tool_call}")
                    logger.debug(f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}")

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            content = self._extract_content(decoded_text, tag, allow_missing_start=True)
                            if not content:
                                full_text = self.tokenizer.decode(curr_inputs[out_idx])
                                content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 更新工具调用计数统计
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == 'length':
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)

                    elif finish_reason == 'stop':  # EOS
                        pass

                if any(tool_requests.values()):
                    logger.info(f"Processing tool requests: {sum(len(reqs) for reqs in tool_requests.values())} total requests")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        logger.debug(f"Processing {len(requests)} requests for tool '{tag}'")
                        tool = self.tools[tag]
                        for req in requests:
                            logger.debug(f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}")
                            future = self.executor.submit(self._execute_tool_with_retry, tool, req["content"])
                            futures[future] = {"index": req["index"], "tag": tag}

                    total_futures = len(futures)
                    completed_futures = 0
                    logger.debug(f"Submitted {total_futures} tool requests for execution")

                    done, pending = concurrent.futures.wait(
                        futures.keys(),
                        timeout=self.tool_timeout,
                        return_when=concurrent.futures.ALL_COMPLETED,
                    )

                    if pending:
                        logger.warning(
                            f"{len(pending)}/{total_futures} tool requests exceeded "
                            f"{self.tool_timeout}s; appending timeout results and continuing rollout."
                        )

                    for future in done:
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # 解析工具执行结果
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]
                            
                            # 更新统计信息
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                logger.info(f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}")
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = result_text or f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s")
                            
                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            
                            # 更新每个工具的时间统计
                            total_time_per_tool[tag] += execution_time
                            
                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s")
                            else:
                                logger.debug(f"Tool({tag}) result: {result_text}")
                                
                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                        
                        logger.debug(f"Tool completion progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)")
                        formatted_result = self._format_tool_result(result_text)
                        result_tokens = self.tokenizer.encode(formatted_result)
                        logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                    for future in pending:
                        future.cancel()
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        result_text = f"Error: Tool({tag}) timed out after {self.tool_timeout}s."
                        tool_metrics["tools/failed_calls"] += 1
                        tool_metrics["tools/total_execution_time"] += self.tool_timeout
                        tool_metrics["tools/max_execution_time"] = max(
                            tool_metrics["tools/max_execution_time"], self.tool_timeout
                        )
                        total_time_per_tool[tag] += self.tool_timeout

                        formatted_result = self._format_tool_result(result_text)
                        result_tokens = self.tokenizer.encode(formatted_result)
                        logger.warning(f"Tool({tag}) for sample {idx} timed out; result tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)
                
                # Apply beam search: split active samples into multiple branches
                new_indices = []
                new_inputs = []
                new_init_inputs = []
                new_result_masks = []
                new_call_counters = []
                new_sample_origins = []  # 记录每个新分支对应的原始样本
                new_branch_counts = []
                new_source_indices = []
                
                # Map from original sample index to its active rollouts
                active_by_sample = {}
                for idx in final_active_indices:
                    # Find which original sample this index belongs to
                    orig_sample = None
                    for sample_idx, indices in sample_to_indices.items():
                        if idx in indices:
                            orig_sample = sample_idx
                            break
                    
                    if orig_sample is not None:
                        if orig_sample not in active_by_sample:
                            active_by_sample[orig_sample] = []
                        active_by_sample[orig_sample].append(idx)
                
        
                for orig_sample, active_idxs in active_by_sample.items():
                    remaining_slots = num_samples - rollouts_per_sample[orig_sample]
                    if remaining_slots <= 0:
                        continue
                    branches_created = 0
                    for source_idx in active_idxs:
                        branches_per_idx = min(beam_size - 1, remaining_slots - branches_created)
                        if branches_per_idx <= 0:
                            break
                        for _ in range(branches_per_idx):
                            # ==== Entropy-based Adaptive Beaming ====
                            
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = initial_entropy_dict.get(source_idx, 0.0)
                            entropy_delta = entropy_now - entropy_init
                            penalty = 0.0
                            if self.branch_penalty_enabled:
                                penalty = max(
                                    0.0,
                                    min(1.0, float(self.branch_penalty_slope) * consecutive_branch_counts[source_idx]),
                                )
                                penalty_values.append(penalty)
                            effective_branch_probability = max(0.0, self.branch_probability * (1.0 - penalty))
                            prob = random.random() - self.entropy_weight * entropy_delta
                    
                            prob = max(0.0, min(1.0, prob))
                            if prob > effective_branch_probability:
                                if penalty > 0.0:
                                    tool_metrics["legacy_online_beam/skipped_by_penalty"] += 1
                                else:
                                    tool_metrics["legacy_online_beam/skipped_by_entropy"] += 1
                                if self.branch_penalty_enabled and self.branch_penalty_on_skip and entropy_delta > 0:
                                    consecutive_branch_counts[source_idx] += 1
                                continue
                            # ==== END ====
                            new_inputs.append(curr_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append(result_masks[source_idx].copy())
                            new_call_counters.append(call_counters[source_idx])
                            consecutive_branch_counts[source_idx] += 1
                            new_branch_counts.append(consecutive_branch_counts[source_idx])
                            new_source_indices.append(source_idx)
                            new_sample_origins.append(orig_sample)
                            tool_metrics["legacy_online_beam/created"] += 1
                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                        if branches_created >= remaining_slots:
                            break


                # Add non-active samples that still need more rollouts
                for orig_sample in range(batch_size):
                    if orig_sample not in active_by_sample and rollouts_per_sample[orig_sample] < num_samples:
                        # 对于不活跃样本，每次只新增一个branch
                        branches_to_add = min(1, num_samples - rollouts_per_sample[orig_sample])
                        if branches_to_add <= 0:
                            continue
                            
                        # Use first index of this sample as template
                        source_idx = sample_to_indices[orig_sample][0]
                        
                        # Create new branches
                        for _ in range(branches_to_add):
                            new_inputs.append(init_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append([])
                            new_call_counters.append(0)
                            new_sample_origins.append(orig_sample)  # 记录原始样本
                            new_branch_counts.append(0)
                            new_source_indices.append(-1)
                            rollouts_per_sample[orig_sample] += 1
                
                # Add new branches to existing lists
                if new_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(new_inputs)
                    init_inputs.extend(new_init_inputs)
                    result_masks.extend(new_result_masks)
                    if len(new_branch_counts) != len(new_inputs):
                        raise RuntimeError("branch count bookkeeping mismatch")
                    consecutive_branch_counts.extend(new_branch_counts)
                    for source_idx in new_source_indices:
                        if 0 <= source_idx < len(entropy_events):
                            entropy_events.append(list(entropy_events[source_idx]))
                        else:
                            entropy_events.append([])
                    for offset, source_idx in enumerate(new_source_indices):
                        new_idx = start_idx + offset
                        if 0 <= source_idx < len(curr_inputs) and source_idx in initial_entropy_dict:
                            initial_entropy_dict[new_idx] = initial_entropy_dict[source_idx]
                    call_counters.extend(new_call_counters)
                    
                    # Add new indices to active list
                    final_active_indices.extend(range(start_idx, start_idx + len(new_inputs)))
                    
                    # 使用正确的原始样本信息更新映射
                    for i, new_idx in enumerate(range(start_idx, start_idx + len(new_inputs))):
                        orig_sample = new_sample_origins[i]
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                
                active_indices = final_active_indices

            # 确保所有序列不超过max_len
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][:offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]
            
            # Reorganize outputs to match original batch structure and select up to num_samples per sample
            output_sequences = []
            output_result_masks = []
            output_entropy_events = []
            for i in range(batch_size):
                # Get all indices for this sample
                sample_indices = sample_to_indices.get(i, [])
                # Ensure we have exactly num_samples outputs per sample
                selected_indices = sample_indices[:num_samples]
                
                # If we have fewer rollouts than requested, duplicate the last one
                while len(selected_indices) < num_samples:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        break  # Should not happen but just in case
                        
                # Extract outputs for selected indices
                for idx in selected_indices:
                    output_sequences.append(curr_inputs[idx][len(prompt_token_ids_list[i]):])
                    output_result_masks.append(result_masks[idx])
                    output_entropy_events.append(list(entropy_events[idx]))

            padded_response_list = []
            padded_result_mask_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")
                
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                
                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                
                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)
            
            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            
            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            if num_samples > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, num_samples)
                attention_mask = _repeat_interleave(attention_mask, num_samples)
                position_ids = _repeat_interleave(position_ids, num_samples)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(value, num_samples, axis=0)
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [item for item in value for _ in range(num_samples)]

            # Prefix payloads are generation inputs, not per-trajectory output
            # metadata. Keeping them would make initial/branch concatenation
            # ambiguous and unnecessarily retain large Python objects.
            non_tensor_batch.pop("tao_prefix_token_ids", None)
            non_tensor_batch.pop("tao_prefix_model_mask", None)
            entropy_event_array = np.empty(len(output_entropy_events), dtype=object)
            entropy_event_array[:] = output_entropy_events
            non_tensor_batch["tao_entropy_events"] = entropy_event_array

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            # NOTE: 不能用 get_response_mask(按首个 eos 截断)。SFT 格式投喂会在响应
            # 中间注入 <|im_end|>(=eos_token_id)，若按首个 eos 截断，工具调用后的续写
            # (含 <answer>)会被整段 mask 掉，训练 loss 覆盖和 reward valid_length 双双失效。
            # 改为按每条轨迹的真实响应长度构造 attention mask（有效区=1，padding=0）。
            # 内联(无中间 eos)情形下该结果与 get_response_mask 完全一致，无回归。
            response_valid_lengths = [len(seq) for seq in output_sequences]
            response_attention_mask = torch.zeros_like(response)
            for _row, _valid_len in enumerate(response_valid_lengths):
                _valid_len = min(_valid_len, response.size(1))
                if _valid_len > 0:
                    response_attention_mask[_row, :_valid_len] = 1
            response_attention_mask = response_attention_mask.to(attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            loss_mask = loss_mask * response_attention_mask
            response_model_mask = loss_mask.clone()
            observation_mask = response_attention_mask * (1 - response_model_mask.to(response_attention_mask.dtype))

            # 自检(env 门控): 确认 SFT 格式注入后的 mask 正确——注入块 attn=1/loss=0、
            # 模型续写 attn=1/loss=1、有效区含 </answer> 未被中途 eos 截断。
            import os as _os
            if _os.environ.get("VERL_ROLLOUT_MASK_DEBUG") and response.size(0) > 0:
                _valid = int(response_attention_mask[0].sum().item())
                _trainable = int(response_model_mask[0].sum().item())
                _text = self.tokenizer.decode(response[0][:_valid]) if _valid > 0 else ""
                logger.info(
                    f"[mask-selfcheck] row0 valid_len(attn=1)={_valid} "
                    f"trainable(loss=1)={_trainable} resp_len={response.size(1)} "
                    f"has_</answer>_in_valid={'</answer>' in _text} "
                    f"has_<tool_response>={'<tool_response>' in _text}"
                )

            post_branch_model_mask = torch.zeros_like(response_model_mask)
            if is_teb_branch:
                boundary_values = non_tensor_batch.get("tao_boundary_response_idx")
                if boundary_values is None:
                    raise ValueError("TAO branch request is missing tao_boundary_response_idx")
                post_branch_model_mask = response_model_mask.clone()
                for row, boundary in enumerate(boundary_values):
                    post_branch_model_mask[row, : int(boundary)] = 0
                if self.teb_advantage_mode != "soft_grpo":
                    # Local-credit variants train only the newly sampled
                    # suffix. Paper ARPO soft attribution intentionally keeps
                    # copied model prefixes in the loss.
                    loss_mask = post_branch_model_mask.clone()

            # 计算平均执行时间
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
            if penalty_values:
                tool_metrics["legacy_online_beam/penalty_mean"] = sum(penalty_values) / len(penalty_values)
                
            # 计算每个工具的平均执行时间和成功率
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            batch = TensorDict({
                "prompts": input_ids,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "loss_mask": loss_mask,
                "response_model_mask": response_model_mask,
                "observation_mask": observation_mask,
                "post_branch_model_mask": post_branch_model_mask,
                "position_ids": final_position_ids,
            }, batch_size=final_batch_size)

            defaults = {
                "tao_steps": None,
                "tao_branch_group_id": "",
                "tao_branch_active": is_teb_branch,
                "tao_boundary_response_idx": -1,
                "tao_boundary_step_idx": -1,
                "branch_score": 0.0,
                "branch_source": "active_branch" if is_teb_branch else "original",
            }
            for key, default in defaults.items():
                if key not in non_tensor_batch:
                    non_tensor_batch[key] = np.asarray([default] * final_batch_size, dtype=object)

        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            
        # 合并所有metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        
        # 将metrics添加到meta_info中
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        return data_proto
