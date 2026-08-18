# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_arpo_soft_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_model_mask: torch.Tensor,
    prompt_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
):
    """Compute the soft advantage attribution used by paper ARPO.

    ARPO places the initial and partial rollouts of a prompt in one GRPO
    reward group.  Exact-prefix copies intentionally keep their model-token
    mask: their identical importance ratios make the shared-prefix gradient
    equal to the average contribution of the descendant trajectories, while
    their newly sampled suffixes retain trajectory-specific advantages.
    Tool observations remain excluded through ``response_model_mask``.
    """

    return compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_model_mask,
        index=prompt_index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std,
    )


def _gigpo_normalize(values: torch.Tensor, mode: str, epsilon: float) -> torch.Tensor:
    centered = values - values.mean()
    if mode == "none":
        return centered
    if mode == "std":
        std = values.std(unbiased=False)
        if std < epsilon:
            return torch.zeros_like(centered)
        return centered / (std + epsilon)
    raise ValueError(f"Unsupported GiGPO normalization: {mode}")


def _gigpo_state_similarity(left: str, right: str) -> float:
    """Longest-matching-subsequence similarity used by similarity GiGPO."""

    lhs = " ".join(str(left).lower().split())
    rhs = " ".join(str(right).lower().split())
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs, autojunk=False).ratio()


def _gigpo_anchor_groups(records, state_match_mode: str, similarity_threshold: float):
    """Group step records by prompt and matching pre-action environment state."""

    if state_match_mode not in {"exact", "similarity", "auto"}:
        raise ValueError(f"Unsupported GiGPO state_match_mode: {state_match_mode}")

    prompt_records = defaultdict(list)
    for record in records:
        prompt_records[record["prompt_id"]].append(record)

    groups = []
    for prompt_group in prompt_records.values():
        representatives = []
        prompt_groups = []
        for record in prompt_group:
            matched_group = None
            for group_index, representative in enumerate(representatives):
                if record["state_source"] != representative["state_source"]:
                    continue
                if record["state_hash"] == representative["state_hash"]:
                    matched_group = group_index
                    break
                use_similarity = state_match_mode == "similarity" or (
                    state_match_mode == "auto" and record["state_source"] == "search"
                )
                if use_similarity and _gigpo_state_similarity(
                    record["state_text"], representative["state_text"]
                ) >= similarity_threshold:
                    matched_group = group_index
                    break
            if matched_group is None:
                representatives.append(record)
                prompt_groups.append([record])
            else:
                prompt_groups[matched_group].append(record)
        groups.extend(prompt_groups)
    return groups


def compute_gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_model_mask: torch.Tensor,
    prompt_index: np.ndarray,
    trajectories,
    gamma: float = 0.95,
    omega: float = 1.0,
    normalization: str = "std",
    min_anchor_group_size: int = 2,
    state_match_mode: str = "auto",
    similarity_threshold: float = 0.9,
    epsilon: float = 1e-6,
    return_metrics: bool = False,
):
    """Compute the two-level Group-in-Group Policy Optimization advantage.

    Episode advantages compare complete rollout returns for the same prompt.
    Step advantages compare discounted returns for actions sharing the same
    pre-action environment state.  The local signal is applied only to model
    tokens belonging to that action; unparsed tokens retain the episode-level
    GRPO signal, so missing anchor groups degrade safely to GRPO.
    """

    if not 0.0 < gamma <= 1.0:
        raise ValueError("GiGPO gamma must be in (0, 1]")
    if omega < 0.0:
        raise ValueError("GiGPO omega must be non-negative")
    if min_anchor_group_size < 2:
        raise ValueError("GiGPO min_anchor_group_size must be at least 2")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("GiGPO similarity_threshold must be in [0, 1]")
    if len(trajectories) != token_level_rewards.shape[0]:
        raise ValueError("GiGPO trajectories and reward rows must have equal length")

    scores = token_level_rewards.sum(dim=-1)
    model_mask = response_model_mask.to(dtype=scores.dtype)
    advantages = torch.zeros_like(model_mask, dtype=scores.dtype)
    prompt_values = np.asarray(prompt_index, dtype=object)

    with torch.no_grad():
        prompt_to_rows = defaultdict(list)
        for row, prompt_id in enumerate(prompt_values):
            prompt_to_rows[prompt_id].append(row)
        for rows in prompt_to_rows.values():
            row_tensor = torch.as_tensor(rows, device=scores.device, dtype=torch.long)
            episode_values = _gigpo_normalize(scores[row_tensor], normalization, epsilon)
            for row, value in zip(rows, episode_values):
                advantages[row] = value * model_mask[row]

        step_records = []
        total_valid_steps = 0
        for row, raw_steps in enumerate(trajectories):
            steps = [step for step in (raw_steps or []) if step.get("parse_valid", False)]
            total_valid_steps += len(steps)
            if not steps:
                continue

            immediate_rewards = []
            response_length = token_level_rewards.shape[1]
            for step_offset, step in enumerate(steps):
                reward_start = max(0, int(step["step_start_token_idx"]))
                reward_end = (
                    max(reward_start, int(steps[step_offset + 1]["step_start_token_idx"]))
                    if step_offset + 1 < len(steps)
                    else response_length
                )
                immediate_rewards.append(token_level_rewards[row, reward_start:reward_end].sum())

            discounted_return = torch.zeros((), device=scores.device, dtype=scores.dtype)
            step_returns = [None] * len(steps)
            for step_offset in range(len(steps) - 1, -1, -1):
                discounted_return = immediate_rewards[step_offset] + gamma * discounted_return
                step_returns[step_offset] = discounted_return.clone()

            for step, step_return in zip(steps, step_returns):
                span_start, span_end = step["model_span"]
                step_records.append(
                    {
                        "row": row,
                        "prompt_id": prompt_values[row],
                        "state_hash": str(step.get("previous_observation_hash", "")),
                        "state_text": str(step.get("previous_observation_text", "")),
                        "state_source": str(step.get("state_source", "initial")),
                        "span_start": max(0, int(span_start)),
                        "span_end": min(response_length, int(span_end)),
                        "return": step_return,
                    }
                )

        anchor_groups = _gigpo_anchor_groups(
            step_records,
            state_match_mode=state_match_mode,
            similarity_threshold=similarity_threshold,
        )
        active_groups = 0
        locally_credited_steps = 0
        group_sizes = []
        for group in anchor_groups:
            if len(group) < min_anchor_group_size:
                continue
            group_returns = torch.stack([record["return"] for record in group])
            local_values = _gigpo_normalize(group_returns, normalization, epsilon)
            active_groups += 1
            group_sizes.append(len(group))
            for record, local_value in zip(group, local_values):
                start, end = record["span_start"], record["span_end"]
                if end <= start:
                    continue
                row = record["row"]
                local_mask = model_mask[row, start:end]
                advantages[row, start:end] += omega * local_value * local_mask
                locally_credited_steps += 1

    metrics = {
        "gigpo/valid_steps": float(total_valid_steps),
        "gigpo/anchor_groups": float(active_groups),
        "gigpo/locally_credited_steps": float(locally_credited_steps),
        "gigpo/local_step_coverage": float(locally_credited_steps / max(1, total_valid_steps)),
        "gigpo/anchor_group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "gigpo/anchor_group_size_max": float(max(group_sizes)) if group_sizes else 0.0,
    }
    if return_metrics:
        return advantages, advantages, metrics
    return advantages, advantages


def _as_metadata_list(value):
    """Normalize scalar/object-array rollout metadata to a Python list."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and not value:
        return []
    return [value]


def _build_exact_prefix_groups(branch_group_id: np.ndarray, branch_boundary: np.ndarray):
    """Expand scalar or multi-node parent metadata into exact-prefix groups."""

    group_values = np.asarray(branch_group_id, dtype=object)
    boundary_values = np.asarray(branch_boundary, dtype=object)
    exact_groups = defaultdict(list)
    row_associations = defaultdict(list)
    group_boundaries = {}

    for row, (group_value, boundary_value) in enumerate(zip(group_values, boundary_values)):
        group_ids = [str(value) for value in _as_metadata_list(group_value) if value is not None and str(value)]
        boundaries = [int(value) for value in _as_metadata_list(boundary_value)]
        if not group_ids:
            continue
        if len(boundaries) == 1 and len(group_ids) > 1:
            boundaries = boundaries * len(group_ids)
        if len(group_ids) != len(boundaries):
            raise ValueError(
                f"branch group/boundary metadata length mismatch at row {row}: "
                f"{len(group_ids)} != {len(boundaries)}"
            )
        for group_id, boundary in zip(group_ids, boundaries):
            if boundary < 0:
                continue
            if row not in exact_groups[group_id]:
                exact_groups[group_id].append(row)
            row_associations[row].append((boundary, group_id))
            previous = group_boundaries.setdefault(group_id, boundary)
            if previous != boundary:
                raise ValueError(f"exact-prefix group {group_id} has inconsistent boundaries")

    for row in row_associations:
        row_associations[row].sort(key=lambda item: item[0])
    return exact_groups, row_associations, group_boundaries


def _token_interval_mask(model_mask: torch.Tensor, start: int, end: int | None = None):
    positions = torch.arange(model_mask.shape[0], device=model_mask.device)
    interval = positions >= int(start)
    if end is not None:
        interval = interval & (positions < int(end))
    return model_mask.bool() & interval


def compute_teb_outcome_advantage(
    token_level_rewards: torch.Tensor,
    outcome_rewards: torch.Tensor,
    response_model_mask: torch.Tensor,
    post_branch_model_mask: torch.Tensor,
    prompt_index: np.ndarray,
    branch_group_id: np.ndarray,
    branch_source: np.ndarray,
    branch_boundary: np.ndarray | None = None,
    epsilon: float = 1e-6,
    omega_local: float = 0.3,
    local_adv_clip: float = 2.0,
    reward_std_threshold: float = 1e-6,
    norm_global_by_std: bool = True,
):
    """Compute global GRPO plus exact-prefix local suffix credit.

    Active branch rows are intentionally excluded from prompt-level GRPO and
    receive only local credit.  Observation, shared-prefix and padding tokens
    remain zero through the two explicit provenance masks.
    """

    scores = token_level_rewards.sum(dim=-1)
    local_scores = outcome_rewards.sum(dim=-1)
    advantages = torch.zeros_like(response_model_mask, dtype=scores.dtype)
    sources = np.asarray(branch_source, dtype=object)
    prompt_ids = np.asarray(prompt_index, dtype=object)
    if branch_boundary is None:
        branch_boundary = np.asarray(
            [
                int(torch.nonzero(mask, as_tuple=False)[0].item()) if torch.any(mask) else -1
                for mask in post_branch_model_mask
            ],
            dtype=object,
        )
    exact_groups, row_associations, _ = _build_exact_prefix_groups(branch_group_id, branch_boundary)

    with torch.no_grad():
        prompt_to_rows = defaultdict(list)
        for row in range(scores.shape[0]):
            if str(sources[row]) != "active_branch":
                prompt_to_rows[prompt_ids[row]].append(row)

        for rows in prompt_to_rows.values():
            group_scores = scores[rows]
            centered = group_scores - group_scores.mean()
            if norm_global_by_std:
                centered = centered / (group_scores.std(unbiased=False) + epsilon)
            for row, value in zip(rows, centered):
                advantages[row] = value * response_model_mask[row]

        local_credit = {}
        for group_id, rows in exact_groups.items():
            if len(rows) < 2:
                continue
            group_scores = local_scores[rows]
            std = group_scores.std(unbiased=False)
            if std < reward_std_threshold:
                continue
            local_values = (group_scores - group_scores.mean()) / (std + epsilon)
            local_values = torch.clamp(local_values, -local_adv_clip, local_adv_clip)
            for row, value in zip(rows, local_values):
                local_credit[(group_id, row)] = omega_local * value

        # Each original trajectory may own multiple ordered nodes.  Attribute
        # each parent segment to its nearest preceding selected node so local
        # credits never overlap or overwrite one another.
        for row, associations in row_associations.items():
            if str(sources[row]) == "active_branch":
                group_id = associations[0][1]
                value = local_credit.get((group_id, row))
                if value is not None:
                    advantages[row] += value * post_branch_model_mask[row]
                continue
            for index, (boundary, group_id) in enumerate(associations):
                value = local_credit.get((group_id, row))
                if value is None:
                    continue
                next_boundary = associations[index + 1][0] if index + 1 < len(associations) else None
                interval = _token_interval_mask(response_model_mask[row], boundary, next_boundary)
                advantages[row] = torch.where(interval, advantages[row] + value, advantages[row])

    return advantages, advantages


def _compute_cbv(group_scores: torch.Tensor, mode: str, epsilon: float) -> torch.Tensor:
    if mode == "binary":
        p_success = (group_scores > 0).float().mean()
        return 4.0 * p_success * (1.0 - p_success)
    if mode == "range":
        return group_scores.max() - group_scores.min()
    if mode == "std":
        return group_scores.std(unbiased=False)
    raise ValueError(f"Unsupported CBV mode: {mode}")


def compute_bc_cbpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    outcome_rewards: torch.Tensor,
    response_model_mask: torch.Tensor,
    post_branch_model_mask: torch.Tensor,
    prompt_index: np.ndarray,
    branch_group_id: np.ndarray,
    branch_source: np.ndarray,
    branch_boundary: np.ndarray | None = None,
    epsilon: float = 1e-6,
    cbv_alpha: float = 0.2,
    cbv_phi: float = 2.0,
    cbv_mode: str = "std",
    reward_std_threshold: float = 1e-6,
    norm_global_by_std: bool = True,
    return_metrics: bool = False,
):
    """Compute shared-prefix and CBV-aware continuation advantages.

    1. Compute the ordinary prompt-level GRPO advantage for every rollout.
    2. Give the canonical parent prefix the mean base advantage of all exact-
       prefix descendants; copied branch prefixes remain masked from the loss.
    3. Compute raw CBV as the sibling outcome-reward standard deviation, then
       normalize CBV across valid nodes and apply EAPO-style signed, clipped
       shaping only to post-branch continuation tokens:

       A_final = A + alpha * sign(A) * clip(z(CBV), -|A|/phi, |A|/phi).

    Entropy is intentionally absent here: it already controls rollout branch
    selection and must not be counted a second time during credit assignment.
    """

    if cbv_phi <= 0:
        raise ValueError("cbv_phi must be positive")
    if not 0 <= cbv_alpha <= 1:
        raise ValueError("cbv_alpha must be in [0, 1]")

    scores = token_level_rewards.sum(dim=-1)
    local_scores = outcome_rewards.sum(dim=-1)
    advantages = torch.zeros_like(response_model_mask, dtype=scores.dtype)
    sources = np.asarray(branch_source, dtype=object)
    prompt_ids = np.asarray(prompt_index, dtype=object)
    if branch_boundary is None:
        branch_boundary = np.asarray(
            [
                int(torch.nonzero(mask, as_tuple=False)[0].item()) if torch.any(mask) else -1
                for mask in post_branch_model_mask
            ],
            dtype=object,
        )
    exact_groups, row_associations, _ = _build_exact_prefix_groups(branch_group_id, branch_boundary)
    metrics = {
        "bc_cbpo/active_branch_groups": 0.0,
        "bc_cbpo/cbv_mean": 0.0,
        "bc_cbpo/cbv_max": 0.0,
        "bc_cbpo/cbv_normalized_mean": 0.0,
        "bc_cbpo/shaped_adv_abs_mean": 0.0,
    }

    with torch.no_grad():
        prompt_to_rows = defaultdict(list)
        for row in range(scores.shape[0]):
            prompt_to_rows[prompt_ids[row]].append(row)

        base_values = torch.zeros_like(scores)
        for rows in prompt_to_rows.values():
            group_scores = scores[rows]
            centered = group_scores - group_scores.mean()
            if norm_global_by_std:
                centered = centered / (group_scores.std(unbiased=False) + epsilon)
            for row, value in zip(rows, centered):
                base_values[row] = value
                advantages[row] = value * response_model_mask[row]

        valid_groups = []
        for group_id, rows in exact_groups.items():
            if len(rows) < 2:
                continue
            cbv = _compute_cbv(local_scores[rows], cbv_mode, epsilon)
            valid_groups.append((group_id, rows, cbv))

        if valid_groups:
            cbv_tensor = torch.stack([item[2] for item in valid_groups])
            cbv_mean = cbv_tensor.mean()
            cbv_std = cbv_tensor.std(unbiased=False)
            normalized_cbv = (
                (cbv_tensor - cbv_mean) / (cbv_std + epsilon)
                if cbv_std >= reward_std_threshold
                else torch.zeros_like(cbv_tensor)
            )
            metrics["bc_cbpo/active_branch_groups"] = float(cbv_tensor.numel())
            metrics["bc_cbpo/cbv_mean"] = cbv_tensor.mean().detach().item()
            metrics["bc_cbpo/cbv_max"] = cbv_tensor.max().detach().item()
            metrics["bc_cbpo/cbv_normalized_mean"] = normalized_cbv.mean().detach().item()

            shaped_abs_values = []
            group_shared_advantage = {}
            group_final_advantage = {}
            for (group_id, rows, _cbv), cbv_credit in zip(valid_groups, normalized_cbv):
                shared_advantage = base_values[rows].mean()
                group_shared_advantage[group_id] = shared_advantage

                for row in rows:
                    base_advantage = base_values[row]
                    limit = base_advantage.abs() / cbv_phi
                    bounded_credit = torch.clamp(cbv_credit, min=-limit, max=limit).detach()
                    final_advantage = base_advantage + cbv_alpha * torch.sign(base_advantage) * bounded_credit
                    group_final_advantage[(group_id, row)] = final_advantage

            # Active branches belong to one node.  Their copied exact prefix is
            # never trained; only their newly sampled suffix gets local credit.
            for row, associations in row_associations.items():
                if str(sources[row]) != "active_branch":
                    continue
                group_id = associations[0][1]
                copied_prefix_mask = response_model_mask[row] * (1.0 - post_branch_model_mask[row])
                advantages[row] = torch.where(
                    copied_prefix_mask.bool(), torch.zeros_like(advantages[row]), advantages[row]
                )
                final_advantage = group_final_advantage.get((group_id, row))
                if final_advantage is not None:
                    suffix_mask = post_branch_model_mask[row].bool()
                    advantages[row] = torch.where(suffix_mask, final_advantage, advantages[row])
                    selected = torch.masked_select(advantages[row], suffix_mask)
                    if selected.numel() > 0:
                        shaped_abs_values.append(selected.abs().mean().detach())

            # A canonical parent can own several ordered branch nodes.  Use a
            # non-overlapping hierarchical partition.  Before the first node,
            # use its shared value.  Between adjacent nodes, average the former
            # node's continuation value and the latter node's shared-prefix
            # value.  After the final node, use its continuation value.  This
            # reduces to the former prefix/suffix rule for a single node.
            for row, associations in row_associations.items():
                if str(sources[row]) != "original":
                    continue
                valid_associations = [
                    (boundary, group_id)
                    for boundary, group_id in associations
                    if group_id in group_shared_advantage and (group_id, row) in group_final_advantage
                ]
                if not valid_associations:
                    continue
                first_boundary, first_group_id = valid_associations[0]
                prefix_mask = _token_interval_mask(response_model_mask[row], 0, first_boundary)
                advantages[row] = torch.where(
                    prefix_mask, group_shared_advantage[first_group_id], advantages[row]
                )

                for index, (boundary, group_id) in enumerate(valid_associations):
                    if index + 1 < len(valid_associations):
                        next_boundary, next_group_id = valid_associations[index + 1]
                        segment_value = 0.5 * (
                            group_final_advantage[(group_id, row)]
                            + group_shared_advantage[next_group_id]
                        )
                    else:
                        next_boundary = None
                        segment_value = group_final_advantage[(group_id, row)]
                    suffix_mask = _token_interval_mask(response_model_mask[row], boundary, next_boundary)
                    advantages[row] = torch.where(suffix_mask, segment_value, advantages[row])
                    selected = torch.masked_select(advantages[row], suffix_mask)
                    if selected.numel() > 0:
                        shaped_abs_values.append(selected.abs().mean().detach())

            if shaped_abs_values:
                metrics["bc_cbpo/shaped_adv_abs_mean"] = torch.stack(shaped_abs_values).mean().item()

    if return_metrics:
        return advantages, advantages, metrics
    return advantages, advantages


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_gspo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float,
    cliprange_low=None,
    cliprange_high=None,
):
    """Compute the Group Sequence Policy Optimization clipped objective.

    GSPO uses one importance ratio per generated sequence instead of one ratio
    per token. The sequence ratio is the geometric mean of token likelihood
    ratios over valid response tokens:

        s(theta) = exp(mean_t(log pi_theta - log pi_old)).

    GRPO advantages are sequence-level values broadcast across response tokens;
    taking their masked mean recovers that scalar and remains well-defined for
    callers that provide token-shaped tensors.
    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    if cliprange_low is None or cliprange_high is None:
        raise ValueError("GSPO requires cliprange or both cliprange_low/cliprange_high")
    if cliprange_low <= 0 or cliprange_high <= 0:
        raise ValueError("GSPO clip ranges must be positive")

    # GSPO commonly uses a very narrow clip range (3e-4). Compute ratios in
    # fp32 so bf16 rounding around 1.0 does not erase the clipping signal.
    mask = response_mask.to(dtype=torch.float32)
    sequence_lengths = mask.sum(dim=-1)
    valid_sequences = sequence_lengths > 0
    if not torch.any(valid_sequences):
        raise ValueError("GSPO requires at least one response with a valid token")
    safe_lengths = sequence_lengths.clamp_min(1.0)

    token_log_ratio = log_prob.float() - old_log_prob.float()
    sequence_log_ratio = (token_log_ratio * mask).sum(dim=-1) / safe_lengths
    sequence_ratio = torch.exp(sequence_log_ratio)
    sequence_advantage = (advantages.float() * mask).sum(dim=-1) / safe_lengths

    pg_losses1 = -sequence_advantage * sequence_ratio
    clipped_ratio = torch.clamp(sequence_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses2 = -sequence_advantage * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_loss = pg_losses[valid_sequences].mean()
    pg_clipfrac = torch.gt(pg_losses2, pg_losses1)[valid_sequences].float().mean()
    ppo_kl = (-sequence_log_ratio[valid_sequences]).mean()
    pg_clipfrac_lower = torch.zeros((), device=pg_loss.device, dtype=pg_loss.dtype)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    advantage_noise_sigma: float = 0.0,
    policy_loss_type: str = "ppo",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        advantage_noise_sigma (float, optional):  
            Standard deviation for random noise applied to advantages. 
            Noise range will be [-sigma, +sigma]. Defaults to 0.0 (no noise).
        policy_loss_type (str, optional):
            ``ppo`` for token-level PPO ratios or ``gspo`` for a single
            length-normalized sequence ratio. Defaults to ``ppo``.
    """
    if policy_loss_type == "gspo":
        return compute_gspo_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=cliprange,
            cliprange_low=cliprange_low,
            cliprange_high=cliprange_high,
        )
    if policy_loss_type != "ppo":
        raise ValueError(f"Unsupported policy_loss_type: {policy_loss_type}")

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # advantage_noise_sigma = 0.1
    # if advantage_noise_sigma > 0:
    #     # 直接对advantage添加噪声
    #     advantage_noise = (2.0 * torch.rand_like(advantages) - 1.0) * advantage_noise_sigma  # [-sigma, sigma]
    #     # 只对有效的token位置（mask为1）添加噪声
    #     advantages = advantages + advantage_noise * response_mask.float()
    
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
