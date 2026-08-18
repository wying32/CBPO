import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_gigpo_outcome_advantage


def _step(index, start, end, state_hash, state_text="", state_source="search"):
    return {
        "step_index": index,
        "step_start_token_idx": start,
        "step_end_token_idx": end,
        "model_span": (start, end),
        "previous_observation_hash": state_hash,
        "previous_observation_text": state_text,
        "state_source": state_source,
        "parse_valid": True,
    }


def test_gigpo_combines_episode_and_anchor_state_advantages():
    rewards = torch.zeros((2, 6))
    rewards[0, 5] = 1.0
    mask = torch.ones_like(rewards)
    trajectories = [
        [_step(0, 0, 2, "initial", state_source="initial"), _step(1, 2, 6, "shared")],
        [_step(0, 0, 2, "initial", state_source="initial"), _step(1, 2, 6, "shared")],
    ]

    advantages, _, metrics = compute_gigpo_outcome_advantage(
        rewards,
        mask,
        np.asarray(["prompt", "prompt"], dtype=object),
        trajectories,
        gamma=0.5,
        omega=1.0,
        normalization="std",
        state_match_mode="exact",
        return_metrics=True,
    )

    # Episode advantages are +/-1. At the shared second state the local
    # advantages add another +/-1; the first step is discounted but preserves
    # the same ordering and also receives local credit.
    assert torch.allclose(advantages[0, 2:], torch.full((4,), 2.0), atol=1e-5)
    assert torch.allclose(advantages[1, 2:], torch.full((4,), -2.0), atol=1e-5)
    assert metrics["gigpo/anchor_groups"] == 2.0
    assert metrics["gigpo/local_step_coverage"] == 1.0


def test_gigpo_degrades_to_grpo_without_repeated_anchor_states():
    rewards = torch.zeros((2, 4))
    rewards[0, 3] = 1.0
    mask = torch.ones_like(rewards)
    trajectories = [
        [_step(0, 0, 4, "state-a", state_source="python")],
        [_step(0, 0, 4, "state-b", state_source="python")],
    ]

    advantages, _, metrics = compute_gigpo_outcome_advantage(
        rewards,
        mask,
        np.asarray(["prompt", "prompt"], dtype=object),
        trajectories,
        normalization="std",
        state_match_mode="auto",
        return_metrics=True,
    )

    assert torch.allclose(advantages[0], torch.ones(4), atol=1e-5)
    assert torch.allclose(advantages[1], -torch.ones(4), atol=1e-5)
    assert metrics["gigpo/anchor_groups"] == 0.0


def test_gigpo_similarity_matching_is_search_only_in_auto_mode():
    rewards = torch.zeros((2, 4))
    rewards[0, 3] = 1.0
    mask = torch.ones_like(rewards)
    similar_left = "retrieved document alpha beta gamma"
    similar_right = "retrieved document alpha beta gamma!"

    search_trajectories = [
        [_step(0, 0, 4, "a", similar_left, "search")],
        [_step(0, 0, 4, "b", similar_right, "search")],
    ]
    _, _, search_metrics = compute_gigpo_outcome_advantage(
        rewards,
        mask,
        np.asarray(["prompt", "prompt"], dtype=object),
        search_trajectories,
        state_match_mode="auto",
        similarity_threshold=0.9,
        return_metrics=True,
    )
    assert search_metrics["gigpo/anchor_groups"] == 1.0

    python_trajectories = [
        [_step(0, 0, 4, "a", similar_left, "python")],
        [_step(0, 0, 4, "b", similar_right, "python")],
    ]
    _, _, python_metrics = compute_gigpo_outcome_advantage(
        rewards,
        mask,
        np.asarray(["prompt", "prompt"], dtype=object),
        python_trajectories,
        state_match_mode="auto",
        similarity_threshold=0.9,
        return_metrics=True,
    )
    assert python_metrics["gigpo/anchor_groups"] == 0.0
