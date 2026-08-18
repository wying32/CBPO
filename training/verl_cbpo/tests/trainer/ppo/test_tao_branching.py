import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_bc_cbpo_outcome_advantage, compute_teb_outcome_advantage
from verl.trainer.ppo.tao_branching import (
    hash_token_prefix,
    parse_tao_trajectory,
    select_entropy_balanced_boundaries,
    select_tao_boundaries,
)


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _tokens(text):
    return [ord(char) for char in text]


def _trajectory(first_action="search", content="alpha", observation="result a"):
    model_1 = f"think <{first_action}>{content}</{first_action}>"
    obs = f" <result>{observation}</result>"
    model_2 = "next <answer>done</answer>ignored"
    token_ids = _tokens(model_1 + obs + model_2)
    model_mask = [1] * len(model_1) + [0] * len(obs) + [1] * len(model_2)
    response_mask = [1] * len(token_ids)
    return token_ids, model_mask, response_mask, len(model_1 + obs)


def test_parser_preserves_boundaries_and_observation_provenance():
    token_ids, model_mask, response_mask, second_boundary = _trajectory()
    steps = parse_tao_trajectory(token_ids, model_mask, response_mask, CharacterTokenizer())

    assert [step["action_type"] for step in steps] == ["search", "answer"]
    assert steps[0]["observation_text"].startswith(" <result>")
    assert steps[1]["step_start_token_idx"] == second_boundary
    assert steps[1]["step_end_token_idx"] < len(token_ids)  # trailing text is invalid suffix


def test_parser_rejects_incomplete_or_multiple_actions():
    text = "think <search>a</search><python>b</python> <result>x</result>"
    ids = _tokens(text)
    obs_start = text.index(" <result>")
    steps = parse_tao_trajectory(
        ids,
        [1] * obs_start + [0] * (len(ids) - obs_start),
        [1] * len(ids),
        CharacterTokenizer(),
    )
    assert steps == []


def test_selector_uses_approximate_candidates_but_hashes_exact_source_prefix():
    tokenizer = CharacterTokenizer()
    rows = [_trajectory(content="alpha"), _trajectory(first_action="python", content="print(1)")]
    parsed = [parse_tao_trajectory(ids, mask, valid, tokenizer) for ids, mask, valid, _ in rows]
    selected = select_tao_boundaries(
        parsed,
        rewards=[0.0, 1.0],
        prompt_indices=[0, 0],
        token_rows=[rows[0][0], rows[1][0]],
    )

    assert len(selected) == 1
    boundary = selected[0]
    prefix = rows[boundary["source_index"]][0][: boundary["boundary_response_idx"]]
    assert boundary["branch_group_id"] == f"0:{hash_token_prefix(prefix)}"


def test_entropy_balanced_selector_penalizes_repeated_path_branches():
    rows = [list(range(20)), list(range(20, 40))]
    events = [
        [{"boundary_response_idx": 5, "entropy_delta": 0.8, "entropy": 0.9, "root_entropy": 0.1}],
        [{"boundary_response_idx": 6, "entropy_delta": 0.6, "entropy": 0.8, "root_entropy": 0.2}],
    ]
    selected, unused = select_entropy_balanced_boundaries(
        events,
        prompt_indices=[0, 0],
        token_rows=rows,
        branch_budget_per_prompt=3,
        base_probability=0.2,
        entropy_scale=0.8,
        branch_threshold=0.3,
        path_power_decay_rho=0.0,
        node_power_decay_rho=1.0,
        max_branch_nodes_per_path=1,
        max_branches_per_node=2,
        max_branches_per_path=2,
    )

    # raw probs 0.84 vs 0.68; the power discount 1/(1+l) hands the second
    # slot to the weaker path before the stronger one takes its second branch.
    counts = {item["source_index"]: item["branch_count"] for item in selected}
    assert counts == {0: 2, 1: 1}
    assert unused[0] == 0


def test_entropy_balanced_selector_keeps_multiple_grid_nodes_per_path():
    rows = [list(range(256)), list(range(256, 512))]
    events = [
        [
            {"boundary_response_idx": 64, "entropy_delta": 0.40, "entropy": 0.50, "root_entropy": 0.10},
            {"boundary_response_idx": 128, "entropy_delta": 0.38, "entropy": 0.48, "root_entropy": 0.10},
            {"boundary_response_idx": 192, "entropy_delta": 0.36, "entropy": 0.46, "root_entropy": 0.10},
        ],
        [
            {"boundary_response_idx": 64, "entropy_delta": 0.40, "entropy": 0.50, "root_entropy": 0.10},
        ],
    ]
    selected, unused = select_entropy_balanced_boundaries(
        events,
        prompt_indices=[0, 0],
        token_rows=rows,
        branch_budget_per_prompt=4,
        base_probability=0.2,
        entropy_scale=2.0,
        branch_threshold=0.3,
        path_power_decay_rho=0.7,
        node_power_decay_rho=0.7,
        max_branch_nodes_per_path=3,
        min_branch_node_distance=64,
        max_branches_per_node=2,
        max_branches_per_path=4,
    )

    first_path_nodes = sorted(
        item["boundary_response_idx"] for item in selected if item["source_index"] == 0
    )
    assert first_path_nodes == [64, 128, 192]
    assert sum(item["branch_count"] for item in selected) == 4
    assert unused[0] == 0


def test_entropy_balanced_selector_requires_tau_above_alpha():
    with pytest.raises(ValueError):
        select_entropy_balanced_boundaries(
            [[]],
            prompt_indices=[0],
            token_rows=[list(range(5))],
            branch_budget_per_prompt=1,
            base_probability=0.3,
            branch_threshold=0.3,
        )


def test_teb_advantage_masks_prefix_observation_and_branch_global_credit():
    rewards = torch.tensor([[1.0], [3.0], [-100.0]])  # may include token-level KL shaping
    outcomes = torch.tensor([[1.0], [3.0], [5.0]])
    model_mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.float32)
    post_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
    advantages, _ = compute_teb_outcome_advantage(
        token_level_rewards=rewards,
        outcome_rewards=outcomes,
        response_model_mask=model_mask,
        post_branch_model_mask=post_mask,
        prompt_index=np.asarray(["p", "p", "p"], dtype=object),
        branch_group_id=np.asarray(["g", "", "g"], dtype=object),
        branch_boundary=np.asarray([3, -1, 3], dtype=object),
        branch_source=np.asarray(["original", "original", "active_branch"], dtype=object),
        omega_local=0.3,
    )

    assert advantages[2, :3].abs().sum() == 0  # active branch has no global/prefix credit
    assert advantages[:, 2].abs().sum() == 0  # observation is never trained
    assert advantages[2, 3] > 0  # better exact-prefix continuation gets local credit


def test_bc_cbpo_uses_cbv_for_exact_prefix_local_credit():
    # Two prompts have identical base advantages but different sibling reward
    # variance. High-CBV continuations must be amplified; low-CBV ones damped.
    rewards = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    outcomes = torch.tensor([[0.0], [1.0], [0.4], [0.6]])
    model_mask = torch.ones((4, 2), dtype=torch.float32)
    post_mask = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.float32)
    advantages, _, metrics = compute_bc_cbpo_outcome_advantage(
        token_level_rewards=rewards,
        outcome_rewards=outcomes,
        response_model_mask=model_mask,
        post_branch_model_mask=post_mask,
        prompt_index=np.asarray(["p", "p", "q", "q"], dtype=object),
        branch_group_id=np.asarray(["g1", "g1", "g2", "g2"], dtype=object),
        branch_boundary=np.asarray([1, 1, 1, 1], dtype=object),
        branch_source=np.asarray(["original", "active_branch", "original", "active_branch"], dtype=object),
        cbv_alpha=0.2,
        cbv_phi=2.0,
        cbv_mode="std",
        return_metrics=True,
    )

    assert metrics["bc_cbpo/active_branch_groups"] == 2.0
    assert metrics["bc_cbpo/cbv_mean"] > 0
    assert advantages[1, 1] > advantages[3, 1] > 0
    assert advantages[0, 1] < advantages[2, 1] < 0
    assert advantages[:, 0].abs().sum() == 0  # shared mean is zero in both nodes


def test_bc_cbpo_keeps_positive_shared_prefix_when_cbv_is_zero():
    rewards = torch.tensor([[1.0], [1.0], [0.0]])
    outcomes = rewards.clone()
    model_mask = torch.ones((3, 2), dtype=torch.float32)
    post_mask = torch.tensor([[0, 1], [0, 1], [0, 0]], dtype=torch.float32)
    advantages, _, metrics = compute_bc_cbpo_outcome_advantage(
        token_level_rewards=rewards,
        outcome_rewards=outcomes,
        response_model_mask=model_mask,
        post_branch_model_mask=post_mask,
        prompt_index=np.asarray(["p", "p", "p"], dtype=object),
        branch_group_id=np.asarray(["g", "g", ""], dtype=object),
        branch_boundary=np.asarray([1, 1, -1], dtype=object),
        branch_source=np.asarray(["original", "active_branch", "global_fallback"], dtype=object),
        return_metrics=True,
    )

    assert metrics["bc_cbpo/active_branch_groups"] == 1.0
    assert metrics["bc_cbpo/cbv_mean"] == 0.0
    assert advantages[0, 0] > 0  # canonical shared prefix keeps descendant value


def test_bc_cbpo_hierarchically_attributes_multiple_parent_nodes():
    # Row 0 is one canonical trajectory with nodes g1@2 and g2@4.  Each node
    # has one active continuation.  The parent must be partitioned into
    # non-overlapping prefix/inter-node/final segments.
    rewards = torch.tensor([[1.0], [0.0], [2.0], [-1.0]])
    outcomes = rewards.clone()
    model_mask = torch.ones((4, 6), dtype=torch.float32)
    post_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    advantages, _, metrics = compute_bc_cbpo_outcome_advantage(
        token_level_rewards=rewards,
        outcome_rewards=outcomes,
        response_model_mask=model_mask,
        post_branch_model_mask=post_mask,
        prompt_index=np.asarray(["p", "p", "p", "p"], dtype=object),
        branch_group_id=np.asarray([["g1", "g2"], "g1", "g2", ""], dtype=object),
        branch_boundary=np.asarray([[2, 4], 2, 4, -1], dtype=object),
        branch_source=np.asarray(["original", "active_branch", "active_branch", "global_fallback"], dtype=object),
        return_metrics=True,
    )

    assert metrics["bc_cbpo/active_branch_groups"] == 2.0
    assert torch.all(advantages[0, :2] == advantages[0, 0])
    assert torch.all(advantages[0, 2:4] == advantages[0, 2])
    assert torch.all(advantages[0, 4:] == advantages[0, 4])
    assert advantages[1, :2].abs().sum() == 0
    assert advantages[2, :4].abs().sum() == 0
