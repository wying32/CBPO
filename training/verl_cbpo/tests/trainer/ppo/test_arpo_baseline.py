import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_arpo_soft_outcome_advantage
from verl.trainer.ppo.tao_branching import select_arpo_boundaries


def test_arpo_selector_uses_paper_threshold_and_partial_budget():
    rows = [list(range(20)), list(range(20, 40))]
    events = [
        [
            {"boundary_response_idx": 5, "entropy_delta": 0.0},
            {"boundary_response_idx": 10, "entropy_delta": 0.5},
        ],
        [{"boundary_response_idx": 6, "entropy_delta": 0.25}],
    ]

    selected, unused = select_arpo_boundaries(
        events,
        prompt_indices=[0, 0],
        token_rows=rows,
        branch_budget_per_prompt=3,
        base_probability=0.5,
        entropy_scale=0.2,
        branch_threshold=0.5,
        branches_per_event=1,
    )

    # P == tau does not branch; both positive post-tool entropy changes do.
    assert [(item["source_index"], item["boundary_response_idx"]) for item in selected] == [
        (1, 6),
        (0, 10),
    ]
    assert unused[0] == 1


def test_arpo_selector_respects_z_and_total_budget():
    selected, unused = select_arpo_boundaries(
        [[{"boundary_response_idx": 4, "entropy_delta": 1.0}]],
        prompt_indices=[0],
        token_rows=[list(range(10))],
        branch_budget_per_prompt=2,
        base_probability=0.5,
        entropy_scale=0.2,
        branch_threshold=0.5,
        branches_per_event=3,
    )

    assert len(selected) == 1
    assert selected[0]["branch_count"] == 2
    assert unused[0] == 0


def test_arpo_soft_advantage_keeps_shared_model_prefix_and_masks_tools():
    rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    # Columns 0 and 2 model a copied shared prefix and a sampled suffix;
    # column 1 is a tool observation and must never receive actor credit.
    model_mask = torch.tensor(
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    )

    advantages, _ = compute_arpo_soft_outcome_advantage(
        token_level_rewards=rewards,
        response_model_mask=model_mask,
        prompt_index=np.asarray(["prompt", "prompt", "prompt"], dtype=object),
    )

    assert torch.all(advantages[:, 1] == 0)
    assert advantages[0, 0] < 0 < advantages[2, 0]
    assert torch.allclose(advantages[:, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[:, 0], advantages[:, 2])
