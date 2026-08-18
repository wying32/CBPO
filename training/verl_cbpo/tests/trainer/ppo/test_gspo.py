import math

import torch

from verl.trainer.ppo.core_algos import compute_gspo_policy_loss


def test_gspo_uses_length_normalized_sequence_ratio():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    old_log_prob = torch.zeros_like(mask)
    log_prob = torch.full_like(mask, math.log(1.1)) * mask
    advantages = torch.ones_like(mask)

    loss, clipfrac, _, _ = compute_gspo_policy_loss(
        old_log_prob, log_prob, advantages, mask, cliprange=0.2
    )

    assert torch.allclose(loss, torch.tensor(-1.1), atol=1e-6)
    assert clipfrac.item() == 0.0


def test_gspo_clips_once_per_sequence_and_ignores_padding():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    old_log_prob = torch.zeros_like(mask)
    log_prob = torch.tensor(
        [[math.log(1.5), math.log(1.5), 100.0], [math.log(0.5), 100.0, 100.0]]
    )
    advantages = torch.tensor([[2.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])

    loss, clipfrac, _, lower_clipfrac = compute_gspo_policy_loss(
        old_log_prob, log_prob, advantages, mask, cliprange=0.2
    )

    # Positive sequence: -2 * 1.2; negative sequence: 1 * 0.8.
    assert torch.allclose(loss, torch.tensor((-2.4 + 0.8) / 2), atol=1e-6)
    assert clipfrac.item() == 1.0
    assert lower_clipfrac.item() == 0.0


def test_gspo_keeps_narrow_clip_signal_with_bfloat16_log_probs():
    mask = torch.ones((1, 2), dtype=torch.bfloat16)
    old_log_prob = torch.zeros_like(mask)
    log_prob = torch.full_like(mask, 0.001)
    advantages = torch.ones_like(mask)

    loss, clipfrac, _, _ = compute_gspo_policy_loss(
        old_log_prob, log_prob, advantages, mask, cliprange=0.0003
    )

    assert loss.dtype == torch.float32
    assert torch.allclose(loss, torch.tensor(-1.0003), atol=2e-6)
    assert clipfrac.item() == 1.0
