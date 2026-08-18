# BC-CBPO Hydra Overrides

Append these overrides to an existing CBPO/TEB training command to enable
entropy-balanced CBV credit assignment:

```bash
algorithm.adv_estimator=grpo \
actor_rollout_ref.rollout.teb.enabled=True \
actor_rollout_ref.rollout.teb.base_probability=0.2 \
actor_rollout_ref.rollout.teb.entropy_scale=0.5 \
actor_rollout_ref.rollout.teb.branch_threshold=0.25 \
actor_rollout_ref.rollout.teb.entropy_window=20 \
actor_rollout_ref.rollout.teb.entropy_top_k=10 \
actor_rollout_ref.rollout.teb.max_window_candidates_per_chunk=3 \
actor_rollout_ref.rollout.teb.max_branch_nodes_per_path=3 \
actor_rollout_ref.rollout.teb.min_branch_node_distance=64 \
actor_rollout_ref.rollout.teb.path_power_decay_rho=0.2 \
actor_rollout_ref.rollout.teb.node_power_decay_rho=0.2 \
actor_rollout_ref.rollout.teb.max_branches_per_node=3 \
actor_rollout_ref.rollout.teb.max_branches_per_path=4 \
algorithm.bc_cbpo.enabled=True \
algorithm.bc_cbpo.alpha=0.2 \
algorithm.bc_cbpo.phi=2.0 \
algorithm.bc_cbpo.cbv_mode=std \
algorithm.bc_cbpo.reward_std_threshold=1.0e-6
```

Entropy is used only for branch selection. Each initial trajectory may retain
several separated nodes, and each node may receive several branches. The
balanced score is
`clip(alpha + gamma * dH) / (1 + L_path)^rho_path / (1 + l_node)^rho_node`
with independent node/path hard caps. `branch_threshold` (tau) must exceed
`base_probability` (alpha). Credit assignment uses normalized CBV with signed
clipping and hierarchical non-overlapping parent segments; no second
token-entropy advantage term is applied.
