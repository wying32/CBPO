"""TAO-step parsing and exact-prefix branch selection.

The selector deliberately keeps approximate candidate grouping separate from
exact-prefix credit groups.  Candidate groups are used only to decide where to
spend rollout budget; a local credit group is always identified by the hash of
one concrete token prefix.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_FAILURE_MARKERS = ("timeout", "timed out", "error:", "failed", "failure")


def _decode(tokenizer: Any, token_ids: Iterable[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(ids)


def _words(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard_distance(left: str, right: str) -> float:
    lhs, rhs = _words(left), _words(right)
    if not lhs and not rhs:
        return 0.0
    return 1.0 - len(lhs & rhs) / max(1, len(lhs | rhs))


def _pairwise_distance(values: list[str]) -> float:
    pairs = list(combinations(values, 2))
    if not pairs:
        return 0.0
    return float(np.mean([_jaccard_distance(a, b) for a, b in pairs]))


def hash_token_prefix(token_ids: Iterable[int]) -> str:
    """Return a stable exact-prefix group id without decoding tokens."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def hash_observation(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def word_at_token_boundary(
    tokenizer: Any,
    token_ids: Iterable[int],
    boundary: int,
    decoded_text: str | None = None,
) -> str:
    """Return the lexical word at (or immediately after) a token boundary.

    Entropy-balanced branching stores token offsets rather than text offsets.
    Decoding the complete response and its prefix lets diagnostics reconstruct
    full words even when the selected token is a BPE/SentencePiece fragment.
    """

    ids = [int(token_id) for token_id in token_ids]
    boundary = max(0, min(int(boundary), len(ids)))
    text = decoded_text if decoded_text is not None else _decode(tokenizer, ids)
    prefix_text = _decode(tokenizer, ids[:boundary])
    char_boundary = min(len(prefix_text), len(text))
    matches = list(_TOKEN_RE.finditer(text))
    for match in matches:
        if match.start() <= char_boundary < match.end():
            return match.group(0)
        if match.start() >= char_boundary:
            return match.group(0)
    return ""


def collect_entropy_balanced_candidates(
    entropy_events: list[list[dict[str, Any]]],
    prompt_indices: Iterable[int],
    token_rows: Iterable[Iterable[int]],
    *,
    base_probability: float = 0.2,
    entropy_scale: float = 0.5,
    branch_threshold: float = 0.25,
    max_branch_nodes_per_path: int = 3,
) -> list[dict[str, Any]]:
    """Build the deduplicated, path-capped candidate pool used by CBPO.

    Records with ``passes_raw_threshold`` false remain in the returned pool for
    diagnostics, but they cannot receive budget because balanced probability
    never exceeds raw probability. The paper word-cloud exporter keeps only
    records that pass this flag.
    """

    prompt_values = [int(value) for value in prompt_indices]
    rows = [[int(token) for token in row] for row in token_rows]
    if not (len(entropy_events) == len(prompt_values) == len(rows)):
        raise ValueError("entropy_events, prompt_indices and token_rows must have equal length")
    if branch_threshold <= base_probability:
        raise ValueError("branch_threshold (tau) must exceed base_probability (alpha)")
    if max_branch_nodes_per_path < 1:
        raise ValueError("max_branch_nodes_per_path must be positive")

    candidates: list[dict[str, Any]] = []
    for row_index, events in enumerate(entropy_events):
        events_by_boundary: dict[int, tuple[float, float, int, dict[str, Any]]] = {}
        for event in events or []:
            boundary = int(event.get("boundary_response_idx", -1))
            if boundary <= 0 or boundary > len(rows[row_index]):
                continue
            delta = float(event.get("entropy_delta", 0.0))
            raw_probability = max(0.0, min(1.0, base_probability + entropy_scale * delta))
            item = (raw_probability, delta, boundary, event)
            previous = events_by_boundary.get(boundary)
            if previous is None or item[:2] > previous[:2]:
                events_by_boundary[boundary] = item

        ranked_events = sorted(
            events_by_boundary.values(), key=lambda item: (item[0], item[1], -item[2]), reverse=True
        )[:max_branch_nodes_per_path]
        for rank, (raw_probability, delta, boundary, event) in enumerate(ranked_events, start=1):
            prefix = rows[row_index][:boundary]
            candidates.append(
                {
                    "prompt_index": prompt_values[row_index],
                    "source_index": row_index,
                    "boundary_response_idx": boundary,
                    "entropy_delta": delta,
                    "raw_branch_probability": raw_probability,
                    "entropy": float(event.get("entropy", 0.0)),
                    "root_entropy": float(event.get("root_entropy", 0.0)),
                    "prefix_token_ids": prefix,
                    "branch_group_id": f"{prompt_values[row_index]}:{hash_token_prefix(prefix)}",
                    "candidate_rank_on_path": rank,
                    "passes_raw_threshold": raw_probability > branch_threshold,
                    "branch_count": 0,
                }
            )
    return candidates


def select_entropy_balanced_boundaries(
    entropy_events: list[list[dict[str, Any]]],
    prompt_indices: Iterable[int],
    token_rows: Iterable[Iterable[int]],
    *,
    branch_budget_per_prompt: int,
    base_probability: float = 0.2,
    entropy_scale: float = 0.5,
    branch_threshold: float = 0.25,
    path_power_decay_rho: float = 0.2,
    node_power_decay_rho: float = 0.2,
    max_branch_nodes_per_path: int = 3,
    min_branch_node_distance: int = 64,
    max_branches_per_node: int | None = 3,
    max_branches_per_path: int | None = 4,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Allocate multi-node exact-prefix branches with hierarchical balancing.

    Each event is an entropy measurement relative to the root-generation
    entropy of the same trajectory. Candidate boundaries lie on a trajectory-
    wide grid with ``min_branch_node_distance`` spacing. Each initial
    trajectory may retain several high-entropy nodes, and each retained node
    may receive several continuations.

    Branches are allocated one at a time with

    ``P_bal = P_raw / (1 + L_path) ** rho_path / (1 + l_node) ** rho_node``.

    The two discounts keep budget from collapsing onto either one trajectory or
    one node.  The path/node hard caps are independent because power decay never
    reaches zero.  Requiring ``tau > alpha`` means a boundary with non-increasing
    entropy can never win a branch.

    Returns selected path/boundary records and the number of unused slots per
    prompt.  Unused slots must be filled with independent global rollouts rather
    than forced low-quality branches.
    """

    prompt_values = [int(value) for value in prompt_indices]
    rows = [[int(token) for token in row] for row in token_rows]
    if not (len(entropy_events) == len(prompt_values) == len(rows)):
        raise ValueError("entropy_events, prompt_indices and token_rows must have equal length")
    if branch_budget_per_prompt < 0:
        raise ValueError("branch_budget_per_prompt must be non-negative")
    if branch_threshold <= base_probability:
        raise ValueError("branch_threshold (tau) must exceed base_probability (alpha)")
    if path_power_decay_rho < 0 or node_power_decay_rho < 0:
        raise ValueError("path/node power decay must be non-negative")
    if max_branch_nodes_per_path < 1:
        raise ValueError("max_branch_nodes_per_path must be positive")
    if min_branch_node_distance < 1:
        raise ValueError("min_branch_node_distance must be positive")
    if max_branches_per_node is not None and max_branches_per_node < 1:
        raise ValueError("max_branches_per_node must be positive when set")
    if max_branches_per_path is not None and max_branches_per_path < 1:
        raise ValueError("max_branches_per_path must be positive when set")

    candidates_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidates = collect_entropy_balanced_candidates(
        entropy_events,
        prompt_values,
        rows,
        base_probability=base_probability,
        entropy_scale=entropy_scale,
        branch_threshold=branch_threshold,
        max_branch_nodes_per_path=max_branch_nodes_per_path,
    )
    for candidate in candidates:
        candidates_by_prompt[int(candidate["prompt_index"])].append(candidate)

    selected: list[dict[str, Any]] = []
    unused_by_prompt: dict[int, int] = {}
    for prompt_index in sorted(set(prompt_values)):
        candidates = candidates_by_prompt.get(prompt_index, [])
        path_branch_counts: Counter[int] = Counter()
        remaining = branch_budget_per_prompt
        while remaining > 0 and candidates:
            scored = []
            for candidate in candidates:
                node_count = int(candidate["branch_count"])
                source_index = int(candidate["source_index"])
                path_count = int(path_branch_counts[source_index])
                if max_branches_per_node is not None and node_count >= max_branches_per_node:
                    continue
                if max_branches_per_path is not None and path_count >= max_branches_per_path:
                    continue
                balanced_probability = (
                    candidate["raw_branch_probability"]
                    / (1.0 + path_count) ** path_power_decay_rho
                    / (1.0 + node_count) ** node_power_decay_rho
                )
                scored.append((balanced_probability, candidate))
            if not scored:
                break
            probability, best = max(
                scored,
                key=lambda item: (item[0], item[1]["entropy_delta"], -item[1]["boundary_response_idx"]),
            )
            if probability <= branch_threshold:
                break
            best["branch_count"] += 1
            path_branch_counts[int(best["source_index"])] += 1
            best["balanced_branch_probability"] = probability
            remaining -= 1

        selected.extend(candidate for candidate in candidates if candidate["branch_count"] > 0)
        unused_by_prompt[prompt_index] = remaining

    return selected, unused_by_prompt


def select_arpo_boundaries(
    entropy_events: list[list[dict[str, Any]]],
    prompt_indices: Iterable[int],
    token_rows: Iterable[Iterable[int]],
    *,
    branch_budget_per_prompt: int,
    base_probability: float = 0.5,
    entropy_scale: float = 0.2,
    branch_threshold: float = 0.5,
    branches_per_event: int = 1,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Select partial-rollout nodes using the ARPO paper rule.

    For each post-tool event, ARPO computes ``P = alpha + beta * delta_H``
    and creates ``Z`` continuations when ``P > tau``. Events are considered
    in trajectory/time order until the per-prompt partial-rollout budget is
    exhausted. Any unused budget is returned for independent global rollouts.
    """

    prompt_values = [int(value) for value in prompt_indices]
    rows = [[int(token) for token in row] for row in token_rows]
    if not (len(entropy_events) == len(prompt_values) == len(rows)):
        raise ValueError("entropy_events, prompt_indices and token_rows must have equal length")
    if branch_budget_per_prompt < 0:
        raise ValueError("branch_budget_per_prompt must be non-negative")
    if not 0.0 <= base_probability <= 1.0:
        raise ValueError("CBPO base_probability (alpha) must be in [0, 1]")
    if entropy_scale < 0.0:
        raise ValueError("CBPO entropy_scale (beta) must be non-negative")
    if not 0.0 <= branch_threshold <= 1.0:
        raise ValueError("CBPO branch_threshold (tau) must be in [0, 1]")
    if branch_threshold < base_probability:
        raise ValueError("CBPO branch_threshold (tau) must be at least base_probability (alpha)")
    if branches_per_event < 1:
        raise ValueError("CBPO branches_per_event (Z) must be positive")

    candidates_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_index, events in enumerate(entropy_events):
        seen_boundaries = set()
        for event in sorted(events or [], key=lambda item: int(item.get("boundary_response_idx", -1))):
            boundary = int(event.get("boundary_response_idx", -1))
            if boundary <= 0 or boundary > len(rows[row_index]) or boundary in seen_boundaries:
                continue
            seen_boundaries.add(boundary)
            delta = float(event.get("entropy_delta", 0.0))
            probability = max(0.0, min(1.0, base_probability + entropy_scale * delta))
            if probability <= branch_threshold:
                continue
            prefix = rows[row_index][:boundary]
            candidates_by_prompt[prompt_values[row_index]].append(
                {
                    "prompt_index": prompt_values[row_index],
                    "source_index": row_index,
                    "boundary_response_idx": boundary,
                    "entropy_delta": delta,
                    "raw_branch_probability": probability,
                    "balanced_branch_probability": probability,
                    "entropy": float(event.get("entropy", 0.0)),
                    "root_entropy": float(event.get("root_entropy", 0.0)),
                    "prefix_token_ids": prefix,
                    "branch_group_id": f"{prompt_values[row_index]}:{hash_token_prefix(prefix)}",
                    "branch_count": 0,
                }
            )

    selected: list[dict[str, Any]] = []
    unused_by_prompt: dict[int, int] = {}
    for prompt_index in sorted(set(prompt_values)):
        remaining = branch_budget_per_prompt
        candidates = sorted(
            candidates_by_prompt.get(prompt_index, []),
            key=lambda item: (item["boundary_response_idx"], item["source_index"]),
        )
        for candidate in candidates:
            if remaining <= 0:
                break
            candidate["branch_count"] = min(branches_per_event, remaining)
            remaining -= int(candidate["branch_count"])
            selected.append(candidate)
        unused_by_prompt[prompt_index] = remaining

    return selected, unused_by_prompt


def _closed_actions(text: str, action_types: tuple[str, ...]) -> list[tuple[int, int, str, str]]:
    actions: list[tuple[int, int, str, str]] = []
    for action_type in action_types:
        pattern = re.compile(
            rf"<{re.escape(action_type)}>(.*?)</{re.escape(action_type)}>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            actions.append((match.start(), match.end(), action_type, match.group(1).strip()))
    return sorted(actions)


def _closing_tag_token_end(tokenizer: Any, token_ids: list[int], start: int, end: int, tag: str) -> int:
    """Find the first token boundary whose decoded prefix closes ``tag``."""

    marker = f"</{tag}>".lower()
    for token_end in range(start + 1, end + 1):
        if marker in _decode(tokenizer, token_ids[start:token_end]).lower():
            return token_end
    return end


def parse_tao_trajectory(
    token_ids: Iterable[int],
    model_mask: Iterable[int],
    response_mask: Iterable[int],
    tokenizer: Any,
    action_types: tuple[str, ...] = ("search", "python", "answer"),
) -> list[dict[str, Any]]:
    """Parse valid TAO steps while preserving response-token boundaries.

    ``model_mask`` is authoritative for provenance: zero-runs inside the valid
    response are tool observations, never actor tokens.  A tool step is valid
    only when it contains one closed action and a closed ``result`` block.
    Incomplete/multiple actions are ignored as candidate steps.  Tokens after a
    closed answer are not represented by a step and therefore receive no local
    branch boundary.
    """

    ids = [int(value) for value in token_ids]
    provenance = [int(value) for value in model_mask]
    validity = [int(value) for value in response_mask]
    if not (len(ids) == len(provenance) == len(validity)):
        raise ValueError("token_ids, model_mask and response_mask must have equal length")

    valid_positions = [idx for idx, flag in enumerate(validity) if flag]
    valid_end = valid_positions[-1] + 1 if valid_positions else 0
    ids, provenance = ids[:valid_end], provenance[:valid_end]

    runs: list[tuple[int, int, int]] = []
    start = 0
    while start < valid_end:
        flag = provenance[start]
        end = start + 1
        while end < valid_end and provenance[end] == flag:
            end += 1
        runs.append((start, end, flag))
        start = end

    steps: list[dict[str, Any]] = []
    previous_observation = ""
    previous_action_type = "initial"
    step_index = 0
    run_index = 0
    while run_index < len(runs):
        model_start, model_end, is_model = runs[run_index]
        if not is_model:
            run_index += 1
            continue

        model_text = _decode(tokenizer, ids[model_start:model_end])
        actions = _closed_actions(model_text, action_types)
        tool_actions = [item for item in actions if item[2] != "answer"]
        answer_actions = [item for item in actions if item[2] == "answer"]

        # An answer terminates the episode. Multiple actions in the same model
        # span are dirty format and are not valid TAO steps.
        if len(actions) == 1 and answer_actions:
            action = answer_actions[0]
            answer_end = _closing_tag_token_end(tokenizer, ids, model_start, model_end, "answer")
            stray_observation = run_index + 1 < len(runs) and runs[run_index + 1][2] == 0
            steps.append(
                {
                    "step_index": step_index,
                    "step_start_token_idx": model_start,
                    "step_end_token_idx": answer_end,
                    "model_span": (model_start, answer_end),
                    "observation_span": (answer_end, answer_end),
                    "model_token_mask": (model_start, answer_end),
                    "observation_mask": (answer_end, answer_end),
                    "thought_text": model_text[: action[0]].strip(),
                    "action_type": "answer",
                    "action_content": action[3],
                    "observation_text": "",
                    "previous_observation_hash": hash_observation(previous_observation),
                    "previous_observation_text": previous_observation,
                    "state_source": previous_action_type,
                    "parse_valid": not stray_observation,
                    "tool_failure": False,
                    "terminal": True,
                }
            )
            break

        has_observation = run_index + 1 < len(runs) and runs[run_index + 1][2] == 0
        if len(actions) != 1 or len(tool_actions) != 1 or not has_observation:
            run_index += 1
            continue

        obs_start, obs_end, _ = runs[run_index + 1]
        observation_text = _decode(tokenizer, ids[obs_start:obs_end])
        result_valid = bool(
            re.search(r"<result(?:\s[^>]*)?>", observation_text, re.IGNORECASE)
            and re.search(r"</result>", observation_text, re.IGNORECASE)
        )
        if not result_valid:
            run_index += 2
            continue

        action = tool_actions[0]
        lowered_observation = observation_text.lower()
        steps.append(
            {
                "step_index": step_index,
                "step_start_token_idx": model_start,
                "step_end_token_idx": obs_end,
                "model_span": (model_start, model_end),
                "observation_span": (obs_start, obs_end),
                "model_token_mask": (model_start, model_end),
                "observation_mask": (obs_start, obs_end),
                "thought_text": model_text[: action[0]].strip(),
                "action_type": action[2],
                "action_content": action[3],
                "observation_text": observation_text,
                "previous_observation_hash": hash_observation(previous_observation),
                "previous_observation_text": previous_observation,
                "state_source": previous_action_type,
                "parse_valid": True,
                "tool_failure": any(marker in lowered_observation for marker in _FAILURE_MARKERS),
                "terminal": False,
            }
        )
        previous_observation = observation_text
        previous_action_type = action[2]
        step_index += 1
        run_index += 2

    return steps


def compute_pre_step_disagreement(
    steps: list[dict[str, Any]], alpha: float = 0.2, beta: float = 0.5, content_weight: float = 0.5
) -> dict[str, float]:
    """Compute the three disagreement components for a candidate group."""

    if not steps:
        return {"think": 0.0, "action_type": 0.0, "action_content": 0.0, "total": 0.0}

    think = _pairwise_distance([step["thought_text"] for step in steps])
    counts = Counter(step["action_type"] for step in steps)
    probabilities = [count / len(steps) for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    action_type = entropy / math.log(3.0)

    within_type: list[float] = []
    for action_name in counts:
        contents = [step["action_content"] for step in steps if step["action_type"] == action_name]
        pairs = list(combinations(contents, 2))
        within_type.extend(_jaccard_distance(a, b) for a, b in pairs)
    action_content = float(np.mean(within_type)) if within_type else 0.0
    total = alpha * think + beta * action_type + content_weight * action_content
    return {"think": think, "action_type": action_type, "action_content": action_content, "total": total}


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std())
    if std < 1e-12:
        return [0.0] * len(values)
    return ((array - array.mean()) / std).tolist()


def select_tao_boundaries(
    trajectories: list[list[dict[str, Any]]],
    rewards: Iterable[float],
    prompt_indices: Iterable[int],
    token_rows: Iterable[Iterable[int]],
    *,
    alpha: float = 0.2,
    beta: float = 0.5,
    content_weight: float = 0.5,
    reward_weight: float = 0.5,
    cost_weight: float = 0.5,
    min_candidate_group_size: int = 2,
) -> list[dict[str, Any]]:
    """Select at most one additive-TEB boundary for each prompt."""

    reward_values = [float(value) for value in rewards]
    prompt_values = [int(value) for value in prompt_indices]
    rows = [[int(token) for token in row] for row in token_rows]
    if not (len(trajectories) == len(reward_values) == len(prompt_values) == len(rows)):
        raise ValueError("all trajectory-level inputs must have equal length")

    grouped: dict[tuple[int, int, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, steps in enumerate(trajectories):
        for step in steps:
            if step.get("parse_valid", False):
                key = (prompt_values[row_index], int(step["step_index"]), step["previous_observation_hash"])
                grouped[key].append((row_index, step))

    candidates_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (prompt_index, step_index, observation_hash), members in grouped.items():
        if len(members) < min_candidate_group_size:
            continue
        member_steps = [member[1] for member in members]
        disagreement = compute_pre_step_disagreement(
            member_steps, alpha=alpha, beta=beta, content_weight=content_weight
        )
        member_rewards = np.asarray([reward_values[member[0]] for member in members], dtype=np.float64)
        remaining_fractions = []
        for row_index, _ in members:
            total_steps = max(1, len(trajectories[row_index]))
            remaining_fractions.append((total_steps - step_index) / total_steps)
        candidates_by_prompt[prompt_index].append(
            {
                "prompt_index": prompt_index,
                "step_index": step_index,
                "observation_hash": observation_hash,
                "members": members,
                "d_pre": disagreement["total"],
                "reward_contrast": float(member_rewards.std()),
                "cost": float(np.mean(remaining_fractions)),
                "disagreement": disagreement,
            }
        )

    selected: list[dict[str, Any]] = []
    for prompt_index, candidates in candidates_by_prompt.items():
        z_disagreement = _zscore([candidate["d_pre"] for candidate in candidates])
        z_reward = _zscore([candidate["reward_contrast"] for candidate in candidates])
        for candidate, z_d, z_r in zip(candidates, z_disagreement, z_reward):
            candidate["branch_score"] = z_d + reward_weight * z_r - cost_weight * candidate["cost"]

        best = max(candidates, key=lambda item: (item["branch_score"], -item["step_index"]))
        # Pick the member closest to the candidate-group mean reward. This avoids
        # choosing an exact prefix merely because its already observed suffix won.
        mean_reward = float(np.mean([reward_values[index] for index, _ in best["members"]]))
        source_index, source_step = min(
            best["members"], key=lambda item: (abs(reward_values[item[0]] - mean_reward), item[0])
        )
        boundary = int(source_step["step_start_token_idx"])
        prefix = rows[source_index][:boundary]
        best.update(
            {
                "source_index": source_index,
                "boundary_response_idx": boundary,
                "prefix_token_ids": prefix,
                # The response prefix is namespaced by prompt because the full
                # exact state is prompt_tokens + response_prefix.
                "branch_group_id": f"{prompt_index}:{hash_token_prefix(prefix)}",
            }
        )
        selected.append(best)

    # The prompt boundary is always legal.  Falling back to it keeps the
    # per-prompt rollout count stable when dirty trajectories contain no valid
    # TAO candidate, while exact-prefix credit remains well-defined.
    selected_prompts = {item["prompt_index"] for item in selected}
    for prompt_index in sorted(set(prompt_values) - selected_prompts):
        source_index = prompt_values.index(prompt_index)
        prefix: list[int] = []
        selected.append(
            {
                "prompt_index": prompt_index,
                "step_index": 0,
                "observation_hash": hash_observation(""),
                "members": [(source_index, {})],
                "d_pre": 0.0,
                "reward_contrast": 0.0,
                "cost": 1.0,
                "disagreement": {"think": 0.0, "action_type": 0.0, "action_content": 0.0, "total": 0.0},
                "branch_score": -cost_weight,
                "source_index": source_index,
                "boundary_response_idx": 0,
                "prefix_token_ids": prefix,
                "branch_group_id": f"{prompt_index}:{hash_token_prefix(prefix)}",
                "fallback_prompt_boundary": True,
            }
        )

    return sorted(selected, key=lambda item: item["prompt_index"])
