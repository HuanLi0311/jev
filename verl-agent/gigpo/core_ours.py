"""Exact-first process credit for variable-length agent trajectories."""

from __future__ import annotations

import collections
import re

import numpy as np
import torch


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
ADMISSIBLE_RE = re.compile(
    r"admissible actions of the current situation are:\s*\[(.*?)\]\.",
    re.I | re.S,
)
GOAL_RE = re.compile(r"your task is to:\s*([^\n]+)", re.I)


def _action(text):
    matches = ACTION_RE.findall(str(text))
    # Match ALFWorld's projection: the environment executes the first action tag.
    return re.sub(r"\s+", " ", (matches[0] if matches else str(text))).strip().lower()


def _role(action):
    for prefix, role in (
        ("go to ", "navigate"), ("take ", "take"), ("put ", "put"),
        ("open ", "open"), ("close ", "close"), ("heat ", "heat"),
        ("clean ", "clean"), ("cool ", "cool"), ("examine ", "examine"),
        ("use ", "use"), ("inventory", "inventory"), ("look", "look"),
    ):
        if action.startswith(prefix):
            return role
    return action.split(maxsplit=1)[0] if action else "parse_error"


def _entity(action):
    value = re.sub(
        r"^(?:go to|take|put|open|close|heat|clean|cool|examine|use)\s+",
        "", action,
    )
    value = re.split(r"\s+(?:from|in/on|with)\s+", value, maxsplit=1)[0]
    return re.sub(r"\s+\d+$", "", value.strip())


def _admissible(prompt):
    match = ADMISSIBLE_RE.search(str(prompt))
    return tuple(re.findall(r"'([^']+)'", match.group(1))) if match else ()


def _goal(prompt):
    match = GOAL_RE.search(str(prompt))
    return match.group(1).strip().lower() if match else ""


def _build_records(
    token_level_rewards,
    step_rewards,
    action_valids,
    visible_prompts,
    anchor_obs,
    action_text,
    task_uids,
    traj_uids,
    turn_indices,
):
    scores = token_level_rewards.sum(dim=-1).detach().cpu().numpy()
    records = {}
    batch_keys = []
    for batch_index, (task, traj, turn) in enumerate(
        zip(task_uids, traj_uids, turn_indices, strict=True)
    ):
        key = (str(task), str(traj), int(turn))
        batch_keys.append(key)
        if key in records:
            continue
        records[key] = {
            "key": key,
            "task": str(task),
            "traj": str(traj),
            "turn": int(turn),
            "prompt": str(visible_prompts[batch_index]),
            "obs": str(anchor_obs[batch_index]),
            "action": _action(action_text[batch_index]),
            "outcome": float(scores[batch_index]),
            "step_reward": float(step_rewards[batch_index]),
            "action_valid": bool(action_valids[batch_index]),
        }

    trajectories = collections.defaultdict(list)
    for record in records.values():
        trajectories[(record["task"], record["traj"])].append(record)

    for rows in trajectories.values():
        rows.sort(key=lambda item: item["turn"])
        seen = set()
        held = None
        location = None
        previous = ("start",)
        for position, record in enumerate(rows):
            admissible = tuple(action.lower() for action in _admissible(record["prompt"]))
            roles = tuple(sorted({_role(action) for action in admissible}))
            goal_text = _goal(record["prompt"]).replace(" ", "")
            held_role = (
                "empty" if held is None
                else "goal_object" if held.replace(" ", "") in goal_text
                else "other_object"
            )
            if set(roles) & {"heat", "clean", "cool", "use"}:
                location_role = "operation_tool"
            elif location and location.replace(" ", "") in goal_text:
                location_role = "goal_location"
            elif any(
                _role(action) == "take"
                and _entity(action).replace(" ", "") in goal_text
                for action in admissible
            ):
                location_role = "target_source"
            elif "put" in roles:
                location_role = "candidate_destination"
            else:
                location_role = "other_location"

            navigation_novelty = previous[2] if previous[0] == "navigate" else None
            record["exact"] = record["prompt"]
            record["struct"] = (
                held_role, location_role, roles, previous[0], navigation_novelty
            )
            record["position"] = position

            next_obs = rows[position + 1]["obs"] if position + 1 < len(rows) else None
            valid = record["action"] in admissible
            novel = next_obs not in seen if next_obs is not None else False
            changed = next_obs != record["obs"] if next_obs is not None else False
            previous = (
                _role(record["action"]),
                "valid" if valid else "invalid",
                "new" if novel else "repeat",
                "changed" if changed else "same",
            )
            seen.add(record["obs"])
            if next_obs is not None:
                seen.add(next_obs)
            entity = _entity(record["action"])
            if previous[0] == "navigate":
                location = entity
            elif previous[0] == "take":
                held = entity
            elif previous[0] == "put":
                held = None

    return records, trajectories, batch_keys


def compute_ours_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_rewards: np.ndarray,
    action_valids: np.ndarray,
    visible_prompts: np.ndarray,
    anchor_obs: np.ndarray,
    action_text: np.ndarray,
    task_uids: np.ndarray,
    traj_uids: np.ndarray,
    turn_indices: np.ndarray,
    minimum_peers: int = 3,
    gamma: float = 0.95,
    invalid_action_penalty: float = 0.1,
    process_weight: float = 1.0,
    outcome_weight: float = 0.2,
    advantage_clip: float = 3.0,
    epsilon: float = 1e-6,
):
    """Return token advantages and mechanism diagnostics.

    Duplicate (trajectory, turn) rows introduced for batch divisibility receive the
    same result but count once in every statistic.
    """
    if minimum_peers < 1 or not 0 <= gamma <= 1 or invalid_action_penalty < 0 or advantage_clip <= 0:
        raise ValueError("invalid process-reward parameters")

    records, trajectories, batch_keys = _build_records(
        token_level_rewards,
        step_rewards,
        action_valids,
        visible_prompts,
        anchor_obs,
        action_text,
        task_uids,
        traj_uids,
        turn_indices,
    )
    by_task = collections.defaultdict(dict)
    for record in records.values():
        previous = by_task[record["task"]].setdefault(record["traj"], record["outcome"])
        if abs(previous - record["outcome"]) > epsilon:
            raise ValueError("terminal outcome is inconsistent within a trajectory")

    exact_groups = collections.defaultdict(list)
    for record in records.values():
        exact_groups[(record["task"], record["exact"])].append(record)
    exact_supported = {
        key: (
            len({item["traj"] for item in items}) >= minimum_peers + 1
            and len({item["action"] for item in items}) >= 2
        )
        for key, items in exact_groups.items()
    }

    groups = collections.defaultdict(list)
    exact_selected = 0
    for record in records.values():
        exact_key = (record["task"], record["exact"])
        if exact_supported[exact_key]:
            record["group"] = ("exact",) + exact_key
            exact_selected += 1
        else:
            record["group"] = ("struct", record["task"], record["struct"])
        groups[record["group"]].append(record)

    for rows in trajectories.values():
        future_return = 0.0
        for record in reversed(rows):
            future_return = (
                record["step_reward"]
                - invalid_action_penalty * (not record["action_valid"])
                + gamma * future_return
            )
            record["return"] = future_return

    rewards = {}
    locally_comparable = set()
    for group_items in groups.values():
        for record in group_items:
            peer_trajectories = {
                item["traj"] for item in group_items
                if item["traj"] != record["traj"] and item["action"] != record["action"]
            }
            if len(peer_trajectories) < minimum_peers:
                continue
            locally_comparable.add(record["key"])
            rewards[record["key"]] = record["return"]

    process_advantages = {}
    rewarded_groups = collections.defaultdict(list)
    for key, reward in rewards.items():
        rewarded_groups[records[key]["group"]].append(records[key])
    for items in rewarded_groups.values():
        eligible = {
            item["key"]
            for item in items
            if len({
                peer["traj"] for peer in items
                if peer["traj"] != item["traj"]
                and peer["action"] != item["action"]
            }) >= minimum_peers
        }
        if not eligible:
            continue
        visits = collections.Counter(item["traj"] for item in items)
        weights = np.asarray([1.0 / visits[item["traj"]] for item in items])
        values = np.asarray([rewards[item["key"]] for item in items])
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        if std <= epsilon:
            continue
        for item, value in zip(items, values, strict=True):
            if item["key"] in eligible:
                process_advantages[item["key"]] = float((value - mean) / (std + epsilon))

    scalar_advantages = {}
    process_broadcast = collections.Counter()
    for (task, _traj), rows in trajectories.items():
        outcomes = list(by_task[task].values())
        outcome_mean = float(np.mean(outcomes))
        outcome_std = float(np.std(outcomes))
        outcome_advantage = (
            (rows[0]["outcome"] - outcome_mean) / (outcome_std + epsilon)
            if outcome_std > epsilon else 0.0
        )
        accumulated = [0.0] * len(rows)
        checkpoints = [
            (record["position"], process_advantages[record["key"]])
            for record in rows if record["key"] in process_advantages
        ]
        previous_checkpoint = -1
        last_negative = -1
        for position, advantage in checkpoints:
            start = last_negative + 1 if advantage > 0 else previous_checkpoint + 1
            for target in range(start, position + 1):
                accumulated[target] += advantage
                process_broadcast[rows[target]["key"]] += 1
            if advantage < 0:
                last_negative = position
            previous_checkpoint = position
        for record, process_advantage in zip(rows, accumulated, strict=True):
            scalar_advantages[record["key"]] = float(np.clip(
                outcome_weight * outcome_advantage + process_weight * process_advantage,
                -advantage_clip,
                advantage_clip,
            ))

    result = torch.zeros_like(response_mask, dtype=torch.float32)
    for batch_index, key in enumerate(batch_keys):
        result[batch_index] = response_mask[batch_index] * scalar_advantages[key]

    token_counts = {}
    for batch_index, key in enumerate(batch_keys):
        token_counts.setdefault(key, float(response_mask[batch_index].sum()))

    total = max(len(records), 1)
    total_tokens = max(sum(token_counts.values()), 1.0)
    stats = {
        "transitions": float(len(records)),
        "exact_selected_fraction": exact_selected / total,
        "comparable_fraction": len(locally_comparable) / total,
        "rewarded_fraction": len(rewards) / total,
        "nonzero_checkpoint_fraction": len(process_advantages) / total,
        "process_broadcast_transition_fraction": len(process_broadcast) / total,
        "process_broadcast_fraction": sum(
            token_counts[key] for key in process_broadcast
        ) / total_tokens,
        "mean_abs_advantage": float(result.abs().sum() / response_mask.sum().clamp_min(1)),
    }

    max_length_by_task = collections.defaultdict(int)
    for (task, _traj), rows in trajectories.items():
        max_length_by_task[task] = max(max_length_by_task[task], len(rows))

    partitions = collections.defaultdict(list)
    for (task, _traj), rows in trajectories.items():
        ratio = len(rows) / max_length_by_task[task]
        length_bin = "short" if ratio <= 0.5 else "medium" if ratio <= 0.8 else "long"
        for record in rows:
            quintile = min(4, 5 * record["position"] // len(rows))
            partitions[f"progress_q{quintile + 1}"].append(record["key"])
            partitions[f"length_{length_bin}"].append(record["key"])

    for partition in [
        "progress_q1", "progress_q2", "progress_q3", "progress_q4", "progress_q5",
        "length_short", "length_medium", "length_long",
    ]:
        keys = partitions[partition]
        count = len(keys)
        stats[f"{partition}_transitions"] = float(count)
        stats[f"{partition}_comparable_fraction"] = (
            sum(key in locally_comparable for key in keys) / count if count else 0.0
        )
        stats[f"{partition}_rewarded_fraction"] = (
            sum(key in rewards for key in keys) / count if count else 0.0
        )
        stats[f"{partition}_checkpoint_fraction"] = (
            sum(key in process_advantages for key in keys) / count if count else 0.0
        )
        partition_tokens = sum(token_counts[key] for key in keys)
        stats[f"{partition}_broadcast_fraction"] = (
            sum(token_counts[key] for key in keys if key in process_broadcast)
            / partition_tokens if partition_tokens else 0.0
        )
    return result, result, stats
