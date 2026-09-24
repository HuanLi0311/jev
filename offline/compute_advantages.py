#!/usr/bin/env python3
"""Compute reward-to-go and empirical group advantages without training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", type=Path)
    parser.add_argument("--source", default="terminal")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judgments", type=Path, help="Complete judge JSONL to join by trajectory and step")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def reward_sequence(trajectory: dict[str, Any], source: str) -> list[float]:
    if source == "terminal" or source.startswith("verifier."):
        rewards = [0.0] * len(trajectory["steps"])
        if rewards:
            key = "terminal_reward" if source == "terminal" else source[len("verifier.") :]
            rewards[-1] = float(trajectory["verifier"][key])
        return rewards
    return [float(step.get("rewards", {}).get(source, 0.0)) for step in trajectory["steps"]]


def reward_to_go(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def add_advantages(
    trajectories: list[dict[str, Any]], source: str, gamma: float
) -> list[dict[str, Any]]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        rewards = reward_sequence(trajectory, source)
        groups[trajectory["group_id"]].append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "rewards": rewards,
                "returns": reward_to_go(rewards, gamma),
            }
        )

    output = []
    for group_id, members in groups.items():
        trajectory_returns = [
            member["returns"][0] if member["returns"] else 0.0 for member in members
        ]
        trajectory_mean = statistics.fmean(trajectory_returns)
        trajectory_std = statistics.pstdev(trajectory_returns)
        for member in members:
            member_trajectory_return = member["returns"][0] if member["returns"] else 0.0
            trajectory_peers = [
                other["returns"][0] if other["returns"] else 0.0
                for other in members
                if other is not member
            ]
            trajectory_rloo = (
                member_trajectory_return - statistics.fmean(trajectory_peers)
                if trajectory_peers
                else None
            )
            trajectory_z = (
                (member_trajectory_return - trajectory_mean) / trajectory_std
                if trajectory_std > 1e-12
                else 0.0
            )
            steps = []
            for step, (reward, return_) in enumerate(
                zip(member["rewards"], member["returns"])
            ):
                peers = [
                    other["returns"][step]
                    for other in members
                    if other is not member and step < len(other["returns"])
                ]
                active = [member["returns"][step], *peers]
                mean = statistics.fmean(active)
                std = statistics.pstdev(active)
                steps.append(
                    {
                        "step": step,
                        "reward": reward,
                        "return": return_,
                        # Standard inference-only group advantage: one trajectory
                        # return compared with peer rollout returns, broadcast to
                        # every decision as it would be in outcome-reward RL.
                        "rloo_advantage": trajectory_rloo,
                        "group_z_advantage": trajectory_z,
                        # Diagnostic process credit: compare reward-to-go at the
                        # same decision index among peers still active there.
                        "stepwise_rloo_advantage": return_ - statistics.fmean(peers)
                        if peers
                        else None,
                        "stepwise_group_z_advantage": (return_ - mean) / std
                        if std > 1e-12
                        else 0.0,
                        "active_group_size": len(active),
                        "group_size": len(members),
                    }
                )
            output.append(
                {
                    "schema_version": 1,
                    "trajectory_id": member["trajectory_id"],
                    "group_id": group_id,
                    "reward_source": source,
                    "gamma": gamma,
                    "steps": steps,
                }
            )
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attach_judgments(trajectories: list[dict[str, Any]], judgments: list[dict[str, Any]], source: str) -> None:
    reward_field = {
        "jev_transition_v1": "jev_reward",
        "jev_transition_v2": "jev_reward",
        "llm_transition_v1": "llm_reward",
    }[source]
    steps = {
        (trajectory["trajectory_id"], index): step
        for trajectory in trajectories
        for index, step in enumerate(trajectory["steps"])
    }
    if len(steps) != sum(len(trajectory["steps"]) for trajectory in trajectories):
        raise ValueError("duplicate trajectory id")
    seen = set()
    for judgment in judgments:
        key = judgment["trajectory_id"], judgment["step_index"]
        reward = float(judgment[reward_field])
        if key not in steps or key in seen or judgment["rubric_version"] != source:
            raise ValueError(f"unknown, duplicate, or wrong-rubric judgment: {key}")
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError(f"invalid judge reward: {key}")
        steps[key].setdefault("rewards", {})[source] = reward
        seen.add(key)
    if seen != set(steps):
        raise ValueError(f"incomplete judge annotation: {len(seen)}/{len(steps)} steps")


def self_check() -> None:
    trajectories = [
        {
            "trajectory_id": "a",
            "group_id": "g",
            "verifier": {"terminal_reward": 1.0},
            "steps": [{}, {}],
        },
        {
            "trajectory_id": "b",
            "group_id": "g",
            "verifier": {"terminal_reward": 0.0},
            "steps": [{}, {}],
        },
    ]
    records = add_advantages(trajectories, "terminal", 1.0)
    assert records[0]["steps"][0]["return"] == 1.0
    assert records[0]["steps"][0]["rloo_advantage"] == 1.0
    assert records[1]["steps"][0]["rloo_advantage"] == -1.0
    assert records[0]["steps"][0]["stepwise_rloo_advantage"] == 1.0
    labels = [
        {"trajectory_id": tid, "step_index": i, "jev_reward": score, "rubric_version": "jev_transition_v1"}
        for tid, values in (("a", [0.5, -0.5]), ("b", [0.0, 0.0]))
        for i, score in enumerate(values)
    ]
    attach_judgments(trajectories, labels, "jev_transition_v1")
    assert add_advantages(trajectories, "jev_transition_v1", 1.0)[0]["steps"][0]["return"] == 0.0
    attach_judgments(trajectories, [
        {**row, "llm_reward": row["jev_reward"], "rubric_version": "llm_transition_v1"}
        for row in labels
    ], "llm_transition_v1")
    assert add_advantages(trajectories, "llm_transition_v1", 1.0)[0]["steps"][0]["return"] == 0.0
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.trajectories is None or args.output is None:
        raise SystemExit("trajectories and --output are required")
    trajectories = read_jsonl(args.trajectories)
    if args.judgments:
        if args.source not in ("jev_transition_v1", "jev_transition_v2", "llm_transition_v1"):
            raise SystemExit("--judgments requires a transition judge source")
        attach_judgments(trajectories, read_jsonl(args.judgments), args.source)
    records = add_advantages(trajectories, args.source, args.gamma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
