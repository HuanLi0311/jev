#!/usr/bin/env python3
"""Summarize reward density and inference-only credit allocation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from compute_advantages import reward_sequence, reward_to_go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", type=Path)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--advantage", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def nonconstant(values: list[float], tolerance: float = 1e-12) -> bool:
    return bool(values) and max(values) - min(values) > tolerance


def effective_support(values: list[float]) -> float | None:
    absolute = [abs(value) for value in values]
    total = sum(absolute)
    if total == 0:
        return None
    return total * total / sum(value * value for value in absolute)


def action_key(step: dict[str, Any]) -> str:
    return json.dumps(step.get("action"), sort_keys=True, ensure_ascii=False)


def reward_metrics(trajectories: list[dict[str, Any]], source: str) -> dict[str, Any]:
    sequences = [reward_sequence(trajectory, source) for trajectory in trajectories]
    returns = [reward_to_go(sequence, 1.0) for sequence in sequences]
    rewards = [reward for sequence in sequences for reward in sequence]
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    failures = [
        sequence
        for trajectory, sequence in zip(trajectories, sequences)
        if not trajectory["verifier"].get("success", False)
    ]
    failure_rewards = [reward for sequence in failures for reward in sequence]

    native_events = (
        len(trajectories)
        if source == "terminal" or source.startswith("verifier.")
        else sum(source in step.get("rewards", {}) for step in steps)
    )
    support = [effective_support(sequence) for sequence in sequences]
    normalized_support = [
        value / len(sequence)
        for value, sequence in zip(support, sequences)
        if value is not None and sequence
    ]
    mass_positions = []
    repeat_mass = 0.0
    total_mass = 0.0
    for trajectory, sequence in zip(trajectories, sequences):
        denominator = max(len(sequence) - 1, 1)
        mass = sum(abs(value) for value in sequence)
        if mass:
            mass_positions.append(
                sum(index / denominator * abs(value) for index, value in enumerate(sequence))
                / mass
            )
        previous_action = None
        for step, reward in zip(trajectory["steps"], sequence):
            key = action_key(step)
            if previous_action == key:
                repeat_mass += abs(reward)
            total_mass += abs(reward)
            previous_action = key

    verifier_key = (
        "terminal_reward" if source == "terminal" else source[len("verifier.") :]
    )
    terminal_errors = [
        abs(sum(sequence) - float(trajectory["verifier"][verifier_key]))
        for trajectory, sequence in zip(trajectories, sequences)
    ] if source == "terminal" or source.startswith("verifier.") else [
        abs(sum(sequence) - float(trajectory["verifier"]["terminal_reward"]))
        for trajectory, sequence in zip(trajectories, sequences)
    ]
    return {
        "source": source,
        "native_label_events": native_events,
        "native_event_density": native_events / len(steps) if steps else None,
        "nonzero_reward_density": mean([float(abs(value) > 1e-12) for value in rewards]),
        "positive_reward_density": mean([float(value > 1e-12) for value in rewards]),
        "negative_reward_density": mean([float(value < -1e-12) for value in rewards]),
        "active_trajectory_rate": mean(
            [float(any(abs(value) > 1e-12 for value in sequence)) for sequence in sequences]
        ),
        "failure_only_nonzero_density": mean(
            [float(abs(value) > 1e-12) for value in failure_rewards]
        ),
        "failure_only_active_trajectory_rate": mean(
            [float(any(abs(value) > 1e-12 for value in sequence)) for sequence in failures]
        ),
        "nonconstant_reward_to_go_rate": mean(
            [float(nonconstant(sequence)) for sequence in returns]
        ),
        "mean_distinct_reward_to_go_values": mean(
            [float(len({round(value, 12) for value in sequence})) for sequence in returns]
        ),
        "mean_effective_credit_steps": mean(
            [value for value in support if value is not None]
        ),
        "mean_normalized_effective_credit_support": mean(normalized_support),
        "mean_absolute_reward_mass_position": mean(mass_positions),
        "absolute_credit_on_consecutive_repeats": repeat_mass / total_mass
        if total_mass
        else None,
        "mean_terminal_consistency_error": mean(terminal_errors),
        "max_terminal_consistency_error": max(terminal_errors, default=None),
    }


def advantage_metrics(
    records: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    success = {
        trajectory["trajectory_id"]: bool(trajectory["verifier"].get("success", False))
        for trajectory in trajectories
    }
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["reward_source"]].append(record)
    output = {}
    for source, source_records in by_source.items():
        steps = [step for record in source_records for step in record["steps"]]
        rloo = [float(step["rloo_advantage"]) for step in steps if step["rloo_advantage"] is not None]
        stepwise_rloo = [
            float(step["stepwise_rloo_advantage"])
            for step in steps
            if step.get("stepwise_rloo_advantage") is not None
        ]
        failure_records = [
            record for record in source_records if not success.get(record["trajectory_id"], False)
        ]
        failure_rloo = [
            float(step["rloo_advantage"])
            for record in failure_records
            for step in record["steps"]
            if step["rloo_advantage"] is not None
        ]
        failure_stepwise = [
            float(step["stepwise_rloo_advantage"])
            for record in failure_records
            for step in record["steps"]
            if step.get("stepwise_rloo_advantage") is not None
        ]
        output[source] = {
            "trajectories": len(source_records),
            "steps": len(steps),
            "rloo_coverage": len(rloo) / len(steps) if steps else None,
            "nonzero_rloo_density": mean([float(abs(value) > 1e-12) for value in rloo]),
            "mean_absolute_rloo_advantage": mean([abs(value) for value in rloo]),
            "within_trajectory_varying_rloo_rate": mean(
                [
                    float(
                        nonconstant(
                            [
                                float(step["rloo_advantage"])
                                for step in record["steps"]
                                if step["rloo_advantage"] is not None
                            ]
                        )
                    )
                    for record in source_records
                ]
            ),
            "within_trajectory_varying_group_z_rate": mean(
                [
                    float(nonconstant([float(step["group_z_advantage"]) for step in record["steps"]]))
                    for record in source_records
                ]
            ),
            "stepwise_rloo_coverage": len(stepwise_rloo) / len(steps) if steps else None,
            "nonzero_stepwise_rloo_density": mean(
                [float(abs(value) > 1e-12) for value in stepwise_rloo]
            ),
            "within_trajectory_varying_stepwise_rloo_rate": mean(
                [
                    float(
                        nonconstant(
                            [
                                float(step["stepwise_rloo_advantage"])
                                for step in record["steps"]
                                if step.get("stepwise_rloo_advantage") is not None
                            ]
                        )
                    )
                    for record in source_records
                ]
            ),
            "failure_only_nonzero_rloo_density": mean(
                [float(abs(value) > 1e-12) for value in failure_rloo]
            ),
            "failure_only_nonzero_stepwise_rloo_density": mean(
                [float(abs(value) > 1e-12) for value in failure_stepwise]
            ),
            "failure_only_active_stepwise_trajectory_rate": mean(
                [
                    float(
                        any(
                            step.get("stepwise_rloo_advantage") is not None
                            and abs(float(step["stepwise_rloo_advantage"])) > 1e-12
                            for step in record["steps"]
                        )
                    )
                    for record in failure_records
                ]
            ),
        }
    return output


def summarize(
    trajectories: list[dict[str, Any]], sources: list[str], advantages: list[dict[str, Any]]
) -> dict[str, Any]:
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    return {
        "trajectories": len(trajectories),
        "groups": len({trajectory["group_id"] for trajectory in trajectories}),
        "steps": len(steps),
        "success_rate": mean(
            [float(trajectory["verifier"].get("success", False)) for trajectory in trajectories]
        ),
        "mean_terminal_reward": mean(
            [float(trajectory["verifier"]["terminal_reward"]) for trajectory in trajectories]
        ),
        "mean_steps": mean([float(len(trajectory["steps"])) for trajectory in trajectories]),
        "mean_elapsed_seconds": mean(
            [float(trajectory.get("elapsed_seconds", 0.0)) for trajectory in trajectories]
        ),
        "reward_sources": {
            source: reward_metrics(trajectories, source) for source in sources
        },
        "advantages": advantage_metrics(advantages, trajectories),
    }


def self_check() -> None:
    trajectories = [
        {
            "trajectory_id": "a",
            "group_id": "g",
            "verifier": {"terminal_reward": 1.0, "success": True},
            "steps": [
                {"action": "a", "rewards": {"dense": 0.5}},
                {"action": "b", "rewards": {"dense": 0.5}},
            ],
        }
    ]
    dense = reward_metrics(trajectories, "dense")
    terminal = reward_metrics(trajectories, "terminal")
    assert dense["nonzero_reward_density"] == 1.0
    assert terminal["native_event_density"] == 0.5
    assert dense["mean_terminal_consistency_error"] == 0.0
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.trajectories is None:
        raise SystemExit("trajectories are required")
    trajectories = read_jsonl(args.trajectories)
    advantages = [record for path in args.advantage for record in read_jsonl(path)]
    result = summarize(trajectories, args.sources or ["terminal"], advantages)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
