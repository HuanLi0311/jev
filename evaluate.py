#!/usr/bin/env python3
"""Summarize ALFWorld trajectories and compare step-level judge signals."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", type=Path)
    parser.add_argument("--judge", action="append", type=Path, default=[])
    parser.add_argument("--jev", action="append", type=Path, default=[])
    parser.add_argument("--llm", action="append", type=Path, default=[])
    parser.add_argument("--write-rlvr", type=Path)
    parser.add_argument("--write-oracle", type=Path)
    parser.add_argument("--write-pairwise", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def population_variance(values: list[float]) -> float | None:
    return statistics.pvariance(values) if len(values) > 1 else 0.0 if values else None


def binary_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    # ponytail: O(n²) is simplest here; replace with rank-sum above ~100k labeled steps.
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def mean_group_auc(groups: dict[str, list[tuple[bool, float]]]) -> tuple[float | None, int]:
    values = [binary_auc([label for label, _ in rows], [score for _, score in rows]) for rows in groups.values()]
    eligible = [value for value in values if value is not None]
    return mean(eligible), len(eligible)


def normalized_edit_distance(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 0.0
    # ponytail: quadratic DP is fine for <=50-step traces; use a C-backed library
    # if much longer rollouts make evaluation slow.
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def brier_score(labels: list[bool], scores: list[float]) -> float | None:
    if not scores:
        return None
    if len(labels) != len(scores) or any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("Calibration labels/scores must align and scores must be in [0, 1]")
    return statistics.fmean((score - float(label)) ** 2 for label, score in zip(labels, scores))


def calibration_error(
    labels: list[bool], scores: list[float], bins: int = 10
) -> float | None:
    if not scores:
        return None
    if len(labels) != len(scores) or any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("Calibration labels/scores must align and scores must be in [0, 1]")
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for label, score in zip(labels, scores):
        buckets[min(int(score * bins), bins - 1)].append((label, score))
    return sum(
        len(bucket) / len(scores)
        * abs(
            statistics.fmean(float(label) for label, _ in bucket)
            - statistics.fmean(score for _, score in bucket)
        )
        for bucket in buckets
        if bucket
    )


def top_tie_mean(candidates: list[tuple[float, float]]) -> float:
    top_score = max(score for score, _ in candidates)
    return statistics.fmean(value for score, value in candidates if score == top_score)


def partial_progress_utility(trajectory: dict[str, Any]) -> float | None:
    if not trajectory["steps"]:
        return None
    start = trajectory["steps"][0].get("oracle_distance_before")
    end = trajectory["steps"][-1].get("oracle_distance_after")
    return None if start is None or end is None else float(start - end)


def hard_step_labels(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    labels = []
    for step in trajectory["steps"]:
        key = (step["observation_before"], step["action"])
        repeated_state_action = key in seen
        seen.add(key)
        delta = step.get("oracle_progress_delta")
        hard_error = (
            not step["action_valid"]
            or repeated_state_action
            or (delta is not None and delta < 0)
        )
        labels.append(
            {
                "step": step["step"],
                "oracle_progress": None if delta is None else delta > 0,
                "hard_error": hard_error,
                "repeated_state_action": repeated_state_action,
            }
        )
    return labels


def trajectory_metrics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    labels = [label for trajectory in trajectories for label in hard_step_labels(trajectory)]
    sequences = [[step["action"] for step in trajectory["steps"]] for trajectory in trajectories]
    sequences_by_game: dict[str, list[list[str]]] = defaultdict(list)
    for trajectory, sequence in zip(trajectories, sequences):
        sequences_by_game[trajectory["game_file"]].append(sequence)
    within_game_distances = [
        normalized_edit_distance(left, right)
        for game_sequences in sequences_by_game.values()
        for left, right in combinations(game_sequences, 2)
    ]
    deltas = [
        step["oracle_progress_delta"]
        for step in steps
        if step.get("oracle_progress_delta") is not None
    ]
    return {
        "trajectories": len(trajectories),
        "transitions": len(steps),
        "games": len(sequences_by_game),
        "task_type_counts": dict(sorted(Counter(t["task_type"] for t in trajectories).items())),
        "policy_model_counts": dict(
            sorted(Counter(t["policy"]["model"] for t in trajectories).items())
        ),
        "termination_counts": dict(
            sorted(Counter(t["terminal"]["termination"] for t in trajectories).items())
        ),
        "success_rate": mean([float(t["terminal"]["won"]) for t in trajectories]),
        "mean_steps": mean([float(len(t["steps"])) for t in trajectories]),
        "valid_action_rate": mean([float(step["action_valid"]) for step in steps]),
        "oracle_next_match_rate": mean(
            [float(step["oracle_next_match"]) for step in steps if step.get("oracle_plan_before")]
        ),
        "oracle_progress_coverage": len(deltas) / len(steps) if steps else None,
        "oracle_progress_rate": mean([float(delta > 0) for delta in deltas]),
        "mean_oracle_progress_delta": mean([float(delta) for delta in deltas]),
        "hard_error_rate": mean([float(label["hard_error"]) for label in labels]),
        "loop_rate": mean([float(label["repeated_state_action"]) for label in labels]),
        "action_sequence_uniqueness": (
            len({tuple(sequence) for sequence in sequences}) / len(sequences)
            if sequences
            else None
        ),
        "mean_unique_state_action_ratio": mean(
            [
                len({(step["observation_before"], step["action"]) for step in t["steps"]})
                / len(t["steps"])
                for t in trajectories
                if t["steps"]
            ]
        ),
        "within_game_action_edit_distance": mean(within_game_distances),
        "within_game_trace_pairs": len(within_game_distances),
        "parse_method_counts": dict(
            sorted(Counter(step["parse_method"] for step in steps).items())
        ),
        "mean_generation_seconds_per_step": mean(
            [float(step["generation_seconds"]) for step in steps]
        ),
        "generated_tokens": sum(int(step["generated_tokens"]) for step in steps),
        "rlvr_native_label_events": len(trajectories),
        "rlvr_native_event_density": len(trajectories) / len(steps) if steps else None,
    }


def rlvr_records(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for trajectory in trajectories:
        reward = float(trajectory["terminal"]["rlvr_reward"])
        records.append(
            {
                "schema_version": 1,
                "trajectory_id": trajectory["trajectory_id"],
                "judge": "rlvr",
                "trajectory_score": reward,
                "native_label_events": 1,
                "latency_seconds": 0.0,
                "cost_usd": 0.0,
                "steps": [
                    {"step": step["step"], "process_score": reward}
                    for step in trajectory["steps"]
                ],
            }
        )
    return records


def transition_records(
    trajectories: list[dict[str, Any]], judgments: list[dict[str, Any]],
    rubric_version: str, reward_field: str,
) -> list[dict[str, Any]]:
    by_id = {trajectory["trajectory_id"]: trajectory for trajectory in trajectories}
    expected = {(trajectory_id, index) for trajectory_id, trajectory in by_id.items() for index in range(len(trajectory["steps"]))}
    scored: dict[tuple[str, int], dict[str, Any]] = {}
    for row in judgments:
        key = row["trajectory_id"], row["step_index"]
        if key in scored or key not in expected or row["rubric_version"] != rubric_version:
            raise ValueError(f"unknown, duplicate, or wrong-rubric judge step: {key}")
        scored[key] = row
    if set(scored) != expected:
        raise ValueError(f"incomplete judge annotation: {len(scored)}/{len(expected)} steps")
    return [
        {
            "schema_version": 1, "trajectory_id": trajectory_id, "judge": rubric_version,
            "native_label_events": len(trajectory["steps"]),
            "latency_seconds": sum(scored[trajectory_id, i]["latency_seconds"] for i in range(len(trajectory["steps"]))),
            "steps": [
                {"step": i, "process_score": (scored[trajectory_id, i][reward_field] + 1) / 2}
                for i in range(len(trajectory["steps"]))
            ],
        }
        for trajectory_id, trajectory in by_id.items()
    ]


def jev_transition_records(trajectories: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    judgments = read_jsonl(path)
    versions = {row.get("rubric_version") for row in judgments}
    if len(versions) != 1 or not versions <= {"jev_transition_v1", "jev_transition_v2"}:
        raise ValueError(f"unsupported or mixed Jev rubric versions: {versions}")
    return transition_records(trajectories, judgments, versions.pop(), "jev_reward")


def pairwise_cases(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for trajectory in trajectories:
        history: list[dict[str, Any]] = []
        for step in trajectory["steps"]:
            oracle_plan = step.get("oracle_plan_before") or []
            oracle_action = oracle_plan[0] if oracle_plan else None
            policy_action = step["action"]
            if oracle_action and policy_action != oracle_action:
                common = {
                    "schema_version": 1,
                    "trajectory_id": trajectory["trajectory_id"],
                    "step": step["step"],
                    "source": "natural_policy_error_vs_textworld_oracle",
                    "state": {
                        "task": trajectory["task"],
                        "initial_observation": trajectory["initial_observation"],
                        "history": list(history),
                        "current_observation": step["observation_before"],
                        "admissible_actions": step["admissible_actions"],
                    },
                }
                cases.extend(
                    [
                        {
                            **common,
                            "case_id": f"{trajectory['trajectory_id']}-s{step['step']:03d}-ab",
                            "options": {"A": policy_action, "B": oracle_action},
                            "correct_option": "B",
                        },
                        {
                            **common,
                            "case_id": f"{trajectory['trajectory_id']}-s{step['step']:03d}-ba",
                            "options": {"A": oracle_action, "B": policy_action},
                            "correct_option": "A",
                        },
                    ]
                )
            history.append(
                {
                    "step": step["step"],
                    "action": policy_action,
                    "observation": step["observation_after"],
                }
            )
    return cases


def oracle_records(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for trajectory in trajectories:
        labels = hard_step_labels(trajectory)
        scores = []
        for step, label in zip(trajectory["steps"], labels):
            delta = step.get("oracle_progress_delta")
            if label["hard_error"]:
                score = 0.0
            elif delta is not None and delta > 0:
                score = 1.0
            else:
                score = 0.5
            scores.append({"step": step["step"], "process_score": score})
        records.append(
            {
                "schema_version": 1,
                "trajectory_id": trajectory["trajectory_id"],
                "judge": "textworld-oracle-anchor",
                "trajectory_score": float(trajectory["terminal"]["won"]),
                "native_label_events": len(scores),
                "latency_seconds": 0.0,
                "cost_usd": 0.0,
                "steps": scores,
            }
        )
    return records


def first_hard_error(labels: list[dict[str, Any]]) -> int | None:
    return next((label["step"] for label in labels if label["hard_error"]), None)


def first_predicted_error(scores: dict[int, float], threshold: float = 0.5) -> int | None:
    return next((index for index in sorted(scores) if scores[index] < threshold), None)


def judge_metrics(
    trajectories: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {trajectory["trajectory_id"]: trajectory for trajectory in trajectories}
    records = [record for record in judgments if record["trajectory_id"] in by_id]
    process_scores: list[float] = []
    progress_labels: list[bool] = []
    progress_scores: list[float] = []
    quality_labels: list[bool] = []
    quality_scores: list[float] = []
    failed_progress_labels: list[bool] = []
    failed_progress_scores: list[float] = []
    failed_quality_labels: list[bool] = []
    failed_quality_scores: list[float] = []
    localization_exact: list[float] = []
    localization_distance: list[float] = []
    adjacent_changes: list[float] = []
    nonconstant: list[float] = []
    game_progress: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    game_quality: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    total_expected_steps = sum(len(by_id[r["trajectory_id"]]["steps"]) for r in records)
    scored_steps = 0

    per_game: dict[str, list[tuple[float, bool, float | None]]] = defaultdict(list)
    for record in records:
        trajectory = by_id[record["trajectory_id"]]
        labels = hard_step_labels(trajectory)
        label_by_step = {label["step"]: label for label in labels}
        step_scores = {
            int(step["step"]): float(step["process_score"])
            for step in record.get("steps", [])
            if step.get("process_score") is not None
        }
        ordered = [step_scores[i] for i in sorted(step_scores)]
        scored_steps += len(ordered)
        process_scores.extend(ordered)
        if ordered:
            adjacent_changes.extend(abs(b - a) for a, b in zip(ordered, ordered[1:]))
            nonconstant.append(float(len({round(score, 6) for score in ordered}) > 1))

        for step_index, score in step_scores.items():
            label = label_by_step.get(step_index)
            if not label:
                continue
            quality_labels.append(not label["hard_error"])
            quality_scores.append(score)
            game_quality[trajectory["game_file"]].append((not label["hard_error"], score))
            if not trajectory["terminal"]["won"]:
                failed_quality_labels.append(not label["hard_error"])
                failed_quality_scores.append(score)
            if label["oracle_progress"] is not None:
                progress_labels.append(bool(label["oracle_progress"]))
                progress_scores.append(score)
                game_progress[trajectory["game_file"]].append((bool(label["oracle_progress"]), score))
                if not trajectory["terminal"]["won"]:
                    failed_progress_labels.append(bool(label["oracle_progress"]))
                    failed_progress_scores.append(score)

        true_first = first_hard_error(labels)
        predicted_first = first_predicted_error(step_scores)
        if true_first is not None and predicted_first is not None:
            localization_exact.append(float(true_first == predicted_first))
            localization_distance.append(float(abs(true_first - predicted_first)))
        elif true_first is None and predicted_first is None:
            localization_exact.append(1.0)
            localization_distance.append(0.0)
        else:
            localization_exact.append(0.0)
            localization_distance.append(float(len(labels)))

        trajectory_score = record.get("trajectory_score")
        if trajectory_score is None and ordered:
            trajectory_score = statistics.fmean(ordered)
        if trajectory_score is not None:
            per_game[trajectory["game_file"]].append(
                (
                    float(trajectory_score),
                    bool(trajectory["terminal"]["won"]),
                    partial_progress_utility(trajectory),
                )
            )

    best_of_n = []
    oracle_best_of_n = []
    failed_partial_selected = []
    failed_partial_oracle = []
    for candidates in per_game.values():
        best_of_n.append(
            top_tie_mean([(score, float(won)) for score, won, _ in candidates])
        )
        oracle_best_of_n.append(float(any(won for _, won, _ in candidates)))
        failed = [
            (score, utility)
            for score, won, utility in candidates
            if not won and utility is not None
        ]
        if len(failed) >= 2:
            failed_partial_selected.append(top_tie_mean(failed))
            failed_partial_oracle.append(max(utility for _, utility in failed))

    mean_game_progress_auc, eligible_progress_games = mean_group_auc(game_progress)
    mean_game_quality_auc, eligible_quality_games = mean_group_auc(game_quality)

    return {
        "judge": records[0].get("judge", "unknown") if records else "unknown",
        "matched_trajectories": len(records),
        "step_score_coverage": scored_steps / total_expected_steps if total_expected_steps else None,
        "native_label_events": sum(int(r.get("native_label_events", len(r.get("steps", [])))) for r in records),
        "native_event_density": (
            sum(int(r.get("native_label_events", len(r.get("steps", [])))) for r in records)
            / total_expected_steps
            if total_expected_steps
            else None
        ),
        "score_mean": mean(process_scores),
        "score_variance": population_variance(process_scores),
        "mean_adjacent_score_change": mean(adjacent_changes),
        "nonconstant_trajectory_rate": mean(nonconstant),
        "oracle_progress_auc": binary_auc(progress_labels, progress_scores),
        "mean_within_game_oracle_progress_auc": mean_game_progress_auc,
        "eligible_oracle_progress_games": eligible_progress_games,
        "oracle_progress_brier": brier_score(progress_labels, progress_scores),
        "oracle_progress_ece_10": calibration_error(progress_labels, progress_scores),
        "hard_step_quality_auc": binary_auc(quality_labels, quality_scores),
        "mean_within_game_hard_step_quality_auc": mean_game_quality_auc,
        "eligible_hard_step_quality_games": eligible_quality_games,
        "hard_step_quality_brier": brier_score(quality_labels, quality_scores),
        "hard_step_quality_ece_10": calibration_error(quality_labels, quality_scores),
        "failed_only_oracle_progress_auc": binary_auc(
            failed_progress_labels, failed_progress_scores
        ),
        "failed_only_hard_step_quality_auc": binary_auc(
            failed_quality_labels, failed_quality_scores
        ),
        "first_hard_error_exact": mean(localization_exact),
        "first_hard_error_mae": mean(localization_distance),
        "best_of_n_success": mean(best_of_n),
        "best_of_n_oracle_ceiling": mean(oracle_best_of_n),
        "failed_only_partial_progress_selected": mean(failed_partial_selected),
        "failed_only_partial_progress_oracle": mean(failed_partial_oracle),
        "failed_only_partial_progress_regret": mean(
            [
                oracle - selected
                for selected, oracle in zip(failed_partial_selected, failed_partial_oracle)
            ]
        ),
        "failed_only_partial_progress_tasks": len(failed_partial_selected),
        "latency_seconds": sum(float(r.get("latency_seconds", 0.0)) for r in records),
        "cost_usd": sum(float(r.get("cost_usd", 0.0)) for r in records),
    }


def self_check() -> None:
    assert binary_auc([True, False], [0.9, 0.1]) == 1.0
    assert binary_auc([True, False], [0.1, 0.9]) == 0.0
    assert binary_auc([True, False], [0.5, 0.5]) == 0.5
    assert mean_group_auc({"a": [(True, 1.0), (False, 0.0)], "b": [(True, 0.0), (False, 1.0)], "c": [(True, 1.0)]}) == (0.5, 2)
    assert normalized_edit_distance([], []) == 0.0
    assert normalized_edit_distance(["a", "b"], ["a", "c"]) == 0.5
    assert brier_score([True, False], [1.0, 0.0]) == 0.0
    assert calibration_error([True, False], [1.0, 0.0]) == 0.0
    assert top_tie_mean([(0.0, 1.0), (0.0, 3.0), (-1.0, 9.0)]) == 2.0
    synthetic = {
        "steps": [
            {
                "step": 0,
                "observation_before": "a",
                "action": "look",
                "action_valid": True,
                "oracle_progress_delta": 0,
            },
            {
                "step": 1,
                "observation_before": "a",
                "action": "look",
                "action_valid": True,
                "oracle_progress_delta": -1,
            },
        ]
    }
    labels = hard_step_labels(synthetic)
    assert not labels[0]["hard_error"] and labels[1]["hard_error"]
    toy = {"trajectory_id": "a", "steps": [{"step": 0}]}
    assert transition_records([toy], [{"trajectory_id": "a", "step_index": 0, "rubric_version": "jev_transition_v1", "jev_reward": 0.5, "latency_seconds": 1.0}], "jev_transition_v1", "jev_reward")[0]["steps"][0]["process_score"] == 0.75
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.trajectories is None:
        raise SystemExit("trajectories path is required")

    trajectories = read_jsonl(args.trajectories)
    if not trajectories:
        raise ValueError("No trajectories found")
    if args.write_rlvr:
        write_jsonl(args.write_rlvr, rlvr_records(trajectories))
    if args.write_oracle:
        write_jsonl(args.write_oracle, oracle_records(trajectories))
    if args.write_pairwise:
        write_jsonl(args.write_pairwise, pairwise_cases(trajectories))

    report = {
        "schema_version": 1,
        "trajectory_metrics": trajectory_metrics(trajectories),
        "judges": [judge_metrics(trajectories, read_jsonl(path)) for path in args.judge]
        + [judge_metrics(trajectories, jev_transition_records(trajectories, path)) for path in args.jev]
        + [judge_metrics(trajectories, transition_records(trajectories, read_jsonl(path), "llm_transition_v1", "llm_reward")) for path in args.llm],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
