#!/usr/bin/env python3
"""Evaluate Jev on frozen, outcome-equivalent failures only."""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def auc(rows):
    positive = [score for score, label in rows if label]
    negative = [score for score, label in rows if not label]
    if not positive or not negative:
        return None
    # ponytail: O(n²) pairs are fine for <=50-step traces; use rank sorting for much longer episodes.
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def repeated_state_action_keys(trajectories):
    repeated = set()
    for trajectory in trajectories:
        seen = set()
        for index, step in enumerate(trajectory["steps"]):
            key = json.dumps(
                [step.get("observation_before"), step.get("action")],
                sort_keys=True, ensure_ascii=False,
            )
            if key in seen:
                repeated.add((trajectory["trajectory_id"], index))
            seen.add(key)
    return repeated


def evaluate(trajectories, judgments, allow_partial=False, score_field="jev_reward", exclude_repeated=False):
    by_id = {x["trajectory_id"]: x for x in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError("duplicate trajectory id")
    groups = defaultdict(list)
    for x in trajectories:
        groups[x["group_id"]].append(x)
    failed_groups = {
        group_id for group_id, members in groups.items()
        if len(members) == 5 and all(not x["verifier"]["success"] for x in members)
    }
    expected = {(x["trajectory_id"], i) for x in trajectories for i in range(len(x["steps"]))}
    labels = {}
    for judgment in judgments:
        key = judgment["trajectory_id"], judgment["step_index"]
        if key in labels or key not in expected:
            raise ValueError(f"duplicate or unknown judgment: {key}")
        labels[key] = judgment
    if not allow_partial and set(labels) != expected:
        raise ValueError(f"incomplete Jev annotation: {len(labels)}/{len(expected)} steps")
    repeated = repeated_state_action_keys(trajectories) if exclude_repeated else set()
    by_trace = defaultdict(list)
    by_group = defaultdict(list)
    for (trajectory_id, index), judgment in labels.items():
        x = by_id[trajectory_id]
        if x["group_id"] not in failed_groups or index == len(x["steps"]) - 1 or (trajectory_id, index) in repeated:
            continue  # The optional control additionally removes repeated state-action pairs.
        step = x["steps"][index]
        source = "scienceworld_score_delta" if x["benchmark"] == "ScienceWorld" else "tool_sandbox_milestone_delta"
        target = step["rewards"][source] > 0
        pair = judgment[score_field], target
        by_trace[trajectory_id].append(pair)
        by_group[x["group_id"]].append(pair)
    trace_auc = [value for rows in by_trace.values() if (value := auc(rows)) is not None]
    group_auc = {group_id: value for group_id, rows in by_group.items() if (value := auc(rows)) is not None}
    return {
        "coverage": {"annotated": len(labels), "total": len(expected), "fraction": len(labels) / len(expected)},
        "all_failed_groups": len(failed_groups),
        "excluded_repeated_state_action_steps": sum(
            (trajectory_id, index) in repeated
            for trajectory_id, index in expected
            if by_id[trajectory_id]["group_id"] in failed_groups
            and index < len(by_id[trajectory_id]["steps"]) - 1
        ),
        "nonterminal_failed_group_steps": sum(map(len, by_trace.values())),
        "eligible_traces": len(trace_auc),
        "mean_within_trace_auc": statistics.mean(trace_auc) if trace_auc else None,
        "eligible_groups": len(group_auc),
        "mean_within_group_auc": statistics.mean(group_auc.values()) if group_auc else None,
        "group_auc_by_id": group_auc,
        "binary_outcome_within_trace_auc": 0.5 if trace_auc else None,
        "continuous_outcome_within_trace_auc": 0.5 if trace_auc else None,
        "mean_latency_seconds": statistics.mean(j["latency_seconds"] for j in judgments) if judgments else None,
        "input_tokens": sum(
            j["response"].get("usage", {}).get("input_tokens", j["response"].get("usage", {}).get("prompt_tokens", 0))
            for j in judgments
        ),
    }


def compare_paired(trajectories, jev, llm, exclude_repeated=False):
    jev_metrics = evaluate(trajectories, jev, exclude_repeated=exclude_repeated)
    llm_metrics = evaluate(
        trajectories, llm, score_field="llm_reward",
        exclude_repeated=exclude_repeated,
    )
    jev_by_key = {(row["trajectory_id"], row["step_index"]): row for row in jev}
    llm_by_key = {(row["trajectory_id"], row["step_index"]): row for row in llm}
    if set(jev_by_key) != set(llm_by_key) or any(jev_by_key[key]["request"]["state"] != llm_by_key[key]["state"] for key in jev_by_key):
        raise ValueError("judges did not score identical public transitions")
    jev_groups = jev_metrics["group_auc_by_id"]
    llm_groups = llm_metrics["group_auc_by_id"]
    if set(jev_groups) != set(llm_groups) or not jev_groups:
        raise ValueError("judges have different eligible task groups")
    deltas = [jev_groups[group] - llm_groups[group] for group in sorted(jev_groups)]
    rng = random.Random(20260919)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(20000))
    groups = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory["group_id"]].append(trajectory)
    sums, means = {}, {}
    for name, records, field in (("jev", jev, "jev_reward"), ("llm", llm, "llm_reward")):
        totals = defaultdict(float)
        counts = Counter()
        for row in records:
            totals[row["trajectory_id"]] += row[field]
            counts[row["trajectory_id"]] += 1
        sums[name] = totals
        means[name] = {trajectory_id: total / counts[trajectory_id] for trajectory_id, total in totals.items()}

    def selection(rows, outcome, rankings):
        result = {"random": statistics.mean(outcome(x) for x in rows), "oracle": max(outcome(x) for x in rows)}
        for name, totals in rankings.items():
            top = max(totals[x["trajectory_id"]] for x in rows)
            result[name] = statistics.mean(outcome(x) for x in rows if totals[x["trajectory_id"]] == top)
        return result

    all_groups = [rows for rows in groups.values() if len(rows) == 5]
    failed_groups = [rows for rows in all_groups if all(not x["verifier"]["success"] for x in rows)]
    def best_of_five(rankings, aggregation):
        all_success = [selection(rows, lambda x: float(x["verifier"]["success"]), rankings) for rows in all_groups]
        failed_continuous = [selection(rows, lambda x: float(x["verifier"]["terminal_reward"]), rankings) for rows in failed_groups]
        return {
            "all_group_success": {name: statistics.mean(row[name] for row in all_success) for name in ("random", "jev", "llm", "oracle")},
            "failed_group_continuous": {name: statistics.mean(row[name] for row in failed_continuous) for name in ("random", "jev", "llm", "oracle")},
            "all_groups": len(all_groups), "failed_groups": len(failed_groups), "aggregation": aggregation,
        }
    return {
        "jev": jev_metrics, "llm": llm_metrics,
        "paired_group_mean_auc_delta": statistics.mean(deltas),
        "jev_wins_eligible_groups": sum(delta > 0 for delta in deltas),
        "paired_group_bootstrap_95": [samples[500], samples[19499]],
        "bootstrap_replicates": 20000, "bootstrap_seed": 20260919,
        "best_of_five": best_of_five(sums, "sum of centered transition rewards"),
        "best_of_five_mean": best_of_five(means, "mean centered transition reward"),
    }


def self_check():
    assert auc([(0.9, True), (0.1, False)]) == 1
    assert auc([(0.5, True), (0.5, False)]) == 0.5
    assert auc([(0.1, True), (0.9, False)]) == 0
    assert auc([(0.9, True)]) is None
    assert repeated_state_action_keys([{
        "trajectory_id": "repeat", "steps": [
            {"observation_before": "x", "action": "a"},
            {"observation_before": "x", "action": "a"},
            {"observation_before": "y", "action": "a"},
        ],
    }]) == {("repeat", 1)}
    trajectories = [
        {"trajectory_id": f"t{i}", "group_id": "g", "benchmark": "ScienceWorld",
         "verifier": {"success": False, "terminal_reward": float(i == 0)},
         "steps": [{"rewards": {"scienceworld_score_delta": int(i == 0)}}, {"rewards": {}}]}
        for i in range(5)
    ]
    jev = [
        {"trajectory_id": f"t{i}", "step_index": step, "jev_reward": int(i == 0),
         "request": {"state": {"i": i, "step": step}}, "latency_seconds": 0, "response": {}}
        for i in range(5) for step in range(2)
    ]
    llm = [
        {"trajectory_id": row["trajectory_id"], "step_index": row["step_index"],
         "llm_reward": 1 - row["jev_reward"], "state": row["request"]["state"],
         "latency_seconds": 0, "response": {}}
        for row in jev
    ]
    report = compare_paired(trajectories, jev, llm)
    assert report["paired_group_bootstrap_95"] == [1.0, 1.0]
    assert report["best_of_five"]["failed_group_continuous"] == {"random": 0.2, "jev": 1.0, "llm": 0.0, "oracle": 1.0}
    assert report["best_of_five_mean"]["failed_group_continuous"] == report["best_of_five"]["failed_group_continuous"]
    trajectories[4]["steps"][1]["rewards"]["scienceworld_score_delta"] = 0
    for step in range(2, 5):
        trajectories[4]["steps"].append({"rewards": {"scienceworld_score_delta": 0}})
        state = {"i": 4, "step": step}
        jev.append({"trajectory_id": "t4", "step_index": step, "jev_reward": 1.0,
                    "request": {"state": state}, "latency_seconds": 0, "response": {}})
        llm.append({"trajectory_id": "t4", "step_index": step, "llm_reward": 0.0,
                    "state": state, "latency_seconds": 0, "response": {}})
    report = compare_paired(trajectories, jev, llm)
    assert report["best_of_five"]["failed_group_continuous"]["jev"] == 0.0
    assert report["best_of_five_mean"]["failed_group_continuous"]["jev"] == 1.0
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", type=Path)
    parser.add_argument("judgments", nargs="?", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--score-field", choices=("jev_reward", "llm_reward"), default="jev_reward")
    parser.add_argument("--compare-llm", type=Path)
    parser.add_argument("--exclude-repeated-state-actions", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.trajectories or not args.judgments:
        parser.error("trajectories and judgments required")
    with args.trajectories.open(encoding="utf-8") as source:
        trajectories = [json.loads(line) for line in source]
    with args.judgments.open(encoding="utf-8") as source:
        judgments = [json.loads(line) for line in source]
    if args.compare_llm:
        if args.allow_partial or args.score_field != "jev_reward":
            parser.error("--compare-llm requires complete Jev primary judgments")
        with args.compare_llm.open(encoding="utf-8") as source:
            llm = [json.loads(line) for line in source]
        report = compare_paired(
            trajectories, judgments, llm, args.exclude_repeated_state_actions,
        )
    else:
        report = evaluate(
            trajectories, judgments, args.allow_partial, args.score_field,
            args.exclude_repeated_state_actions,
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
