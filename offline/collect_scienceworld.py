#!/usr/bin/env python3
"""Collect ScienceWorld trajectories from an OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv


DEFAULT_TASKS = [
    "change-the-state-of-matter-of",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color",
    "find-animal",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "freeze",
    "identify-life-stages-2",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "melt",
    "power-component",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollouts-per-task", type=int, default=5)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def post_chat(base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"chat server returned HTTP {error.code}: {detail}") from error


def parse_action(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    line = re.sub(r"^(?:action|command)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip("`'\" ").lower()


def safe_goal_progress(env: ScienceWorldEnv) -> str | None:
    # Some official task variations throw on this optional diagnostic accessor.
    try:
        return env.get_goal_progress()
    except Exception:
        return None


def prompt_actions(
    valid_actions: list[str], task_description: str, observation: str, look: str
) -> list[str]:
    if len(valid_actions) <= 120:
        return valid_actions
    ignored = {
        "and",
        "create",
        "determine",
        "first",
        "focus",
        "for",
        "from",
        "into",
        "move",
        "next",
        "place",
        "task",
        "the",
        "then",
        "this",
        "to",
        "turn",
        "using",
        "with",
        "your",
    }

    def words(text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 2 and word not in ignored
        }

    task_words = words(task_description)
    state_words = words(observation + " " + look)

    def relevance(item: tuple[int, str]) -> tuple[int, int, int]:
        index, action = item
        action_words = words(action)
        navigation = int(action.startswith(("go to ", "look", "teleport to ")))
        score = 100 * len(task_words & action_words) + 10 * navigation
        return (-score, -len(state_words & action_words), index)

    return [action for _, action in sorted(enumerate(valid_actions), key=relevance)[:120]]


def choose_action(
    base_url: str,
    model: str,
    task_description: str,
    observation: str,
    look: str,
    inventory: str,
    valid_actions: list[str],
    history: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> tuple[str, dict[str, Any], list[str]]:
    shown_actions = prompt_actions(valid_actions, task_description, observation, look)
    recent = [
        {"action": step["action"], "observation": step["observation_after"][:500]}
        for step in history
    ]
    prompt = (
        f"TASK:\n{task_description}\n\n"
        f"RECENT HISTORY:\n{json.dumps(recent, ensure_ascii=False)}\n\n"
        f"CURRENT OBSERVATION:\n{observation[:4000]}\n\n"
        f"CURRENT LOCATION VIEW:\n{look[:4000]}\n\n"
        f"INVENTORY:\n{inventory[:2000]}\n\n"
        "VALID ACTIONS (choose exactly one):\n"
        + "\n".join(shown_actions)
    )
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You control a text-world agent. Choose the single best next action from "
                    "the supplied valid-action list. Make concrete progress on every requested subgoal "
                    "and do not repeat an action when it left the state unchanged. Output only that "
                    "exact action, with no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = post_chat(base_url, body)
    content = response["choices"][0]["message"].get("content") or ""
    return parse_action(content), response, shown_actions


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def self_check() -> None:
    assert parse_action("Action: OPEN DOOR\n") == "open door"
    scores = [0, 10, 10, -100]
    deltas = [(max(b, 0) - max(a, 0)) / 100 for a, b in zip(scores, scores[1:])]
    assert sum(deltas) == 0.0
    print("self-check passed")


def build_task_manifest(
    env: ScienceWorldEnv,
    tasks: list[str],
    variation: int,
    simplification: str,
) -> list[dict[str, Any]]:
    records = []
    for task in tasks:
        if variation >= env.get_max_variations(task):
            raise ValueError(f"variation {variation} is invalid for {task}")
        env.load(task, variation, simplification, generateGoldPath=True)
        env.reset()
        records.append(
            {
                "task": task,
                "variation": variation,
                "simplification": simplification,
                "task_description": env.get_task_description(),
                "gold_action_sequence": env.get_gold_action_sequence(),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    if args.rollouts_per_task < 1 or args.max_steps < 1:
        raise SystemExit("rollout and step counts must be positive")

    tasks = args.tasks or DEFAULT_TASKS
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    config = {
        "schema_version": 1,
        "benchmark": "ScienceWorld",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "history_steps": args.history_steps,
        "seed": args.seed,
        "rollouts_per_task": args.rollouts_per_task,
        "tasks": tasks,
        "variation": args.variation,
        "simplification": args.simplification,
        "terminal_reward": "max(final_score, 0) / 100",
        "prompt_version": "observable-budget-task-priority-v2",
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise SystemExit("existing manifest does not match this run")
    else:
        write_json_exclusive(manifest_path, config)

    jsonl_path = output_dir / "trajectories.jsonl"
    completed: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            completed = {json.loads(line)["trajectory_id"] for line in handle if line.strip()}

    env = ScienceWorldEnv("", None, envStepLimit=args.max_steps)
    try:
        task_manifest_path = output_dir / "tasks.json"
        if task_manifest_path.exists():
            task_manifest = json.loads(task_manifest_path.read_text())
        else:
            task_manifest = build_task_manifest(env, tasks, args.variation, args.simplification)
            write_json_exclusive(task_manifest_path, task_manifest)
        task_metadata = {record["task"]: record for record in task_manifest}

        with jsonl_path.open("a", encoding="utf-8") as output:
            for task_index, task in enumerate(tasks):
                metadata = task_metadata[task]
                for rollout_index in range(args.rollouts_per_task):
                    trajectory_id = f"{task_index:02d}_{task}_v{args.variation}__r{rollout_index:02d}"
                    if trajectory_id in completed:
                        continue
                    seed = args.seed + task_index * 10_000 + rollout_index * 1_000
                    started = time.time()
                    env.load(task, args.variation, args.simplification, generateGoldPath=False)
                    observation, reset_info = env.reset()
                    initial_observation = observation
                    current_info = reset_info
                    score = float(reset_info.get("score", 0))
                    previous_goal_progress = safe_goal_progress(env)
                    steps: list[dict[str, Any]] = []
                    raw_responses: list[dict[str, Any]] = []
                    completed_episode = False

                    for step_index in range(args.max_steps):
                        valid_actions = env.get_valid_action_object_combinations()
                        action, response, shown_actions = choose_action(
                            args.base_url,
                            args.model,
                            metadata["task_description"],
                            observation,
                            str(current_info.get("look", "")),
                            str(current_info.get("inv", "")),
                            valid_actions,
                            steps[-args.history_steps :],
                            args.temperature,
                            args.top_p,
                            args.max_tokens,
                            seed + step_index,
                        )
                        raw_responses.append(response)
                        score_before = score
                        observation_before = observation
                        observation, native_reward, completed_episode, info = env.step(action)
                        score = float(info["score"])
                        goal_progress = safe_goal_progress(env)
                        gold_actions = metadata["gold_action_sequence"]
                        steps.append(
                            {
                                "step_index": step_index,
                                "action": action,
                                "action_in_valid_menu": action in valid_actions,
                                "observation_before": observation_before,
                                "observation_after": observation,
                                "look_before": current_info.get("look"),
                                "inventory_before": current_info.get("inv"),
                                "valid_actions": valid_actions,
                                "prompt_valid_actions": shown_actions,
                                "score_before": score_before,
                                "score_after": score,
                                "oracle": {
                                    "goal_progress": goal_progress,
                                    "goal_progress_changed": goal_progress != previous_goal_progress,
                                    "gold_action_at_position": gold_actions[step_index]
                                    if step_index < len(gold_actions)
                                    else None,
                                    "gold_action_match_at_position": step_index < len(gold_actions)
                                    and action == gold_actions[step_index],
                                },
                                "rewards": {
                                    "scienceworld_score_delta": float(native_reward) / 100.0,
                                    "scienceworld_clipped_score_delta": (
                                        max(score, 0.0)
                                        - (0.0 if step_index == 0 else max(score_before, 0.0))
                                    )
                                    / 100.0,
                                    "scienceworld_reset_score_offset": (
                                        max(score_before, 0.0) / 100.0
                                        if step_index == 0
                                        else 0.0
                                    ),
                                    "scienceworld_goal_progress_change": float(
                                        goal_progress != previous_goal_progress
                                    ),
                                },
                            }
                        )
                        previous_goal_progress = goal_progress
                        current_info = info
                        if completed_episode:
                            break

                    terminal_reward = max(score, 0.0) / 100.0
                    dense_sum = sum(
                        step["rewards"]["scienceworld_clipped_score_delta"] for step in steps
                    )
                    if abs(dense_sum - terminal_reward) > 1e-9:
                        raise AssertionError(
                            f"dense reward does not telescope: {dense_sum} != {terminal_reward}"
                        )
                    trajectory_dir = output_dir / "trajectories" / trajectory_id
                    trajectory_dir.mkdir(exist_ok=False)
                    raw_path = trajectory_dir / "raw.json"
                    write_json_exclusive(
                        raw_path,
                        {
                            "initial_observation": initial_observation,
                            "reset_info": reset_info,
                            "task": metadata,
                            "steps": steps,
                            "model_responses": raw_responses,
                        },
                    )
                    record = {
                        "schema_version": 1,
                        "trajectory_id": trajectory_id,
                        "group_id": f"{task}:v{args.variation}:{args.simplification}",
                        "benchmark": "ScienceWorld",
                        "model": args.model,
                        "seed": seed,
                        "task": metadata,
                        "verifier": {
                            "terminal_reward": terminal_reward,
                            "binary_success_reward": float(score >= 100.0),
                            "success": score >= 100.0,
                            "raw_final_score": score,
                            "episode_completed": completed_episode,
                        },
                        "steps": steps,
                        "raw": {"trajectory": str(raw_path)},
                        "elapsed_seconds": time.time() - started,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    print(
                        f"{trajectory_id}: reward={terminal_reward:.3f} "
                        f"steps={len(steps)} seconds={record['elapsed_seconds']:.1f}",
                        flush=True,
                    )
    finally:
        env.close()


if __name__ == "__main__":
    main()
