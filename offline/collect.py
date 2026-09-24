#!/usr/bin/env python3
"""Collect Qwen3-1.7B trajectories from ALFWorld TextWorld tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable


SYSTEM_PROMPT = """You control an agent in the ALFWorld text environment.
At each turn, select exactly one command from the admissible actions. Base the choice only on the task and observations; never invent object names or ids. Think briefly about progress and failed attempts. End with exactly one line in this form:
Action: <exact admissible command>
Do not write anything after the Action line.

Operational rules:
- Match goal object names exactly. For example, a saltshaker is not a peppershaker.
- If a searched location contains the wrong objects, leave them alone and search a different location.
- Never repeat an action from the same state; it cannot create new progress.
- Once you hold the exact goal object, carry out the requested clean, heat, cool, examine, or placement operation instead of continuing to search."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/alfworld")
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".cache/huggingface/hub/Qwen3-1.7B"),
    )
    parser.add_argument(
        "--split",
        choices=("eval_in_distribution", "eval_out_of_distribution"),
        default="eval_out_of_distribution",
    )
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--episode-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stratify-task-types", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def alfworld_config(data_dir: Path, max_steps: int) -> dict[str, Any]:
    root = str(data_dir.resolve())
    return {
        "dataset": {
            "data_path": f"{root}/json_2.1.1/train",
            "eval_id_data_path": f"{root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{root}/logic/alfred.pddl",
            "grammar": f"{root}/logic/alfred.twl2",
        },
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_type": "planner",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


def task_metadata(game_file: str) -> dict[str, str]:
    game_path = Path(game_file)
    with (game_path.parent / "traj_data.json").open() as handle:
        data = json.load(handle)
    return {
        "task_type": data["task_type"],
        "scene": str(data.get("scene", {})),
    }


def make_env(game_file: str, max_steps: int):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    # policy_commands is TextWorld's shortest remaining plan. It is recorded for
    # analysis but never shown to the policy model.
    request_infos = textworld.EnvInfos(
        won=True,
        score=True,
        max_score=True,
        intermediate_reward=True,
        admissible_commands=True,
        policy_commands=True,
    )
    env_id = textworld.gym.register_game(
        game_file,
        request_infos,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler()],
    )
    return textworld.gym.make(env_id)


def extract_task(initial_observation: str) -> str:
    match = re.search(r"Your task is to:\s*(.+)", initial_observation, re.I | re.S)
    return match.group(1).strip() if match else initial_observation.strip()


def parse_model_output(text: str, admissible: list[str]) -> tuple[str, str]:
    matches = re.findall(r"(?im)^\s*Action:\s*(.+?)\s*$", text)
    candidate = matches[-1] if matches else text.strip().splitlines()[-1] if text.strip() else ""
    candidate = candidate.strip().strip("`\"'").rstrip(". ")
    by_lower = {action.casefold(): action for action in admissible}
    if candidate.casefold() in by_lower:
        return by_lower[candidate.casefold()], "action_line_exact"

    lower_text = text.casefold()
    mentioned = [action for action in admissible if action.casefold() in lower_text]
    if mentioned:
        # The final mention is usually the decision; longest breaks substring ties.
        action = max(mentioned, key=lambda value: (lower_text.rfind(value.casefold()), len(value)))
        return action, "last_admissible_mention"

    if candidate:
        return candidate, "freeform_invalid"
    return "look", "empty_fallback_look"


def split_reasoning(text: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.S)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def policy_plan(infos: dict[str, Any]) -> list[str] | None:
    value = infos.get("policy_commands")
    return None if value is None else as_list(value)


def make_user_prompt(
    task: str,
    initial_observation: str,
    current_observation: str,
    admissible: list[str],
    history: list[dict[str, Any]],
    history_turns: int,
) -> str:
    recent = history[-history_turns:]
    trace = "\n".join(
        f"Step {step['step']}: {step['action']}\nObservation: {step['observation_after']}"
        for step in recent
    ) or "(no previous steps)"
    actions = "\n".join(f"- {action}" for action in admissible)
    return f"""Task:
{task}

Initial observation:
{initial_observation}

Recent trajectory:
{trace}

Current observation:
{current_observation}

Admissible actions:
{actions}"""


def generate_action(
    model,
    tokenizer,
    prompt: str,
    admissible: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.no_thinking,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - started
    generated = output_ids[0, inputs.input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    action, parse_method = parse_model_output(text, admissible)
    reasoning, final_text = split_reasoning(text)
    return {
        "model_output": text,
        "reasoning": reasoning,
        "final_text": final_text,
        "action": action,
        "parse_method": parse_method,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(generated.shape[0]),
        "generation_seconds": latency,
    }


def collect_one(
    model,
    tokenizer,
    game_file: str,
    task_index: int,
    sample_index: int,
    rollout_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)
    torch.cuda.manual_seed_all(rollout_seed)

    env = make_env(game_file, args.max_steps)
    initial_observation, infos = env.reset()
    task = extract_task(initial_observation)
    current_observation = initial_observation
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    won = False
    timed_out = False
    episode_started = time.monotonic()

    try:
        for step_index in range(args.max_steps):
            if (
                args.episode_timeout_seconds > 0
                and time.monotonic() - episode_started >= args.episode_timeout_seconds
            ):
                timed_out = True
                break
            admissible = sorted(as_list(infos.get("admissible_commands")))
            oracle_before = policy_plan(infos)
            prompt = make_user_prompt(
                task,
                initial_observation,
                current_observation,
                admissible,
                steps,
                args.history_turns,
            )
            decision = generate_action(model, tokenizer, prompt, admissible, args)
            action = decision["action"]
            next_observation, reward, done, next_infos = env.step(action)
            oracle_after = policy_plan(next_infos)
            won = bool(next_infos.get("won", False))
            reward = float(reward)
            total_reward += reward

            step = {
                "step": step_index,
                "observation_before": current_observation,
                "admissible_actions": admissible,
                "oracle_plan_before": oracle_before,
                **decision,
                "action_valid": action in admissible,
                "oracle_next_match": bool(oracle_before and action == oracle_before[0]),
                "observation_after": next_observation,
                "environment_reward": reward,
                "oracle_plan_after": oracle_after,
                "oracle_distance_before": len(oracle_before) if oracle_before is not None else None,
                "oracle_distance_after": len(oracle_after) if oracle_after is not None else None,
                "oracle_progress_delta": (
                    len(oracle_before) - len(oracle_after)
                    if oracle_before is not None and oracle_after is not None
                    else None
                ),
                "done": bool(done),
                "won": won,
            }
            steps.append(step)
            current_observation, infos = next_observation, next_infos
            if done:
                break
    finally:
        env.close()

    game_path = Path(game_file)
    data_root = args.data_dir.resolve()
    return {
        "schema_version": 1,
        "run_id": args.output_dir.name,
        "trajectory_id": (
            f"{args.output_dir.name}-task{task_index:03d}-sample{sample_index:02d}"
        ),
        "benchmark": "ALFWorld",
        "split": args.split,
        "game_file": str(game_path.resolve().relative_to(data_root)),
        **task_metadata(game_file),
        "task": task,
        "initial_observation": initial_observation,
        "rollout_seed": rollout_seed,
        "policy": {
            "model": args.model,
            "thinking": not args.no_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "history_turns": args.history_turns,
            "system_prompt": SYSTEM_PROMPT,
        },
        "steps": steps,
        "terminal": {
            "done": bool(done),
            "won": won,
            "termination": (
                "success"
                if won
                else "wall_timeout"
                if timed_out
                else "max_steps"
                if len(steps) >= args.max_steps
                else "environment_done"
            ),
            "num_steps": len(steps),
            "environment_return": total_reward,
            "rlvr_reward": float(won),
            "wall_seconds": time.monotonic() - episode_started,
        },
    }


def package_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in ("alfworld", "textworld", "torch", "transformers")
    }


def balanced_sample(
    items: list[str], count: int, seed: int, key: Callable[[str], str]
) -> list[str]:
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    while len(selected) < count:
        for name in sorted(groups):
            if groups[name]:
                selected.append(groups[name].pop())
                if len(selected) == count:
                    break
    return selected


def self_check() -> None:
    admissible = ["look", "go to countertop 1", "take apple 1 from countertop 1"]
    action, method = parse_model_output(
        "<think>Inspect first.</think>\nAction: go to countertop 1", admissible
    )
    assert (action, method) == ("go to countertop 1", "action_line_exact")
    action, method = parse_model_output("I choose take apple 1 from countertop 1", admissible)
    assert (action, method) == (
        "take apple 1 from countertop 1",
        "last_admissible_mention",
    )
    assert extract_task("Room.\nYour task is to: put the apple away") == "put the apple away"
    items = ["a0", "a1", "b0", "b1", "c0", "c1"]
    selected = balanced_sample(items, 6, 7, lambda item: item[0])
    assert [item[0] for item in selected] == ["a", "b", "c", "a", "b", "c"]
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this collector on air-node-02/03/04")
    if args.tasks < 1 or args.samples_per_task < 1 or args.max_steps < 1:
        raise ValueError("tasks, samples-per-task, and max-steps must be positive")
    if args.episode_timeout_seconds < 0:
        raise ValueError("episode-timeout-seconds cannot be negative")
    required = args.data_dir / "json_2.1.1"
    if not required.is_dir():
        raise FileNotFoundError(f"Missing ALFWorld data at {required}; run alfworld-download")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = alfworld_config(args.data_dir, args.max_steps)
    manager = AlfredTWEnv(config, train_eval=args.split)
    games = sorted(manager.game_files)
    if args.tasks > len(games):
        raise ValueError(f"Requested {args.tasks} tasks, split only has {len(games)}")
    selected = (
        balanced_sample(
            games,
            args.tasks,
            args.seed,
            lambda game: task_metadata(game)["task_type"],
        )
        if args.stratify_task_types
        else random.Random(args.seed).sample(games, args.tasks)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()

    manifest = {
        "schema_version": 1,
        "run_id": args.output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "command": sys.argv,
        "versions": package_versions(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_games": [str(Path(game).resolve().relative_to(args.data_dir.resolve())) for game in selected],
        "selected_task_types": [task_metadata(game)["task_type"] for game in selected],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    output_path = args.output_dir / "trajectories.jsonl"
    with output_path.open("x") as output:
        for task_index, game_file in enumerate(selected):
            for sample_index in range(args.samples_per_task):
                rollout_seed = args.seed + task_index * 10_000 + sample_index
                trajectory = collect_one(
                    model,
                    tokenizer,
                    game_file,
                    task_index,
                    sample_index,
                    rollout_seed,
                    args,
                )
                output.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    trajectory["trajectory_id"],
                    "won=" + str(trajectory["terminal"]["won"]),
                    "steps=" + str(trajectory["terminal"]["num_steps"]),
                    flush=True,
                )


if __name__ == "__main__":
    main()
