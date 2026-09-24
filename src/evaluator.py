#!/usr/bin/env python3
"""Evaluate any formal ALFWorld actor checkpoint on the two fixed panels."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "verl-agent"
DEFAULT_CONFIG = PROJECT / "config" / "config.yaml"
EXPECTED_PANELS = {
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="global_step_N directory, or 'base'")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    evaluation = config.get("evaluation", {})
    if evaluation.get("seed") != 1000:
        raise ValueError("formal evaluation seed must be 1000")
    if evaluation.get("panels") != EXPECTED_PANELS:
        raise ValueError(f"evaluation.panels must be {EXPECTED_PANELS}")
    if config.get("training", {}).get("invalid_action_shaping") is not False:
        raise ValueError("formal comparison requires invalid_action_shaping=false")
    model_path = Path(config.get("model_path", ""))
    if not (model_path / "config.json").is_file():
        raise ValueError(f"model snapshot is unavailable: {model_path}")
    validation_files = config.get("data", {}).get("validation_files", [])
    if not validation_files or any(not Path(path).is_file() for path in validation_files):
        raise ValueError("data.validation_files must contain existing files")
    generation = config.get("generation", {}).get("evaluation", {})
    if generation.get("do_sample") is not False or generation.get("temperature") != 0.0:
        raise ValueError("formal evaluation must use greedy decoding")
    return config


def bool_text(value: Any, name: str) -> str:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return str(value).lower()


def runtime_value(config: dict[str, Any], key: str, env_name: str) -> str:
    value = os.environ.get(env_name, config["runtime"][key])
    return str(value).lower() if type(value) is bool else str(value)


def checkpoint_details(value: str, milestones: list[int]) -> tuple[Path | None, int]:
    if value == "base":
        return None, 0
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_dir() or not (checkpoint / "actor").is_dir():
        raise ValueError(f"invalid actor checkpoint: {checkpoint}")
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    if not match:
        raise ValueError("checkpoint directory must be named global_step_N")
    step = int(match.group(1))
    if step not in milestones:
        raise ValueError(f"checkpoint step must be one of {milestones}")
    return checkpoint, step


def gpu_count() -> int:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) not in (2, 4) or any(not item.isdigit() for item in devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to two or four GPU indices")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU indices must be unique")
    return len(devices)


def build_command(
    config: dict[str, Any], checkpoint: Path | None, output: Path, devices: int
) -> list[str]:
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]
    optimization = config["optimization"]
    generation = config["generation"]
    eval_generation = generation["evaluation"]
    runtime = config["runtime"]
    resume_mode = "disable" if checkpoint is None else "resume_path"
    resume_path = "null" if checkpoint is None else str(checkpoint)
    actor_micro_batch = runtime_value(
        config, "actor_micro_batch_size_per_gpu", "ACTOR_MICRO_BATCH"
    )
    log_prob_micro_batch = runtime_value(
        config, "log_prob_micro_batch_size_per_gpu", "LOG_PROB_MICRO_BATCH"
    )
    tensor_parallel = runtime_value(
        config, "tensor_model_parallel_size", "TENSOR_MODEL_PARALLEL_SIZE"
    )
    gpu_util = runtime_value(
        config, "rollout_gpu_memory_utilization", "ROLLOUT_GPU_UTIL"
    )
    ray_cpus = runtime_value(config, "ray_num_cpus", "RAY_NUM_CPUS")
    optimizer_offload = runtime_value(
        config, "optimizer_offload", "OPTIMIZER_OFFLOAD"
    )
    persistent_rollout = runtime_value(
        config, "persistent_rollout", "PERSISTENT_ROLLOUT"
    )

    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "+algorithm.grpo_cross_steps=false",
        f"data.train_files={json.dumps(data['train_files'], separators=(',', ':'))}",
        f"data.val_files={json.dumps(data['validation_files'], separators=(',', ':'))}",
        f"data.train_batch_size={training['tasks']}",
        f"data.val_batch_size={evaluation['tasks']}",
        f"data.shuffle={bool_text(data['shuffle'], 'data.shuffle')}",
        f"+data.seed={evaluation['seed']}",
        "+data.dataloader_num_workers=0",
        f"data.max_prompt_length={generation['max_prompt_length']}",
        f"data.max_response_length={generation['max_response_length']}",
        f"data.filter_overlong_prompts={bool_text(data['filter_overlong_prompts'], 'data.filter_overlong_prompts')}",
        f"data.truncation={data['truncation']}",
        f"data.return_raw_chat={bool_text(data['return_raw_chat'], 'data.return_raw_chat')}",
        f"+data.apply_chat_template_kwargs.enable_thinking={bool_text(generation['enable_thinking'], 'enable_thinking')}",
        f"actor_rollout_ref.model.path={config['model_path']}",
        f"actor_rollout_ref.actor.optim.lr={optimization['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={optimization['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor_micro_batch}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        f"actor_rollout_ref.actor.use_torch_compile={bool_text(runtime['use_torch_compile'], 'use_torch_compile')}",
        f"actor_rollout_ref.model.use_remove_padding={bool_text(runtime['use_remove_padding'], 'use_remove_padding')}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={bool_text(runtime['gradient_checkpointing'], 'gradient_checkpointing')}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={bool_text(runtime['actor_parameter_offload'], 'actor_parameter_offload')}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optimizer_offload}",
        "actor_rollout_ref.rollout.name=vllm",
        f"+actor_rollout_ref.rollout.seed={evaluation['seed']}",
        f"actor_rollout_ref.rollout.temperature={eval_generation['temperature']}",
        f"actor_rollout_ref.rollout.top_p={eval_generation['top_p']}",
        f"actor_rollout_ref.rollout.top_k={eval_generation['top_k']}",
        f"actor_rollout_ref.rollout.do_sample={bool_text(eval_generation['do_sample'], 'evaluation.do_sample')}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={eval_generation['temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={eval_generation['top_p']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={eval_generation['top_k']}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={bool_text(eval_generation['do_sample'], 'evaluation.do_sample')}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_util}",
        f"actor_rollout_ref.rollout.enforce_eager={bool_text(runtime['enforce_eager'], 'enforce_eager')}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={bool_text(runtime['enable_chunked_prefill'], 'enable_chunked_prefill')}",
        f"actor_rollout_ref.rollout.free_cache_engine={bool_text(runtime['free_cache_engine'], 'free_cache_engine')}",
        f"+actor_rollout_ref.rollout.persistent_across_turns={persistent_rollout}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch}",
        f"actor_rollout_ref.actor.use_invalid_action_penalty={bool_text(training['invalid_action_shaping'], 'invalid_action_shaping')}",
        f"algorithm.use_kl_in_reward={bool_text(optimization['use_kl_in_reward'], 'use_kl_in_reward')}",
        "env.env_name=alfworld/AlfredTWEnv",
        f"env.seed={evaluation['seed']}",
        f"env.history_length={training['history_length']}",
        f"env.max_steps={training['max_steps']}",
        "env.rollout.n=1",
        f"+env.alfworld.eval_panels={json.dumps(evaluation['panels'], separators=(',', ':'))}",
        f"+env.alfworld.eval_seed={evaluation['seed']}",
        f"+env.alfworld.no_thinking={str(not generation['enable_thinking']).lower()}",
        "env.resources_per_worker.num_cpus=0.1",
        f"ray_init.num_cpus={ray_cpus}",
        "+ray_init.address=local",
        "+ray_init.include_dashboard=false",
        'trainer.logger=["console"]',
        "trainer.project_name=jev_alfworld_evaluation",
        f"trainer.experiment_name={output.name}",
        f"trainer.n_gpus_per_node={devices}",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=true",
        "trainer.val_only=true",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.resume_from_path={resume_path}",
        "trainer.rollout_data_dir=null",
        f"trainer.validation_data_dir={output}",
        f"trainer.default_local_dir={output / 'unused-checkpoints'}",
    ]


def source_selection(checkpoint: Path | None) -> dict[str, str]:
    if checkpoint is None:
        return {"arm": "base", "seed": "base"}
    selection_path = checkpoint.parent.parent / "run-selection.txt"
    if not selection_path.is_file():
        return {}
    return dict(
        line.split("=", 1)
        for line in selection_path.read_text().splitlines()
        if "=" in line
    )


def aggregate_panel(raw_path: Path, task_path: Path) -> dict[str, Any]:
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with raw_path.open() as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                trajectories[str(row["traj_uid"])].append(row)
    records = []
    for trajectory_id, rows in trajectories.items():
        task_ids = {str(row["task_uid"]) for row in rows}
        if len(task_ids) != 1:
            raise ValueError(f"trajectory {trajectory_id} spans multiple tasks")
        rows.sort(key=lambda row: int(row["turn_index"]))
        scores = {float(row["score"]) for row in rows}
        if len(scores) != 1:
            raise ValueError(f"trajectory {trajectory_id} has inconsistent scores")
        score = scores.pop()
        records.append(
            {
                "task_uid": task_ids.pop(),
                "trajectory_id": trajectory_id,
                "success": score > 0,
                "score": score,
                "steps": len(rows),
                "valid_actions": sum(bool(row["is_action_valid"]) for row in rows),
            }
        )
    records.sort(key=lambda row: (row["task_uid"], row["trajectory_id"]))
    with task_path.open("x") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    transitions = sum(record["steps"] for record in records)
    return {
        "tasks": len(records),
        "success_rate": sum(record["success"] for record in records) / len(records),
        "mean_score": sum(record["score"] for record in records) / len(records),
        "mean_steps": transitions / len(records),
        "valid_action_rate": (
            sum(record["valid_actions"] for record in records) / transitions
        ),
        "transitions": transitions,
        "task_records": str(task_path),
        "raw_records": str(raw_path),
    }


def run() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    checkpoint, step = checkpoint_details(
        args.checkpoint, config["evaluation"]["milestones"]
    )
    output = args.output.expanduser().resolve()
    devices = gpu_count()
    command = build_command(config, checkpoint, output, devices)
    if args.check_only:
        print(shlex.join(command))
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"evaluation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "experiment-config.yaml")
    manifest_path = output / "manifest.json"
    manifest = {
        "status": "running",
        "checkpoint": "base" if checkpoint is None else str(checkpoint),
        "checkpoint_step": step,
        "source": source_selection(checkpoint),
        "config": str(config_path),
        "evaluation_seed": config["evaluation"]["seed"],
        "panels": config["evaluation"]["panels"],
        "tasks_per_panel": config["evaluation"]["tasks"],
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = os.environ.copy()
    python_paths = [str(PROJECT / "src"), str(REPO)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "ALFWORLD_DATA": str(PROJECT.parents[1] / ".cache" / "alfworld"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "TORCHDYNAMO_DISABLE": "1",
            "RAY_TMPDIR": environment.get("RAY_TMPDIR", "/dev/shm"),
        }
    )
    try:
        subprocess.run(command, cwd=REPO, env=environment, check=True)
        summaries = {}
        for panel in EXPECTED_PANELS:
            raw_path = output / panel / f"{step}.jsonl"
            if not raw_path.is_file():
                raise FileNotFoundError(f"missing panel records: {raw_path}")
            task_path = raw_path.with_name(f"tasks-{step}.jsonl")
            summary = aggregate_panel(raw_path, task_path)
            if summary["tasks"] != config["evaluation"]["tasks"]:
                raise ValueError(
                    f"{panel} produced {summary['tasks']} tasks, expected "
                    f"{config['evaluation']['tasks']}"
                )
            summaries[panel] = summary
        manifest.update({"status": "complete", "results": summaries})
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    run()
