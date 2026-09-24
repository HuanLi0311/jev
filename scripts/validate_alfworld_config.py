#!/usr/bin/env python3
"""Validate and render one formal ALFWorld training configuration."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as parquet
import yaml

path, selected_seed = sys.argv[1:]
config = yaml.safe_load(Path(path).read_text())
if not isinstance(config, dict):
    raise SystemExit("config must be a mapping")


def integer(section, key, *, positive=True):
    value = section.get(key)
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise SystemExit(f"{key} must be a {qualifier} integer")
    return value


def number(section, key, *, minimum=None, maximum=None):
    value = section.get(key)
    if type(value) not in (int, float):
        raise SystemExit(f"{key} must be numeric")
    value = float(value)
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise SystemExit(f"{key} is outside the allowed range")
    return value


def boolean(section, key):
    value = section.get(key)
    if type(value) is not bool:
        raise SystemExit(f"{key} must be boolean")
    return value


def rendered(value):
    if type(value) is bool:
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


training = config.get("training", {})
groups = integer(training, "tasks")
rollouts = integer(training, "rollouts")
max_steps = integer(training, "max_steps")
history_length = integer(training, "history_length", positive=False)
updates = integer(training, "updates")
invalid_action_shaping = boolean(training, "invalid_action_shaping")
if invalid_action_shaping:
    raise SystemExit("formal comparison requires invalid_action_shaping=false")
seeds = training.get("paired_seeds")
if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
    raise SystemExit("paired_seeds must contain distinct seeds")
if any(type(value) is not int or value <= 0 for value in seeds):
    raise SystemExit("paired_seeds must be positive integers and exclude seed 0")
try:
    selected_seed = int(selected_seed)
except ValueError as error:
    raise SystemExit("SEED must be an integer") from error
if selected_seed not in seeds:
    raise SystemExit(f"SEED must be one of {seeds}")

evaluation = config.get("evaluation", {})
panel_size = integer(evaluation, "tasks")
eval_seed = integer(evaluation, "seed", positive=False)
if eval_seed != 1000:
    raise SystemExit("formal evaluation seed must be 1000")
panels = evaluation.get("panels")
expected_panels = {
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}
if panels != expected_panels:
    raise SystemExit(f"evaluation.panels must be {expected_panels}")
milestones = evaluation.get("milestones")
if (
    not isinstance(milestones, list)
    or any(type(value) is not int for value in milestones)
    or milestones != sorted(set(milestones))
    or not milestones
    or milestones[0] != 0
    or milestones[-1] != updates
):
    raise SystemExit("milestones must be sorted, unique, start at 0, and end at updates")

model_path = config.get("model_path")
if not isinstance(model_path, str) or not model_path.startswith("/"):
    raise SystemExit("model_path must be absolute")
data = config.get("data", {})
train_files = data.get("train_files")
validation_files = data.get("validation_files")
for name, paths in (("train_files", train_files), ("validation_files", validation_files)):
    if not isinstance(paths, list) or not paths or any(not Path(item).is_file() for item in paths):
        raise SystemExit(f"data.{name} must contain existing files")


def validate_slots(name, paths, expected, split):
    resolved = [str(Path(item).resolve()) for item in paths]
    if len(resolved) != len(set(resolved)):
        raise SystemExit(f"data.{name} contains duplicate parquet paths")
    rows = [
        row
        for item in resolved
        for row in parquet.read_table(item).to_pylist()
    ]
    infos = [row.get("extra_info") for row in rows]
    slots = [info.get("slot") if isinstance(info, dict) else None for info in infos]
    if len(rows) != expected or None in slots or len(set(slots)) != expected:
        raise SystemExit(f"data.{name} must contain {expected} unique slot IDs")
    if any(row.get("data_source") != "alfworld" for row in rows):
        raise SystemExit(f"data.{name} data_source must be alfworld")
    if any(info.get("split") != split for info in infos):
        raise SystemExit(f"data.{name} split must be {split}")
    prompt = [{"role": "user", "content": ""}]
    if any(row.get("prompt") != prompt for row in rows):
        raise SystemExit(f"data.{name} contains a non-placeholder prompt")


validate_slots("train_files", train_files, groups, "train")
validate_slots("validation_files", validation_files, panel_size, "evaluation")
data_shuffle = boolean(data, "shuffle")
filter_overlong_prompts = boolean(data, "filter_overlong_prompts")
truncation = data.get("truncation")
if truncation not in {"error", "left", "right", "middle"}:
    raise SystemExit("unsupported data.truncation")
return_raw_chat = boolean(data, "return_raw_chat")

optimization = config.get("optimization", {})
learning_rate = number(optimization, "learning_rate", minimum=0)
if learning_rate == 0:
    raise SystemExit("learning_rate must be positive")
ppo_mini_batch_size = integer(optimization, "ppo_mini_batch_size")
use_kl_loss = boolean(optimization, "use_kl_loss")
use_kl_in_reward = boolean(optimization, "use_kl_in_reward")
kl_loss_coef = number(optimization, "kl_loss_coef", minimum=0)
kl_loss_type = optimization.get("kl_loss_type")
if kl_loss_type not in {"kl", "abs", "mse", "low_var_kl", "full"}:
    raise SystemExit("unsupported kl_loss_type")

generation = config.get("generation", {})
max_prompt_length = integer(generation, "max_prompt_length")
max_response_length = integer(generation, "max_response_length")
enable_thinking = boolean(generation, "enable_thinking")
train_generation = generation.get("train", {})
eval_generation = generation.get("evaluation", {})
for name, section in (("train", train_generation), ("evaluation", eval_generation)):
    number(section, "temperature", minimum=0)
    number(section, "top_p", minimum=0, maximum=1)
    if type(section.get("top_k")) is not int:
        raise SystemExit(f"generation.{name}.top_k must be an integer")
    boolean(section, "do_sample")
if eval_generation["do_sample"] and eval_generation["temperature"] <= 0:
    raise SystemExit("sampled evaluation requires positive temperature")

runtime = config.get("runtime", {})
for key in (
    "actor_micro_batch_size_per_gpu", "log_prob_micro_batch_size_per_gpu",
    "tensor_model_parallel_size", "ray_num_cpus",
):
    integer(runtime, key)
for key in (
    "optimizer_offload", "reference_parameter_offload", "persistent_rollout",
    "use_remove_padding", "use_torch_compile", "gradient_checkpointing",
    "actor_parameter_offload",
    "enforce_eager", "enable_chunked_prefill", "free_cache_engine",
):
    boolean(runtime, key)
number(runtime, "rollout_gpu_memory_utilization", minimum=0, maximum=1)

methods = config.get("methods", {})
required_methods = {"grpo", "gigpo", "hgpo", "graphgpo"}
if not required_methods <= methods.keys():
    raise SystemExit(f"methods must define {sorted(required_methods)}")

values = {
    "model_path": model_path,
    "train_files": train_files,
    "validation_files": validation_files,
    "data_shuffle": data_shuffle,
    "filter_overlong_prompts": filter_overlong_prompts,
    "truncation": truncation,
    "return_raw_chat": return_raw_chat,
    "groups": groups,
    "rollouts": rollouts,
    "max_steps": max_steps,
    "history_length": history_length,
    "updates": updates,
    "panel_size": panel_size,
    "eval_seed": eval_seed,
    "eval_panels": panels,
    "milestones": milestones,
    "milestone_count": len(milestones),
    "invalid_action_shaping": invalid_action_shaping,
    "learning_rate": learning_rate,
    "ppo_mini_batch_size": ppo_mini_batch_size,
    "use_kl_loss": use_kl_loss,
    "use_kl_in_reward": use_kl_in_reward,
    "kl_loss_coef": kl_loss_coef,
    "kl_loss_type": kl_loss_type,
    "max_prompt_length": max_prompt_length,
    "max_response_length": max_response_length,
    "enable_thinking": enable_thinking,
    "train_temperature": train_generation["temperature"],
    "train_top_p": train_generation["top_p"],
    "train_top_k": train_generation["top_k"],
    "train_do_sample": train_generation["do_sample"],
    "eval_temperature": eval_generation["temperature"],
    "eval_top_p": eval_generation["top_p"],
    "eval_top_k": eval_generation["top_k"],
    "eval_do_sample": eval_generation["do_sample"],
    **runtime,
}
for method_name, method_config in methods.items():
    if not isinstance(method_config, dict):
        raise SystemExit(f"methods.{method_name} must be a mapping")
    for key, value in method_config.items():
        values[f"method.{method_name}.{key}"] = value
for key, value in values.items():
    text = rendered(value)
    if "\t" in text or "\n" in text:
        raise SystemExit(f"unsupported whitespace in {key}")
    print(f"{key}\t{text}")
