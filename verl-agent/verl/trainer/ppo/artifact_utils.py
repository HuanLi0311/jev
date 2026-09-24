"""Small, shared writers for comparable PPO training artifacts."""

import json
from pathlib import Path

import numpy as np
import torch


NON_TENSOR_FIELDS = (
    "uid",
    "task_uid",
    "traj_uid",
    "visible_prompt",
    "anchor_obs",
    "action_text",
    "observed_result",
    "turn_index",
    "is_action_valid",
    "done",
    "rewards",
    "episode_rewards",
    "episode_success",
    "episode_lengths",
    "tool_callings",
    "jev_effect_scores",
    "jev_confidences",
    "data_source",
)


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _masked_values(tensor, mask, reduction):
    values = tensor.detach().float().cpu()
    if values.ndim == 1:
        return values.tolist()
    mask = mask.detach().bool().cpu()
    if reduction == "sum":
        return (values * mask).sum(-1).tolist()
    counts = mask.sum(-1).clamp_min(1)
    return ((values * mask).sum(-1) / counts).tolist()


def dump_training_transitions(batch, tokenizer, dump_path, global_step, extra_infos=None):
    """Write one compact JSON record per transition used by an update."""
    target = Path(dump_path)
    target.mkdir(parents=True, exist_ok=True)
    output = target / f"{global_step}.jsonl"

    response_mask = batch.batch["response_mask"].bool()
    response_tokens = response_mask.sum(-1).detach().cpu().tolist()
    records = {
        "schema_version": [1] * len(batch),
        "update": [int(global_step)] * len(batch),
        "input": tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True),
        "output": tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True),
        "response_tokens": response_tokens,
    }
    if "attention_mask" in batch.batch:
        prompt_length = batch.batch["prompts"].shape[-1]
        prompt_tokens = (
            batch.batch["attention_mask"][:, :prompt_length]
            .sum(-1).detach().cpu().tolist()
        )
        records["prompt_tokens"] = prompt_tokens
        records["total_tokens"] = [
            prompt + response
            for prompt, response in zip(prompt_tokens, response_tokens, strict=True)
        ]
    tensor_fields = {
        "token_level_scores": ("score", "sum"),
        "token_level_rewards": ("reward", "sum"),
        "advantages": ("advantage", "mean"),
        "returns": ("return", "mean"),
        "old_log_probs": ("old_log_prob_mean", "mean"),
        "ref_log_prob": ("ref_log_prob_mean", "mean"),
        "rollout_log_probs": ("rollout_log_prob_mean", "mean"),
        "step_rewards": ("step_reward", "mean"),
    }
    for source, (name, reduction) in tensor_fields.items():
        if source in batch.batch:
            records[name] = _masked_values(batch.batch[source], response_mask, reduction)

    for name in NON_TENSOR_FIELDS:
        if name in batch.non_tensor_batch and len(batch.non_tensor_batch[name]) == len(batch):
            records[name] = [_json_value(value) for value in batch.non_tensor_batch[name]]
    for name, values in (extra_infos or {}).items():
        if name not in records and len(values) == len(batch):
            records[name] = [_json_value(value) for value in values]

    identities = list(zip(
        records.get("traj_uid", [""] * len(batch)),
        records.get("turn_index", range(len(batch))),
    ))
    seen = set()
    duplicates = []
    for identity in identities:
        duplicates.append(identity in seen)
        seen.add(identity)
    records["padding_duplicate"] = duplicates

    with output.open("x", encoding="utf-8") as sink:
        for index in range(len(batch)):
            sink.write(json.dumps(
                {name: values[index] for name, values in records.items()},
                ensure_ascii=False,
            ) + "\n")
    print(f"Dumped training transitions to {output}")
