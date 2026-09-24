#!/usr/bin/env bash
set -euo pipefail

usage='usage: run_alfworld.sh grpo|jev|gigpo|hgpo|graphgpo RUN_TAG SEED'
arm=${1:?$usage}
run_tag=${2:?$usage}
seed=${3:?$usage}
(( $# == 3 )) || { echo "$usage" >&2; exit 2; }
case $arm in
    baseline|sparse_grpo) arm=grpo ;;
    jev-only) arm=jev ;;
    grpo|jev|gigpo|hgpo|graphgpo) ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
[[ $run_tag =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run tag' >&2; exit 2; }
[[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+,){1,3}[0-9]+$ ]] || { echo 'set CUDA_VISIBLE_DEVICES to two or four GPU indices' >&2; exit 2; }
IFS=, read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
gpu_count=${#cuda_devices[@]}
(( gpu_count == 2 || gpu_count == 4 )) || { echo 'use exactly two or four GPUs' >&2; exit 2; }
[[ $(printf '%s\n' "${cuda_devices[@]}" | sort -u | wc -l) -eq $gpu_count ]] || { echo 'GPU indices must be unique' >&2; exit 2; }

root=/home/JJ_Group/lih2511
project=$root/test/jev
repo=$project/verl-agent
python=$root/.conda/envs/verl/bin/python
config_path=${CONFIG_PATH:-$project/config/config.yaml}
[[ -f $config_path ]] || { echo "config missing: $config_path" >&2; exit 2; }

config_output=$(
    "$python" - "$config_path" "$seed" <<'PY'
import json
import sys
from pathlib import Path

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
if eval_generation["do_sample"]:
    raise SystemExit("formal evaluation must be greedy")

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
PY
)
declare -A cfg
while IFS=$'\t' read -r key value; do
    cfg[$key]=$value
done <<< "$config_output"

model_path=${cfg[model_path]}
[[ -f $model_path/config.json ]] || { echo "model config missing: $model_path" >&2; exit 2; }

case $arm in
    grpo)
        entrypoint=verl.trainer.main_ppo
        method_args=(algorithm.adv_estimator=grpo +algorithm.grpo_cross_steps="${cfg[method.grpo.cross_steps]}")
        ;;
    jev)
        entrypoint=verl.trainer.main_ppo
        method_args=(algorithm.adv_estimator=jev_step_grpo)
        ;;
    gigpo)
        entrypoint=verl.trainer.main_ppo
        method_args=(
            algorithm.adv_estimator=gigpo
            algorithm.gamma="${cfg[method.gigpo.gamma]}"
            algorithm.gigpo.step_advantage_w="${cfg[method.gigpo.step_advantage_w]}"
            algorithm.gigpo.mode="${cfg[method.gigpo.mode]}"
            algorithm.gigpo.enable_similarity="${cfg[method.gigpo.enable_similarity]}"
            algorithm.gigpo.similarity_thresh="${cfg[method.gigpo.similarity_thresh]}"
        )
        ;;
    hgpo)
        entrypoint=recipe.hgpo.main_hgpo
        method_args=(
            algorithm.adv_estimator=hgpo
            algorithm.gamma="${cfg[method.hgpo.gamma]}"
            algorithm.hgpo.mode="${cfg[method.hgpo.mode]}"
            algorithm.hgpo.weight_type="${cfg[method.hgpo.weight_type]}"
            algorithm.hgpo.length_weight_alpha="${cfg[method.hgpo.length_weight_alpha]}"
            algorithm.hgpo.base_group="${cfg[method.hgpo.base_group]}"
        )
        ;;
    graphgpo)
        entrypoint=recipe.GraphGPO.main_graphgpo
        method_args=(
            algorithm.adv_estimator=graphgpo
            algorithm.gamma="${cfg[method.graphgpo.gamma]}"
            algorithm.graphgpo.step_advantage_w="${cfg[method.graphgpo.step_advantage_w]}"
            algorithm.graphgpo.episode_advantage_w="${cfg[method.graphgpo.episode_advantage_w]}"
            algorithm.graphgpo.mode="${cfg[method.graphgpo.mode]}"
            algorithm.graphgpo.enable_similarity="${cfg[method.graphgpo.enable_similarity]}"
            algorithm.graphgpo.similarity_thresh="${cfg[method.graphgpo.similarity_thresh]}"
            algorithm.graphgpo.normalize_distance="${cfg[method.graphgpo.normalize_distance]}"
        )
        ;;
esac

optimizer_offload=${OPTIMIZER_OFFLOAD:-${cfg[optimizer_offload]}}
ref_param_offload=${REF_PARAM_OFFLOAD:-${cfg[reference_parameter_offload]}}
persistent_rollout=${PERSISTENT_ROLLOUT:-${cfg[persistent_rollout]}}
for value in "$optimizer_offload" "$ref_param_offload" "$persistent_rollout"; do
    [[ $value == true || $value == false ]] || { echo 'boolean runtime overrides must be true or false' >&2; exit 2; }
done
actor_micro_batch=${ACTOR_MICRO_BATCH:-${cfg[actor_micro_batch_size_per_gpu]}}
log_prob_micro_batch=${LOG_PROB_MICRO_BATCH:-${cfg[log_prob_micro_batch_size_per_gpu]}}
tensor_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-${cfg[tensor_model_parallel_size]}}
ray_num_cpus=${RAY_NUM_CPUS:-${cfg[ray_num_cpus]}}
for value in "$actor_micro_batch" "$log_prob_micro_batch" "$tensor_parallel_size" "$ray_num_cpus"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'integer runtime overrides must be positive' >&2; exit 2; }
done
rollout_gpu_util=${ROLLOUT_GPU_UTIL:-${cfg[rollout_gpu_memory_utilization]}}
[[ $rollout_gpu_util =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo 'ROLLOUT_GPU_UTIL must be in [0,1]' >&2; exit 2; }

if [[ ${CHECK_CONFIG_ONLY:-false} == true ]]; then
    printf 'arm=%s entrypoint=%s seed=%s data_seed=%s rollout_seed=%s groups=%s rollouts=%s steps=%s updates=%s panels=%s milestones=%s invalid_shaping=%s\n' \
        "$arm" "$entrypoint" "$seed" "$seed" "$seed" "${cfg[groups]}" "${cfg[rollouts]}" \
        "${cfg[max_steps]}" "${cfg[updates]}" "${cfg[eval_panels]}" "${cfg[milestones]}" \
        "${cfg[invalid_action_shaping]}"
    exit 0
fi

run_dir=$project/runs/grpo-alfworld-$run_tag
resume_mode=${RESUME_MODE:-disable}
[[ $resume_mode == disable || $resume_mode == auto || $resume_mode == resume_path ]] || { echo 'invalid RESUME_MODE' >&2; exit 2; }
selection=$(printf 'arm=%s\nseed=%s\n' "$arm" "$seed")
if [[ $resume_mode == disable ]]; then
    [[ ! -e $run_dir ]] || { echo "run directory exists: $run_dir" >&2; exit 2; }
    mkdir -p "$run_dir"
    cp "$config_path" "$run_dir/experiment-config.yaml"
    printf '%s' "$selection" > "$run_dir/run-selection.txt"
else
    [[ -d $run_dir ]] || { echo "resume run directory missing: $run_dir" >&2; exit 2; }
    cmp -s "$config_path" "$run_dir/experiment-config.yaml" || { echo 'config differs from the original run' >&2; exit 2; }
    [[ $(<"$run_dir/run-selection.txt") == "$selection" ]] || { echo 'arm or seed differs from the original run' >&2; exit 2; }
fi

runtime_selection=$(
    printf 'arm=%s\nseed=%s\ndata_seed=%s\nrollout_seed=%s\n' "$arm" "$seed" "$seed" "$seed"
    printf 'gpu_count=%s\nactor_micro_batch_size_per_gpu=%s\nlog_prob_micro_batch_size_per_gpu=%s\n' "$gpu_count" "$actor_micro_batch" "$log_prob_micro_batch"
    printf 'optimizer_offload=%s\nreference_parameter_offload=%s\npersistent_rollout=%s\n' "$optimizer_offload" "$ref_param_offload" "$persistent_rollout"
    printf 'tensor_model_parallel_size=%s\nrollout_gpu_memory_utilization=%s\nray_num_cpus=%s\n' "$tensor_parallel_size" "$rollout_gpu_util" "$ray_num_cpus"
)
if [[ $resume_mode == disable ]]; then
    printf '%s' "$runtime_selection" > "$run_dir/resolved-runtime.txt"
else
    [[ $(<"$run_dir/resolved-runtime.txt") == "$runtime_selection" ]] || {
        echo 'runtime settings differ from the original run' >&2; exit 2;
    }
fi

export ALFWORLD_DATA=$root/.cache/alfworld
export PYTHONPATH=$project/src:$repo${PYTHONPATH:+:$PYTHONPATH}
export PATH=$root/.conda/envs/verl/bin:$PATH
python_flags=()
if [[ ${PYTHON_NO_SITE:-false} == true ]]; then
    # ponytail: skip unrelated editable .pth hooks during concurrent Ray worker startup.
    export PYTHONPATH=$root/.conda/envs/verl/lib/python3.10/site-packages:$PYTHONPATH
    python_flags=(-S)
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASHINFER TORCHDYNAMO_DISABLE=1
# ponytail: a per-run tmpfs path lets two local Ray clusters coexist safely.
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm}
mkdir -p "$RAY_TMPDIR"
if [[ $arm == jev && -z ${TYPESAFE_API_KEY:-} ]]; then
    read -r -s -p 'Jev API key: ' TYPESAFE_API_KEY
    printf '\n'
    export TYPESAFE_API_KEY
fi
if [[ $arm == jev ]]; then
    # ponytail: the cluster's localhost proxy is absent on compute nodes; direct HTTPS works.
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
fi
if [[ $resume_mode == disable ]]; then
    exec > >(tee "$run_dir/train.log") 2>&1
else
    exec > >(tee -a "$run_dir/train.log") 2>&1
fi

stdlib_shm=${STDLIB_SHM:-false}
[[ $stdlib_shm == true || $stdlib_shm == false ]] || { echo 'STDLIB_SHM must be true or false' >&2; exit 2; }
if [[ $stdlib_shm == true ]]; then
    # ponytail: tmpfs cache assumes an immutable conda env; delete it after Python upgrades.
    stdlib_cache=/dev/shm/jev-python-stdlib
    flock /dev/shm/jev-python-stdlib.lock bash -e -c '
        if [[ ! -f "$2/.complete" ]]; then
            rsync -a --exclude=site-packages "$1/" "$2/"
            touch "$2/.complete"
        fi
    ' _ "$root/.conda/envs/verl/lib/python3.10" "$stdlib_cache"
    export PYTHONPATH=$stdlib_cache:$PYTHONPATH
fi
model_shm=${MODEL_SHM:-false}
[[ $model_shm == true || $model_shm == false ]] || { echo 'MODEL_SHM must be true or false' >&2; exit 2; }
if [[ $model_shm == true ]]; then
    # ponytail: one lock serializes cold copies; use per-model locks if staging many models.
    model_hash=$(printf %s "$model_path" | md5sum | cut -d' ' -f1)
    model_cache=/dev/shm/verl-cache/$model_hash/$(basename "$model_path")
    mkdir -p "$model_cache"
    flock /dev/shm/jev-model-cache.lock rsync -aL --delete "$model_path/" "$model_cache/"
    model_path=$model_cache
fi

jev_process_reward=false
[[ $arm == jev ]] && jev_process_reward=true
no_thinking=true
[[ ${cfg[enable_thinking]} == true ]] && no_thinking=false
common_args=(
    "data.train_files=${cfg[train_files]}"
    "data.val_files=${cfg[validation_files]}"
    "data.train_batch_size=${cfg[groups]}"
    "data.val_batch_size=${cfg[panel_size]}"
    "data.shuffle=${cfg[data_shuffle]}"
    "+data.seed=$seed"
    +data.dataloader_num_workers=0
    "data.max_prompt_length=${cfg[max_prompt_length]}"
    "data.max_response_length=${cfg[max_response_length]}"
    "data.filter_overlong_prompts=${cfg[filter_overlong_prompts]}"
    "data.truncation=${cfg[truncation]}"
    "data.return_raw_chat=${cfg[return_raw_chat]}"
    "+data.apply_chat_template_kwargs.enable_thinking=${cfg[enable_thinking]}"
    "actor_rollout_ref.model.path=$model_path"
    "actor_rollout_ref.actor.optim.lr=${cfg[learning_rate]}"
    "actor_rollout_ref.actor.use_torch_compile=${cfg[use_torch_compile]}"
    "actor_rollout_ref.model.use_remove_padding=${cfg[use_remove_padding]}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${cfg[ppo_mini_batch_size]}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$actor_micro_batch"
    "actor_rollout_ref.actor.use_kl_loss=${cfg[use_kl_loss]}"
    "actor_rollout_ref.actor.kl_loss_coef=${cfg[kl_loss_coef]}"
    "actor_rollout_ref.actor.kl_loss_type=${cfg[kl_loss_type]}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=${cfg[gradient_checkpointing]}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${cfg[actor_parameter_offload]}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload"
    actor_rollout_ref.rollout.name=vllm
    "+actor_rollout_ref.rollout.seed=$seed"
    "actor_rollout_ref.rollout.temperature=${cfg[train_temperature]}"
    "actor_rollout_ref.rollout.top_p=${cfg[train_top_p]}"
    "actor_rollout_ref.rollout.top_k=${cfg[train_top_k]}"
    "actor_rollout_ref.rollout.do_sample=${cfg[train_do_sample]}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${cfg[eval_temperature]}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${cfg[eval_top_p]}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${cfg[eval_top_k]}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${cfg[eval_do_sample]}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_parallel_size"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_util"
    "actor_rollout_ref.rollout.enforce_eager=${cfg[enforce_eager]}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${cfg[enable_chunked_prefill]}"
    "actor_rollout_ref.rollout.free_cache_engine=${cfg[free_cache_engine]}"
    "+actor_rollout_ref.rollout.persistent_across_turns=$persistent_rollout"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch"
    "actor_rollout_ref.ref.fsdp_config.param_offload=$ref_param_offload"
    "actor_rollout_ref.actor.use_invalid_action_penalty=${cfg[invalid_action_shaping]}"
    "algorithm.use_kl_in_reward=${cfg[use_kl_in_reward]}"
    env.env_name=alfworld/AlfredTWEnv
    "env.seed=$seed"
    "env.history_length=${cfg[history_length]}"
    "env.max_steps=${cfg[max_steps]}"
    "env.rollout.n=${cfg[rollouts]}"
    "+env.alfworld.no_thinking=$no_thinking"
    "+env.alfworld.jev_process_reward=$jev_process_reward"
    "+env.alfworld.jev_log_path=$run_dir/jev-process.jsonl"
    env.resources_per_worker.num_cpus=0.1
    "ray_init.num_cpus=$ray_num_cpus"
    +ray_init.address=local
    +ray_init.include_dashboard=false
    'trainer.logger=["console"]'
    trainer.project_name=jev_alfworld
    "trainer.experiment_name=$run_tag"
    "trainer.n_gpus_per_node=$gpu_count"
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=false
    +trainer.external_evaluator=true
    "+trainer.checkpoint_steps=${cfg[milestones]}"
    '+trainer.eval_steps=[]'
    "trainer.max_actor_ckpt_to_keep=${cfg[milestone_count]}"
    "trainer.total_epochs=${cfg[updates]}"
    "trainer.total_training_steps=${cfg[updates]}"
    "trainer.resume_mode=$resume_mode"
    "trainer.resume_from_path=${RESUME_FROM_PATH:-null}"
    "trainer.rollout_data_dir=$run_dir/rollouts"
    trainer.validation_data_dir=null
    "trainer.default_local_dir=$run_dir/checkpoints"
)

cd "$repo"
exec "$python" "${python_flags[@]}" -m "$entrypoint" "${common_args[@]}" "${method_args[@]}"
