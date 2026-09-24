#!/usr/bin/env bash
set -euo pipefail

usage='usage: run_grpo_alfworld.sh baseline|jev RUN_TAG SEED [off|on]'
arm=${1:?$usage}
run_tag=${2:?$usage}
seed=${3:?$usage}
shaping_mode=${4:-}
[[ $arm == baseline || $arm == jev ]] || { echo 'arm must be baseline or jev' >&2; exit 2; }
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
    "$python" - "$config_path" "$seed" "$shaping_mode" <<'PY'
import json
import sys
from pathlib import Path

import yaml

path, selected_seed, selected_mode = sys.argv[1:]
config = yaml.safe_load(Path(path).read_text())
if not isinstance(config, dict):
    raise SystemExit("config must be a mapping")

def positive(section, key):
    value = section.get(key)
    if type(value) is not int or value <= 0:
        raise SystemExit(f"{key} must be a positive integer")
    return value

training = config.get("training", {})
groups = positive(training, "task")
rollouts = positive(training, "samples")
max_steps = positive(training, "max_steps")
history_length = training.get("history_length")
if type(history_length) is not int or history_length < 0:
    raise SystemExit("history_length must be a nonnegative integer")
updates = positive(training, "updates")
seeds = training.get("paired_seeds")
if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
    raise SystemExit("paired_seeds must contain three distinct seeds")
if any(type(value) is not int or value <= 0 for value in seeds):
    raise SystemExit("paired_seeds must be positive integers and exclude seed 0")
try:
    selected_seed = int(selected_seed)
except ValueError as error:
    raise SystemExit("SEED must be an integer") from error
if selected_seed not in seeds:
    raise SystemExit(f"SEED must be one of {seeds}")

evaluation = config.get("evaluation", {})
panel_size = positive(evaluation, "tasks_per_panel")
eval_seed = evaluation.get("seed")
if type(eval_seed) is not int or eval_seed < 0:
    raise SystemExit("evaluation.seed must be a nonnegative integer")
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

shaping = config.get("invalid_action_shaping", {})
modes = shaping.get("modes")
if modes != {"off": False, "on": True}:
    raise SystemExit("invalid_action_shaping.modes must define off=false and on=true")
selected_mode = selected_mode or shaping.get("default", "")
if selected_mode not in modes:
    raise SystemExit("invalid-action shaping mode must be off or on")
coefficient = shaping.get("coefficient")
if type(coefficient) not in (int, float) or coefficient < 0:
    raise SystemExit("invalid-action shaping coefficient must be nonnegative")

model_path = config.get("model_path")
if not isinstance(model_path, str) or not model_path.startswith("/"):
    raise SystemExit("model_path must be absolute")
data = config.get("data", {})
train_files = data.get("train_files")
validation_files = data.get("validation_files")
for name, paths in (("train_files", train_files), ("validation_files", validation_files)):
    if not isinstance(paths, list) or not paths or any(not Path(item).is_file() for item in paths):
        raise SystemExit(f"data.{name} must contain existing files")

values = [
    model_path, json.dumps(train_files), json.dumps(validation_files),
    groups, rollouts, max_steps, history_length, updates,
    panel_size, eval_seed, json.dumps(panels, separators=(",", ":")),
    json.dumps(milestones, separators=(",", ":")), len(milestones),
    selected_mode, str(modes[selected_mode]).lower(), coefficient,
]
print("\n".join(map(str, values)))
PY
)
mapfile -t config_values <<< "$config_output"
(( ${#config_values[@]} == 16 )) || { echo 'config parser returned incomplete data' >&2; exit 2; }
model_path=${config_values[0]}
train_files=${config_values[1]}
validation_files=${config_values[2]}
train_batch_size=${config_values[3]}
rollouts_per_group=${config_values[4]}
max_steps=${config_values[5]}
history_length=${config_values[6]}
updates=${config_values[7]}
val_batch_size=${config_values[8]}
eval_seed=${config_values[9]}
eval_panels=${config_values[10]}
milestones=${config_values[11]}
milestone_count=${config_values[12]}
shaping_mode=${config_values[13]}
invalid_action_shaping=${config_values[14]}
invalid_action_penalty=${config_values[15]}
[[ -f $model_path/config.json ]] || { echo "model config missing: $model_path" >&2; exit 2; }
if [[ ${CHECK_CONFIG_ONLY:-false} == true ]]; then
    printf 'seed=%s groups=%s rollouts=%s steps=%s updates=%s panels=%s milestones=%s shaping=%s\n' \
        "$seed" "$train_batch_size" "$rollouts_per_group" "$max_steps" "$updates" \
        "$eval_panels" "$milestones" "$shaping_mode"
    exit 0
fi
run_dir=$project/runs/grpo-alfworld-$run_tag
resume_mode=${RESUME_MODE:-disable}
[[ $resume_mode == disable || $resume_mode == auto || $resume_mode == resume_path ]] || { echo 'invalid RESUME_MODE' >&2; exit 2; }
if [[ $resume_mode == disable ]]; then
    [[ ! -e $run_dir ]] || { echo "run directory exists: $run_dir" >&2; exit 2; }
    mkdir -p "$run_dir"
    cp "$config_path" "$run_dir/experiment-config.yaml"
    printf 'seed=%s\ninvalid_action_shaping=%s\n' "$seed" "$shaping_mode" > "$run_dir/run-selection.txt"
else
    [[ -d $run_dir ]] || { echo "resume run directory missing: $run_dir" >&2; exit 2; }
    cmp -s "$config_path" "$run_dir/experiment-config.yaml" || { echo 'config differs from the original run' >&2; exit 2; }
    [[ $(<"$run_dir/run-selection.txt") == $(printf 'seed=%s\ninvalid_action_shaping=%s' "$seed" "$shaping_mode") ]] || {
        echo 'seed or shaping mode differs from the original run' >&2; exit 2;
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

if [[ $arm == jev ]]; then
    adv_estimator=jev_step_grpo
    jev_process_reward=true
else
    adv_estimator=grpo
    jev_process_reward=false
fi
optimizer_offload=${OPTIMIZER_OFFLOAD:-true}
[[ $optimizer_offload == true || $optimizer_offload == false ]] || { echo 'OPTIMIZER_OFFLOAD must be true or false' >&2; exit 2; }
actor_micro_batch=${ACTOR_MICRO_BATCH:-1}
log_prob_micro_batch=${LOG_PROB_MICRO_BATCH:-1}
for value in "$actor_micro_batch" "$log_prob_micro_batch"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'microbatch sizes must be positive integers' >&2; exit 2; }
done
ref_param_offload=${REF_PARAM_OFFLOAD:-true}
[[ $ref_param_offload == true || $ref_param_offload == false ]] || { echo 'REF_PARAM_OFFLOAD must be true or false' >&2; exit 2; }
persistent_rollout=${PERSISTENT_ROLLOUT:-false}
[[ $persistent_rollout == true || $persistent_rollout == false ]] || { echo 'PERSISTENT_ROLLOUT must be true or false' >&2; exit 2; }
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
rollout_gpu_util=${ROLLOUT_GPU_UTIL:-0.40}
ray_num_cpus=${RAY_NUM_CPUS:-32}
[[ $ray_num_cpus =~ ^[1-9][0-9]*$ ]] || { echo 'RAY_NUM_CPUS must be a positive integer' >&2; exit 2; }

cd "$repo"
exec "$root/.conda/envs/verl/bin/python" "${python_flags[@]}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$adv_estimator" +algorithm.grpo_cross_steps=false \
    data.train_files="$train_files" data.val_files="$validation_files" \
    data.train_batch_size="$train_batch_size" data.val_batch_size="$val_batch_size" data.shuffle=false \
    +data.dataloader_num_workers=0 \
    data.max_prompt_length=2048 data.max_response_length=256 \
    data.filter_overlong_prompts=true data.truncation=left data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$actor_micro_batch" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$optimizer_offload" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_util" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    +actor_rollout_ref.rollout.persistent_across_turns="$persistent_rollout" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$ref_param_offload" \
    actor_rollout_ref.actor.use_invalid_action_penalty="$invalid_action_shaping" \
    actor_rollout_ref.actor.invalid_action_penalty_coef="$invalid_action_penalty" \
    algorithm.use_kl_in_reward=false \
    env.env_name=alfworld/AlfredTWEnv env.seed="$seed" \
    env.history_length="$history_length" env.max_steps="$max_steps" env.rollout.n="$rollouts_per_group" \
    +env.alfworld.eval_panels="$eval_panels" +env.alfworld.eval_seed="$eval_seed" \
    +env.alfworld.no_thinking=true \
    +env.alfworld.jev_process_reward="$jev_process_reward" \
    +env.alfworld.jev_log_path="$run_dir/jev-process.jsonl" \
    env.resources_per_worker.num_cpus=0.1 ray_init.num_cpus="$ray_num_cpus" +ray_init.address=local +ray_init.include_dashboard=false \
    trainer.logger='["console"]' trainer.project_name=jev_alfworld \
    trainer.experiment_name="$run_tag" trainer.n_gpus_per_node="$gpu_count" trainer.nnodes=1 \
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=false \
    +trainer.checkpoint_steps="$milestones" +trainer.eval_steps="$milestones" \
    trainer.max_actor_ckpt_to_keep="$milestone_count" \
    trainer.total_epochs="$updates" trainer.total_training_steps="$updates" \
    trainer.resume_mode="$resume_mode" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    trainer.rollout_data_dir="$run_dir/rollouts" \
    trainer.validation_data_dir="$run_dir/evaluations" \
    trainer.default_local_dir="$run_dir/checkpoints"
