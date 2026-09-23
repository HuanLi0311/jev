#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: run_grpo_alfworld.sh baseline|jev RUN_TAG}
run_tag=${2:?usage: run_grpo_alfworld.sh baseline|jev RUN_TAG}
[[ $arm == baseline || $arm == jev ]] || { echo 'arm must be baseline or jev' >&2; exit 2; }
[[ $run_tag =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run tag' >&2; exit 2; }
[[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+,){1,3}[0-9]+$ ]] || { echo 'set CUDA_VISIBLE_DEVICES to two or four GPU indices' >&2; exit 2; }
IFS=, read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
gpu_count=${#cuda_devices[@]}
(( gpu_count == 2 || gpu_count == 4 )) || { echo 'use exactly two or four GPUs' >&2; exit 2; }
[[ $(printf '%s\n' "${cuda_devices[@]}" | sort -u | wc -l) -eq $gpu_count ]] || { echo 'GPU indices must be unique' >&2; exit 2; }

root=/home/JJ_Group/lih2511
repo=$root/test/dllm/iclr_4/verl-agent
model_path=${MODEL_PATH:-$root/.cache/huggingface/hub/Qwen3-1.7B}
[[ -f $model_path/config.json ]] || { echo "model config missing: $model_path" >&2; exit 2; }
run_dir=$root/test/jev/runs/grpo-alfworld-$run_tag
resume_mode=${RESUME_MODE:-disable}
[[ $resume_mode == disable || $resume_mode == auto || $resume_mode == resume_path ]] || { echo 'invalid RESUME_MODE' >&2; exit 2; }
if [[ $resume_mode == disable ]]; then
    [[ ! -e $run_dir ]] || { echo "run directory exists: $run_dir" >&2; exit 2; }
    mkdir -p "$run_dir"
else
    [[ -d $run_dir ]] || { echo "resume run directory missing: $run_dir" >&2; exit 2; }
fi

export ALFWORLD_DATA=$root/.cache/alfworld
export PYTHONPATH=$root/test/jev:$repo${PYTHONPATH:+:$PYTHONPATH}
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

jev_weight=0
if [[ $arm == jev ]]; then
    # ponytail: fixed pilot coefficient; tune only on development tasks before held-out claims.
    jev_weight=0.1
fi
updates=${TRAIN_UPDATES:-1}
max_steps=${MAX_STEPS:-10}
val_batch_size=${VAL_BATCH_SIZE:-64}
[[ $val_batch_size =~ ^[1-9][0-9]*$ ]] || { echo 'VAL_BATCH_SIZE must be a positive integer' >&2; exit 2; }
test_freq=${TEST_FREQ:-1}
save_freq=${SAVE_FREQ:--1}
val_before_train=${VAL_BEFORE_TRAIN:-true}
[[ $val_before_train == true || $val_before_train == false ]] || { echo 'VAL_BEFORE_TRAIN must be true or false' >&2; exit 2; }
optimizer_offload=${OPTIMIZER_OFFLOAD:-true}
[[ $optimizer_offload == true || $optimizer_offload == false ]] || { echo 'OPTIMIZER_OFFLOAD must be true or false' >&2; exit 2; }
actor_micro_batch=${ACTOR_MICRO_BATCH:-1}
log_prob_micro_batch=${LOG_PROB_MICRO_BATCH:-1}
for value in "$actor_micro_batch" "$log_prob_micro_batch"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'microbatch sizes must be positive integers' >&2; exit 2; }
done
remove_padding=${REMOVE_PADDING:-false}
ref_param_offload=${REF_PARAM_OFFLOAD:-true}
for value in "$remove_padding" "$ref_param_offload"; do
    [[ $value == true || $value == false ]] || { echo 'REMOVE_PADDING and REF_PARAM_OFFLOAD must be true or false' >&2; exit 2; }
done
rollout_gpu_util=${ROLLOUT_GPU_UTIL:-0.40}
ray_num_cpus=${RAY_NUM_CPUS:-16}
[[ $ray_num_cpus =~ ^[1-9][0-9]*$ ]] || { echo 'RAY_NUM_CPUS must be a positive integer' >&2; exit 2; }
adv_estimator=${ADV_ESTIMATOR:-grpo}
[[ $adv_estimator == grpo || $adv_estimator == jev_step_grpo || $adv_estimator == jev_group_grpo ]] || { echo 'invalid ADV_ESTIMATOR' >&2; exit 2; }
jev_reward_mode=${JEV_REWARD_MODE:-trajectory_mean}
[[ $jev_reward_mode == trajectory_mean || $jev_reward_mode == step_advantage || $jev_reward_mode == hindsight_step_advantage || $jev_reward_mode == hindsight_group_advantage || $jev_reward_mode == hindsight_step_only_advantage ]] || { echo 'invalid JEV_REWARD_MODE' >&2; exit 2; }
if [[ $adv_estimator == jev_step_grpo && $jev_reward_mode != step_advantage && $jev_reward_mode != hindsight_step_advantage && $jev_reward_mode != hindsight_step_only_advantage ]]; then
    echo 'jev_step_grpo requires a step-advantage JEV_REWARD_MODE' >&2
    exit 2
fi
jev_verifier_weight=${JEV_VERIFIER_WEIGHT:-0.1}
[[ $jev_verifier_weight =~ ^(0|[0-9]+([.][0-9]+)?)$ ]] || { echo 'JEV_VERIFIER_WEIGHT must be nonnegative' >&2; exit 2; }
if [[ $adv_estimator == jev_group_grpo && $jev_reward_mode != hindsight_group_advantage ]]; then
    echo 'jev_group_grpo requires JEV_REWARD_MODE=hindsight_group_advantage' >&2
    exit 2
fi
seed=${SEED:-0}
[[ $seed =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
eval_split=${EVAL_SPLIT:-eval_in_distribution}
[[ $eval_split == eval_in_distribution || $eval_split == eval_out_of_distribution ]] || { echo 'invalid EVAL_SPLIT' >&2; exit 2; }

cd "$repo"
exec "$root/.conda/envs/verl/bin/python" "${python_flags[@]}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$adv_estimator" +algorithm.grpo_cross_steps=false \
    algorithm.jev_step.verifier_weight="$jev_verifier_weight" \
    data.train_files="$root/data/verl-agent/text/train.parquet" \
    data.val_files="$root/data/verl-agent/text/test.parquet" \
    data.train_batch_size=4 data.val_batch_size="$val_batch_size" data.shuffle=false \
    data.max_prompt_length=2048 data.max_response_length=256 \
    data.filter_overlong_prompts=true data.truncation=left data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.use_remove_padding="$remove_padding" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$log_prob_micro_batch" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$ref_param_offload" \
    actor_rollout_ref.actor.use_invalid_action_penalty=false \
    algorithm.use_kl_in_reward=false \
    env.env_name=alfworld/AlfredTWEnv env.seed="$seed" \
    env.history_length=2 env.max_steps="$max_steps" env.rollout.n=4 \
    env.alfworld.eval_dataset="$eval_split" \
    +env.alfworld.no_thinking=true \
    +env.alfworld.jev_weight="$jev_weight" \
    +env.alfworld.jev_reward_mode="$jev_reward_mode" \
    +env.alfworld.jev_log_path="$run_dir/jev-online.jsonl" \
    env.resources_per_worker.num_cpus=0.1 ray_init.num_cpus="$ray_num_cpus" +ray_init.address=local +ray_init.include_dashboard=false \
    trainer.logger='["console"]' trainer.project_name=jev_alfworld \
    trainer.experiment_name="$run_tag" trainer.n_gpus_per_node="$gpu_count" trainer.nnodes=1 \
    trainer.save_freq="$save_freq" trainer.test_freq="$test_freq" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs="$updates" trainer.total_training_steps="$updates" \
    trainer.val_before_train="$val_before_train" trainer.resume_mode="$resume_mode" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    trainer.rollout_data_dir="$run_dir/rollouts" \
    trainer.default_local_dir="$run_dir/checkpoints"
