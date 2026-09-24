#!/usr/bin/env bash
set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/JJ_Group/lih2511/.cache/alfworld}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

MODEL_PATH=${MODEL_PATH:-/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-4}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-32}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TEST_FREQ=${TEST_FREQ:-2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ours_qwen2.5_1.5b_seed0_pilot}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/ours_alfworld_1p5b}

if [[ "${SKIP_PREPARE:-0}" != 1 ]]; then
    python -m examples.data_preprocess.prepare \
        --mode text \
        --train_data_size "$TRAIN_DATA_SIZE" \
        --val_data_size "$VAL_DATA_SIZE"
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ours \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    algorithm.ours.minimum_peers=3 \
    algorithm.gamma=0.95 \
    algorithm.ours.process_weight=1.0 \
    algorithm.ours.outcome_weight=0.2 \
    algorithm.ours.advantage_clip=3.0 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.history_length=2 \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=ours_alfworld_1p5b \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.val_before_train=True \
    trainer.rollout_data_dir="$OUTPUT_ROOT/rollouts" \
    trainer.default_local_dir="$OUTPUT_ROOT/checkpoints" \
    trainer.resume_mode=disable \
    "$@"
