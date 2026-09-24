#!/usr/bin/env python3
"""No-GPU integration check for the single Jev process-reward path."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "verl-agent"))

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluator import aggregate_panel
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld.envs import worker_seed_and_offset
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from score import public_completed_trajectory, questions_for_steps
from verl.trainer.ppo.artifact_utils import dump_training_transitions
from verl.trainer.ppo.core_algos import compute_jev_step_grpo_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.logger.aggregate_logger import LocalLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def check_formal_config():
    import pyarrow.parquet as parquet

    config = yaml.safe_load(
        (Path(__file__).resolve().parent / "config" / "config.yaml").read_text()
    )
    training = config["training"]
    evaluation = config["evaluation"]
    assert training["tasks"] > 0 and training["rollouts"] > 0
    assert training["paired_seeds"] == [1]
    assert 0 not in training["paired_seeds"]
    assert len(training["paired_seeds"]) == len(set(training["paired_seeds"]))
    assert evaluation["milestones"][0] == 0
    assert evaluation["milestones"][-1] == training["updates"]
    assert evaluation["seed"] == 1000
    assert evaluation["panels"] == {
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
    }
    assert training["invalid_action_shaping"] is False
    assert config["data"]["shuffle"] is False
    assert config["data"]["truncation"] == "left"
    assert config["generation"]["max_response_length"] == 512
    assert config["generation"]["enable_thinking"] is True
    assert config["generation"]["evaluation"] == {
        "temperature": 0.4,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": True,
    }
    assert config["optimization"]["learning_rate"] > 0
    assert config["optimization"]["ppo_mini_batch_size"] == 256
    assert config["optimization"]["use_kl_in_reward"] is False
    runtime = config["runtime"]
    assert runtime["rollout_gpu_memory_utilization"] == 0.20
    assert runtime["tensor_model_parallel_size"] == 2
    assert runtime["use_remove_padding"] is True
    assert runtime["enforce_eager"] is False
    assert runtime["enable_chunked_prefill"] is False
    assert runtime["free_cache_engine"] is False
    for files, expected in (
        (config["data"]["train_files"], training["tasks"]),
        (config["data"]["validation_files"], evaluation["tasks"]),
    ):
        assert len(files) == len(set(files))
        rows = [row for path in files for row in parquet.read_table(path).to_pylist()]
        slots = [row["extra_info"]["slot"] for row in rows]
        assert len(rows) == expected
        assert len(set(slots)) == expected
        assert all(row["data_source"] == "alfworld" for row in rows)


def check_eval_task_assignment():
    config = yaml.safe_load(
        (Path(__file__).resolve().parent / "config" / "config.yaml").read_text()
    )
    eval_tasks = config["evaluation"]["tasks"]
    train_tasks = config["training"]["tasks"]
    eval_seed = config["evaluation"]["seed"]
    train_seed = config["training"]["paired_seeds"][0]
    assert [worker_seed_and_offset(eval_seed, i, False) for i in range(eval_tasks)] == [
        (eval_seed, i) for i in range(eval_tasks)
    ]
    assert [worker_seed_and_offset(train_seed, i, True) for i in range(train_tasks)] == [
        (train_seed + i, 0) for i in range(train_tasks)
    ]


def check_public_input():
    leak = "SECRET_ORACLE_SENTINEL"
    trajectory = {
        "trajectory_id": "trace",
        "task": "put apple in fridge",
        "outcome": {
            "reward": 0,
            "success": False,
            "reward_definition": "10 iff every condition is satisfied, otherwise 0",
            "private_verifier": leak,
        },
        "steps": [{
            "observation": "An apple is here.",
            "action": "take apple",
            "observed_result": "You take the apple.",
            "oracle": leak,
        }],
        "gold_path": leak,
    }
    state = public_completed_trajectory(trajectory)
    assert leak not in json.dumps(state)
    assert state["verified_outcome"]["success"] is False
    assert "verified final outcome" in questions_for_steps(1)["step_0000"]["instructions"]


def check_action_format():
    manager = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
    manager.config = SimpleNamespace(
        env=SimpleNamespace(history_length=0, alfworld={"no_thinking": False})
    )
    prompt = manager.build_text_obs(
        ["Your task is to: find the lamp"], [["look"]], init=True
    )[0]
    assert "<think>" in prompt and "<action>" in prompt
    assert alfworld_projection(
        ["<think>I should inspect the room.</think><action>look</action>"],
        [["look"]],
        require_think=True,
    ) == (["look"], [1])
    assert alfworld_projection(
        ["<action>look</action>"], [["look"]], require_think=True
    )[1] == [0]


def check_advantage():
    masks = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    kwargs = {
        "response_mask": masks,
        "effect_scores": np.array([0.9, 0.5, 0.2], dtype=np.float32),
        "confidences": np.array([0.25, 1.0, 0.5], dtype=np.float32),
        "index": np.array(["task"] * 3),
        "traj_index": np.array(["a", "b", "c"]),
        "turn_index": np.zeros(3, dtype=np.int32),
    }
    advantages, returns, metrics = compute_jev_step_grpo_advantage(
        token_level_rewards=torch.zeros_like(masks), **kwargs
    )
    expected = torch.tensor([
        [0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0],
        [-0.3, -0.3, -0.3],
    ])
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.equal(advantages, returns)
    assert metrics["nonzero_transition_fraction"] == 2 / 3

    # The verifier outcome conditions Jev upstream, but is not added to A again.
    changed_outcomes, _, _ = compute_jev_step_grpo_advantage(
        token_level_rewards=torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]),
        **kwargs,
    )
    assert torch.equal(advantages, changed_outcomes)


def check_post_episode_annotation():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    with tempfile.TemporaryDirectory() as directory:
        log_path = str(Path(directory) / "jev-process.jsonl")
        collector.config = SimpleNamespace(
            env=SimpleNamespace(alfworld={"jev_log_path": log_path})
        )
        rollout_rows = [[
            {
                "active_masks": True,
                "anchor_obs": "room",
                "action_text": "look",
                "task_uid": "game",
                "jev_effect_scores": 0.5,
                "jev_confidences": 0.0,
            },
            {
                "active_masks": True,
                "anchor_obs": "kitchen",
                "action_text": "take apple",
                "task_uid": "game",
                "jev_effect_scores": 0.5,
                "jev_confidences": 0.0,
            },
        ]]
        rollout_infos = [[
            {"observation_text": "You enter the kitchen."},
            {"observation_text": "You take the apple."},
        ]]

        def fake_score(key, trajectory):
            assert key == "test-only"
            assert trajectory["outcome"] == {
                "reward": 0.0,
                "success": False,
                "reward_definition": (
                    "ALFWorld sparse verifier: 10 iff every task condition is "
                    "satisfied, otherwise 0."
                ),
            }
            assert len(trajectory["steps"]) == 2
            return {
                "trajectory_id": trajectory["trajectory_id"],
                "rubric_version": "test",
                "request": {},
                "response": {},
                "step_credit": [
                    {"step_index": 0, "jev_score": 0.8, "jev_confidence": 0.7},
                    {"step_index": 1, "jev_score": 0.2, "jev_confidence": 0.6},
                ],
                "latency_seconds": 0.0,
            }

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch(
            "score.score_completed_trajectory", side_effect=fake_score
        ):
            collector._annotate_jev_process_rewards(
                SimpleNamespace(tasks=["put apple in fridge"]),
                rollout_rows,
                rollout_infos,
                np.array([0.0]),
                {"success_rate": np.array([0.0])},
                np.array(["trace"]),
            )
        assert rollout_rows[0][0]["jev_effect_scores"] == np.float32(0.8)
        assert rollout_rows[0][1]["jev_confidences"] == np.float32(0.6)
        record = json.loads(Path(log_path).read_text())
        assert record["trajectory_id"] == "trace" and record["task_uid"] == "game"


def check_persistent_rollout():
    calls = []
    manager = SimpleNamespace(
        __enter__=lambda: calls.append("wake"),
        __exit__=lambda *_: calls.append("sleep"),
    )
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker._is_rollout = True
    worker._rollout_open = False
    worker.rollout_sharding_manager = manager
    worker.begin_rollout()
    worker.end_rollout()
    assert calls == ["wake", "sleep"] and not worker._rollout_open

    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout={"persistent_across_turns": True})
    )
    collector._vanilla_multi_turn_loop = lambda *_: "rows"
    group = SimpleNamespace(
        begin_rollout=lambda: calls.append("begin"),
        end_rollout=lambda: calls.append("end"),
    )
    assert collector.vanilla_multi_turn_loop(None, group, None) == "rows"
    assert calls[-2:] == ["begin", "end"]

    def fail(*_):
        raise RuntimeError("rollout failed")

    collector._vanilla_multi_turn_loop = fail
    try:
        collector.vanilla_multi_turn_loop(None, group, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("rollout error was swallowed")
    assert calls[-2:] == ["begin", "end"]


def check_validation_panels():
    closed = []

    class Panel:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.val_envs = {
        "valid_seen": lambda: Panel("valid_seen"),
        "valid_unseen": lambda: Panel("valid_unseen"),
    }
    trainer._validate_one = lambda panel, name: {f"val/{name}/score": panel.name}
    assert trainer._validate() == {
        "val/valid_seen/score": "valid_seen",
        "val/valid_unseen/score": "valid_unseen",
    }
    assert closed == ["valid_seen", "valid_unseen"]


def check_evaluator_summary():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw_path = root / "0.jsonl"
        rows = [
            {"traj_uid": "a", "task_uid": "task-a", "turn_index": 0,
             "score": 1, "is_action_valid": True},
            {"traj_uid": "a", "task_uid": "task-a", "turn_index": 1,
             "score": 1, "is_action_valid": True},
            {"traj_uid": "b", "task_uid": "task-b", "turn_index": 0,
             "score": 0, "is_action_valid": False},
        ]
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = aggregate_panel(raw_path, root / "tasks-0.jsonl")
        assert summary["tasks"] == 2
        assert summary["unique_tasks"] == 2
        assert summary["success_rate"] == 0.5
        assert summary["transitions"] == 3
        assert summary["valid_action_rate"] == 2 / 3


def check_training_artifacts():
    class Batch:
        def __init__(self):
            self.batch = {
                "prompts": torch.tensor([[1, 2], [1, 2]]),
                "responses": torch.tensor([[3, 4], [3, 0]]),
                "response_mask": torch.tensor([[1, 1], [1, 0]]),
                "attention_mask": torch.tensor([
                    [1, 1, 1, 1], [1, 1, 1, 0]
                ]),
                "token_level_scores": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
                "advantages": torch.tensor([[0.2, 0.4], [-0.1, 0.0]]),
                "step_rewards": torch.tensor([0.75, -0.25]),
            }
            self.non_tensor_batch = {
                "traj_uid": np.array(["trace", "trace"], dtype=object),
                "turn_index": np.array([0, 0]),
                "task_uid": np.array(["task", "task"], dtype=object),
                "episode_success": np.array([True, True]),
            }

        def __len__(self):
            return 2

    tokenizer = SimpleNamespace(
        batch_decode=lambda values, skip_special_tokens: ["decoded"] * len(values)
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dump_training_transitions(Batch(), tokenizer, root / "rollouts", 1)
        rows = [
            json.loads(line)
            for line in (root / "rollouts" / "1.jsonl").read_text().splitlines()
        ]
        assert abs(rows[0]["advantage"] - 0.3) < 1e-6
        assert rows[0]["step_reward"] == 0.75
        assert rows[0]["prompt_tokens"] == 2 and rows[0]["total_tokens"] == 4
        assert [row["padding_duplicate"] for row in rows] == [False, True]

        metrics_path = root / "metrics.jsonl"
        with patch.dict(os.environ, {"VERL_METRICS_FILE": str(metrics_path)}):
            LocalLogger().log({"loss": np.float32(1.5), "text": "ignored"}, 7)
        metrics = json.loads(metrics_path.read_text())
        assert metrics["step"] == 7 and metrics["loss"] == 1.5
        assert "time_unix" in metrics and "text" not in metrics
def main():
    check_formal_config()
    check_eval_task_assignment()
    check_public_input()
    check_action_format()
    check_advantage()
    check_post_episode_annotation()
    check_persistent_rollout()
    check_validation_panels()
    check_evaluator_summary()
    check_training_artifacts()
    print("Jev process-reward integration check passed")


if __name__ == "__main__":
    main()
