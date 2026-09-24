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
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from score_jev import public_completed_trajectory, questions_for_steps
from verl.trainer.ppo.core_algos import compute_jev_step_grpo_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def check_formal_config():
    import pyarrow.parquet as parquet

    config = yaml.safe_load(
        (Path(__file__).resolve().parent / "config" / "config.yaml").read_text()
    )
    training = config["training"]
    evaluation = config["evaluation"]
    assert training["tasks"] > 0 and training["rollouts"] > 0
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
    assert config["generation"]["enable_thinking"] is False
    assert config["generation"]["evaluation"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": False,
    }
    assert config["optimization"]["learning_rate"] > 0
    assert config["optimization"]["ppo_mini_batch_size"] > 0
    assert config["optimization"]["use_kl_in_reward"] is False
    assert sum(
        parquet.read_metadata(path).num_rows for path in config["data"]["train_files"]
    ) == training["tasks"]
    assert sum(
        parquet.read_metadata(path).num_rows
        for path in config["data"]["validation_files"]
    ) == evaluation["tasks"]


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
        env=SimpleNamespace(history_length=0, alfworld={"no_thinking": True})
    )
    prompt = manager.build_text_obs(
        ["Your task is to: find the lamp"], [["look"]], init=True
    )[0]
    assert "<think>" not in prompt and "Reply only as <action>" in prompt
    assert alfworld_projection(
        ["<action>look</action>"], [["look"]], require_think=False
    ) == (["look"], [1])
    assert alfworld_projection(
        ["look elsewhere<|im_end|>"], [["look"]], require_think=False
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
            "score_jev.score_completed_trajectory", side_effect=fake_score
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


def main():
    check_formal_config()
    check_public_input()
    check_action_format()
    check_advantage()
    check_post_episode_annotation()
    check_persistent_rollout()
    check_validation_panels()
    print("Jev process-reward integration check passed")


if __name__ == "__main__":
    main()
