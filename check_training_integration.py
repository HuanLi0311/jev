#!/usr/bin/env python3
"""One no-GPU check for online Jev reward and turn-level GRPO advantage."""

import io
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dllm/iclr_4/verl-agent"))

import numpy as np
import torch

from agent_system.environments.env_package.alfworld import envs as alf
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from score_jev_v1 import online_alfworld_transition, parse_effect_response
from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_jev_group_grpo_advantage,
    compute_jev_step_grpo_advantage,
)


class Remote:
    def __init__(self, result):
        self.result = result

    def remote(self, *args):
        return deepcopy(self.result)


class Worker:
    def __init__(self):
        self.reset = Remote((
            ["You are here. Your task is to: find the lamp"],
            {"admissible_commands": [["look"]], "won": [False]},
        ))
        self.step = Remote((
            ["You see the lamp."], [0], [False],
            {"admissible_commands": [["take lamp"]], "won": [False]},
        ))


def main():
    environment = alf.AlfworldEnvs.__new__(alf.AlfworldEnvs)
    environment.workers = [Worker()]
    environment.num_processes = 1
    environment.prev_admissible_commands = [None]
    environment.multi_modal = False
    environment.jev_enabled = True
    environment.jev_weight = 0.1
    environment.jev_reward_mode = "trajectory_mean"
    environment.jev_key = "test-only"
    environment.jev_rubric_version = "jev_transition_v2"
    environment.jev_log = io.StringIO()
    environment.jev_rollout = -1
    environment.jev_transition = online_alfworld_transition
    responses = iter([
        {"model": "jev-1.13.0", "answers": {"effect": {"score": 1.0, "confidence": 1.0}}},
        {"model": "jev-1.13.0", "answers": {"effect": {"score": 0.75, "confidence": 0.5}}},
    ])
    environment.jev_query = lambda key, state: next(responses)
    environment.jev_effect = parse_effect_response
    original_get = alf.ray.get
    alf.ray.get = lambda values: values
    try:
        environment.reset()
        _, _, rewards, _, _ = environment.step(["look"])
        _, _, next_rewards, _, _ = environment.step(["take lamp"])
    finally:
        alf.ray.get = original_get
    assert rewards == [0.1]
    assert abs(next_rewards[0] + 0.0375) < 1e-9
    records = [json.loads(line) for line in environment.jev_log.getvalue().splitlines()]
    assert [row["jev_score"] for row in records] == [1.0, 0.75]
    assert [row["jev_confidence"] for row in records] == [1.0, 0.5]
    assert [row["jev_reward"] for row in records] == [1.0, 0.25]
    assert sum(row["jev_aggregate_increment"] for row in records) == 0.625
    assert abs(sum(row["combined_reward"] for row in records) - 0.0625) < 1e-9
    assert records[0]["request_state"]["current_transition"]["action"] == "look"
    assert records[0]["request_state"]["success_criteria"]
    assert all(not any(key in json.dumps(row["request_state"]).lower() for key in ("verifier", "oracle", "reward")) for row in records)

    environment.jev_reward_mode = "step_advantage"
    environment.jev_log = io.StringIO()
    environment.jev_history = [[]]
    environment.jev_sum = [0.0]
    environment.jev_done = [False]
    environment.jev_tasks = ["find the lamp"]
    environment.jev_obs = ["You are here."]
    environment.jev_query = lambda key, state: {
        "model": "jev-1.13.0",
        "answers": {"effect": {"score": 0.75, "confidence": 0.4}},
    }
    alf.ray.get = lambda values: values
    try:
        _, _, step_mode_rewards, _, step_mode_infos = environment.step(["look"])
    finally:
        alf.ray.get = original_get
    assert step_mode_rewards == [0.0]
    assert step_mode_infos[0]["jev_effect_score"] == 0.75
    assert step_mode_infos[0]["jev_confidence"] == 0.4

    manager = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
    manager.config = SimpleNamespace(env=SimpleNamespace(history_length=0, alfworld={"no_thinking": True}))
    prompt = manager.build_text_obs(["Your task is to: find the lamp"], [["look"]], init=True)[0]
    assert "<think>" not in prompt and "Reply only as <action>" in prompt
    assert alfworld_projection(["<action>look</action>"], [["look"]], require_think=False) == (["look"], [1])
    assert alfworld_projection(["LOOK<|im_end|>"], [["look"]], require_think=False) == (["look"], [1])
    assert alfworld_projection(["look elsewhere<|im_end|>"], [["look"]], require_think=False)[1] == [0]
    assert alfworld_projection(["<action>look</action>"], [["look"]])[1] == [0]
    assert alfworld_projection(["look<|im_end|>"], [["look"]])[1] == [0]

    rewards = torch.tensor([[1.0], [1.0], [1.0], [3.0]])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards),
        index=np.array(["task"] * 4),
        traj_index=np.array(["a", "a", "a", "b"]),
        norm_adv_by_std_in_grpo=False,
        compute_mean_std_cross_steps=False,
    )
    assert advantages.flatten().tolist() == [-1.0, -1.0, -1.0, 1.0]

    task_ids = np.array(["task"] * 12)
    traj_ids = np.repeat(np.array(["a", "b", "c", "d"]), 3)
    turns = np.tile(np.arange(3), 4)
    masks = torch.ones((12, 1))
    zero_outcomes = torch.zeros((12, 1))
    effect_scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0, 0.5, 0.5], dtype=np.float32)
    confidences = np.array([0.25] + [1] * 11, dtype=np.float32)
    step_advantages, _, metrics = compute_jev_step_grpo_advantage(
        token_level_rewards=zero_outcomes,
        response_mask=masks,
        effect_scores=effect_scores,
        confidences=confidences,
        index=task_ids,
        traj_index=traj_ids,
        turn_index=turns,
    )
    assert abs(step_advantages[0, 0].item() - 0.2) < 1e-6
    assert len(set(step_advantages[:3, 0].tolist())) > 1
    assert metrics["nonconstant_trajectory_fraction"] > 0
    assert metrics["outcome_equivalent_nonzero_fraction"] > 0

    group_scores = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float32)
    group_confidences = np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)
    group_advantages, _, group_metrics = compute_jev_group_grpo_advantage(
        token_level_rewards=torch.zeros((4, 1)),
        response_mask=torch.ones((4, 1)),
        effect_scores=group_scores,
        confidences=group_confidences,
        index=np.array(["group"] * 4),
        traj_index=np.array(["a", "b", "c", "d"]),
        turn_index=np.zeros(4, dtype=np.int32),
        verifier_weight=0.0,
    )
    weighted_mean = np.average(group_scores, weights=group_confidences)
    expected = group_confidences * (group_scores - weighted_mean)
    assert np.allclose(group_advantages[:, 0].numpy(), expected)
    assert abs(float(np.dot(group_confidences, group_scores - weighted_mean))) < 1e-6
    assert group_metrics["peer_group_fraction"] == 1.0

    # Same trajectory's different turns must never share a Jev baseline.
    per_turn, _, _ = compute_jev_group_grpo_advantage(
        token_level_rewards=torch.zeros((4, 1)),
        response_mask=torch.ones((4, 1)),
        effect_scores=np.array([0.9, 0.1, 0.5, 0.7]),
        confidences=np.ones(4),
        index=np.array(["group"] * 4),
        traj_index=np.array(["a", "a", "b", "b"]),
        turn_index=np.array([0, 1, 0, 1]),
        verifier_weight=0.0,
    )
    assert np.allclose(per_turn[:, 0].numpy(), [0.2, -0.3, -0.2, 0.3])

    singleton, _, singleton_metrics = compute_jev_group_grpo_advantage(
        token_level_rewards=torch.zeros((1, 1)),
        response_mask=torch.ones((1, 1)),
        effect_scores=np.array([0.8]),
        confidences=np.array([0.25]),
        index=np.array(["group"]),
        traj_index=np.array(["last"]),
        turn_index=np.array([9]),
        verifier_weight=0.0,
    )
    assert abs(singleton[0, 0].item() - 0.15) < 1e-6
    assert singleton_metrics["singleton_v2_fallback_fraction"] == 1.0

    outcome_by_traj = np.array([0, 0, 0, 10], dtype=np.float32)
    repeated_outcomes = torch.tensor(np.repeat(outcome_by_traj, 3)).unsqueeze(-1)
    step_baseline, _, _ = compute_jev_step_grpo_advantage(
        token_level_rewards=repeated_outcomes,
        response_mask=masks,
        effect_scores=np.full(12, 0.5, dtype=np.float32),
        confidences=np.ones(12, dtype=np.float32),
        index=task_ids,
        traj_index=traj_ids,
        turn_index=turns,
        verifier_weight=1.0,
    )
    standard_baseline, _ = compute_grpo_outcome_advantage(
        token_level_rewards=repeated_outcomes,
        response_mask=masks,
        index=task_ids,
        traj_index=traj_ids,
        norm_adv_by_std_in_grpo=True,
        compute_mean_std_cross_steps=False,
    )
    assert torch.allclose(step_baseline, standard_baseline)

    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    with tempfile.TemporaryDirectory() as directory:
        log_path = str(Path(directory) / "hindsight.jsonl")
        collector.config = SimpleNamespace(env=SimpleNamespace(alfworld={
            "jev_reward_mode": "hindsight_step_advantage",
            "jev_log_path": log_path,
        }))
        rollout_rows = [[
            {
                "active_masks": True, "anchor_obs": "room", "action_text": "look",
                "task_uid": "game", "jev_effect_scores": 0.5, "jev_confidences": 0.0,
            },
            {
                "active_masks": True, "anchor_obs": "kitchen", "action_text": "take apple",
                "task_uid": "game", "jev_effect_scores": 0.5, "jev_confidences": 0.0,
            },
        ]]
        rollout_infos = [[
            {"observation_text": "You enter the kitchen."},
            {"observation_text": "You take the apple."},
        ]]

        def fake_score(key, trajectory):
            assert key == "test-only" and trajectory["outcome"]["reward"] == 0.0
            assert len(trajectory["steps"]) == 2
            return {
                "trajectory_id": trajectory["trajectory_id"],
                "rubric_version": "test",
                "request": {}, "response": {},
                "step_credit": [
                    {"step_index": 0, "jev_score": 0.8, "jev_confidence": 0.7, "jev_advantage": 0.42},
                    {"step_index": 1, "jev_score": 0.2, "jev_confidence": 0.6, "jev_advantage": -0.36},
                ],
                "latency_seconds": 0.0,
            }

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch(
            "score_jev_v2.score_completed_trajectory", side_effect=fake_score
        ):
            collector._annotate_hindsight_jev(
                SimpleNamespace(tasks=["put apple in fridge"]),
                rollout_rows, rollout_infos, np.array([0.0]),
                {"success_rate": np.array([0.0])}, np.array(["trace"]),
            )
        assert rollout_rows[0][0]["jev_effect_scores"] == np.float32(0.8)
        assert rollout_rows[0][1]["jev_confidences"] == np.float32(0.6)
        hindsight_log = json.loads(Path(log_path).read_text())
        assert hindsight_log["trajectory_id"] == "trace"
        assert hindsight_log["task_uid"] == "game"
    print("online Jev trajectory and V2/V3 step-level GRPO self-check passed")


if __name__ == "__main__":
    main()
