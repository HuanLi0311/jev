import numpy as np
import subprocess
import sys
import torch

from gigpo.core_ours import _action, _goal, compute_ours_advantage


def test_text_alfworld_import_does_not_eagerly_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import agent_system.environments.env_package.alfworld.envs; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_action_matches_alfworld_first_tag_projection():
    text = "<think><action>go to desk 1</action></think><action>look</action>"
    assert _action(text) == "go to desk 1"


def test_exact_successors_rank_actions_and_broadcast():
    assert _goal("history apple\nYour task is to: put mug in cabinet\ncurrent apple") == "put mug in cabinet"

    prompts, observations, actions, tasks, trajectories, turns, outcomes = [], [], [], [], [], [], []
    for trajectory in range(8):
        success = trajectory < 4
        prompts.extend([
            "Your admissible actions of the current situation are: ['go to kitchen 1', 'go to desk 1'].",
            (
                "Your admissible actions of the current situation are: ['look']."
                + (" good" if success else " bad")
            ),
        ])
        observations.extend(["start", "good" if success else "bad"])
        actions.extend([
            "<action>go to kitchen 1</action>" if success else "<action>go to desk 1</action>",
            "<action>look</action>",
        ])
        tasks.extend(["task"] * 2)
        trajectories.extend([f"traj-{trajectory}"] * 2)
        turns.extend([0, 1])
        outcomes.extend([float(success)] * 2)

    rewards = torch.zeros(16, 4)
    rewards[:, -1] = torch.tensor(outcomes)
    step_rewards = np.asarray([
        reward for trajectory in range(8)
        for reward in (0.0, float(trajectory < 4))
    ])
    mask = torch.ones_like(rewards)
    advantages, _, stats = compute_ours_advantage(
        rewards,
        mask,
        step_rewards,
        np.ones(16, dtype=bool),
        np.asarray(prompts, dtype=object),
        np.asarray(observations, dtype=object),
        np.asarray(actions, dtype=object),
        np.asarray(tasks, dtype=object),
        np.asarray(trajectories, dtype=object),
        np.asarray(turns),
    )

    assert advantages[0, 0] > 0
    assert advantages[8, 0] < 0
    assert stats["nonzero_checkpoint_fraction"] > 0
    assert stats["process_broadcast_fraction"] > 0
    assert stats["process_broadcast_transition_fraction"] > 0
    assert stats["progress_q1_transitions"] == 8
    assert stats["length_long_transitions"] == 16


def test_same_successor_does_not_reverse_reward_sign():
    prompts = np.asarray([
        "Your admissible actions of the current situation are: ['go to kitchen 1', 'take apple 1 from table 1'].",
        "Your admissible actions of the current situation are: ['look']. shared",
        "Your admissible actions of the current situation are: ['go to kitchen 1', 'take apple 1 from table 1'].",
        "Your admissible actions of the current situation are: ['look']. shared",
        "Your admissible actions of the current situation are: ['go to kitchen 1', 'take apple 1 from table 1'].",
        "Your admissible actions of the current situation are: ['look']. unique",
    ], dtype=object)
    observations = np.asarray(["start", "shared", "start", "shared", "start", "unique"], dtype=object)
    actions = np.asarray([
        "<action>go to kitchen 1</action>", "<action>look</action>",
        "<action>go to kitchen 1</action>", "<action>look</action>",
        "<action>take apple 1 from table 1</action>", "<action>look</action>",
    ], dtype=object)
    trajectories = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    rewards = torch.zeros(6, 2)
    rewards[:, -1] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    advantages, _, stats = compute_ours_advantage(
        rewards,
        torch.ones_like(rewards),
        np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.ones(6, dtype=bool),
        prompts,
        observations,
        actions,
        np.asarray(["task"] * 6, dtype=object),
        trajectories,
        np.asarray([0, 1, 0, 1, 0, 1]),
        minimum_peers=1,
        outcome_weight=0.0,
    )

    assert stats["rewarded_fraction"] > 0
    assert advantages[0, 0] > 0
    assert advantages[4, 0] < 0


def test_invalid_action_signal_when_all_terminal_outcomes_tie():
    rewards = torch.zeros(4, 2)
    advantages, _, stats = compute_ours_advantage(
        rewards,
        torch.ones_like(rewards),
        np.zeros(4),
        np.asarray([True, True, False, False]),
        np.asarray(["Your admissible actions of the current situation are: ['look'].\n"] * 4, dtype=object),
        np.asarray(["start"] * 4, dtype=object),
        np.asarray(["<action>look</action>"] * 2 + ["<action>bogus</action>"] * 2, dtype=object),
        np.asarray(["task"] * 4, dtype=object),
        np.asarray(["a", "b", "c", "d"], dtype=object),
        np.zeros(4, dtype=int),
        minimum_peers=1,
    )
    assert stats["nonzero_checkpoint_fraction"] == 1.0
    assert advantages[0, 0] > 0 > advantages[2, 0]
