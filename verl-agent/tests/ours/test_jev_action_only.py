import numpy as np
import torch

from agent_system.multi_turn_rollout.rollout_loop import jev_action_token_mask
from verl.trainer.ppo.core_algos import compute_jev_step_grpo_advantage


class _CharacterTokenizer:
    all_special_ids = [0, 999]

    def __call__(self, texts, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [[ord(char) for char in text] for text in texts],
            "offset_mapping": [
                [(index, index + 1) for index in range(len(text))]
                for text in texts
            ],
        }


def test_jev_credit_updates_only_action_content():
    text = "<think>inspect first</think><action>go left</action>"
    response_ids = [ord(char) for char in text] + [999, 0]
    responses = torch.tensor([response_ids])
    response_mask = torch.tensor([[1] * (len(response_ids) - 1) + [0]])
    action_mask = jev_action_token_mask(
        _CharacterTokenizer(), responses, response_mask, [text]
    )

    selected = "".join(
        chr(token_id)
        for token_id, selected in zip(response_ids, action_mask[0].tolist(), strict=True)
        if selected
    )
    assert selected == "go left"

    advantages, _, metrics = compute_jev_step_grpo_advantage(
        token_level_rewards=torch.zeros_like(responses, dtype=torch.float32),
        response_mask=response_mask,
        action_mask=action_mask,
        effect_scores=np.asarray([0.25]),
        confidences=np.asarray([0.8]),
        index=np.asarray(["task"]),
        traj_index=np.asarray(["trajectory"]),
        turn_index=np.asarray([0]),
    )
    assert torch.allclose(
        advantages[action_mask.bool()], torch.full((7,), -0.4)
    )
    assert torch.count_nonzero(advantages[~action_mask.bool()]) == 0
    assert metrics["transitions_with_action_fraction"] == 1.0


if __name__ == "__main__":
    test_jev_credit_updates_only_action_content()
