#!/usr/bin/env python3
"""Collect ToolSandbox trajectories from an OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.scenarios import named_scenarios


DEFAULT_SCENARIOS = [
    "cellular_off_3_distraction_tools",
    "search_phone_number_with_name",
    "add_contact_with_name_and_phone_number",
    "send_message_with_phone_number_and_content",
    "search_message_with_recency_latest_3_distraction_tools",
    "search_message_with_recency_oldest",
    "update_contact_relationship_with_relationship",
    "remove_contact_by_phone",
    "turn_on_wifi_low_battery_mode_3_distraction_tools",
    "turn_on_cellular_low_battery_mode",
    "send_message_with_contact_content_cellular_off",
    "find_days_till_holiday_wifi_off",
    "find_days_till_holiday_3_distraction_tools",
    "search_reminder_with_recency_upcoming",
    "add_reminder_content_and_date_and_time",
    "remove_reminder_with_recency_latest",
    "find_days_till_holiday_insufficient_information",
    "modify_contact_with_message_recency_insufficient_information",
    "remove_contact_by_phone_no_remove_contact_insufficient_information_3_distraction_tools",
    "search_reminder_with_recency_upcoming_insufficient_information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollouts-per-scenario", type=int, default=5)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--max-messages", type=int, default=30)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


class LocalQwenAgent(OpenAIAPIAgent):
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> None:
        self.openai_client = OpenAI(base_url=base_url, api_key="local", timeout=120.0)
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.call_index = 0
        self.raw_responses: list[dict[str, Any]] = []

    def model_inference(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: Iterable[ChatCompletionToolParam] | Any,
    ) -> ChatCompletion:
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=cast(list[ChatCompletionMessageParam], openai_messages),
            tools=openai_tools,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed + self.call_index,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.call_index += 1
        self.raw_responses.append(response.model_dump(mode="json"))
        # vLLM serializes "no tool selected" as [] while ToolSandbox's older
        # OpenAI adapter expects None. Without this normalization it retries the
        # same turn forever because no message is appended.
        if not response.choices[0].message.tool_calls:
            response.choices[0].message.tool_calls = None
        else:
            # ToolSandbox embeds call IDs in temporary Python variable names.
            # vLLM's default IDs contain hyphens, which are invalid identifiers.
            for tool_call in response.choices[0].message.tool_calls:
                tool_call.id = "call_" + re.sub(r"\W", "_", tool_call.id)
        return response


class EndAfterAnswerUser(BaseRole):
    """End a single-user-turn scenario after the agent gives its final answer."""

    role_type = RoleType.USER

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return
        self.add_messages(
            [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content="print(repr(end_conversation()))",
                )
            ]
        )


def evaluation_dict(result: Any) -> dict[str, Any]:
    return {
        "similarity": float(result.similarity),
        "milestone_similarity": float(result.milestone_similarity),
        "minefield_similarity": float(result.minefield_similarity),
        "turn_count": int(result.turn_count),
        "milestone_mapping": {str(k): list(v) for k, v in result.milestone_mapping.items()},
        "minefield_mapping": {str(k): list(v) for k, v in result.minefield_mapping.items()},
    }


def conversation_steps(
    conversation: list[dict[str, Any]], number_of_milestones: int, terminal_reward: float
) -> list[dict[str, Any]]:
    assistant_turns = [i for i, turn in enumerate(conversation) if turn["role"] == "assistant"]
    if not assistant_turns:
        raise ValueError("trajectory contains no assistant decision")

    steps: list[dict[str, Any]] = []
    turn_to_step: dict[int, int] = {}
    current_step = 0
    for turn_index, turn in enumerate(conversation):
        if turn["role"] == "assistant":
            current_step = len(steps)
            action = {key: turn[key] for key in ("content", "tool_calls") if key in turn}
            steps.append(
                {
                    "step_index": current_step,
                    "action": action,
                    "outcomes": [],
                    "oracle": {"milestone_matches": [], "minefield_matches": []},
                    "rewards": {},
                }
            )
        turn_to_step[turn_index] = current_step
        if turn["role"] != "assistant" and steps:
            steps[current_step]["outcomes"].append(turn)

    # Evaluation can map a zero-similarity milestone to the initial user message.
    # Only positive matches carry credit; pre-action positive matches go to step zero.
    milestone_matches: dict[int, tuple[int, float]] = {}
    minefield_matches: dict[int, tuple[int, float]] = {}
    for turn_index, turn in enumerate(conversation):
        details = turn.get(f"{turn['role']}_details", {})
        step_index = turn_to_step.get(turn_index, 0)
        for match in details.get("milestone_matches", []):
            similarity = float(match["milestone_similarity"])
            if similarity > 0:
                milestone_matches[int(match["milestone_index"])] = (step_index, similarity)
        for match in details.get("minefield_matches", []):
            similarity = float(match["minefield_similarity"])
            if similarity > 0:
                minefield_matches[int(match["minefield_index"])] = (step_index, similarity)

    for milestone_index, (step_index, similarity) in milestone_matches.items():
        steps[step_index]["oracle"]["milestone_matches"].append(
            {"index": milestone_index, "similarity": similarity}
        )
    for minefield_index, (step_index, similarity) in minefield_matches.items():
        steps[step_index]["oracle"]["minefield_matches"].append(
            {"index": minefield_index, "similarity": similarity}
        )

    potential = 0.0
    poisoned = False
    denominator = max(number_of_milestones, 1)
    for step in steps:
        previous = potential
        credit = sum(x["similarity"] for x in step["oracle"]["milestone_matches"]) / denominator
        minefield = bool(step["oracle"]["minefield_matches"])
        if not poisoned:
            potential += credit
        if minefield:
            potential = 0.0
            poisoned = True
        step["oracle"]["potential"] = potential
        step["rewards"]["tool_sandbox_milestone_delta"] = potential - previous
        step["rewards"]["tool_sandbox_new_milestone_credit"] = credit
        step["rewards"]["tool_sandbox_minefield_trigger"] = float(minefield)

    # Insufficient-information scenarios can define success purely as avoiding a
    # minefield and therefore have no positive milestone to map. Attribute that
    # official residual to the final decision, explicitly rather than hiding it.
    residual = terminal_reward - potential
    steps[-1]["rewards"]["tool_sandbox_terminal_reconciliation"] = residual
    steps[-1]["rewards"]["tool_sandbox_milestone_delta"] += residual
    steps[-1]["oracle"]["potential"] = terminal_reward
    if abs(
        sum(step["rewards"]["tool_sandbox_milestone_delta"] for step in steps)
        - terminal_reward
    ) > 1e-6:
        raise AssertionError("dense reward does not telescope to verifier")
    return steps


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def self_check() -> None:
    conversation = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "assistant_details": {
                "milestone_matches": [
                    {"milestone_index": 0, "milestone_similarity": 1.0}
                ]
            },
        },
        {"role": "tool", "content": "ok"},
        {
            "role": "assistant",
            "content": "done",
            "assistant_details": {
                "milestone_matches": [
                    {"milestone_index": 1, "milestone_similarity": 1.0}
                ]
            },
        },
    ]
    steps = conversation_steps(conversation, 2, 1.0)
    assert [step["rewards"]["tool_sandbox_milestone_delta"] for step in steps] == [0.5, 0.5]
    assert sum(step["rewards"]["tool_sandbox_milestone_delta"] for step in steps) == 1.0
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    if args.rollouts_per_scenario < 1:
        raise SystemExit("--rollouts-per-scenario must be positive")

    scenario_names = args.scenarios or DEFAULT_SCENARIOS
    random.seed(args.seed)
    scenarios = named_scenarios(ToolBackend.DEFAULT)
    missing = sorted(set(scenario_names) - set(scenarios))
    if missing:
        raise SystemExit(f"unknown scenarios: {missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    config = {
        "schema_version": 1,
        "benchmark": "ToolSandbox",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "rollouts_per_scenario": args.rollouts_per_scenario,
        "scenarios": scenario_names,
        "max_messages": args.max_messages,
        "process_reward": "tool_sandbox_milestone_delta",
        "process_reward_telescopes_to": "verifier.terminal_reward",
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise SystemExit("existing manifest does not match this run")
    else:
        write_json_exclusive(manifest_path, config)

    jsonl_path = output_dir / "trajectories.jsonl"
    completed: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            completed = {json.loads(line)["trajectory_id"] for line in handle if line.strip()}

    with jsonl_path.open("a", encoding="utf-8") as output:
        for scenario_index, scenario_name in enumerate(scenario_names):
            scenario = scenarios[scenario_name]
            scenario.max_messages = args.max_messages
            for rollout_index in range(args.rollouts_per_scenario):
                trajectory_id = f"{scenario_index:02d}_{scenario_name}__r{rollout_index:02d}"
                if trajectory_id in completed:
                    continue
                trajectory_seed = args.seed + scenario_index * 10_000 + rollout_index * 1_000
                agent = LocalQwenAgent(
                    args.base_url,
                    args.model,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    trajectory_seed,
                )
                started = time.time()
                result = scenario.play_and_evaluate(
                    roles={
                        RoleType.AGENT: agent,
                        RoleType.USER: EndAfterAnswerUser(),
                        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                    },
                    output_directory=output_dir,
                    scenario_name=trajectory_id,
                )
                trajectory_dir = output_dir / "trajectories" / trajectory_id
                conversation = json.loads((trajectory_dir / "conversation.json").read_text())
                evaluation = evaluation_dict(result.evaluation_result)
                steps = conversation_steps(
                    conversation,
                    len(scenario.evaluation.milestone_matcher.milestones),
                    evaluation["similarity"],
                )
                write_json_exclusive(trajectory_dir / "model_responses.json", agent.raw_responses)
                record = {
                    "schema_version": 1,
                    "trajectory_id": trajectory_id,
                    "group_id": scenario_name,
                    "benchmark": "ToolSandbox",
                    "model": args.model,
                    "seed": trajectory_seed,
                    "task": {
                        "scenario": scenario_name,
                        "categories": [category.value for category in scenario.categories],
                        "allowed_tools": scenario.starting_context.tool_allow_list,
                    },
                    "verifier": {
                        "terminal_reward": evaluation["similarity"],
                        "success": evaluation["similarity"] == 1.0,
                        "binary_success_reward": float(evaluation["similarity"] == 1.0),
                        **evaluation,
                    },
                    "steps": steps,
                    "raw": {
                        "conversation": str(trajectory_dir / "conversation.json"),
                        "execution_context": str(trajectory_dir / "execution_context.json"),
                        "model_responses": str(trajectory_dir / "model_responses.json"),
                    },
                    "elapsed_seconds": time.time() - started,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"{trajectory_id}: reward={evaluation['similarity']:.3f} "
                    f"steps={len(steps)} seconds={record['elapsed_seconds']:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
