# Jev-as-a-judge trajectory lab

This directory freezes agent trajectories for a comparison of terminal RLVR,
an LLM judge, and Jev as a process judge. Offline scoring does no policy or
environment rollout; the separate GRPO launcher is an online training experiment.

## Benchmarks and policy panel

- **ToolSandbox:** structured APIs, official milestone DAGs and minefields.
- **ScienceWorld:** text actions, native per-step score deltas and official final score.
- **ALFWorld:** the earlier hard-failure stress set remains in `runs/`.
- **Policies:** Qwen3-1.7B, 4B and 8B in non-thinking mode.
- **Formal sampling:** 20 matched task configurations × 5 stochastic rollouts =
  100 trajectories per policy per new benchmark.

All model inference runs on `air-node-02/03/04`; the current formal runs use
free GPUs 3 and 4 on `air-node-04`. Each vLLM server is capped below half of a
40 GB GPU.

## RL-style records without training

Every trajectory contains an official verifier reward and step records. The
same five rollouts for a task form an inference-only RL group.

`compute_advantages.py` produces two deliberately distinct quantities:

- `rloo_advantage`: the trajectory return minus the other four trajectory
  returns, broadcast to every decision as outcome-reward RL would do;
- `stepwise_rloo_advantage`: reward-to-go minus the mean reward-to-go of peer
  rollouts still active at the same decision index. This is a process-credit
  diagnostic, not GAE and not a learned value estimate.

ToolSandbox's `tool_sandbox_milestone_delta` is derived from the official
milestone mapping and minefield rule. Its sum exactly equals the official
terminal similarity. ScienceWorld retains both its raw native
`scienceworld_score_delta` (including failure penalties) and a
`scienceworld_clipped_score_delta` whose sum equals the nonnegative normalized
final score.

## Collection

The benchmark environments have isolated local environments:

```bash
.venv-toolsandbox/bin/python collect_toolsandbox.py \
  --base-url http://127.0.0.1:PORT/v1 --model qwen3-4b \
  --output-dir runs/toolsandbox-qwen3-4b-r100-s20260919 \
  --rollouts-per-scenario 5

.venv-scienceworld/bin/python collect_scienceworld.py \
  --base-url http://127.0.0.1:PORT/v1 --model qwen3-4b \
  --output-dir runs/scienceworld-qwen3-4b-r100-v2-s20260919 \
  --rollouts-per-task 5
```

Collectors are resumable: a matching manifest is required and completed
trajectory IDs are skipped. Each run keeps `trajectories.jsonl` plus the raw
conversation, environment state and model response under `trajectories/`.
ScienceWorld gold paths and ToolSandbox oracle mappings are saved for analysis
but never shown to the policy.

## Rewards, advantages and process metrics

```bash
python3 compute_advantages.py RUN/trajectories.jsonl \
  --source terminal --output RUN/advantages/terminal.jsonl

python3 compute_advantages.py RUN/trajectories.jsonl \
  --source PROCESS_REWARD_NAME --output RUN/advantages/process.jsonl

python3 process_metrics.py RUN/trajectories.jsonl \
  --source terminal --source PROCESS_REWARD_NAME \
  --advantage RUN/advantages/terminal.jsonl \
  --advantage RUN/advantages/process.jsonl \
  --output RUN/metrics-process.json
```

`process_metrics.py` reports native-event density, nonzero reward density,
failure-only coverage, temporal variation in reward-to-go, effective credit
support, reward mass timing, credit on repeated actions, verifier consistency,
and both trajectory-level and stepwise RLOO behavior. Binary success reward is
available as `verifier.binary_success_reward`.

Small runnable checks:

```bash
.venv-toolsandbox/bin/python collect_toolsandbox.py --model dummy --self-check
.venv-scienceworld/bin/python collect_scienceworld.py --model dummy --self-check
python3 compute_advantages.py --self-check
python3 process_metrics.py --self-check
python3 evaluate.py --self-check
```

The pre-registered comparison and interpretation rules are in
[`DESIGN.md`](DESIGN.md). The 600-trajectory matched-panel inventory and
source checksums are in [`runs/DATASET.md`](runs/DATASET.md); measured verifier
reward and credit allocation are in
[`runs/PROCESS_RESULTS.md`](runs/PROCESS_RESULTS.md). Jev and LLM-judge
annotations are separate from the immutable source trajectories. Jev API access
has been checked with `jev-1.13.0`; completed scored subsets are recorded in
`runs/PROCESS_RESULTS.md`. Never put an API key in a script, shell history,
or output file. The scorer prompts for it when `TYPESAFE_API_KEY` is absent:

```bash
python3 score_jev_v1.py RUN/trajectories.jsonl --output RUN/jev-transition-v1.jsonl
python3 evaluate_jev.py RUN/trajectories.jsonl RUN/jev-transition-v1.jsonl
```

`score_jev_v1.py` sends only task text, public prefix (up to three previous
transitions), the current action, and the observed result. Its self-check
asserts that `verifier`, `rewards`, `oracle`, `gold_action_sequence`, hidden
tool details, and future steps never enter the request. The evaluator refuses
incomplete annotations by default and scores nonterminal steps within groups
where all five rollouts failed. Run `python3 score_jev_v1.py --self-check` and
`python3 evaluate_jev.py --self-check` before changing either protocol.
`evaluate_jev.py --exclude-repeated-state-actions` is an exploratory control
that drops later occurrences of an exact `(observation_before, action)` pair;
it does not replace the preregistered primary slice.

Offline judge comparisons reuse these previously sampled trajectories; they
never call the policy or environment to generate another rollout. Join Jev and
LLM-judge records by `(trajectory_id, step_index)`, then associate the saved
trajectory-level baseline reward by `trajectory_id`. Verify each run's
`SOURCE_SHA256SUMS` before analysis. Keep baseline labels in
the source records for evaluation, but never include them in judge requests.
ToolSandbox and ScienceWorld store baseline outcome reward in `verifier`;
ALFWorld stores it in `terminal.rlvr_reward`. The separate `jev-online.jsonl`
from GRPO training is not part of this frozen offline comparison.

The ICML 2026 paper draft and official template are in `../../dllm/assets/icml_1/`.
Offline annotation can establish only process-credit measurement quality;
policy improvement requires the later matched GRPO training experiment.

## V4 online efficiency checks

Use `run_grpo_alfworld.sh` for one-update timing checks before a full run. Set
`TRAIN_UPDATES=1 MAX_STEPS=10 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 SAVE_FREQ=-1`;
the launcher then creates only one unused validation actor. Compare
`timing_s/gen`, `timing_s/update_actor`, `timing_s/step`, rollout tokens, and GPU
memory/utilization after the first training step. Run artifacts stay under
`runs/grpo-alfworld-RUN_TAG/`.

For the four-GPU V4 Qwen2.5-1.5B pilot, set `REF_PARAM_OFFLOAD=false`,
`OPTIMIZER_OFFLOAD=false`, and `PERSISTENT_ROLLOUT=true`. The persistent rollout
keeps vLLM weights resident across ALFWorld turns and releases them before the
actor update. The fastest measured short-run setting also uses
`TRAIN_BATCH_SIZE=8 ACTOR_MICRO_BATCH=4 LOG_PROB_MICRO_BATCH=8 OMP_NUM_THREADS=4`.
Batch 8 changes the training batch, so keep batch size matched for algorithm
comparisons. `MODEL_SHM=true` stages a model snapshot under `/dev/shm/verl-cache`;
`STDLIB_SHM=true` stages the small Python standard library to avoid intermittent
Ray worker import failures on the shared filesystem. Both caches are optional.
Use these as an H200 starting point, then remeasure batch and microbatch limits.

On air-node-03 (four A100s, batch 4, ten ALFWorld turns), the second update
provides a matched comparison. Both runs used actor microbatch 2, log-prob
microbatch 4, and a GPU-resident reference model:

| Rollout scheduling | Generation | Full update | Processed tokens | Peak reserved/GPU |
| --- | ---: | ---: | ---: | ---: |
| Per-turn weight sync | 132 s | 311 s | 90,971 | 25.5 GB |
| Persistent rollout | 55 s | 240 s | 89,876 | 25.5 GB |

Persistent scheduling cut the matched full update by 23%.

At batch 8 with the same 125,828 processed tokens in each one-update run,
raising actor/log-prob microbatches from 2/4 to 4/8 reduced actor update time
from 267 to 147 seconds and full update time from 426 to 276 seconds. Peak
PyTorch reserved memory rose from 25.4 to 27.9 GB per A100. This is 54% more
processed tokens per training second; both runs completed without OOM or
DataLoader shutdown errors.

Sampled GPU utilization fell from 77% to 68% in that comparison even as
throughput rose, so utilization alone is not a throughput measure. Launch to
first training step took about 14–15 minutes with one validation actor, zero
DataLoader workers, and four CPU threads; earlier warm launches took about
16–19 minutes. Shared-node startup varies, and staging the model alone did not
remove Ray or FSDP initialization cost.
