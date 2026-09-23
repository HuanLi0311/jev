# Jev as a process judge on agent trajectories

Research date: 2026-09-19; updated 2026-09-24

## Question

Can Jev turn an agent trajectory into process supervision that is not only denser than terminal RLVR, but also accurate enough to locate mistakes and rank rollouts, at substantially higher throughput than an LLM judge?

The first stage uses previously sampled, frozen trajectories with their
existing verifier rewards and advantages. Jev and LLM-judge annotation reads
those same records; it does not resample the policy or environment. It does
not train a policy or reward model, and makes no on-policy/off-policy claim.
ALFWorld is retained as a hard-failure set; ToolSandbox and ScienceWorld add
native process-verifier anchors.

## Inference-only RL protocol

For ToolSandbox and ScienceWorld, each task/variation is sampled five times with
the same policy and sampling configuration. That five-rollout set is treated as
an RL group, but no gradient update is performed.

For reward sequence `r[t]`, store reward-to-go
`G[t] = r[t] + gamma * G[t+1]` (`gamma=1` in the primary analysis). Report:

1. **Outcome RLOO advantage.** `G[0]` minus the mean `G[0]` of the other four
   rollouts, broadcast over that trajectory. This matches the usual
   outcome-reward group baseline used during RL.
2. **Stepwise RLOO diagnostic.** `G[t]` minus the mean `G[t]` of peer rollouts
   that are still active at decision index `t`. This exposes temporal credit,
   but is an offline diagnostic rather than GAE or a learned value estimate.
3. **Group z advantage.** The corresponding population-standardized form;
   report zero when group variance is zero rather than dividing by zero.

The primary reward channels are intentionally kept separate:

| Benchmark | Outcome verifier | Native/process verifier |
|---|---|---|
| ToolSandbox | Official final `EvaluationResult.similarity`; strict full-score bit as a binary RLVR view | Official milestone/minefield mapping converted to a telescoping potential delta |
| ScienceWorld | Nonnegative normalized official final score; 100-point completion bit as a binary RLVR view | Native score delta, plus a clipped-potential delta that telescopes to the outcome score |
| ALFWorld | Environment success bit | TextWorld plan-distance anchors, evaluation-only |

ToolSandbox's process rewards sum exactly to its outcome similarity. A minefield
drops the potential to zero. Insufficient-information tasks can succeed by
avoiding all minefields and have no positive milestone; their unavoidable
outcome residual is explicitly assigned to the final answer and labeled
`tool_sandbox_terminal_reconciliation`.

ScienceWorld's raw score delta is the most informative failure signal but need
not sum to the clipped outcome reward because reset can start above zero and a
failure can set a negative terminal score. The clipped-potential channel is
provided separately for equal-total comparisons. Neither channel, the gold
path, nor hidden goal-progress diagnostics are placed in the policy prompt.

## Pre-registered comparisons

All judges receive the same frozen trajectory and the same semantic rubric.

| Judge | Native signal | Step output used for comparison |
|---|---|---|
| RLVR | One verified terminal success bit | Terminal bit broadcast to steps only for API compatibility; native event count remains one |
| LLM judge | Generated three-level probabilities | `p(progress)-p(harm)` in `[-1, 1]`, plus raw response |
| Jev | Typed Score distribution | Expected level minus one in `[-1, 1]`, plus raw response |

Do not claim superiority from density alone: a judge can emit a number at every step and still be useless. The primary quality anchors are environment-derived and hidden from the policy:

1. **Oracle-progress discrimination.** AUROC for separating steps that reduce TextWorld's shortest remaining plan from steps that do not.
2. **First-fault localization.** Exact accuracy and step-distance error for the first invalid, looping, or oracle-regressing action.
3. **Best-of-N utility.** Success rate when a judge selects one rollout per task, compared with the oracle ceiling among the sampled rollouts.
4. **Pairwise action discrimination.** On natural Qwen errors, choose between Qwen's action and TextWorld's oracle next action. Swap option order to measure positional bias.

The key delayed-credit endpoint is the same AUROC computed **only among terminal
failures**. RLVR assigns every one of those steps the same zero, while a useful
process judge should still distinguish partial progress from the first damaging
or looping action. This failure-only slice is declared before Jev scores are seen.
For online/localization claims, a judge sees only the prefix through the scored
transition—never later steps, terminal success, or any oracle field.

The likely Jev advantage should be tested on a Pareto plot, not selected after seeing results:

- x-axis: wall-clock seconds or dollars;
- y-axis: oracle-progress AUROC / pairwise action accuracy / first-fault accuracy;
- annotation: scored transitions per second.

The strongest defensible result would be **matched quality with much higher validated-label throughput**, or higher quality at matched cost. “More scores” by itself is not a win.

## Frozen datasets produced

The earlier ALFWorld source set contains 54 complete trajectories and 1,544 transitions
from Qwen3 0.6B, 1.7B, 4B, and 8B. It covers all six task types, contains 5
successful trajectories, and yields 2,888 order-swapped pairwise cases from
1,444 natural policy errors. Exact files and SHA-256 hashes are recorded in
[`runs/DATASET.md`](runs/DATASET.md).

The matched ToolSandbox and ScienceWorld panel adds 600 primary trajectories
and 5,743 transitions: 100 rollouts for each of three model sizes on each
benchmark. Their frozen source files, reward/advantage artifacts, and measured
results are indexed in [`runs/DATASET.md`](runs/DATASET.md) and
[`runs/PROCESS_RESULTS.md`](runs/PROCESS_RESULTS.md). All primary ScienceWorld
runs use the same v2 prompt; earlier v1 or mixed-prompt runs are exploratory.

Two observations motivate, but do not prove, the Jev hypothesis:

- The original 1.7B set has 360 steps and twelve terminal RLVR events (native
  density 0.033); all rewards are zero, while environment anchors still mark
  18.6% of steps as progress and 67.5% as hard errors.
- On the mixed 8B set, RLVR reaches 0.724 progress AUROC over all steps only
  because successful trajectories receive ones. On terminal failures alone it
  falls to 0.5 exactly. The failure-only slice is therefore the cleanest test of
  whether Jev resolves delayed credit.

The TextWorld oracle judgment files are implementation anchors, not competing
learned judges: their hard-quality scores are constructed from the same labels
used for evaluation and must never be reported as an empirical model result.

## Process-supervision measurements

### Density and temporal resolution

- `native_event_density = native judge evaluations / trajectory transitions`.
- `step_score_coverage = non-null step scores / transitions`.
- `mean_adjacent_score_change` and `nonconstant_trajectory_rate` detect a formally dense but constant signal.
- Report trajectory-length-stratified coverage so truncation is visible.
- `failure_only_active_trajectory_rate` asks whether a signal distinguishes any
  steps when binary RLVR is identically zero.
- `nonconstant_reward_to_go_rate` and distinct reward-to-go counts test whether
  a dense API actually changes temporal credit.
- Effective credit support `(sum |r|)^2 / sum r^2`, its length-normalized form,
  and reward-mass time locate how concentrated and delayed credit is.
- Report reward/advantage mass on consecutive repeated actions to detect false
  credit for loops.

### Agent-trace diversity

- Cover all six ALFWorld task types and report their counts rather than treating
  the benchmark as one homogeneous pool.
- Sample the same games from multiple policy scales so task variation and policy
  variation can be separated.
- Report exact action-sequence uniqueness, within-game normalized action edit
  distance, and per-trace unique state/action ratio. These measure different
  things and must not be collapsed into a single “diversity” claim.
- Always report diversity beside success, validity, loop, and hard-error rates:
  random or broken traces can be diverse without being useful judge inputs.

### Grounded usefulness

- AUROC against `oracle_progress_delta > 0`.
- AUROC against hard step quality (valid, non-looping, non-regressing).
- First-fault exact match and mean absolute localization error.
- Best-of-N success and regret from the sampled oracle ceiling.
- Among terminal-equivalent failures, oracle-distance progress of the selected
  rollout and regret from the best failed rollout in the sampled set; tied
  judge scores use their expected value rather than file order.
- Probability calibration (Brier score and ECE) where a binary environment label exists.

### Robustness

- Re-score with candidate order swapped.
- Re-score after irrelevant formatting/paraphrase changes.
- Break out metrics by task type and trajectory length.
- Keep the model version, rubric version, raw response, latency, token counts, and price snapshot with every judgment.

### Efficiency

- End-to-end latency per trajectory and per scored transition.
- Input/output tokens and dollars per trajectory.
- Correctly discriminated or correctly localized steps per second and per dollar; always show its quality and throughput components separately.

## Data contract

`collect.py` writes one trajectory per line. Each step retains:

- pre-action observation and admissible actions;
- raw Qwen generation, extracted reasoning, parsed action, and parsing method;
- post-action observation and environment reward;
- validity, oracle plan before/after, oracle distance delta, termination, and success;
- inference tokens and latency.

Judge annotations are separate JSONL records keyed by `(trajectory_id,
step_index)`, with the filtered `request.state`, raw `response`, rubric version,
numeric judge reward, and latency. Baseline verifier labels remain only in the
frozen source files and are joined for evaluation, never sent to a judge.

`pairwise_cases.jsonl` is a labeled evaluation artifact. A judge adapter must
send only `state` and `options`; it must withhold `correct_option` and all oracle
fields. The AB/BA twins are scored separately, then combined to expose position
bias.

## Matched online GRPO pilot

The primary training comparison starts from the same cached
Qwen2.5-1.5B-Instruct checkpoint and seed 0. It uses four training prompts,
four rollouts per prompt, a 30-action horizon, ten updates, and verifier-only
`valid_seen` evaluation before training and after updates 5 and 10. The sparse
RLVR baseline uses standard trajectory-level GRPO on
`R_i = 10 * success_i`, broadcasting one group-standardized `A_out[i]` to every
action in trajectory `i`. The revised Jev rubric explicitly receives the
static success criteria and returns a continuous effect score `q[i,t]` and
confidence `c[i,t]`, both in `[0, 1]`. It assigns

`A_jev[i,t] = c[i,t] * (2 * q[i,t] - 1)`,

`A[i,t] = A_jev[i,t] + 0.1 * A_out[i]`.

The Jev term is not group-standardized or propagated across future turns,
because its sign and confidence magnitude directly determine the update for
turn `t`. Only that turn's action tokens receive `A[i,t]`; standard PPO then
applies its clipped importance ratio. The shared parser, optimizer,
sampling budget, model, and held-out verifier are unchanged. Jev reads only
the success criteria, current prefix, and observed transition; it never reads
the current trajectory's verifier result, future steps, or frozen annotations. The earlier
six-game `valid_unseen` slice is development-only. This remains a one-seed
pilot until replicated.

A completed matched run now evaluates the revised confidence-preserving
estimator. Both arms start at 0/64 held-out successes, and their first 480
sampled transitions are identical after removing run identifiers. All 16
first-batch trajectories fail. Standard GRPO therefore has zero advantage and
policy-gradient loss 0.000, whereas Jev gives 480/480 transitions nonzero
advantage, every trajectory has varying within-trajectory advantage, and the
policy-gradient loss is 0.571. At update 5, held-out success is 8/64 for Jev
and 0/64 for sparse GRPO; at update 10 it is 19/64 versus 1/64, with mean
verifier scores 1.423 versus 0.037. The full 4,327-call Jev log contains no API
errors or forbidden request fields. This passes the pilot's three go/no-go
gates, but it remains one training seed and one 64-task evaluation; formal
sample-efficiency or benchmark-superiority claims require multi-seed
replication. Exact artifacts and the leakage audit are recorded in
[`runs/PROCESS_RESULTS.md`](runs/PROCESS_RESULTS.md).

The earlier one-update reward-to-go smoke test remains an ablation. It also
produced dense turn-level advantages in an all-failure batch, but same-turn
standardization discarded the confidence scale. That run was stopped after
seven completed updates and is not used for the primary result.

### Outcome-conditioned hindsight follow-up

A matched single-arm follow-up keeps the prefix-only run frozen and changes
only when and with what context Jev labels transitions. After each episode
terminates, Jev receives the full public trajectory plus the verified terminal
reward and success bit, and returns one continuous effect score and confidence
per transition in a single request. It uses the same direct advantage and PPO
objective above. Oracle state, gold paths, stored process rewards, and hidden
verifier internals remain excluded. This is a post-episode hindsight annotator,
not an online causal PRM: future steps and terminal outcome are intentionally
visible while constructing training labels, but neither Jev nor those fields
are inputs to the deployed policy.

The hindsight arm starts from the same 0/64 checkpoint. At update 5 it reaches
16/64 held-out successes and mean verifier score 1.214, compared with 8/64 and
0.427 for the frozen prefix-only arm. At update 10 it reaches 22/64 and 1.250,
compared with 19/64 and 1.423. Across 4,350 labeled transitions, 99.6% receive
nonzero advantage and every update has nonconstant within-trajectory credit.
Thus strict success favors hindsight at both checkpoints, especially early,
but the final continuous score does not. This one-seed mixed endpoint supports
multi-seed replication rather than a general superiority claim. Exact artifacts
are indexed in [`runs/PROCESS_RESULTS.md`](runs/PROCESS_RESULTS.md).

The previous online run is retained only as a trajectory-aggregate ablation.
It used `10 * success + 0.1 * mean_t(j_t)`, standardized the four episode
returns, and broadcast one scalar advantage to all turns. It produced optimizer
signal when all rollouts failed and had 2/64 held-out successes at update 5
versus 1/64 for sparse GRPO, but it did not implement within-trajectory credit;
the one-success difference is inconclusive and is not the primary method.

## Sources and design rationale

- ALFWorld's official configuration defines the `valid_seen` and `valid_unseen` splits, six task types, and a 50-step default evaluation horizon. We use the text environment and start with `valid_unseen`; our pilot caps at 30 steps to control inference cost, and records that cap in the manifest. [Official ALFWorld configuration](https://github.com/alfworld/alfworld/blob/master/configs/base_config.yaml)
- Qwen3 supports explicit thinking and non-thinking modes and recommends sampling rather than greedy decoding. A smoke run showed that the 1.7B model often exhausted a short generation budget before emitting an action in thinking mode, so the pilot uses non-thinking mode. Temperature/top-p are raised from the official 0.7/0.8 recommendation to 0.9/0.95 to obtain multiple distinct rollouts; this deviation is recorded in the run manifest. [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)
- ProcessBench makes earliest-error identification a direct PRM evaluation task. We adapt that protocol to the first environment-grounded fault in an agent trace. [ProcessBench](https://arxiv.org/abs/2412.06559)
- ToolPRMBench evaluates a correct action against a plausible incorrect alternative at a fixed interaction history and separates offline local errors from online rollout failures. This motivates the oracle-vs-natural-error pairwise test. [ToolPRMBench](https://arxiv.org/abs/2601.12294)
- AgentPRM specifically studies process rewards on ALFWorld and highlights test-time scaling and reward hacking, supporting ALFWorld as a relevant first environment while warning against trusting reward magnitude alone. [AgentPRM](https://arxiv.org/abs/2502.10325)
- Plan-RewardBench reports degradation on longer trajectories and uses natural rollouts plus rule/minimal-edit hard negatives. We therefore stratify by length and retain natural failures before adding controlled perturbations. [Plan-RewardBench](https://arxiv.org/abs/2604.08178)
- RewardBench 2 emphasizes that standalone reward-model accuracy must correlate with downstream selection or optimization. Best-of-N success is therefore a primary endpoint rather than an optional demo. [RewardBench 2](https://arxiv.org/abs/2506.01937)

Confidence: high for environment fields and collection protocol; medium for whether oracle plan-distance is a complete definition of action quality; low until measured for any expected Jev advantage.

## Known limitations

- Shortest-plan progress is a strong but incomplete proxy: exploration or recovery can be useful without immediately reducing plan length.
- ALFWorld actions are structured textual commands, not heterogeneous real-world APIs.
- Qwen3-1.7B may yield mostly failed rollouts. If every sample fails identically, expand sampling temperature/task count before changing the judge rubric.
- The pilot evaluates observable action/observation traces, not hidden chain-of-thought. Raw generations are retained, but non-thinking mode is used so the small policy reliably emits environment actions.
- Jev's primary training language is English, so the policy and judge rubrics are English in this pilot.
- No post-hoc metric or rubric changes should replace the primary endpoints above. Exploratory analyses must be labeled exploratory.

## Research methodology

- Type: technical audit / experiment design; standard-depth target expanded to seven primary pages.
- Queries: ALFWorld official evaluation and splits; Qwen3-1.7B official sampling; PRM step-error localization; agent/tool-use PRM evaluation; best-of-N reward-model evaluation; dense process reward on ALFWorld.
- Sources read: seven primary sources (official repositories/model card and research papers), found through web search; secondary summaries were excluded from design claims.
- Gaps: no public Jev-specific PRM study was found. Jev API access is working;
  the fixed transition rubric is being measured on the frozen panel. Offline
  judge comparisons cannot establish an RL-training benefit.
