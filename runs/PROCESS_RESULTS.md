# Verifier reward and offline RL credit results

Research date: 2026-09-19; updated 2026-09-24

These are inference-only measurements. Each row contains 20 task groups with
five stochastic rollouts per group. No policy, value model, or reward model was
trained. `rloo_advantage` is the trajectory return minus the mean return of the
other four rollouts and is broadcast over the trajectory, as in outcome-reward
RL. `stepwise_rloo_advantage` applies the same leave-one-out baseline to
reward-to-go at each decision index; it is an offline process-credit diagnostic,
not GAE.

## Outcome results

| Benchmark | Model | N | Steps | Success | Mean verifier reward | Mean length |
|---|---:|---:|---:|---:|---:|---:|
| ToolSandbox | 1.7B | 100 | 146 | 16.0% | 0.4166 | 1.46 |
| ToolSandbox | 4B | 100 | 139 | 20.0% | 0.4111 | 1.39 |
| ToolSandbox | 8B | 100 | 215 | 19.0% | 0.5298 | 2.15 |
| ScienceWorld | 1.7B | 100 | 1,596 | 4.0% | 0.0593 | 15.96 |
| ScienceWorld | 4B | 100 | 1,578 | 8.0% | 0.1249 | 15.78 |
| ScienceWorld | 8B | 100 | 2,069 | 14.0% | 0.2422 | 20.69 |

The continuous verifier reward matters: for example, ScienceWorld 1.7B has
4% strict successes and mean official normalized score 0.0593. Both the
continuous outcome reward and the strict completion bit have separate reward
and advantage files.

## Failure-only credit allocation

The process source is ToolSandbox's official milestone/minefield delta or
ScienceWorld's native score delta. “Reward active” means that a failed
trajectory contains at least one nonzero reward. “Step-adv active” means that it
contains at least one nonzero stepwise RLOO advantage.

| Benchmark | Model | Binary failure reward active | Process failure reward active | Process nonzero step reward | Varying process RTG | Binary failure step-adv active | Process failure step-adv active |
|---|---:|---:|---:|---:|---:|---:|---:|
| ToolSandbox | 1.7B | 0.0% | 59.5% | 60.3% | 23.0% | 4.8% | 52.4% |
| ToolSandbox | 4B | 0.0% | 56.2% | 63.3% | 21.0% | 0.0% | 50.0% |
| ToolSandbox | 8B | 0.0% | 72.8% | 65.1% | 45.0% | 1.2% | 56.8% |
| ScienceWorld | 1.7B | 0.0% | 87.5% | 6.0% | 20.0% | 1.0% | 27.1% |
| ScienceWorld | 4B | 0.0% | 94.6% | 9.3% | 52.0% | 13.0% | 34.8% |
| ScienceWorld | 8B | 0.0% | 100.0% | 9.1% | 79.0% | 1.2% | 41.9% |

This isolates the intended delayed-credit failure mode. A failed rollout has
zero native binary reward. It can receive nonzero binary RLOO advantage only
when another rollout in its five-sample group succeeds, and that signal remains
trajectory-constant. In contrast, native process rewards remain active on many
failed trajectories and create temporally varying reward-to-go.

The largest clean gaps are ToolSandbox 8B (56.8% versus 1.2% failure-trajectory
step-advantage coverage) and ScienceWorld 8B (41.9% versus 1.2%). These are
properties of the environment verifiers, not Jev results. They define the
pre-registered opportunity: Jev is useful only if its scores recover this
failure-only structure with good oracle agreement and substantially better
throughput than an LLM judge.

## Reward channels and caveats

- ToolSandbox `tool_sandbox_milestone_delta` is computed from the official
  milestone and minefield mappings and sums exactly to official final
  similarity.
- ScienceWorld `scienceworld_score_delta` retains native positive and negative
  events. Reset scores and negative terminal scores mean it need not sum to the
  clipped nonnegative outcome reward.
- ScienceWorld `scienceworld_clipped_score_delta` is the equal-total control. It
  telescopes to `max(final_score, 0) / 100`; its maximum observed conservation
  error is `1.11e-16`.
- Density alone is not treated as judge quality. Later Jev and LLM-judge scores
  must also pass failure-only discrimination, first-fault localization,
  best-of-five selection, calibration, and latency/cost measurements described
  in `../DESIGN.md`.

## Validation and artifacts

Across the six primary runs there are 600 trajectories and 5,743 transitions.
Every run has exactly 20 groups of five, unique trajectory IDs, valid raw-file
references, and zero-sum group RLOO advantages. `SOURCE_SHA256SUMS` in each run
freezes the manifest, source JSONL, and per-trajectory raw files. The compact
machine-readable summary is `metrics-rl-credit.json`; per-source decision-level
records are under `advantages/`.

The earlier `scienceworld-qwen3-1.7b-r100-s20260919` and
`scienceworld-qwen3-4b-r100-s20260919` runs and the hybrid-prompt
`scienceworld-qwen3-8b-r100-s20260919` run are retained as exploratory data
but excluded from this table. All primary ScienceWorld rows use uniformly
budgeted `*-v2-*` prompt runs.

## Jev v1 transition judge: first complete frozen panel

The ToolSandbox Qwen3-8B panel has now been scored by `jev-1.13.0` using the
fixed `jev_transition_v1` rubric: all 215 transitions in all 100 frozen
trajectories have a separate Jev record, with no source trajectory rewritten.
The API request contains only task text, public prefix, current action, and
observed result; verifier, stored rewards, oracle mappings, hidden tool details,
gold paths, and future steps are excluded. The SHA-256 source manifest still
passes. The request/response and latency are preserved in
`toolsandbox-qwen3-8b-r100-s20260919/jev-transition-v1.jsonl`.

On the predeclared all-five-failed groups, binary RLVR has zero group advantage.
Jev yields a nonzero trajectory RLOO advantage for all 80 failed trajectories,
and a varying reward-to-go in 50/80. **This is density, not a quality result.**
Against positive official milestone deltas on nonterminal steps, its mean
within-trajectory AUROC is only 0.304 over 14 eligible trajectories; mean
within-group AUROC is 0.353 over just three eligible groups. Both are below
the 0.5 constant-reward baseline, and the very small number of independent
task groups precludes a general conclusion. In particular, on date/time tasks
a call can earn hidden milestone credit even when the visible tool response
contains a connection error; the current public-only rubric penalizes that
response. Future rubric changes must be chosen on a development split and
evaluated on untouched task groups, not selected post hoc from this panel.

The matched local Qwen3-8B non-thinking LLM judge has also scored all 215
ToolSandbox steps using the same public state and three-level rubric. Its
mean within-trajectory AUROC is 0.679 over the same 14 eligible traces;
within-group AUROC is 0.720 over the same three task groups. Mean observed
single-step latency is 1.195 s for this local LLM service versus 0.908 s for
the remote Jev API from this server. These timings are not a controlled
hardware-matched speed comparison, and the local LLM's compute cost has not
yet been normalized. Nevertheless, on this panel Jev has no demonstrated
quality advantage over even this 8B judge. ScienceWorld's matched judge
comparison is reported below. No policy-training
or agentic-RL improvement claim is supported yet.

## ScienceWorld Qwen3-8B: complete paired judge comparison

All 2,069 transitions of the 100 previously sampled Qwen3-8B v2 rollouts
have Jev scores, and the source `SOURCE_SHA256SUMS` still passes. On the 17
groups whose five rollouts all fail, the mean within-trajectory AUROC for
positive native score changes is 0.900 over 45 eligible trajectories;
mean within-group AUROC is 0.876 over 10 eligible task groups. The outcome
reward is constant within each trajectory, giving 0.500 for that particular
step-localization comparison, though its continuous final score remains a
meaningful cross-trajectory baseline. Jev produces nonzero trajectory RLOO
advantage on 83/85 all-failed-group trajectories and varying reward-to-go
on 64/85; density is reported separately from accuracy. These are frozen-trace
measurements, not a training result. The matched non-thinking Qwen3-8B LLM
judge scored the exact same 2,069 public states and rubric. Its mean
within-trajectory AUROC is 0.490 on the same 45 eligible traces, and mean
within-group AUROC is 0.502 on the same 10 eligible groups. Jev's group-level
advantage is 0.374 AUROC; nine of ten eligible groups favor Jev. A
task-resampling bootstrap (20,000 replicates, seed 20260919) gives an
exploratory 95% interval of [0.236, 0.507] for the paired group mean
difference. This comparison supports a task-conditional process-measurement
advantage over this specific local 8B judge, not over all LLM judges or in
trained policy success. Requests were independently checked for identical
public state and complete `(trajectory_id, step_index)` coverage.
As an exploratory robustness slice, excluding repeated
`(observation_before, action)` pairs leaves 627 nonterminal failed-group
steps; mean within-group AUROC is 0.876 for Jev and 0.416 for the LLM over
the same ten eligible groups. The ScienceWorld gap is therefore not solely
a loop-detection effect, although this slice was inspected after the primary
result.
Nine of the ten eligible groups exceed 0.5 AUROC; the
`identify-life-stages-2` group is below chance at 0.456. Group positives range
from five to 25, so these per-task estimates remain noisy.

The predeclared best-of-five utility check exposes a distinct limitation.
Using the sum of centered transition scores to choose one rollout from each
of the 17 all-failed groups, the mean official *continuous final score* is
0.079 for Jev, versus 0.120 for a random rollout, 0.129 for the LLM judge,
and 0.151 for the oracle best sampled rollout. On all 20 groups, strict
success selection is 0.15 for Jev, 0.14 random, 0.10 LLM, and 0.15 oracle;
only three groups contain a success. High local AUROC therefore does not
validate naive summed Jev reward as a trajectory-ranking reward. This
negative endpoint remains 0.079 if scores are averaged per trajectory, so
length normalization alone does not rescue it on ScienceWorld. It must inform,
not be hidden by, the later training comparison.

## ScienceWorld Qwen3-4B: frozen policy-size replication

All 1,578 transitions from the previously sampled 100-trajectory v2 panel
were scored by `jev-1.13.0` and the same local Qwen3-8B judge. The source
checksum still passes, both annotation files have complete coverage, and the
paired evaluator confirms identical public input at every step. Sixteen of 20
five-rollout groups all fail. Over seven groups with both positive and
non-positive transitions, mean within-group AUROC for detecting positive
native score changes is 0.865 for Jev versus
0.560 for the LLM; Jev wins six groups. The paired mean difference is 0.305,
with an exploratory task-bootstrap 95% interval of [0.126, 0.493]. On the
same 25 eligible individual traces, mean AUROC is 0.809 versus 0.482. This
replicates the *direction* of the 8B process-measurement result across policy
sizes, not across independent tasks or judge models.

Among all failed trajectories, including those in mixed-outcome groups, the
nonzero stepwise RLOO density is 0.988 for Jev and 0.579 for the LLM; density
is distinct from the AUROC quality endpoint. Observed mean call latency is
1.015 s for remote Jev and 0.848 s for the local LLM, not a hardware-matched
speed comparison. On the 16 all-failed groups, selecting by summed Jev scores
gives mean official continuous final reward 0.010 versus 0.0196 random,
0.0206 LLM, and 0.0394 sampled oracle. Mean aggregation also gives 0.010 for
Jev here. The second policy panel therefore strengthens the local-credit
finding but repeats the negative whole-trajectory reward-ranking result.

## ScienceWorld Qwen3-1.7B: frozen policy-size replication

All 1,596 transitions from the previously sampled 100-trajectory v2 panel
were scored by `jev-1.13.0` and the same local Qwen3-8B judge. The source
checksum passes, both annotation files have complete coverage, and the paired
evaluator confirms identical public input at every step. Nineteen of 20
five-rollout groups all fail, so binary outcome GRPO assigns zero group
advantage to all 95 trajectories in that slice. Only three task groups contain
both positive and non-positive transitions; over those groups, mean within-group AUROC for
detecting positive native score changes is 0.989 for Jev versus 0.388 for the
LLM. Jev wins all three, with a
paired mean difference of 0.601 and an exploratory task-bootstrap 95% interval
of [0.427, 0.867]. Across the same 15 eligible traces, mean AUROC is 0.989
versus 0.388. The near-ceiling value is encouraging but rests on only three
correlated ScienceWorld tasks and must not be presented as broad coverage.
An exploratory control removes every repeated `(observation_before, action)`
pair after its first occurrence, leaving 298 of 1,435 nonterminal failed-group
steps. Mean group AUROC remains 0.882 for Jev versus 0.262 for the LLM over
the same three groups; the paired difference is 0.620 with task-bootstrap
95% interval [0.449, 0.910]. The result is therefore not explained solely by
Jev assigning low scores to repeated loops.

Among failed trajectories, nonzero stepwise RLOO density is 0.991 for Jev
versus 0.346 for the LLM; the corresponding active-trajectory rates are 0.927
and 0.271. Observed mean call latency is 0.675 s for remote Jev and 0.808 s
for the local LLM, again not a hardware-matched speed comparison. The 19
all-failed groups provide no best-of-five discrimination in official
continuous final reward: random, Jev, LLM, and sampled oracle all obtain
0.0142 because the sampled rollouts within each group have equal verifier
scores. Thus this panel strengthens local process discrimination and dense
credit coverage, but supplies no trajectory-ranking evidence.

## Earlier ALFWorld stress set: matched frozen-trace comparison

On the previously collected Qwen3-1.7B `diverse` run (10 failed trajectories,
300 transitions, five games), Jev and the local Qwen3-8B judge each scored
the same 300 public transitions; neither policy nor environment was rerun.
For failure-only TextWorld shortest-plan progress, the pooled AUROC is 0.500
for terminal RLVR, 0.650 for Jev, and 0.629 for the LLM judge. For the
environment-grounded hard-step-quality label, the corresponding AUROCs are
0.500, 0.769, and 0.544. Jev's first-hard-error exact localization is 3/10,
versus 0/10 for both baselines. These are encouraging measurement signals on
a very small, correlated, all-failure set; they neither erase the negative
ToolSandbox result nor establish a general Jev advantage or training benefit.

The pooled progress result changes interpretation when games are weighted
equally: mean within-game progress AUROC is 0.500/0.563/0.628 for
RLVR/Jev/LLM across four eligible games, and Jev is below the LLM in three
of those four. For hard-step quality the corresponding within-game means are
0.500/0.761/0.551 across all five games; Jev exceeds the LLM in every game.
This is the narrower, task-balanced ALFWorld signal. The exact five-game
sample is too small for a generalization or training-success claim.

## ALFWorld 8B frozen development slice: aggregation matters

The 12 previously sampled Qwen3-8B `diverse` traces (306 transitions, six
games, four successes) now have complete `jev-1.13.0` annotations; the source
hashes still pass. Mean within-game AUROC is 0.892 for shortest-plan progress
over five eligible games and 0.773 for hard-step quality over all six. A
matched local non-thinking Qwen3-8B LLM judge scored the **same 306 public
states**, with corresponding AUROCs 0.797 and 0.660. First-hard-error exact
localization is 6/12 for Jev versus 1/12 for the LLM. This is an exploratory
development slice, not an independent policy-training result; six correlated
games are too few for a general superiority claim. Paired game-bootstrap
95% intervals (20,000 resamples, seed 20260919) for Jev minus LLM AUROC are
[-0.007, 0.220] for progress (five games) and [0.031, 0.209] for hard-step
quality (six games).
Mean observed call latency is 1.136 s for the remote Jev API versus 0.871 s
for this local LLM service; the hardware and network differ, so this slice
does not establish a throughput advantage.
For best-of-two success selection, averaging centered Jev step scores chooses
a success in 3/6 games, matching the sampled oracle ceiling; summing the same
scores chooses only 1/6, below random selection's 2/6. The LLM's mean-score
selection also chooses 3/6. The distinction is
consistent with the ScienceWorld finding that a locally useful score is not
automatically a useful summed trajectory reward. These six games come from
`valid_unseen`; any training-reward choice informed by them requires a
downstream test that excludes them or uses a disjoint split.

## Trajectory-aggregate online ablation

The resource-matched rerun starts both Qwen2.5-1.5B-Instruct arms from the
same checkpoint and seed, with optimizer offload disabled and vLLM reservation
0.30 in both arms. Their update-0 `valid_seen` success is identically 0/64.
At update 1, all 16 training trajectories fail in both arms. The binary-outcome
arm therefore records zero reward, zero advantage, and policy-gradient loss
0.000. After excluding reward fields and random identifiers, the two arms'
480 recorded transitions have the same SHA-256 digest, confirming that this
first-update comparison uses identical sampled behavior. The Jev arm scores
all 480 newly generated transitions online, with no
API errors or forbidden judge-input fields, and records episode rewards from
-0.076 to -0.032, advantages from -1.355 to 1.449, policy-gradient loss 0.076,
and gradient norm 15.452. Peak reserved GPU memory is 28.1 GiB per card in
both arms. This proves that Jev-derived variation reaches a real optimizer
update where sparse binary GRPO has no task-reward contrast.

Updates 2--5 preserve the contrast: every training trajectory in both arms
fails, the outcome-only arm has zero reward, advantage, and policy-gradient
loss at every update, while the Jev arm has nonzero advantages and
policy-gradient loss at every update. At update 5, held-out `valid_seen`
success is 1/64 (1.6%) for the outcome-only arm and 2/64 (3.1%) for the Jev
arm; mean continuous verifier score is 0.037 versus 0.101. The extra Jev
success is in `pick_and_place`; both arms solve one
`pick_two_obj_and_place` game. This one-success difference at one seed is
directionally encouraging but far too small to establish a downstream gain.
This ablation was superseded at update 5 by the turn-level estimator below; no
update-10 claim is made from it.

This standard-GRPO pilot does **not** assign a different optimizer advantage
to each transition. Jev's transition scores are averaged into one episode
return; the four returns for a task are standardized, and each trajectory's
resulting scalar advantage is broadcast to all of its action tokens. The
experiment therefore tests whether Jev recovers contrast *between* failed
trajectories. Offline reward-to-go analyses measure the stronger within-
trajectory temporal-credit property, which would require a separately
declared turn-level training arm to optimize directly.

## Step-level online GRPO: completed prefix-only one-seed pilot

The revised primary arm gives Jev the explicit static success criteria and
uses its native continuous effect score `q[i,t]` and confidence `c[i,t]`:

`A_jev[i,t] = c[i,t] * (2 * q[i,t] - 1)`,

`A[i,t] = A_jev[i,t] + 0.1 * A_out[i]`.

`A_out[i]` is the standard trajectory-level verifier GRPO advantage. The Jev
term is neither group-standardized nor propagated across future turns, so its
confidence controls the absolute gradient magnitude for the current action.

The controlled run used the same Qwen2.5-1.5B-Instruct checkpoint, seed 0,
four prompts, four rollouts per prompt, 30-step horizon, optimizer, sampling
settings, and `valid_seen` evaluation set in both arms. Both started at 0/64
held-out successes. The first batch contains 480 transitions and all 16
trajectories fail in both arms. A canonical digest over task ID, turn, action,
observation, input, and output is identical in the two runs
(`b5a51e524035baf42061488be604ff5d85fb57a91837f98615a846055187426a`),
so this first-update contrast is on exactly the same sampled behavior.

Sparse GRPO gives every first-batch token zero advantage and reports
policy-gradient loss 0.000. Jev gives 480/480 transitions nonzero advantage,
all 16 trajectories have nonconstant within-trajectory advantages, and the
outcome-equivalent nonzero fraction is 1.000. Its advantages range from
-0.893 to 0.951, mean absolute advantage is 0.668, policy-gradient loss is
0.571, and gradient norm is 7.160. Thus the process scores are not merely
logged: distinct `A[i,t]` values enter the PPO loss in an all-failure group.
Across ten updates the Jev arm retains 0.986--1.000 nonzero transition
coverage and 1.000 nonconstant-trajectory coverage. Sparse GRPO has zero
training success, zero advantage, and zero policy-gradient loss in nine of ten
updates; its only active batch contains one successful trajectory at update 9.

| Held-out `valid_seen` evaluation | Sparse GRPO | Jev step GRPO |
|---|---:|---:|
| Before training: success | 0/64 (0.0%) | 0/64 (0.0%) |
| Update 5: success | 0/64 (0.0%) | 8/64 (12.5%) |
| Update 5: mean verifier score | 0.000 | 0.427 |
| Update 10: success | 1/64 (1.6%) | 19/64 (29.7%) |
| Update 10: mean verifier score | 0.037 | 1.423 |

The full Jev log contains 4,327 transition records. Every record has both a
continuous score and confidence; scores span 0.01--1.00, confidences span
0.00--0.99, 4,287 signed Jev rewards are nonzero, and no response error is
recorded. Every request has exactly four top-level state fields: `task`,
`success_criteria`, `recent_history`, and `current_transition`. Recursive
request keys contain only public actions, observations, and observed results;
there is no verifier output, stored reward, oracle field, gold path, or future
step. The environment verifier is used only to form the small outcome anchor
and to evaluate the held-out benchmark.

This completes the three pilot gates: all-failure groups remain supervised,
the supervision is dense and reaches turn-specific PPO advantages, and the
trained policy is better on the matched held-out benchmark. The final 19/64
versus 1/64 gap is large enough to justify a formal multi-seed study, but this
single training seed and 64-task evaluation do not establish a general
sample-efficiency or benchmark-superiority claim. The authoritative artifacts
are `grpo-alfworld-formal6_v2_baseline4_vb64_q25_15b_s0_t30_u10_20260923/`
and
`grpo-alfworld-formal7_v2_jev4_vb64_retry520_q25_15b_s0_t30_u10_20260923/`.
Each retains all ten rollout JSONLs, the console metrics, and the final model
checkpoint; the Jev directory additionally retains the complete request and
response log.

The superseded reward-to-go estimator remains an implementation ablation. It
standardized same-turn scores and therefore discarded the intended confidence
scale; at update 5 it obtained 0/64 held-out successes versus 1/64 for sparse
GRPO. It is not included in the primary table above.

## Outcome-conditioned hindsight Jev: matched single-arm follow-up

The prefix-only result above is frozen. A single new arm tests the user's
post-episode alternative under the same checkpoint, seed, four-prompt by
four-rollout batches, 30-step horizon, optimizer, ten updates, and 64-task
`valid_seen` evaluation. It does not rerun either the sparse baseline or the
prefix-only Jev arm. After a trajectory terminates, one Jev request receives
the task, static success criteria, complete public action--observation trace,
and verified terminal `{reward, success, reward_definition}`. It contains one
typed Score question per transition and returns continuous `q[i,t]` and
`c[i,t]` for every step. The PPO advantage is unchanged:

`A[i,t] = c[i,t] * (2 * q[i,t] - 1) + 0.1 * A_out[i]`.

This is deliberately outcome- and future-conditioned hindsight labeling. It
is valid as post-episode RL supervision because the policy never consumes the
judge at deployment, but it is not evidence for an online or causal
prefix-only PRM. The request whitelist contains no oracle state, gold path, or
hidden verifier internals.

The first 480 transitions are identical to the frozen prefix-only first batch
under the same canonical digest, and all 16 trajectories fail. Hindsight Jev
still gives nonzero advantage to 479/480 transitions, all 16 trajectories have
nonconstant within-trajectory advantages, mean absolute advantage is 0.803,
policy-gradient loss is 0.726, and gradient norm is 10.326. The prefix-only
values on the same behavior are 480/480, 1.000, 0.668, 0.571, and 7.160,
respectively. Thus both variants satisfy the all-failure and optimizer-path
checks; the hindsight signal is stronger in this first batch, not merely
denser.

| Held-out `valid_seen` evaluation | Sparse GRPO | Prefix-only Jev | Hindsight Jev |
|---|---:|---:|---:|
| Before training: success | 0/64 (0.0%) | 0/64 (0.0%) | 0/64 (0.0%) |
| Update 5: success | 0/64 (0.0%) | 8/64 (12.5%) | 16/64 (25.0%) |
| Update 5: mean verifier score | 0.000 | 0.427 | 1.214 |
| Update 10: success | 1/64 (1.6%) | 19/64 (29.7%) | 22/64 (34.4%) |
| Update 10: mean verifier score | 0.037 | 1.423 | 1.250 |

The prefix-only training-batch success counts over updates 1--10 are
`0, 0, 3, 3, 4, 1, 0, 2, 6, 11` out of 16; the matched hindsight counts are
`0, 0, 4, 1, 5, 2, 3, 1, 10, 9`. These on-policy batch counts are descriptive,
not held-out evaluation. Across the hindsight run, 160 completed-trajectory
requests return 4,350 step labels; 4,332 signed advantages are nonzero
(99.6%), every update has a 1.000 nonconstant-trajectory fraction, and no API
error occurs. Mean and median observed request latency are 3.303 s and 1.755 s.
The request count is about 27 times lower than the prefix arm's 4,327
one-transition requests, although the batched requests are larger and this is
not a controlled cost or throughput comparison.

The success endpoint favors hindsight at both checkpoints: +8 successes at
update 5 and +3 at update 10 relative to prefix-only Jev. The final continuous
verifier score moves in the opposite direction (1.250 versus 1.423), so the
result does not establish uniform superiority. It shows an earlier and modestly
higher strict-success curve at this one seed, while retaining dense turn-level
credit. Multi-seed runs are required before attributing the difference to
outcome conditioning rather than optimization variance. The authoritative new
artifacts are in
`grpo-alfworld-formal8_hindsight4_vb64_q25_15b_s0_t30_u10_20260923/`, including
all ten rollout files, the 160 request/response records, final checkpoint, and
archived TaskRunner metrics. The launcher log contains two infrastructure
tracebacks: an auxiliary Ray worker fails Python codec initialization during
startup, and a DataLoader worker is killed during teardown after final metrics
and checkpoint saving. The main TaskRunner nevertheless records updates 1--10,
the final validation metrics, and all 12 expected rank-sharded model, optimizer,
and extra-state files for checkpoint 10. These warnings do not remove any
reported result, but the run should not be described as having a clean process
exit.

## Current outcome-conditioned Jev-only estimator

The maintained method uses the same completed public trajectory and explicit
verified outcome as the hindsight follow-up, but removes the separately added
`0.1 * A_out` anchor. Its training signal is exactly
`A[i,t] = c[i,t] * (2*q[i,t] - 1)` on the response tokens of turn `t`.

The run starts at 0/64 held-out success. It reaches 16/64 with mean verifier
score 1.067 at update 5 and 21/64 with mean score 1.535 at update 10. Training
batch successes over updates 1--10 are `0, 0, 4, 1, 6, 1, 2, 0, 4, 9` out of
16. Across 4,363 labeled transitions, 99.51% receive nonzero advantage and the
nonconstant-trajectory fraction is 100%; mean Jev confidence is approximately
0.705. The corresponding sparse GRPO endpoint remains 1/64 and 0.037.

The historical outcome-anchored run is 22/64 with mean score 1.250, so the
one-success difference does not identify a reliable winner, while the current
Jev-only run is better on continuous verifier score. The same-turn
group-relative ablation reaches 2/64 with mean score 0.106 and is not retained
as an active method. The authoritative current artifact is
`grpo-alfworld-formal10_v4step_only4_vb64_q25_15b_s0_t30_u10_20260924/`;
its historical directory name predates the code cleanup.
