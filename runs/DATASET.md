# Frozen trajectory index

These `trajectories.jsonl` files are the immutable source traces for later Jev
and LLM judging. Judge annotations use distinct files such as
`jev-transition-v1.jsonl` and `llm-transition-v1.jsonl` beside each run; do not
add judge output to the source records. Offline verifier advantages live
in `advantages/` and do not modify source traces.

## Matched ToolSandbox and ScienceWorld panel

Each primary run has 20 task groups × 5 stochastic rollouts. ToolSandbox uses
the official milestone/minefield verifier; ScienceWorld uses the official
score. All primary ScienceWorld runs use the same budgeted v2 prompt protocol.

| Run | Traces | Steps | Strict success |
|---|---:|---:|---:|
| `toolsandbox-qwen3-1.7b-r100-s20260919` | 100 | 146 | 16/100 |
| `toolsandbox-qwen3-4b-r100-s20260919` | 100 | 139 | 20/100 |
| `toolsandbox-qwen3-8b-r100-s20260919` | 100 | 215 | 19/100 |
| `scienceworld-qwen3-1.7b-r100-v2-s20260919` | 100 | 1,596 | 4/100 |
| `scienceworld-qwen3-4b-r100-v2-s20260919` | 100 | 1,578 | 8/100 |
| `scienceworld-qwen3-8b-r100-v2-s20260919` | 100 | 2,069 | 14/100 |
| **Total** | **600** | **5,743** | **81/600** |

Each run has `manifest.json`, raw per-trajectory files, `advantages/`,
`metrics-rl-credit.json`, and `SOURCE_SHA256SUMS`. Verify a run from its own
directory with `sha256sum -c SOURCE_SHA256SUMS`. Source benchmark revisions:
ToolSandbox `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`; ScienceWorld
`e8216d6044e8e39be9fcb185e3b2dfb602584b52`. See
[`PROCESS_RESULTS.md`](PROCESS_RESULTS.md) for reward and credit-allocation
results.

The v1 ScienceWorld 1.7B and 4B runs, plus the mixed-prompt 8B run, remain in
`runs/` as exploratory records and are excluded from the primary panel.

## Earlier ALFWorld stress set

| Run | Complete traces | Steps | Success | Loop rate | Unique state/action | Natural-error pair cases |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3-1.7b-ood-s20260919` | 12 | 360 | 0/12 | 56.9% | 43.1% | 708 |
| `qwen3-0.6b-diverse-s20260919` | 10 | 300 | 0/10 | 82.3% | 17.7% | 598 |
| `qwen3-1.7b-diverse-s20260919` | 10 | 300 | 0/10 | 55.0% | 45.0% | 554 |
| `qwen3-4b-diverse-s20260919` | 10 | 278 | 1/10 | 36.0% | 65.8% | 484 |
| `qwen3-8b-diverse-s20260919` | 12 | 306 | 4/12 | 38.6% | 65.7% | 544 |
| **Total** | **54** | **1,544** | **5/54** | — | — | **2,888** |

The first run is the original four-game random OOD pilot. The `diverse` runs
share the same stratified games for matched comparison. The 0.6B run contains
the five completed single-object task types; its first two-object episode was
discarded after exceeding the wall-clock cap, as recorded in its manifest. The
8B run retains two complete two-object traces. The 1.7B and 4B runs intentionally
stop after the same five faster task types.

Verify source integrity from this directory with:

```bash
sha256sum -c SHA256SUMS
```
