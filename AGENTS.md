# Jev project instructions

- Read `DESIGN.md` before changing an experiment. Its formal open-source
  comparison section is the current source of truth.
- The maintained training method is outcome-conditioned Jev-only credit:
  `A = confidence * (2 * score - 1)`. Historical reward variants are frozen
  ablations, not executable modes to revive.
- Treat seed 0 and the repeatedly inspected 64-task evaluation as development
  evidence. Do not tune on them or include them as confirmatory seeds.
- Keep frozen source trajectories immutable and keep judge annotations in
  separate files.
- Use the locally adapted `verl-agent/`; do not replace it with a fresh clone.
- Keep formal scale, paired seeds, and evaluation panels/milestones only in
  `config/config.yaml`; do not duplicate them in launchers. Invalid-action
  shaping is disabled in the maintained launcher.
- The paper source is `assets/paper/main.tex`; offline utilities are under
  `offline/`, and the maintained Jev scorer is `src/score.py`.
- Launch online runs through `scripts/run_alfworld.sh` and preserve exact
  configs, per-task evaluation records, and raw judge responses.
