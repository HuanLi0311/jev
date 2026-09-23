#!/usr/bin/env python3
"""V3: reuse V2 hindsight labels; center them across same-turn rollouts."""

from score_jev_v2 import score_completed_trajectory as _score_completed_trajectory


RUBRIC_VERSION = "jev_hindsight_group_relative_v1"


def score_completed_trajectory(key, trajectory):
    """Return raw q/confidence; group-relative advantage is computed batch-side."""
    record = _score_completed_trajectory(key, trajectory)
    record["rubric_version"] = RUBRIC_VERSION
    for credit in record["step_credit"]:
        credit.pop("jev_advantage", None)
    return record
