#!/usr/bin/env python3
"""V4: V2 hindsight Jev credit with no explicit outcome-advantage anchor."""

from score_jev_v2 import score_completed_trajectory as _score_completed_trajectory


RUBRIC_VERSION = "jev_hindsight_step_only_v1"


def score_completed_trajectory(key, trajectory):
    record = _score_completed_trajectory(key, trajectory)
    record["rubric_version"] = RUBRIC_VERSION
    return record
