from __future__ import annotations

import sys
from pathlib import Path


BEHAVIOR_DIR = Path(__file__).resolve().parents[2] / "behavior_algo"
sys.path.insert(0, str(BEHAVIOR_DIR))

from behavior_detector import TemporalSmoother  # noqa: E402


def test_temporal_smoother_clears_behavior_on_first_missing_frame():
    smoother = TemporalSmoother(window=4, activate=1, deactivate=1)

    assert smoother.update(["phone_use"], 0.0) == ["phone_use"]
    assert smoother.update([], 0.3) == []
    assert smoother.update([], 0.6) == []
