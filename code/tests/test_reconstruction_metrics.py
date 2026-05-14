from __future__ import annotations

import numpy as np

from vascular_reconstruction.evaluation.reconstruction_metrics import (
    ReconstructionGeometry,
    active_gaussian_count_from_schedule,
    select_active_geometry,
)


def test_active_gaussian_count_from_schedule_uses_latest_applicable_stage():
    schedule = [[0, 4], [10, 6], [20, 8]]

    assert active_gaussian_count_from_schedule(12, schedule, iteration=0) == 4
    assert active_gaussian_count_from_schedule(12, schedule, iteration=15) == 6
    assert active_gaussian_count_from_schedule(12, schedule, iteration=25) == 8


def test_select_active_geometry_keeps_highest_opacity_subset():
    geometry = ReconstructionGeometry(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        scales=np.ones((4, 3), dtype=np.float32),
        opacities=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
    )

    active_geometry = select_active_geometry(geometry, active_count=2)

    assert len(active_geometry.xyz) == 2
    assert {tuple(point.tolist()) for point in active_geometry.xyz} == {
        (1.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    }
