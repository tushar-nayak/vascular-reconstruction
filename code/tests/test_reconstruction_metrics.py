from __future__ import annotations

import numpy as np

from vascular_reconstruction.evaluation.reconstruction_metrics import (
    ReconstructionGeometry,
    active_gaussian_count_from_schedule,
    gate_and_score_from_metrics,
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


def test_gate_and_score_from_metrics_applies_thresholds():
    metrics = {
        "largest_component_fraction": 0.52,
        "component_count": 180,
        "voxel_largest_component_fraction": 0.61,
        "voxel_component_count": 420,
        "occupancy_fill_ratio": 0.02,
        "mesh_vertex_chamfer_p95": 150.0,
        "mst_p95": 2.75,
    }
    training_config = {
        "gate_min_graph_largest_component_fraction": 0.45,
        "gate_max_graph_component_count": 300,
        "gate_min_voxel_largest_component_fraction": 0.45,
        "gate_max_voxel_component_count": 900,
        "gate_max_occupancy_fill_ratio": 0.03,
        "gate_max_mesh_vertex_chamfer_p95": 180.0,
    }

    result = gate_and_score_from_metrics(metrics, training_config=training_config)

    assert result["gate_pass"] is True
    assert result["score"] == [-0.61, 420.0, 150.0, 0.02, 2.75]
