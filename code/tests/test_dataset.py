from __future__ import annotations

import json

import numpy as np
from PIL import Image

from vascular_reconstruction.data.dataset import ProjectionDataset


def test_projection_dataset_caches_case_and_distance_transform(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    image_path = root / "case1_AP.png"
    image = np.full((4, 4), 255, dtype=np.uint8)
    image[1:3, 1:3] = 0
    Image.fromarray(image).save(image_path)

    manifest = {
        "case1_AP.png": {
            "mesh_source": "case1",
            "view_name": "AP",
            "angles_deg": [0.0, 0.0],
            "projection_matrix": [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
        }
    }
    with (root / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    dataset = ProjectionDataset(root, compute_dt=True, cache_cases=True)
    first = dataset.get_case(0)
    second = dataset.get_case(0)

    assert first is second
    assert "distance_transform" in first["views"][0]
    assert first["views"][0]["distance_transform"].shape == (4, 4)
