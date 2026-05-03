from __future__ import annotations

from pathlib import Path

from vascular_reconstruction.data.dataset_generation import ViewSpec, _mesh_outputs_complete


def test_mesh_outputs_complete_requires_all_views(tmp_path):
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    mesh_path = str(Path("/tmp/example_mesh.stl"))
    views = (ViewSpec("AP", 0.0, 0.0), ViewSpec("Lateral", 90.0, 0.0))

    (output_dir / "example_mesh_AP.png").write_bytes(b"ok")
    assert not _mesh_outputs_complete(output_dir, mesh_path, views)

    (output_dir / "example_mesh_Lateral.png").write_bytes(b"ok")
    assert _mesh_outputs_complete(output_dir, mesh_path, views)
