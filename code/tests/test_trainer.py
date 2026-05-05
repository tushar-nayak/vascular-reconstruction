from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.data.dataset import ProjectionDataset
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.rendering import downsample_mask, render_gaussian_silhouette
from vascular_reconstruction.training.trainer import Trainer


def _build_dataset(root):
    image_path = root / "case1_AP.png"
    image = np.full((32, 32), 255, dtype=np.uint8)
    image[10:22, 14:18] = 0
    Image.fromarray(image).save(image_path)

    manifest = {
        "case1_AP.png": {
            "mesh_source": "case1",
            "view_name": "AP",
            "angles_deg": [0.0, 0.0],
            "projection_matrix": [[10.0, 0.0, 16.0], [0.0, 10.0, 16.0], [0.0, 0.0, 1.0]],
        }
    }
    with (root / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return ProjectionDataset(root, compute_dt=True, cache_cases=True)


def _build_trainer(tmp_path, num_gaussians: int = 16) -> Trainer:
    dataset = _build_dataset(tmp_path)
    model_config = ModelConfig(num_gaussians=num_gaussians, sh_degree=2, pinn_hidden_dim=16, pinn_num_layers=3)
    train_config = TrainingConfig(
        iterations=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug_projection_dir=str(tmp_path / "debug"),
        debug_projection_interval=1,
        device="cpu",
        repulsion_num_samples=min(num_gaussians, 16),
        render_image_size=16,
        physics_warmup_iterations=10,
    )
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config={"hidden_dim": model_config.pinn_hidden_dim, "num_layers": model_config.pinn_num_layers},
        sh_degree=model_config.sh_degree,
    )
    return Trainer(model=model, dataset=dataset, train_config=train_config, model_config=model_config)


def test_trainer_checkpoint_contains_configs_and_optimizer_state(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.save_checkpoint(7)
    checkpoint = torch.load(tmp_path / "checkpoints" / "checkpoint_7.pt", map_location="cpu")

    assert checkpoint["iteration"] == 7
    assert checkpoint["model_config"]["num_gaussians"] == 8
    assert checkpoint["training_config"]["device"] == "cpu"
    assert checkpoint["case_id"] == "case1"
    assert "gs_optimizer_state_dict" in checkpoint
    assert "pinn_optimizer_state_dict" in checkpoint


def test_trainer_can_restore_checkpoint_state(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    with torch.no_grad():
        trainer.model.gs._xyz.fill_(3.0)
    trainer.save_checkpoint(9)

    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    restored = _build_trainer(restored_root, num_gaussians=8)
    restored_iteration = restored.load_checkpoint(tmp_path / "checkpoints" / "checkpoint_9.pt")

    assert restored_iteration == 9
    assert torch.allclose(restored.model.gs._xyz, torch.full_like(restored.model.gs._xyz, 3.0))


def test_trainer_bootstraps_gaussians_from_dataset(tmp_path):
    trainer = _build_trainer(tmp_path)
    xyz = trainer.model.gs.get_xyz.detach().cpu().numpy()

    assert xyz.shape == (16, 3)
    assert np.std(xyz[:, 1]) > 0.0
    assert np.std(xyz[:, 2]) > 0.0


def test_geometry_regularization_penalizes_collapsed_points(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)

    with torch.no_grad():
        trainer.model.gs._xyz.fill_(0.0)

    active_indices = torch.arange(8)
    reg_loss, stats = trainer._geometry_regularization(0, active_indices)
    assert reg_loss.item() > 0.0
    assert stats["repulsion"] > 0.0
    assert stats["xyz_std_mean"] == 0.0


def test_line_structure_penalty_prefers_line_over_blob(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)

    line_points = torch.stack(
        [
            torch.linspace(-4.0, 4.0, steps=8),
            torch.zeros(8),
            torch.zeros(8),
        ],
        dim=-1,
    )
    blob_points = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(line_points)
    active_indices = torch.arange(8)
    _, line_stats = trainer._geometry_regularization(0, active_indices)

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(blob_points)
    _, blob_stats = trainer._geometry_regularization(0, active_indices)

    assert line_stats["line_structure"] < blob_stats["line_structure"]


def test_graph_connectivity_penalty_prefers_connected_chain(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.config.graph_connectivity_weight = 1.0
    trainer.config.graph_sample_size = 8
    trainer.config.graph_edge_target = 2.0
    trainer.config.graph_bridge_edges = 2
    active_indices = torch.arange(8)

    chain_points = torch.stack(
        [
            torch.linspace(-7.0, 7.0, steps=8),
            torch.zeros(8),
            torch.zeros(8),
        ],
        dim=-1,
    )
    fragmented_points = torch.tensor(
        [
            [-12.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [-6.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(chain_points)
    _, chain_stats = trainer._geometry_regularization(0, active_indices)

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(fragmented_points)
    _, fragmented_stats = trainer._geometry_regularization(0, active_indices)

    assert chain_stats["graph_connectivity"] < fragmented_stats["graph_connectivity"]
    assert chain_stats["graph_edge_p90"] < fragmented_stats["graph_edge_p90"]
    assert chain_stats["graph_bridge_mean"] < fragmented_stats["graph_bridge_mean"]


def test_active_gaussian_schedule_changes_count(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=16)
    trainer.config.active_gaussian_schedule = [[0, 4], [10, 12]]

    assert trainer._active_gaussian_count(0) == 4
    assert trainer._active_gaussian_count(9) == 4
    assert trainer._active_gaussian_count(10) == 12


def test_skeleton_loss_prefers_thin_centerline(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    target_mask = torch.zeros((16, 16), dtype=torch.float32)
    target_mask[:, 7:9] = 1.0
    skeleton_mask = torch.zeros((16, 16), dtype=torch.float32)
    skeleton_mask[:, 8] = 1.0

    thin_render = skeleton_mask.clone()
    thick_render = target_mask.clone()

    thin_loss = trainer._skeleton_loss(thin_render, target_mask, skeleton_mask)
    thick_loss = trainer._skeleton_loss(thick_render, target_mask, skeleton_mask)

    assert thin_loss.item() < thick_loss.item()


def test_volume_thickness_loss_prefers_line_over_blob(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.config.volume_grid_size = 16
    trainer.config.volume_sample_size = 8
    active_indices = torch.arange(8)

    line_points = torch.stack(
        [
            torch.linspace(-4.0, 4.0, steps=8),
            torch.zeros(8),
            torch.zeros(8),
        ],
        dim=-1,
    )
    blob_points = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        trainer.model.gs._xyz.copy_(line_points)
        trainer.model.gs._scaling.fill_(np.log(0.4))
        trainer.model.gs._opacity.fill_(2.0)
    line_loss, line_stats = trainer._volume_thickness_loss(active_indices)

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(blob_points)
    blob_loss, blob_stats = trainer._volume_thickness_loss(active_indices)

    assert line_loss.item() < blob_loss.item()
    assert line_stats["volume_core_fill"] < blob_stats["volume_core_fill"]


def test_densify_clones_active_structure_into_inactive_slots(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    active_indices = torch.arange(4)

    with torch.no_grad():
        trainer.model.gs._xyz[:4] = torch.tensor(
            [
                [-4.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        trainer.model.gs._xyz[4:] = 99.0
        trainer.model.gs._opacity[:4] = 2.0
        trainer.model.gs._opacity[4:] = -10.0
        trainer.model.gs._scaling[:4] = np.log(0.5)

    trainer._densify_to_count(active_indices, 8)

    new_xyz = trainer.model.gs.get_xyz[4:].detach()
    assert torch.all(new_xyz.abs() < 20.0)
    assert torch.std(new_xyz[:, 0]).item() > 0.0


def test_densify_targets_gap_edges(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.config.graph_sample_size = 4
    trainer.config.densify_edge_knn = 1
    trainer.config.densify_spacing_scale = 0.1
    trainer.config.densify_jitter_scale = 0.01
    active_indices = torch.arange(4)

    with torch.no_grad():
        trainer.model.gs._xyz[:4] = torch.tensor(
            [
                [-10.0, 0.0, 0.0],
                [-9.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        trainer.model.gs._xyz[4:] = 99.0
        trainer.model.gs._opacity[:4] = 2.0
        trainer.model.gs._opacity[4:] = -10.0
        trainer.model.gs._scaling[:4] = np.log(0.5)

    trainer._densify_to_count(active_indices, 8)

    new_x = trainer.model.gs.get_xyz[4:, 0].detach().cpu().numpy()
    assert np.max(np.abs(new_x)) < 6.0


def test_edge_multiview_support_prefers_supported_bridge(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.config.densify_support_views = 1
    trainer.config.densify_support_samples = 5
    trainer.config.densify_support_radius_px = 0

    supported_start = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float32)
    supported_end = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)
    unsupported_start = torch.tensor([-10.0, 400.0, 0.0], dtype=torch.float32)
    unsupported_end = torch.tensor([10.0, 400.0, 0.0], dtype=torch.float32)

    supported_score = trainer._edge_multiview_support(supported_start, supported_end)
    unsupported_score = trainer._edge_multiview_support(unsupported_start, unsupported_end)

    assert supported_score > unsupported_score


def test_point_multiview_support_prefers_centerline_points(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    trainer.config.point_support_weight = 1.0
    trainer.config.point_skeleton_weight = 1.0
    trainer.config.point_support_views = 1
    trainer.config.point_support_sample_size = 8
    trainer.config.point_vessel_min_ratio = 0.8

    centered_points = torch.stack(
        [
            torch.linspace(-8.0, 8.0, steps=8),
            torch.zeros(8),
            torch.zeros(8),
        ],
        dim=-1,
    )
    shifted_points = centered_points.clone()
    shifted_points[:, 1] = 250.0
    active_indices = torch.arange(8)

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(centered_points)
        trainer.model.gs._opacity.fill_(2.0)
    centered_loss, centered_stats = trainer._point_multiview_support_loss(active_indices)

    with torch.no_grad():
        trainer.model.gs._xyz.copy_(shifted_points)
    shifted_loss, shifted_stats = trainer._point_multiview_support_loss(active_indices)

    assert centered_loss.item() < shifted_loss.item()
    assert centered_stats["point_vessel_support"] > shifted_stats["point_vessel_support"]
    assert centered_stats["point_skeleton_support"] > shifted_stats["point_skeleton_support"]


def test_silhouette_renderer_returns_soft_mask(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    view = trainer.case_data["views"][0]
    view_matrix = trainer.model.get_view_matrix(0.0, 0.0, device="cpu")
    projection_matrix = torch.tensor(view["projection_matrix"], dtype=torch.float32)

    rendered = render_gaussian_silhouette(
        model=trainer.model,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        source_image_size=view["vessel_mask"].shape,
        render_size=16,
        chunk_size=8,
        min_sigma=0.05,
        max_sigma=2.0,
    )
    target = downsample_mask(torch.from_numpy(view["vessel_mask"]), 16)

    assert rendered.shape == (16, 16)
    assert torch.all(rendered >= 0.0)
    assert torch.all(rendered <= 1.0)
    assert target.shape == (16, 16)


def test_silhouette_renderer_keeps_scale_gradients_near_sigma_floor(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    view = trainer.case_data["views"][0]
    view_matrix = trainer.model.get_view_matrix(0.0, 0.0, device="cpu")
    projection_matrix = torch.tensor(view["projection_matrix"], dtype=torch.float32)

    trainer.model.gs._scaling.grad = None
    rendered = render_gaussian_silhouette(
        model=trainer.model,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        source_image_size=view["vessel_mask"].shape,
        render_size=16,
        chunk_size=8,
        min_sigma=0.5,
        max_sigma=2.0,
    )
    rendered.mean().backward()

    assert trainer.model.gs._scaling.grad is not None
    assert torch.count_nonzero(trainer.model.gs._scaling.grad).item() > 0


def test_train_step_writes_debug_projection(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    loss, sil, phys, reg, stats = trainer.train_step(0)

    assert loss >= 0.0
    assert sil >= 0.0
    assert phys == 0.0
    assert reg >= 0.0
    assert stats["xyz_std_mean"] > 0.0
    assert "continuity" in stats
    assert "line_structure" in stats
    assert "volume_core_fill" in stats
    debug_files = sorted((tmp_path / "debug").glob("*.png"))
    assert len(debug_files) == 1
