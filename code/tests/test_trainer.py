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

    reg_loss, stats = trainer._geometry_regularization()
    assert reg_loss.item() > 0.0
    assert stats["repulsion"] > 0.0
    assert stats["xyz_std_mean"] == 0.0


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
    )
    target = downsample_mask(torch.from_numpy(view["vessel_mask"]), 16)

    assert rendered.shape == (16, 16)
    assert torch.all(rendered >= 0.0)
    assert torch.all(rendered <= 1.0)
    assert target.shape == (16, 16)


def test_train_step_writes_debug_projection(tmp_path):
    trainer = _build_trainer(tmp_path, num_gaussians=8)
    loss, sil, phys, reg, stats = trainer.train_step(0)

    assert loss >= 0.0
    assert sil >= 0.0
    assert phys == 0.0
    assert reg >= 0.0
    assert stats["xyz_std_mean"] > 0.0
    debug_files = sorted((tmp_path / "debug").glob("*.png"))
    assert len(debug_files) == 1
