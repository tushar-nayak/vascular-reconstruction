from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.data.dataset import ProjectionDataset
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.training.trainer import Trainer


def _build_dataset(root):
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
    return ProjectionDataset(root, compute_dt=True, cache_cases=True)


def test_trainer_checkpoint_contains_configs_and_optimizer_state(tmp_path):
    dataset = _build_dataset(tmp_path)
    model_config = ModelConfig(num_gaussians=8, sh_degree=2, pinn_hidden_dim=16, pinn_num_layers=3)
    train_config = TrainingConfig(
        iterations=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config={"hidden_dim": model_config.pinn_hidden_dim, "num_layers": model_config.pinn_num_layers},
        sh_degree=model_config.sh_degree,
    )
    trainer = Trainer(model=model, dataset=dataset, train_config=train_config, model_config=model_config)

    trainer.save_checkpoint(7)
    checkpoint = torch.load(tmp_path / "checkpoints" / "checkpoint_7.pt", map_location="cpu")

    assert checkpoint["iteration"] == 7
    assert checkpoint["model_config"]["num_gaussians"] == 8
    assert checkpoint["training_config"]["device"] == "cpu"
    assert "gs_optimizer_state_dict" in checkpoint
    assert "pinn_optimizer_state_dict" in checkpoint


def test_trainer_bootstraps_gaussians_from_dataset(tmp_path):
    dataset = _build_dataset(tmp_path)
    model_config = ModelConfig(num_gaussians=16, sh_degree=2, pinn_hidden_dim=16, pinn_num_layers=3)
    train_config = TrainingConfig(
        iterations=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug_projection_dir=str(tmp_path / "debug"),
        device="cpu",
    )
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config={"hidden_dim": model_config.pinn_hidden_dim, "num_layers": model_config.pinn_num_layers},
        sh_degree=model_config.sh_degree,
    )

    trainer = Trainer(model=model, dataset=dataset, train_config=train_config, model_config=model_config)
    xyz = trainer.model.gs.get_xyz.detach().cpu().numpy()

    assert xyz.shape == (16, 3)
    assert np.std(xyz[:, 1]) > 0.0
    assert np.std(xyz[:, 2]) > 0.0


def test_geometry_regularization_penalizes_collapsed_points(tmp_path):
    dataset = _build_dataset(tmp_path)
    model_config = ModelConfig(num_gaussians=8, sh_degree=2, pinn_hidden_dim=16, pinn_num_layers=3)
    train_config = TrainingConfig(
        iterations=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug_projection_dir=str(tmp_path / "debug"),
        device="cpu",
        repulsion_num_samples=8,
    )
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config={"hidden_dim": model_config.pinn_hidden_dim, "num_layers": model_config.pinn_num_layers},
        sh_degree=model_config.sh_degree,
    )
    trainer = Trainer(model=model, dataset=dataset, train_config=train_config, model_config=model_config)

    with torch.no_grad():
        trainer.model.gs._xyz.fill_(0.0)

    reg_loss, stats = trainer._geometry_regularization(torch.zeros((8, 2), dtype=torch.float32))
    assert reg_loss.item() > 0.0
    assert stats["repulsion"] > 0.0
    assert stats["xyz_std_mean"] == 0.0
