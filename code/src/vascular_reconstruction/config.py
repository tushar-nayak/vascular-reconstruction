"""Shared configuration types and helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class for all configuration objects."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        # Filter out keys that are not in the dataclass
        fields = cls.__dataclass_fields__
        filtered_data = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered_data)

    @classmethod
    def load(cls: Type[T], path: str | Path) -> T:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for the training loop."""

    experiment_name: str = "default"
    iterations: int = 30000
    learning_rate: float = 0.001
    pinn_learning_rate: float = 0.0001
    physics_loss_weight: float = 0.05
    repulsion_weight: float = 0.02
    min_gaussian_separation: float = 1.5
    repulsion_num_samples: int = 512
    std_floor_weight: float = 0.05
    axis_std_floor: float = 6.0
    opacity_weight: float = 0.01
    opacity_mean_target: float = 0.65
    scale_weight: float = 0.01
    scale_mean_target: float = 1.2
    silhouette_loss_weight: float = 1.0
    mask_bce_weight: float = 0.7
    mask_dice_weight: float = 0.3
    render_image_size: int = 128
    gaussian_chunk_size: int = 512
    physics_warmup_iterations: int = 5000
    save_interval: int = 5000
    eval_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_projection_dir: str = "logs/projections"
    debug_projection_interval: int = 1000
    device: str = "auto"
    max_failures: int = 5
    init_from_case_index: int = 0
    init_depth_mm: float = 20.0
    init_jitter_mm: float = 4.0
    max_init_views: int = 4
    train_case_index: int = 0


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for the hybrid PINN-GS model."""

    num_gaussians: int = 50000
    sh_degree: int = 3
    pinn_hidden_dim: int = 128
    pinn_num_layers: int = 4
