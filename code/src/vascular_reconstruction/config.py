"""Shared configuration types and helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
    opacity_mean_target: float = 0.8
    scale_weight: float = 0.02
    scale_mean_target: float = 0.25
    silhouette_loss_weight: float = 1.0
    mask_bce_weight: float = 0.7
    mask_dice_weight: float = 0.3
    skeleton_loss_weight: float = 0.15
    skeleton_focus_weight: float = 0.7
    skeleton_thickness_weight: float = 0.3
    volume_thickness_weight: float = 0.03
    volume_core_weight: float = 2.0
    volume_grid_size: int = 24
    volume_sample_size: int = 2048
    volume_chunk_size: int = 256
    outside_mask_weight: float = 0.5
    mass_match_weight: float = 0.1
    continuity_weight: float = 0.02
    continuity_knn: int = 6
    continuity_max_distance: float = 3.5
    graph_connectivity_weight: float = 0.0
    graph_sample_size: int = 192
    graph_edge_target: float = 2.5
    graph_bridge_edges: int = 8
    line_structure_weight: float = 0.05
    line_structure_knn: int = 8
    point_support_weight: float = 0.0
    point_skeleton_weight: float = 0.0
    point_support_sample_size: int = 1024
    point_support_views: int = 6
    point_vessel_min_ratio: float = 0.8
    point_skeleton_dilation_radius_px: int = 3
    render_image_size: int = 128
    gaussian_chunk_size: int = 512
    render_min_sigma: float = 0.12
    render_max_sigma: float = 4.0
    physics_warmup_iterations: int = 5000
    save_interval: int = 5000
    eval_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_projection_dir: str = "logs/projections"
    debug_projection_interval: int = 1000
    device: str = "auto"
    max_failures: int = 5
    active_gaussian_schedule: list[list[int]] = field(default_factory=list)
    densify_opacity_scale: float = 0.85
    densify_scale_shrink: float = 0.7
    densify_spacing_scale: float = 0.75
    densify_jitter_scale: float = 0.2
    densify_edge_knn: int = 4
    densify_support_views: int = 6
    densify_support_samples: int = 5
    densify_support_radius_px: int = 3
    densify_min_support_ratio: float = 0.35
    densify_support_vessel_weight: float = 0.35
    densify_support_skeleton_weight: float = 0.65
    init_from_case_index: int = 0
    init_depth_mm: float = 8.0
    init_jitter_mm: float = 1.5
    max_init_views: int = 6
    train_case_index: int = 0


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for the hybrid PINN-GS model."""

    num_gaussians: int = 50000
    sh_degree: int = 3
    pinn_hidden_dim: int = 128
    pinn_num_layers: int = 4
