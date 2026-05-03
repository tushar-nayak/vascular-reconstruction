"""Training loop for vascular reconstruction with geometry diagnostics and anti-collapse regularization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.optim as optim
from tqdm import tqdm

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.data.dataset import ProjectionDataset
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.simulation.equations import navier_stokes_loss


class Trainer:
    """Handles the optimization of GS geometry and PINN physics."""

    def __init__(
        self,
        model: PINN_GS,
        dataset: ProjectionDataset,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        device: str | None = None,
    ):
        resolved_device = device or self._resolve_device(train_config.device)
        self.dataset = dataset
        self.config = train_config
        self.model_config = model_config
        self.device = resolved_device
        self.model = model.to(self.device)
        self.failure_count = 0

        self.debug_projection_dir = Path(self.config.debug_projection_dir)
        self.debug_projection_dir.mkdir(parents=True, exist_ok=True)

        # Separate optimizers for GS and PINN
        self.gs_optimizer = optim.Adam(self.model.gs.parameters(), lr=self.config.learning_rate)
        self.pinn_optimizer = optim.Adam(self.model.pinn.parameters(), lr=self.config.pinn_learning_rate)

        self._initialize_gaussians_from_dataset()

    @staticmethod
    def _resolve_device(configured_device: str) -> str:
        if configured_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured_device

    @staticmethod
    def _projection_matrix_from_view(view: Mapping[str, object], device: str) -> torch.Tensor:
        matrix = view["projection_matrix"]
        return torch.tensor(matrix, dtype=torch.float32, device=device)

    @staticmethod
    def _grid_coords(projected_coords: torch.Tensor, width: int, height: int) -> torch.Tensor:
        x = (projected_coords[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
        y = (projected_coords[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1).view(1, 1, -1, 2)

    @staticmethod
    def _view_rotation(lao: float, cran: float) -> np.ndarray:
        lao_rad = np.radians(lao)
        cran_rad = np.radians(cran)
        ry = np.array(
            [
                [np.cos(lao_rad), 0.0, np.sin(lao_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(lao_rad), 0.0, np.cos(lao_rad)],
            ],
            dtype=np.float32,
        )
        rz = np.array(
            [
                [np.cos(cran_rad), -np.sin(cran_rad), 0.0],
                [np.sin(cran_rad), np.cos(cran_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rz @ ry

    def _initialize_gaussians_from_dataset(self) -> None:
        if len(self.dataset) == 0:
            return

        case_index = min(max(self.config.init_from_case_index, 0), len(self.dataset) - 1)
        case_data = self.dataset.get_case(case_index)
        bootstrap_points = self._bootstrap_points_from_case(case_data)
        if bootstrap_points is None or len(bootstrap_points) == 0:
            return

        with torch.no_grad():
            self.model.gs.initialize_from_points(
                bootstrap_points,
                scale_value=self.config.scale_mean_target,
                opacity_value=self.config.opacity_mean_target,
            )

    def _bootstrap_points_from_case(self, case_data: Mapping[str, object]) -> torch.Tensor | None:
        views = list(case_data["views"])[: self.config.max_init_views]
        if not views:
            return None

        points_per_view = max(self.model_config.num_gaussians // len(views), 1)
        world_points: list[np.ndarray] = []

        for view in views:
            image = np.asarray(view["image"])
            vessel_pixels = np.argwhere(image < 180)
            if len(vessel_pixels) == 0:
                continue

            sample_count = min(points_per_view, len(vessel_pixels))
            rng = np.random.default_rng(int(np.sum(image) % (2**32 - 1)))
            sampled_pixels = vessel_pixels[rng.choice(len(vessel_pixels), size=sample_count, replace=len(vessel_pixels) < sample_count)]

            projection_matrix = np.asarray(view["projection_matrix"], dtype=np.float32)
            focal_x = projection_matrix[0, 0]
            focal_y = projection_matrix[1, 1]
            center_x = projection_matrix[0, 2]
            center_y = projection_matrix[1, 2]

            x_cam = rng.normal(loc=0.0, scale=self.config.init_depth_mm, size=sample_count).astype(np.float32)
            x_dist = np.clip(600.0 + x_cam, 1.0, None)
            pixel_x = sampled_pixels[:, 1].astype(np.float32)
            pixel_y = sampled_pixels[:, 0].astype(np.float32)
            y_cam = ((pixel_x - center_x) / focal_x) * x_dist
            z_cam = -((pixel_y - center_y) / focal_y) * x_dist

            jitter = rng.normal(loc=0.0, scale=self.config.init_jitter_mm, size=(sample_count, 3)).astype(np.float32)
            cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1) + jitter

            lao, cran = view["angles"]
            rotation = self._view_rotation(float(lao), float(cran))
            world_points.append(cam_points @ rotation)

        if not world_points:
            return None

        points_np = np.concatenate(world_points, axis=0)
        if len(points_np) < self.model_config.num_gaussians:
            repeat_count = (self.model_config.num_gaussians + len(points_np) - 1) // len(points_np)
            points_np = np.tile(points_np, (repeat_count, 1))

        points_np = points_np[: self.model_config.num_gaussians]
        return torch.from_numpy(points_np)

    def _geometry_regularization(self, projected_points: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        xyz = self.model.gs.get_xyz
        opacity = self.model.gs.get_opacity.squeeze(-1)
        scaling = self.model.gs.get_scaling

        sample_count = min(self.config.repulsion_num_samples, len(xyz))
        sample_idx = torch.randperm(len(xyz), device=xyz.device)[:sample_count]
        sampled_xyz = xyz[sample_idx]
        pairwise_dist = torch.cdist(sampled_xyz, sampled_xyz)
        valid_mask = ~torch.eye(sample_count, dtype=torch.bool, device=xyz.device)
        repulsion = torch.relu(self.config.min_gaussian_separation - pairwise_dist[valid_mask]).pow(2).mean()

        axis_std = torch.std(xyz, dim=0)
        std_floor = torch.relu(self.config.axis_std_floor - axis_std).pow(2).mean()

        opacity_mean = opacity.mean()
        opacity_reg = (opacity_mean - self.config.opacity_mean_target).pow(2)

        scaling_mean = scaling.mean()
        scale_reg = (scaling_mean - self.config.scale_mean_target).pow(2)

        projected_std = torch.std(projected_points, dim=0).mean()
        projected_std_reg = torch.relu(8.0 - projected_std).pow(2)

        total_reg = (
            self.config.repulsion_weight * repulsion
            + self.config.std_floor_weight * std_floor
            + self.config.opacity_weight * opacity_reg
            + self.config.scale_weight * scale_reg
            + 0.01 * projected_std_reg
        )
        stats = {
            "repulsion": float(repulsion.item()),
            "std_floor": float(std_floor.item()),
            "opacity_mean": float(opacity_mean.item()),
            "scale_mean": float(scaling_mean.item()),
            "xyz_std_mean": float(axis_std.mean().item()),
            "projected_std": float(projected_std.item()),
        }
        return total_reg, stats

    def _save_debug_projection(
        self,
        iteration: int,
        case_data: Mapping[str, object],
        first_projected_coords: torch.Tensor,
    ) -> None:
        if iteration % self.config.debug_projection_interval != 0:
            return

        view = case_data["views"][0]
        image = np.asarray(view["image"], dtype=np.uint8)
        canvas = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        points = first_projected_coords.detach().cpu().numpy()
        for x_coord, y_coord in points[: min(5000, len(points))]:
            draw.ellipse((x_coord - 1, y_coord - 1, x_coord + 1, y_coord + 1), fill=(255, 0, 0))

        output_path = self.debug_projection_dir / f"iter_{iteration:06d}_{case_data['case_id']}.png"
        canvas.save(output_path)

    def train_step(self, case_data: Mapping[str, object], iteration: int) -> tuple[float, float, float, float, dict[str, float]]:
        self.gs_optimizer.zero_grad()
        self.pinn_optimizer.zero_grad()

        views = case_data["views"]
        total_dt_loss = torch.tensor(0.0, device=self.device)
        first_projected_coords: torch.Tensor | None = None

        for view_index, view in enumerate(views):
            lao, cran = view["angles"]
            view_matrix = self.model.get_view_matrix(lao, cran, device=self.device)
            dt_map = torch.from_numpy(view["distance_transform"]).to(self.device)
            height, width = dt_map.shape[-2:]
            projection_matrix = self._projection_matrix_from_view(view, self.device)
            projected_coords = self.model.project_points(
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
            )
            if view_index == 0:
                first_projected_coords = projected_coords

            grid_coords = self._grid_coords(projected_coords, width=width, height=height)
            sampled_dt = torch.nn.functional.grid_sample(
                dt_map.unsqueeze(0).unsqueeze(0),
                grid_coords,
                align_corners=True,
                padding_mode="border",
            ).squeeze()
            total_dt_loss += torch.mean(sampled_dt)

        loss_image = total_dt_loss / len(views)

        raw_coords = torch.rand(1024, 4, device=self.device, requires_grad=True)
        coords_xyz = (raw_coords[:, :3] - 0.5) * 120.0
        coords_t = raw_coords[:, 3:4]
        coords = torch.cat([coords_xyz, coords_t], dim=-1)

        pinn_out = self.model(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4])
        loss_physics = navier_stokes_loss(pinn_out, coords)

        if first_projected_coords is None:
            raise RuntimeError("Projected coordinates were not computed.")

        loss_reg, reg_stats = self._geometry_regularization(first_projected_coords)
        total_loss = loss_image + self.config.physics_loss_weight * loss_physics + loss_reg

        total_loss.backward()
        self.gs_optimizer.step()
        self.pinn_optimizer.step()

        self._save_debug_projection(iteration, case_data, first_projected_coords)
        return total_loss.item(), loss_image.item(), loss_physics.item(), loss_reg.item(), reg_stats

    def train(self) -> None:
        print(f"Starting training for {self.config.iterations} iterations...")

        pbar = tqdm(range(self.config.iterations))
        for i in pbar:
            case_idx = i % len(self.dataset)
            case_data = self.dataset.get_case(case_idx)

            try:
                loss, l_img, l_phys, l_reg, reg_stats = self.train_step(case_data, i)
                self.failure_count = 0
                if i % 10 == 0:
                    pbar.set_description(
                        "Loss: "
                        f"{loss:.4f} | DT: {l_img:.4f} | Phys: {l_phys:.4f} | Reg: {l_reg:.4f} "
                        f"| XYZstd: {reg_stats['xyz_std_mean']:.2f}"
                    )
            except Exception as exc:
                self.failure_count += 1
                print(f"Training failed at iteration {i}: {exc}")
                if self.failure_count >= self.config.max_failures:
                    raise RuntimeError(
                        f"Training aborted after {self.failure_count} consecutive failures."
                    ) from exc

            if i > 0 and i % self.config.save_interval == 0:
                self.save_checkpoint(i)

        self.save_checkpoint(self.config.iterations)

    def save_checkpoint(self, iteration: int) -> None:
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{iteration}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "gs_optimizer_state_dict": self.gs_optimizer.state_dict(),
                "pinn_optimizer_state_dict": self.pinn_optimizer.state_dict(),
                "training_config": self.config.to_dict(),
                "model_config": self.model_config.to_dict(),
            },
            path,
        )
        print(f"Saved checkpoint to {path}")
