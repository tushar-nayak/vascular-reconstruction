"""Training loop for vascular reconstruction with improved projection and DT loss."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

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

        # Separate optimizers for GS and PINN
        self.gs_optimizer = optim.Adam(self.model.gs.parameters(), lr=self.config.learning_rate)
        self.pinn_optimizer = optim.Adam(self.model.pinn.parameters(), lr=self.config.pinn_learning_rate)

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

    def train_step(self, case_data: Mapping[str, object]) -> tuple[float, float, float]:
        self.gs_optimizer.zero_grad()
        self.pinn_optimizer.zero_grad()

        # 1. Geometry Loss (Distance Transform on Projections)
        views = case_data["views"]
        
        total_dt_loss = torch.tensor(0.0, device=self.device)

        for view in views:
            lao, cran = view["angles"]
            view_matrix = self.model.get_view_matrix(lao, cran, device=self.device)
            dt_map = torch.from_numpy(view["distance_transform"]).to(self.device)
            height, width = dt_map.shape[-2:]
            projection_matrix = self._projection_matrix_from_view(view, self.device)
            projected_coords = self.model.project_points(
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
            )
            grid_coords = self._grid_coords(projected_coords, width=width, height=height)

            sampled_dt = torch.nn.functional.grid_sample(
                dt_map.unsqueeze(0).unsqueeze(0),
                grid_coords,
                align_corners=True,
                padding_mode="border",
            ).squeeze()

            total_dt_loss += torch.mean(sampled_dt)

        loss_image = total_dt_loss / len(views)

        # 2. Physics Loss (Navier-Stokes)
        raw_coords = torch.rand(1024, 4, device=self.device, requires_grad=True)
        # Match Gaussian volume (-60 to 60)
        coords_xyz = (raw_coords[:, :3] - 0.5) * 120.0
        coords_t = raw_coords[:, 3:4]
        coords = torch.cat([coords_xyz, coords_t], dim=-1)
        
        pinn_out = self.model(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4])
        loss_physics = navier_stokes_loss(pinn_out, coords)

        # 3. Regularization: Keep Gaussians compact
        # loss_reg = torch.mean(self.model.gs.get_scaling()) * 0.1
        
        # Total Loss
        total_loss = loss_image + self.config.physics_loss_weight * loss_physics

        total_loss.backward()

        self.gs_optimizer.step()
        self.pinn_optimizer.step()

        return total_loss.item(), loss_image.item(), loss_physics.item()

    def train(self) -> None:
        print(f"Starting training for {self.config.iterations} iterations...")

        pbar = tqdm(range(self.config.iterations))
        for i in pbar:
            case_idx = i % len(self.dataset)
            case_data = self.dataset.get_case(case_idx)

            try:
                loss, l_img, l_phys = self.train_step(case_data)
                self.failure_count = 0
                if i % 10 == 0:
                    pbar.set_description(f"Loss: {loss:.4f} | DT: {l_img:.4f} | Phys: {l_phys:.4f}")
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
