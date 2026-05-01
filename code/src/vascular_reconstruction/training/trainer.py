"""Training loop for vascular reconstruction."""

from __future__ import annotations

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.simulation.equations import navier_stokes_loss
from vascular_reconstruction.data.dataset import ProjectionDataset


class Trainer:
    """Handles the optimization of GS geometry and PINN physics."""

    def __init__(
        self,
        model: PINN_GS,
        dataset: ProjectionDataset,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.config = train_config
        self.model_config = model_config
        self.device = device

        # Separate optimizers for GS and PINN
        self.gs_optimizer = optim.Adam(self.model.gs.parameters(), lr=0.005)
        self.pinn_optimizer = optim.Adam(self.model.pinn.parameters(), lr=1e-4)

    def get_view_matrix(self, lao: float, cran: float) -> torch.Tensor:
        """Compute camera view matrix from angles."""
        lao_rad = np.radians(lao)
        cran_rad = np.radians(cran)
        
        ry = torch.tensor([
            [np.cos(lao_rad), 0, np.sin(lao_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(lao_rad), 0, np.cos(lao_rad), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        rz = torch.tensor([
            [np.cos(cran_rad), -np.sin(cran_rad), 0, 0],
            [np.sin(cran_rad), np.cos(cran_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        return rz @ ry

    def train_step(self, case_data: dict):
        self.gs_optimizer.zero_grad()
        self.pinn_optimizer.zero_grad()

        # 1. Geometry Loss (Image Projection)
        views = case_data["views"]
        view = views[torch.randint(0, len(views), (1,)).item()]
        
        target_image = torch.from_numpy(view["image"]).float().to(self.device) / 255.0
        lao, cran = view["angles"]
        view_matrix = self.get_view_matrix(lao, cran)
        
        xyz = self.model.gs.get_xyz
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
        xyz_cam = (xyz_hom @ view_matrix.T)[:, :3]
        
        # Simple point projection
        x_proj = xyz_cam[:, 0] / (xyz_cam[:, 2] + 600) * 1000 + 512
        y_proj = xyz_cam[:, 1] / (xyz_cam[:, 2] + 600) * 1000 + 512
        
        mask = (x_proj >= 0) & (x_proj < 1024) & (y_proj >= 0) & (y_proj < 1024)
        
        if torch.sum(mask) > 0:
            vessel_mask = 1.0 - target_image
            grid_coords = torch.stack([
                (x_proj[mask] / 512.0) - 1.0,
                (y_proj[mask] / 512.0) - 1.0
            ], dim=-1).unsqueeze(0).unsqueeze(0)
            
            sampled_values = torch.nn.functional.grid_sample(
                vessel_mask.unsqueeze(0).unsqueeze(0), 
                grid_coords, 
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros'
            ).squeeze()
            
            loss_image = 1.0 - torch.mean(sampled_values)
        else:
            # Penalty for being out of bounds
            loss_image = torch.mean(torch.abs(xyz)) * 0.01

        # 2. Physics Loss (Navier-Stokes)
        raw_coords = torch.rand(512, 4, device=self.device, requires_grad=True)
        # Avoid in-place modification
        coords = torch.zeros_like(raw_coords)
        coords_xyz = (raw_coords[:, :3] - 0.5) * 200.0
        coords_t = raw_coords[:, 3:4]
        coords = torch.cat([coords_xyz, coords_t], dim=-1)
        
        pinn_out = self.model(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4])
        loss_physics = navier_stokes_loss(pinn_out, coords)

        # Total Loss
        total_loss = loss_image + 0.01 * loss_physics
        
        total_loss.backward()
        
        self.gs_optimizer.step()
        self.pinn_optimizer.step()

        return total_loss.item(), loss_image.item(), loss_physics.item()

    def train(self):
        print(f"Starting training for {self.config.iterations} iterations...")
        
        pbar = tqdm(range(self.config.iterations))
        for i in pbar:
            case_idx = i % len(self.dataset)
            case_data = self.dataset.get_case(case_idx)
            
            try:
                loss, l_img, l_phys = self.train_step(case_data)
                if i % 10 == 0:
                    pbar.set_description(f"Loss: {loss:.4f} | Img: {l_img:.4f} | Phys: {l_phys:.4f}")
            except Exception as e:
                # print(f"Error at iteration {i}: {e}")
                continue
            
            if i > 0 and i % self.config.save_interval == 0:
                self.save_checkpoint(i)

    def save_checkpoint(self, iteration: int):
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{iteration}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")
