"""Training loop for vascular reconstruction with improved projection and DT loss."""

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
        self.gs_optimizer = optim.Adam(self.model.gs.parameters(), lr=0.01)
        self.pinn_optimizer = optim.Adam(self.model.pinn.parameters(), lr=1e-4)

    def train_step(self, case_data: dict):
        self.gs_optimizer.zero_grad()
        self.pinn_optimizer.zero_grad()

        # 1. Geometry Loss (Distance Transform on Projections)
        views = case_data["views"]
        
        total_dt_loss = 0
        
        for view in views:
            lao, cran = view["angles"]
            view_matrix = self.model.get_view_matrix(lao, cran, device=self.device)
            
            # Project points to 2D
            # projected_coords: [N, 2] in pixel space
            projected_coords = self.model.project_points(view_matrix, img_size=1024)
            
            # Distance Transform of the vessel mask
            # dt_map: [1024, 1024], values are distance to nearest vessel pixel
            dt_map = torch.from_numpy(view["distance_transform"]).to(self.device)
            
            # Sample DT values at projected locations
            # grid_sample expects coordinates in [-1, 1]
            grid_coords = (projected_coords / 512.0) - 1.0
            grid_coords = grid_coords.unsqueeze(0).unsqueeze(0) # [1, 1, N, 2]
            
            # sampled_dt: [N]
            sampled_dt = torch.nn.functional.grid_sample(
                dt_map.unsqueeze(0).unsqueeze(0), 
                grid_coords, 
                align_corners=True,
                padding_mode='border'
            ).squeeze()
            
            # Minimize distance to vessels
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
        total_loss = loss_image + 0.05 * loss_physics # Increased physics weight slightly
        
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
                    pbar.set_description(f"Loss: {loss:.4f} | DT: {l_img:.4f} | Phys: {l_phys:.4f}")
            except Exception as e:
                # print(f"Error at iteration {i}: {e}")
                continue
            
            if i > 0 and i % self.config.save_interval == 0:
                self.save_checkpoint(i)
                
        self.save_checkpoint(self.config.iterations)

    def save_checkpoint(self, iteration: int):
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{iteration}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")
