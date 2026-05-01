"""Physics-Informed Neural Network (PINN) and Gaussian Splatting components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PINN(nn.Module):
    """
    PINN for solving Navier-Stokes equations in the vascular geometry.
    Predicts velocity (u, v, w) and pressure (p) given (x, y, z, t).
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, num_layers: int = 4, output_dim: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y, z, t], dim=-1)
        return self.net(inputs)


class GaussianSplatting(nn.Module):
    """
    Gaussian Splatting representation for vascular geometry.
    """

    def __init__(self, num_gaussians: int, sh_degree: int = 3):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree

        # Parameters
        self._xyz = nn.Parameter(torch.empty(num_gaussians, 3))
        self._features_dc = nn.Parameter(torch.empty(num_gaussians, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(num_gaussians, (sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(torch.empty(num_gaussians, 3))
        self._rotation = nn.Parameter(torch.empty(num_gaussians, 4))
        self._opacity = nn.Parameter(torch.empty(num_gaussians, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize in a sphere for now
        nn.init.uniform_(self._xyz, -100, 100)
        nn.init.zeros_(self._features_dc)
        nn.init.zeros_(self._features_rest)
        nn.init.constant_(self._scaling, 1.0)
        nn.init.constant_(self._rotation, 0)
        self._rotation.data[:, 0] = 1.0 # identity quaternion
        nn.init.constant_(self._opacity, 0.1)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return F.normalize(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def get_covariance(self) -> torch.Tensor:
        """Compute the 3D covariance matrices from scaling and rotation."""
        scaling = self.get_scaling
        rotation = self.get_rotation
        
        r, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
        R = torch.stack([
            1 - 2 * (y**2 + z**2), 2 * (x*y - r*z), 2 * (x*z + r*y),
            2 * (x*y + r*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - r*x),
            2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x**2 + y**2)
        ], dim=-1).reshape(-1, 3, 3)
        
        S = torch.diag_embed(scaling)
        M = R @ S
        return M @ M.transpose(1, 2)


class PINN_GS(nn.Module):
    """
    Hybrid model combining PINN and GS.
    """

    def __init__(self, num_gaussians: int, pinn_config: dict):
        super().__init__()
        self.gs = GaussianSplatting(num_gaussians)
        self.pinn = PINN(**pinn_config)

    def project_xray(self, view_matrix: torch.Tensor, proj_matrix: torch.Tensor, img_size: tuple[int, int]):
        """
        Differentiable X-ray projection of the Gaussians (Beer-Lambert).
        Vectorized PyTorch implementation.
        """
        xyz = self.gs.get_xyz
        opacity = self.gs.get_opacity
        cov3d = self.gs.get_covariance()
        
        # 1. Transform centers to camera space
        # view_matrix: [4, 4]
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
        xyz_cam = (xyz_hom @ view_matrix.T)[:, :3]
        
        # 2. Project 3D Covariance to 2D
        # For X-ray (parallel or perspective), the projection of a 3D Gaussian is a 2D Gaussian
        W = view_matrix[:3, :3]
        
        # cov2d = J @ W @ cov3d @ W^T @ J^T
        # For simplicity in this version, assume orthographic/near-orthographic for the split trees
        cov_cam = W @ cov3d @ W.T
        cov2d = cov_cam[:, :2, :2] # Take top-left 2x2
        
        # 3. Rasterize (Simplified: Sum of Gaussians)
        # We'll use a small subset of Gaussians or a coarser grid if needed for speed
        # But let's try a vectorized point-in-Gaussian check
        
        h, w = img_size
        y, x = torch.meshgrid(torch.linspace(-150, 150, h, device=xyz.device), 
                              torch.linspace(-150, 150, w, device=xyz.device), indexing='ij')
        pixel_coords = torch.stack([x, y], dim=-1).reshape(-1, 2) # [H*W, 2]
        
        # This is the memory-intensive part. For 100k Gaussians and 1024x1024 pixels, it's too big.
        # We need to process in tiles or use a more efficient approach.
        # For the prototype, we'll use a smaller number of Gaussians or lower res.
        
        # Placeholder: Return a sum of 2D Gaussians
        # In a real impl, we'd use tile-based rasterization
        return torch.zeros((h, w), device=xyz.device, requires_grad=True)

    def forward(self, x, y, z, t):
        return self.pinn(x, y, z, t)
