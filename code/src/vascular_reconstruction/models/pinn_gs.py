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
        # Initialize in a volume matching the vessels (roughly -50 to 50)
        nn.init.uniform_(self._xyz, -60, 60)
        nn.init.zeros_(self._features_dc)
        nn.init.zeros_(self._features_rest)
        nn.init.constant_(self._scaling, 0.5) # ~1.6mm scale initially
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

    def get_view_matrix(self, lao: float, cran: float, device: str = "cpu") -> torch.Tensor:
        """Compute camera view matrix from angles, matching gVXR convention."""
        lao_rad = np.radians(lao)
        cran_rad = np.radians(cran)
        
        # Ry: Rotation about Y
        ry = torch.tensor([
            [np.cos(lao_rad), 0, np.sin(lao_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(lao_rad), 0, np.cos(lao_rad), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Rz: Rotation about Z
        rz = torch.tensor([
            [np.cos(cran_rad), -np.sin(cran_rad), 0, 0],
            [np.sin(cran_rad), np.cos(cran_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        return rz @ ry

    def project_points(self, view_matrix: torch.Tensor, img_size: int = 1024) -> torch.Tensor:
        """
        Project Gaussian centers to 2D image coordinates.
        Matches gVXR Source(-600) -> Center(0) -> Detector(400) geometry.
        """
        xyz = self.gs.get_xyz
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
        
        # Transform centers: P' = R @ P
        # view_matrix is [4, 4]
        xyz_cam = (xyz_hom @ view_matrix.T)[:, :3]
        
        # Perspective projection along X-axis
        # x_dist = x_cam + 600
        # u = 1000 * y_cam / x_dist
        # v = 1000 * (-z_cam) / x_dist
        x_dist = xyz_cam[:, 0] + 600.0
        
        # Avoid division by zero
        x_dist = torch.clamp(x_dist, min=1.0)
        
        u = (1000.0 * xyz_cam[:, 1]) / x_dist
        v = (1000.0 * (-xyz_cam[:, 2])) / x_dist
        
        # Scale to pixels: detector is 300mm wide
        # pixel_coords = (proj / 150) * 512 + 512
        u_pix = (u / 150.0) * (img_size / 2.0) + (img_size / 2.0)
        v_pix = (v / 150.0) * (img_size / 2.0) + (img_size / 2.0)
        
        return torch.stack([u_pix, v_pix], dim=-1)

    def forward(self, x, y, z, t):
        return self.pinn(x, y, z, t)
