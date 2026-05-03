"""Differentiable low-resolution silhouette rendering for projected Gaussians."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vascular_reconstruction.models.pinn_gs import PINN_GS


def _scaled_projection_matrix(projection_matrix: torch.Tensor, source_size: tuple[int, int], target_size: int) -> torch.Tensor:
    height, width = source_size
    scale_x = target_size / max(width, 1)
    scale_y = target_size / max(height, 1)
    scaled = projection_matrix.clone()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def render_gaussian_silhouette(
    model: PINN_GS,
    view_matrix: torch.Tensor,
    projection_matrix: torch.Tensor,
    source_image_size: tuple[int, int],
    render_size: int,
    chunk_size: int = 512,
    min_sigma: float = 0.12,
    max_sigma: float = 4.0,
) -> torch.Tensor:
    """Render a soft silhouette by splatting projected Gaussians onto a low-resolution image."""
    scaled_projection = _scaled_projection_matrix(projection_matrix, source_image_size, render_size)
    projected_coords, x_dist = model.project_gaussians(view_matrix, scaled_projection)

    scaling = model.gs.get_scaling
    world_radius = torch.mean(scaling[:, 1:], dim=-1)
    focal = 0.5 * (scaled_projection[0, 0] + scaled_projection[1, 1])
    raw_sigma = (focal * world_radius) / x_dist
    # Use a leaky clamp so sigma keeps gradients near the bounds instead of freezing.
    screen_sigma = torch.where(
        raw_sigma < min_sigma,
        min_sigma + 0.25 * (raw_sigma - min_sigma),
        raw_sigma,
    )
    screen_sigma = torch.where(
        screen_sigma > max_sigma,
        max_sigma + 0.25 * (screen_sigma - max_sigma),
        screen_sigma,
    ).clamp_min(1e-3)
    weights = model.gs.get_opacity.squeeze(-1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(render_size, device=projected_coords.device, dtype=projected_coords.dtype),
        torch.arange(render_size, device=projected_coords.device, dtype=projected_coords.dtype),
        indexing="ij",
    )

    accumulation = torch.zeros((render_size, render_size), device=projected_coords.device, dtype=projected_coords.dtype)
    for start in range(0, len(projected_coords), chunk_size):
        end = min(start + chunk_size, len(projected_coords))
        coords_chunk = projected_coords[start:end]
        sigma_chunk = screen_sigma[start:end].view(-1, 1, 1)
        weight_chunk = weights[start:end].view(-1, 1, 1)

        dx = grid_x.unsqueeze(0) - coords_chunk[:, 0].view(-1, 1, 1)
        dy = grid_y.unsqueeze(0) - coords_chunk[:, 1].view(-1, 1, 1)
        exponent = -(dx.square() + dy.square()) / (2.0 * sigma_chunk.square())
        contribution = torch.exp(exponent) * weight_chunk
        accumulation = accumulation + contribution.sum(dim=0)

    silhouette = 1.0 - torch.exp(-accumulation)
    return torch.clamp(silhouette, 0.0, 1.0)


def downsample_mask(mask: torch.Tensor, render_size: int) -> torch.Tensor:
    """Downsample a binary vessel mask to the renderer size."""
    if mask.ndim != 2:
        raise ValueError("Expected mask with shape [H, W].")
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    height, width = mask.shape
    if height % render_size == 0 and width % render_size == 0:
        kernel_h = height // render_size
        kernel_w = width // render_size
        resized = F.max_pool2d(mask_4d, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    else:
        resized = F.adaptive_max_pool2d(mask_4d, output_size=(render_size, render_size))
    return resized.squeeze(0).squeeze(0).clamp(0.0, 1.0)
