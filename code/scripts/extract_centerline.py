"""Extract a coarse 3D centerline candidate set from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.config import ModelConfig
from vascular_reconstruction.models.pinn_gs import PINN_GS


def _load_model(checkpoint_path: Path) -> tuple[dict[str, object], PINN_GS]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config={"hidden_dim": model_config.pinn_hidden_dim, "num_layers": model_config.pinn_num_layers},
        sh_degree=model_config.sh_degree,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _voxelize(
    xyz: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    grid_size: int = 96,
    sigma_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = sigma_scale * np.max(scales, axis=1)
    mins = np.min(xyz - radii[:, None], axis=0)
    maxs = np.max(xyz + radii[:, None], axis=0)

    axes = [np.linspace(mins[i], maxs[i], grid_size, dtype=np.float32) for i in range(3)]
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    for point, scale, opacity in zip(xyz, scales, opacities, strict=False):
        radius = sigma_scale * float(np.max(scale))
        lower = [max(int(np.searchsorted(axes[d], point[d] - radius) - 1), 0) for d in range(3)]
        upper = [min(int(np.searchsorted(axes[d], point[d] + radius) + 1), grid_size - 1) for d in range(3)]

        xs = axes[0][lower[0] : upper[0] + 1]
        ys = axes[1][lower[1] : upper[1] + 1]
        zs = axes[2][lower[2] : upper[2] + 1]
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            continue

        dx = ((xs - point[0]) / max(float(scale[0]), 1e-3)) ** 2
        dy = ((ys - point[1]) / max(float(scale[1]), 1e-3)) ** 2
        dz = ((zs - point[2]) / max(float(scale[2]), 1e-3)) ** 2
        local = np.exp(-(dx[:, None, None] + dy[None, :, None] + dz[None, None, :]) / 2.0)
        grid[
            lower[0] : upper[0] + 1,
            lower[1] : upper[1] + 1,
            lower[2] : upper[2] + 1,
        ] += opacity * local.astype(np.float32)

    return grid, mins, maxs


def _extract_centerline_points(
    density: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    density_quantile: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(np.quantile(density[density > 0], density_quantile)) if np.any(density > 0) else 0.0
    occupancy = density >= threshold
    distance = distance_transform_edt(occupancy)
    ridge_mask = occupancy & (distance == maximum_filter(distance, size=3)) & (distance >= 1.0)

    coords = np.argwhere(ridge_mask)
    if len(coords) == 0:
        return occupancy, np.empty((0, 3), dtype=np.float32)

    spacing = (maxs - mins) / np.maximum(np.array(density.shape) - 1, 1)
    points = mins + coords.astype(np.float32) * spacing
    return occupancy, points


def _save_debug_image(output_path: Path, occupancy: np.ndarray, centerline_points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    mip_xy = occupancy.max(axis=2)
    mip_xz = occupancy.max(axis=1)
    mip_yz = occupancy.max(axis=0)

    panels = [
        (mip_xy, 0, 1, "XY Occupancy"),
        (mip_xz, 0, 2, "XZ Occupancy"),
        (mip_yz, 1, 2, "YZ Occupancy"),
    ]
    for ax, (panel, a_idx, b_idx, title) in zip(axes, panels, strict=False):
        ax.imshow(
            panel.T,
            origin="lower",
            cmap="gray",
            extent=(mins[a_idx], maxs[a_idx], mins[b_idx], maxs[b_idx]),
            aspect="auto",
        )
        if len(centerline_points):
            ax.scatter(
                centerline_points[:, a_idx],
                centerline_points[:, b_idx],
                s=2,
                c="#ff3b30",
                alpha=0.6,
                linewidths=0,
            )
        ax.set_title(title)
        ax.set_xlim(mins[a_idx], maxs[a_idx])
        ax.set_ylim(mins[b_idx], maxs[b_idx])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract centerline candidates from a reconstruction checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("centerline_extraction"), help="Output directory.")
    parser.add_argument("--grid-size", type=int, default=96, help="Voxel grid size per axis.")
    parser.add_argument("--density-quantile", type=float, default=0.9, help="Density quantile for occupancy threshold.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model = _load_model(args.checkpoint)
    xyz = model.gs.get_xyz.detach().cpu().numpy()
    scales = model.gs.get_scaling.detach().cpu().numpy()
    opacities = model.gs.get_opacity.detach().cpu().numpy().squeeze(-1)

    density, mins, maxs = _voxelize(xyz, scales, opacities, grid_size=args.grid_size)
    occupancy, centerline_points = _extract_centerline_points(
        density,
        mins,
        maxs,
        density_quantile=args.density_quantile,
    )

    np.savez_compressed(
        output_dir / f"centerline_iter_{checkpoint['iteration']}.npz",
        density=density,
        occupancy=occupancy.astype(np.uint8),
        centerline_points=centerline_points,
        bounds_min=mins,
        bounds_max=maxs,
    )
    _save_debug_image(
        output_dir / f"centerline_iter_{checkpoint['iteration']}.png",
        occupancy,
        centerline_points,
        mins,
        maxs,
    )
    npz_path = output_dir / f"centerline_iter_{checkpoint['iteration']}.npz"
    print(f"Saved {len(centerline_points):,} centerline candidate points to {npz_path}")


if __name__ == "__main__":
    main()
