"""Visualization tools for vascular reconstruction with Ground Truth comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.config import ModelConfig
from vascular_reconstruction.models.pinn_gs import PINN_GS

PROJECTION_AXES = (
    ("XY Projection", 0, 1, "X", "Y"),
    ("XZ Projection", 0, 2, "X", "Z"),
    ("YZ Projection", 1, 2, "Y", "Z"),
)


def _load_model_from_checkpoint(checkpoint_path: Path) -> tuple[dict[str, object], PINN_GS]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    saved_model_config = checkpoint.get("model_config")
    if saved_model_config is not None:
        model_config = ModelConfig.from_dict(saved_model_config)
        num_gaussians = model_config.num_gaussians
        pinn_config = {
            "hidden_dim": model_config.pinn_hidden_dim,
            "num_layers": model_config.pinn_num_layers,
        }
        sh_degree = model_config.sh_degree
    else:
        num_gaussians = state_dict["gs._xyz"].shape[0]
        pinn_config = {"hidden_dim": 128, "num_layers": 4}
        sh_degree = 3

    model = PINN_GS(num_gaussians=num_gaussians, pinn_config=pinn_config, sh_degree=sh_degree)
    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint, model


def _prepare_mesh_vertices(mesh_path: Path | None) -> np.ndarray | None:
    if mesh_path is None or not mesh_path.exists():
        return None

    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    if np.max(mesh.bounding_box.extents) < 50.0:
        mesh.apply_scale(10.0)
    return np.asarray(mesh.vertices)


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def _robust_limits(points: np.ndarray, a_idx: int, b_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
    a = points[:, a_idx]
    b = points[:, b_idx]
    a_low, a_high = np.percentile(a, [1, 99])
    b_low, b_high = np.percentile(b, [1, 99])
    a_margin = max((a_high - a_low) * 0.15, 1.0)
    b_margin = max((b_high - b_low) * 0.15, 1.0)
    return (a_low - a_margin, a_high + a_margin), (b_low - b_margin, b_high + b_margin)


def _flow_field_samples(xyz: np.ndarray) -> np.ndarray:
    lower = np.percentile(xyz, 2, axis=0) - 2.0
    upper = np.percentile(xyz, 98, axis=0) + 2.0
    grid_axes = [np.linspace(lower[i], upper[i], 12) for i in range(3)]
    xx, yy, zz = np.meshgrid(*grid_axes, indexing="ij")
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sample_xyz = _sample_points(xyz, max_points=min(6000, len(xyz)), seed=1)
    distances = np.sqrt(((coords[:, None, :] - sample_xyz[None, :, :]) ** 2).sum(axis=-1))
    nearest = distances.min(axis=1)
    return coords[nearest <= 4.0]


def _compute_flow(model: PINN_GS, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = _flow_field_samples(xyz)
    if len(coords) == 0:
        return coords, np.empty((0, 3), dtype=np.float32)

    coords_torch = torch.from_numpy(coords).float()
    t = torch.zeros((len(coords), 1), dtype=torch.float32)
    with torch.no_grad():
        flow = model.pinn(coords_torch[:, 0:1], coords_torch[:, 1:2], coords_torch[:, 2:3], t)[:, :3].numpy()

    speed = np.linalg.norm(flow, axis=1)
    keep = speed >= np.percentile(speed, 40)
    return coords[keep], flow[keep]


def _plot_geometry_projection(ax, xyz: np.ndarray, mesh_vertices: np.ndarray | None, title: str, a_idx: int, b_idx: int, a_label: str, b_label: str) -> None:
    sampled_xyz = _sample_points(xyz, max_points=20000, seed=0)
    axis_limits = _robust_limits(sampled_xyz, a_idx, b_idx)

    ax.scatter(
        sampled_xyz[:, a_idx],
        sampled_xyz[:, b_idx],
        c=sampled_xyz[:, 2],
        cmap="autumn",
        s=5,
        alpha=0.35,
        linewidths=0,
    )
    if mesh_vertices is not None:
        sampled_mesh = _sample_points(mesh_vertices, max_points=12000, seed=2)
        ax.scatter(
            sampled_mesh[:, a_idx],
            sampled_mesh[:, b_idx],
            c="#4c6ef5",
            s=1,
            alpha=0.08,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel(a_label)
    ax.set_ylabel(b_label)
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _plot_flow_projection(ax, coords: np.ndarray, flow: np.ndarray, title: str, a_idx: int, b_idx: int, a_label: str, b_label: str) -> None:
    if len(coords) == 0:
        ax.set_title(f"{title}\n(no valid samples)")
        ax.set_xlabel(a_label)
        ax.set_ylabel(b_label)
        return

    speed = np.linalg.norm(flow, axis=1)
    norm = np.linalg.norm(flow[:, [a_idx, b_idx]], axis=1, keepdims=True)
    direction = flow[:, [a_idx, b_idx]] / np.clip(norm, 1e-6, None)
    display_flow = direction * 1.5
    colors = plt.cm.viridis((speed - speed.min()) / max(float(np.ptp(speed)), 1e-6))

    ax.quiver(
        coords[:, a_idx],
        coords[:, b_idx],
        display_flow[:, 0],
        display_flow[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=colors,
        width=0.003,
    )

    axis_limits = _robust_limits(coords, a_idx, b_idx)
    ax.set_title(title)
    ax.set_xlabel(a_label)
    ax.set_ylabel(b_label)
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def visualize_checkpoint(checkpoint_path: Path, output_dir: Path, mesh_path: Path | None = None) -> None:
    """Load a checkpoint and visualize geometry and flow in orthographic projections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, model = _load_model_from_checkpoint(checkpoint_path)
    xyz = model.gs.get_xyz.detach().numpy()
    mesh_vertices = _prepare_mesh_vertices(mesh_path)
    flow_coords, flow_vectors = _compute_flow(model, xyz)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(f"Checkpoint {checkpoint['iteration']} Projections", fontsize=18)

    for column, (title, a_idx, b_idx, a_label, b_label) in enumerate(PROJECTION_AXES):
        geometry_title = title
        if mesh_vertices is not None and column == 0:
            geometry_title = f"{title} (blue = GT)"
        _plot_geometry_projection(
            axes[0, column],
            xyz,
            mesh_vertices,
            geometry_title,
            a_idx,
            b_idx,
            a_label,
            b_label,
        )
        _plot_flow_projection(
            axes[1, column],
            flow_coords,
            flow_vectors,
            f"{title} Flow",
            a_idx,
            b_idx,
            a_label,
            b_label,
        )

    output_path = output_dir / f"reconstruction_comparison_iter_{checkpoint['iteration']}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstruction checkpoints with GT mesh.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--mesh", type=Path, help="Path to ground truth STL mesh.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"), help="Output directory.")
    args = parser.parse_args()

    visualize_checkpoint(args.checkpoint, args.output_dir, args.mesh)


if __name__ == "__main__":
    main()
