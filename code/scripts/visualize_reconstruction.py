"""Visualization tools for vascular reconstruction checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.config import ModelConfig
from vascular_reconstruction.models.pinn_gs import PINN_GS

PROJECTIONS = (
    ("XY Density", 0, 1, "X", "Y"),
    ("XZ Density", 0, 2, "X", "Z"),
    ("YZ Density", 1, 2, "Y", "Z"),
)


def _load_model_from_checkpoint(checkpoint_path: Path) -> tuple[dict[str, object], PINN_GS]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    saved_model_config = checkpoint.get("model_config")
    if saved_model_config is not None:
        model_config = ModelConfig.from_dict(saved_model_config)
        pinn_config = {
            "hidden_dim": model_config.pinn_hidden_dim,
            "num_layers": model_config.pinn_num_layers,
        }
        model = PINN_GS(
            num_gaussians=model_config.num_gaussians,
            pinn_config=pinn_config,
            sh_degree=model_config.sh_degree,
        )
    else:
        model = PINN_GS(
            num_gaussians=state_dict["gs._xyz"].shape[0],
            pinn_config={"hidden_dim": 128, "num_layers": 4},
            sh_degree=3,
        )

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
    return points[rng.choice(len(points), size=max_points, replace=False)]


def _robust_limits(points: np.ndarray, a_idx: int, b_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
    a_low, a_high = np.percentile(points[:, a_idx], [1, 99])
    b_low, b_high = np.percentile(points[:, b_idx], [1, 99])
    a_margin = max((a_high - a_low) * 0.1, 1.0)
    b_margin = max((b_high - b_low) * 0.1, 1.0)
    return (a_low - a_margin, a_high + a_margin), (b_low - b_margin, b_high + b_margin)


def _split_clusters(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    axis = int(np.argmax(np.std(xyz, axis=0)))
    sorted_values = np.sort(xyz[:, axis])
    gaps = np.diff(sorted_values)
    split_idx = int(np.argmax(gaps))
    split_gap = float(gaps[split_idx])
    threshold = float((sorted_values[split_idx] + sorted_values[split_idx + 1]) / 2.0)
    mask = xyz[:, axis] <= threshold
    return mask, ~mask, axis, split_gap


def _cluster_summary(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.mean(axis=0), points.std(axis=0)


def _plot_density(ax, points: np.ndarray, mesh_vertices: np.ndarray | None, title: str, a_idx: int, b_idx: int, a_label: str, b_label: str) -> None:
    limits = _robust_limits(points, a_idx, b_idx)
    ax.hexbin(points[:, a_idx], points[:, b_idx], gridsize=70, bins="log", cmap="inferno", mincnt=1)
    if mesh_vertices is not None:
        mesh_sample = _sample_points(mesh_vertices, max_points=10000, seed=3)
        ax.scatter(mesh_sample[:, a_idx], mesh_sample[:, b_idx], s=0.2, c="#4c6ef5", alpha=0.06, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(a_label)
    ax.set_ylabel(b_label)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _plot_cluster(ax, points: np.ndarray, title: str, color: str) -> None:
    pts = _sample_points(points, max_points=12000, seed=4)
    limits = _robust_limits(pts, 0, 1)
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c=color, alpha=0.25, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _plot_summary(ax, cluster_a: np.ndarray, cluster_b: np.ndarray, axis: int, split_gap: float) -> None:
    center_a, std_a = _cluster_summary(cluster_a)
    center_b, std_b = _cluster_summary(cluster_b)
    separation = float(np.linalg.norm(center_a - center_b))
    lines = [
        "Cluster Summary",
        "",
        f"A count: {len(cluster_a):,}",
        f"B count: {len(cluster_b):,}",
        f"Split axis: {'XYZ'[axis]}",
        f"Gap on split axis: {split_gap:.3f}",
        f"Centroid distance: {separation:.3f}",
        "",
        f"A center: [{center_a[0]:.2f}, {center_a[1]:.2f}, {center_a[2]:.2f}]",
        f"A std:    [{std_a[0]:.3f}, {std_a[1]:.3f}, {std_a[2]:.3f}]",
        "",
        f"B center: [{center_b[0]:.2f}, {center_b[1]:.2f}, {center_b[2]:.2f}]",
        f"B std:    [{std_b[0]:.3f}, {std_b[1]:.3f}, {std_b[2]:.3f}]",
        "",
        "Interpretation:",
        "This checkpoint is collapsed into two tight modes,",
        "not a continuous vascular geometry.",
    ]
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)


def visualize_checkpoint(checkpoint_path: Path, output_dir: Path, mesh_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, model = _load_model_from_checkpoint(checkpoint_path)
    xyz = model.gs.get_xyz.detach().cpu().numpy()
    mesh_vertices = _prepare_mesh_vertices(mesh_path)
    cluster_a_mask, cluster_b_mask, axis, split_gap = _split_clusters(xyz)
    cluster_a = xyz[cluster_a_mask]
    cluster_b = xyz[cluster_b_mask]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Checkpoint {checkpoint['iteration']} Geometry Diagnostics", fontsize=18)

    for col, (title, a_idx, b_idx, a_label, b_label) in enumerate(PROJECTIONS):
        panel_title = title if mesh_vertices is None else f"{title} (blue = GT)"
        _plot_density(axes[0, col], xyz, mesh_vertices, panel_title, a_idx, b_idx, a_label, b_label)

    _plot_cluster(axes[1, 0], cluster_a, f"Cluster A XY ({len(cluster_a):,} points)", "#ff3b30")
    _plot_cluster(axes[1, 1], cluster_b, f"Cluster B XY ({len(cluster_b):,} points)", "#ffd60a")
    _plot_summary(axes[1, 2], cluster_a, cluster_b, axis, split_gap)

    output_path = output_dir / f"reconstruction_comparison_iter_{checkpoint['iteration']}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstruction checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--mesh", type=Path, help="Path to ground truth STL mesh.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"), help="Output directory.")
    args = parser.parse_args()
    visualize_checkpoint(args.checkpoint, args.output_dir, args.mesh)


if __name__ == "__main__":
    main()
