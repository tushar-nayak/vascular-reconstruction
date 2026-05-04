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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

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

    try:
        import trimesh
    except ModuleNotFoundError:
        return None

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


def _build_graph_diagnostics(points: np.ndarray, knn: int = 6) -> dict[str, object]:
    sample = _sample_points(points, max_points=3500, seed=11)
    tree = cKDTree(sample)
    distances, indices = tree.query(sample, k=min(knn + 1, len(sample)))

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    rows = np.repeat(np.arange(len(sample)), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1)
    data = neighbor_distances.reshape(-1)
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(sample), len(sample)))
    adjacency = adjacency.minimum(adjacency.T)

    component_count, labels = connected_components(adjacency, directed=False)
    mst = minimum_spanning_tree(adjacency)
    mst_lengths = mst.data

    offsets = sample[neighbor_indices] - sample[:, None, :]
    covariance = np.einsum("nki,nkj->nij", offsets, offsets) / max(offsets.shape[1], 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    line_scores = 1.0 - ((eigenvalues[:, 0] + eigenvalues[:, 1]) / (eigenvalues.sum(axis=1) + 1e-6))

    component_sizes = np.bincount(labels, minlength=component_count)
    largest_component = int(component_sizes.max()) if len(component_sizes) else 0

    return {
        "sample": sample,
        "adjacency": adjacency,
        "labels": labels,
        "component_count": int(component_count),
        "largest_component_fraction": float(largest_component / max(len(sample), 1)),
        "neighbor_distance_mean": float(neighbor_distances.mean()),
        "neighbor_distance_p95": float(np.percentile(neighbor_distances, 95)),
        "mst_mean": float(mst_lengths.mean()) if len(mst_lengths) else 0.0,
        "mst_p95": float(np.percentile(mst_lengths, 95)) if len(mst_lengths) else 0.0,
        "line_score_mean": float(line_scores.mean()),
        "line_scores": line_scores,
    }


def _plot_density(
    ax,
    points: np.ndarray,
    mesh_vertices: np.ndarray | None,
    title: str,
    a_idx: int,
    b_idx: int,
    a_label: str,
    b_label: str,
) -> None:
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


def _plot_graph_projection(
    ax,
    sample: np.ndarray,
    adjacency: csr_matrix,
    labels: np.ndarray,
    title: str,
    a_idx: int,
    b_idx: int,
    a_label: str,
    b_label: str,
) -> None:
    limits = _robust_limits(sample, a_idx, b_idx)
    graph = adjacency.tocoo()
    if graph.nnz:
        edge_mask = graph.row < graph.col
        for start, end in zip(graph.row[edge_mask], graph.col[edge_mask], strict=False):
            ax.plot(
                [sample[start, a_idx], sample[end, a_idx]],
                [sample[start, b_idx], sample[end, b_idx]],
                color="#adb5bd",
                alpha=0.06,
                linewidth=0.4,
            )
    scatter = ax.scatter(
        sample[:, a_idx],
        sample[:, b_idx],
        c=labels,
        cmap="tab20",
        s=4,
        alpha=0.7,
        linewidths=0,
    )
    scatter.set_clim(0, max(int(labels.max()), 1))
    ax.set_title(title)
    ax.set_xlabel(a_label)
    ax.set_ylabel(b_label)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _plot_line_score_hist(ax, line_scores: np.ndarray) -> None:
    ax.hist(line_scores, bins=30, color="#ff922b", alpha=0.9)
    ax.set_title("Local Line-Likeness")
    ax.set_xlabel("1 = line-like, 0 = blob-like")
    ax.set_ylabel("Count")
    ax.grid(False)


def _plot_summary(ax, checkpoint: dict[str, object], xyz: np.ndarray, diagnostics: dict[str, object]) -> None:
    center = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    lines = [
        f"Checkpoint {checkpoint['iteration']}",
        "",
        f"Points: {len(xyz):,}",
        f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]",
        f"Std:    [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]",
        "",
        f"Graph components: {diagnostics['component_count']}",
        f"Largest component frac: {diagnostics['largest_component_fraction']:.3f}",
        f"kNN mean dist: {diagnostics['neighbor_distance_mean']:.3f}",
        f"kNN p95 dist: {diagnostics['neighbor_distance_p95']:.3f}",
        f"MST mean edge: {diagnostics['mst_mean']:.3f}",
        f"MST p95 edge: {diagnostics['mst_p95']:.3f}",
        f"Mean line score: {diagnostics['line_score_mean']:.3f}",
        "",
        "Interpretation:",
        "Higher line score and one dominant component",
        "are better proxies for vessel-like 3D structure.",
    ]
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)


def visualize_checkpoint(checkpoint_path: Path, output_dir: Path, mesh_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, model = _load_model_from_checkpoint(checkpoint_path)
    xyz = model.gs.get_xyz.detach().cpu().numpy()
    mesh_vertices = _prepare_mesh_vertices(mesh_path)
    diagnostics = _build_graph_diagnostics(xyz)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Checkpoint {checkpoint['iteration']} Geometry Diagnostics", fontsize=18)

    for col, (title, a_idx, b_idx, a_label, b_label) in enumerate(PROJECTIONS):
        panel_title = title if mesh_vertices is None else f"{title} (blue = GT)"
        _plot_density(axes[0, col], xyz, mesh_vertices, panel_title, a_idx, b_idx, a_label, b_label)
        _plot_graph_projection(
            axes[1, col],
            diagnostics["sample"],
            diagnostics["adjacency"],
            diagnostics["labels"],
            f"{a_label}{b_label} kNN Graph",
            a_idx,
            b_idx,
            a_label,
            b_label,
        )

    _plot_line_score_hist(axes[2, 0], diagnostics["line_scores"])
    _plot_summary(axes[2, 1], checkpoint, xyz, diagnostics)
    axes[2, 2].axis("off")

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
