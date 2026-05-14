"""Visualization tools for vascular reconstruction checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.config import ModelConfig
from vascular_reconstruction.evaluation.reconstruction_metrics import (
    ReconstructionGeometry,
    active_gaussian_count_from_schedule,
    build_graph_diagnostics,
    evaluate_reconstruction,
    select_active_geometry,
    sample_points,
)
from vascular_reconstruction.models.pinn_gs import PINN_GS

PROJECTIONS = (
    ("XY Density", 0, 1, "X", "Y"),
    ("XZ Density", 0, 2, "X", "Z"),
    ("YZ Density", 1, 2, "Y", "Z"),
)


def _default_output_dir(checkpoint_path: Path) -> Path:
    run_name = checkpoint_path.parent.name or checkpoint_path.stem
    return ROOT / "outputs" / "visualization" / run_name


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


def _robust_limits(points: np.ndarray, a_idx: int, b_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
    a_low, a_high = np.percentile(points[:, a_idx], [1, 99])
    b_low, b_high = np.percentile(points[:, b_idx], [1, 99])
    a_margin = max((a_high - a_low) * 0.1, 1.0)
    b_margin = max((b_high - b_low) * 0.1, 1.0)
    return (a_low - a_margin, a_high + a_margin), (b_low - b_margin, b_high + b_margin)


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
        mesh_sample = sample_points(mesh_vertices, max_points=10000, seed=3)
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


def _plot_summary(ax, checkpoint: dict[str, object], metrics: dict[str, object]) -> None:
    lines = [
        f"Checkpoint {checkpoint['iteration']}",
        "",
        f"Points: {int(metrics['point_count']):,}",
        f"Center: [{metrics['center'][0]:.2f}, {metrics['center'][1]:.2f}, {metrics['center'][2]:.2f}]",
        f"Std:    [{metrics['std'][0]:.2f}, {metrics['std'][1]:.2f}, {metrics['std'][2]:.2f}]",
        "",
        f"Graph components: {int(metrics['component_count'])}",
        f"Largest component frac: {metrics['largest_component_fraction']:.3f}",
        f"kNN mean dist: {metrics['neighbor_distance_mean']:.3f}",
        f"kNN p95 dist: {metrics['neighbor_distance_p95']:.3f}",
        f"MST mean edge: {metrics['mst_mean']:.3f}",
        f"MST p95 edge: {metrics['mst_p95']:.3f}",
        f"Mean line score: {metrics['line_score_mean']:.3f}",
        "",
        f"Voxel components: {int(metrics['voxel_component_count'])}",
        f"Voxel largest frac: {metrics['voxel_largest_component_fraction']:.3f}",
        f"Occupancy fill: {metrics['occupancy_fill_ratio']:.5f}",
        f"Surface ratio: {metrics['occupancy_surface_ratio']:.3f}",
        f"Compactness: {metrics['occupancy_compactness']:.4f}",
        f"Mesh Chamfer p95: {metrics['mesh_vertex_chamfer_p95']:.3f}",
        "",
        "Interpretation:",
        "Pass only if graph, voxel, and mesh",
        "metrics all support one thin connected tree.",
    ]
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)


def _save_metrics(
    output_path: Path,
    metrics: dict[str, object],
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def visualize_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    mesh_path: Path | None = None,
    save_figure: bool = True,
) -> dict[str, float | int | str | list[float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, model = _load_model_from_checkpoint(checkpoint_path)
    geometry = ReconstructionGeometry(
        xyz=model.gs.get_xyz.detach().cpu().numpy(),
        scales=model.gs.get_scaling.detach().cpu().numpy(),
        opacities=model.gs.get_opacity.detach().cpu().numpy().squeeze(-1),
    )
    training_config = checkpoint.get("training_config") or {}
    active_count = active_gaussian_count_from_schedule(
        total_count=len(geometry.xyz),
        active_gaussian_schedule=training_config.get("active_gaussian_schedule"),
        iteration=int(checkpoint["iteration"]),
    )
    active_geometry = select_active_geometry(geometry, active_count=active_count)
    xyz = active_geometry.xyz
    scales = active_geometry.scales
    opacities = active_geometry.opacities
    mesh_vertices = _prepare_mesh_vertices(mesh_path)
    metrics = evaluate_reconstruction(active_geometry, gt_mesh_path=mesh_path)
    diagnostics = build_graph_diagnostics(xyz)
    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "iteration": int(checkpoint["iteration"]),
        "active_gaussians": int(len(xyz)),
        **metrics,
    }

    metrics_path = output_dir / f"reconstruction_comparison_iter_{checkpoint['iteration']}.json"
    _save_metrics(metrics_path, metrics)
    print(f"Metrics saved to {metrics_path}")

    if not save_figure:
        return metrics

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
    _plot_summary(axes[2, 1], checkpoint, metrics)
    axes[2, 2].axis("off")

    output_path = output_dir / f"reconstruction_comparison_iter_{checkpoint['iteration']}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstruction checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--mesh", type=Path, help="Path to ground truth STL mesh.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to code/outputs/visualization/<checkpoint-dir>/.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir(args.checkpoint)
    visualize_checkpoint(args.checkpoint, output_dir, args.mesh)


if __name__ == "__main__":
    main()
