"""Shared reconstruction evaluation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from skimage import measure


@dataclass(frozen=True)
class ReconstructionGeometry:
    xyz: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray


def active_gaussian_count_from_schedule(
    total_count: int,
    active_gaussian_schedule: list[list[int]] | None,
    iteration: int,
) -> int:
    if not active_gaussian_schedule:
        return total_count

    active_count = total_count
    for start_iteration, count in active_gaussian_schedule:
        if iteration >= int(start_iteration):
            active_count = int(count)
    return min(max(active_count, 1), total_count)


def select_active_geometry(
    geometry: ReconstructionGeometry,
    active_count: int,
) -> ReconstructionGeometry:
    if len(geometry.xyz) <= active_count:
        return geometry

    active_count = min(max(int(active_count), 1), len(geometry.xyz))
    active_indices = np.argsort(geometry.opacities)[-active_count:]
    return ReconstructionGeometry(
        xyz=geometry.xyz[active_indices],
        scales=geometry.scales[active_indices],
        opacities=geometry.opacities[active_indices],
    )


def gate_and_score_from_metrics(
    metrics: Mapping[str, object],
    training_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = training_config or {}
    gate_pass = (
        float(metrics["largest_component_fraction"]) >= float(config.get("gate_min_graph_largest_component_fraction", 0.0))
        and float(metrics["component_count"]) <= float(config.get("gate_max_graph_component_count", 10_000))
        and float(metrics["voxel_largest_component_fraction"]) >= float(config.get("gate_min_voxel_largest_component_fraction", 0.0))
        and float(metrics["voxel_component_count"]) <= float(config.get("gate_max_voxel_component_count", 10_000))
        and float(metrics["occupancy_fill_ratio"]) <= float(config.get("gate_max_occupancy_fill_ratio", 1.0))
        and (
            float(config.get("gate_max_mesh_vertex_chamfer_p95", -1.0)) < 0.0
            or float(metrics["mesh_vertex_chamfer_p95"]) <= float(config.get("gate_max_mesh_vertex_chamfer_p95", -1.0))
        )
    )
    score = (
        -float(metrics["voxel_largest_component_fraction"]),
        float(metrics["voxel_component_count"]),
        float(metrics["mesh_vertex_chamfer_p95"]) if float(metrics["mesh_vertex_chamfer_p95"]) >= 0.0 else float("inf"),
        float(metrics["occupancy_fill_ratio"]),
        float(metrics["mst_p95"]),
    )
    return {"gate_pass": bool(gate_pass), "score": list(score)}


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def voxelize_geometry(
    geometry: ReconstructionGeometry,
    grid_size: int = 96,
    sigma_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = geometry.xyz
    scales = geometry.scales
    opacities = geometry.opacities
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


def occupancy_from_density(density: np.ndarray, density_quantile: float = 0.9) -> np.ndarray:
    if not np.any(density > 0):
        return np.zeros_like(density, dtype=bool)
    threshold = float(np.quantile(density[density > 0], density_quantile))
    return density >= threshold


def load_stl_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
        if len(count_bytes) < 4:
            raise ValueError(f"Invalid STL file: {path}")
        triangle_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 84 + triangle_count * 50
        file_size = path.stat().st_size
        if file_size == expected_size:
            vertices = np.empty((triangle_count * 3, 3), dtype=np.float32)
            for index in range(triangle_count):
                record = f.read(50)
                if len(record) < 50:
                    raise ValueError(f"Unexpected EOF in binary STL: {path}")
                vals = struct.unpack("<12fH", record)
                vertices[index * 3 : (index + 1) * 3] = np.array(vals[3:12], dtype=np.float32).reshape(3, 3)
            return vertices

    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            _, x, y, z = stripped.split()
            vertices.append([float(x), float(y), float(z)])
    if not vertices:
        raise ValueError(f"Could not parse STL vertices from {path}")
    return np.asarray(vertices, dtype=np.float32)


def build_graph_diagnostics(points: np.ndarray, knn: int = 6, max_points: int = 3500) -> dict[str, object]:
    sample = sample_points(points, max_points=max_points, seed=11)
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


def compute_voxel_metrics(
    density: np.ndarray,
    occupancy: np.ndarray,
) -> dict[str, float]:
    occupancy_count = int(occupancy.sum())
    if occupancy_count == 0:
        return {
            "voxel_component_count": 0.0,
            "voxel_largest_component_fraction": 0.0,
            "occupancy_fill_ratio": 0.0,
            "occupancy_surface_ratio": 0.0,
            "occupancy_compactness": 0.0,
        }
    labels, component_count = label(occupancy)
    component_sizes = np.bincount(labels.reshape(-1))[1:]
    largest_component = int(component_sizes.max()) if len(component_sizes) else 0
    eroded = binary_erosion(occupancy)
    surface_count = int((occupancy & ~eroded).sum())
    fill_ratio = occupancy_count / occupancy.size
    surface_ratio = surface_count / occupancy_count
    compactness = largest_component / max(surface_count, 1)
    return {
        "voxel_component_count": float(component_count),
        "voxel_largest_component_fraction": float(largest_component / occupancy_count),
        "occupancy_fill_ratio": float(fill_ratio),
        "occupancy_surface_ratio": float(surface_ratio),
        "occupancy_compactness": float(compactness),
    }


def compute_mesh_metrics(
    occupancy: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    gt_mesh_path: Path | None,
) -> dict[str, float]:
    metrics = {
        "mesh_vertex_chamfer_mean": -1.0,
        "mesh_vertex_chamfer_p95": -1.0,
    }
    if gt_mesh_path is None or not gt_mesh_path.exists() or not np.any(occupancy):
        return metrics
    verts, _faces, _normals, _values = measure.marching_cubes(occupancy.astype(np.float32), level=0.5)
    spacing = (maxs - mins) / np.maximum(np.array(occupancy.shape) - 1, 1)
    recon_vertices = mins + verts.astype(np.float32) * spacing
    gt_vertices = load_stl_vertices(gt_mesh_path)
    recon_sample = sample_points(recon_vertices, 12000, seed=17)
    gt_sample = sample_points(gt_vertices, 12000, seed=19)
    if len(recon_sample) == 0 or len(gt_sample) == 0:
        return metrics
    gt_tree = cKDTree(gt_sample)
    recon_tree = cKDTree(recon_sample)
    recon_to_gt = gt_tree.query(recon_sample, k=1)[0]
    gt_to_recon = recon_tree.query(gt_sample, k=1)[0]
    chamfer_all = np.concatenate([recon_to_gt, gt_to_recon], axis=0)
    metrics["mesh_vertex_chamfer_mean"] = float(chamfer_all.mean())
    metrics["mesh_vertex_chamfer_p95"] = float(np.percentile(chamfer_all, 95))
    return metrics


def evaluate_reconstruction(
    geometry: ReconstructionGeometry,
    gt_mesh_path: Path | None = None,
    graph_knn: int = 6,
    voxel_grid_size: int = 96,
    density_quantile: float = 0.9,
    sigma_scale: float = 2.0,
) -> dict[str, float]:
    graph = build_graph_diagnostics(geometry.xyz, knn=graph_knn)
    density, mins, maxs = voxelize_geometry(geometry, grid_size=voxel_grid_size, sigma_scale=sigma_scale)
    occupancy = occupancy_from_density(density, density_quantile=density_quantile)
    voxel = compute_voxel_metrics(density, occupancy)
    mesh = compute_mesh_metrics(occupancy, mins, maxs, gt_mesh_path)
    center = geometry.xyz.mean(axis=0)
    std = geometry.xyz.std(axis=0)
    return {
        "point_count": int(len(geometry.xyz)),
        "center": [float(value) for value in center.tolist()],
        "std": [float(value) for value in std.tolist()],
        "component_count": float(graph["component_count"]),
        "largest_component_fraction": float(graph["largest_component_fraction"]),
        "neighbor_distance_mean": float(graph["neighbor_distance_mean"]),
        "neighbor_distance_p95": float(graph["neighbor_distance_p95"]),
        "mst_mean": float(graph["mst_mean"]),
        "mst_p95": float(graph["mst_p95"]),
        "line_score_mean": float(graph["line_score_mean"]),
        **voxel,
        **mesh,
    }
