"""Training loop for per-case vascular reconstruction with differentiable silhouette rendering."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label
from skimage.morphology import skeletonize
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.data.dataset import ProjectionDataset
from vascular_reconstruction.evaluation.reconstruction_metrics import (
    ReconstructionGeometry,
    active_gaussian_count_from_schedule,
    evaluate_reconstruction,
    select_active_geometry,
)
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.rendering import downsample_mask, render_gaussian_silhouette
from vascular_reconstruction.simulation.equations import navier_stokes_loss


class Trainer:
    """Optimizes one case reconstruction at a time."""

    def __init__(
        self,
        model: PINN_GS,
        dataset: ProjectionDataset,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        device: str | None = None,
    ):
        resolved_device = device or self._resolve_device(train_config.device)
        self.dataset = dataset
        self.config = train_config
        self.model_config = model_config
        self.device = resolved_device
        self.model = model.to(self.device)
        self.failure_count = 0

        self.debug_projection_dir = Path(self.config.debug_projection_dir)
        self.debug_projection_dir.mkdir(parents=True, exist_ok=True)

        self.gs_optimizer = optim.Adam(self.model.gs.parameters(), lr=self.config.learning_rate)
        self.pinn_optimizer = optim.Adam(self.model.pinn.parameters(), lr=self.config.pinn_learning_rate)

        self.case_index = min(max(self.config.train_case_index, 0), len(self.dataset) - 1)
        self.case_data = self.dataset.get_case(self.case_index)
        self.case_id = str(self.case_data["case_id"])
        self.gt_mesh_path = Path(self.case_data["mesh_path"]) if self.case_data.get("mesh_path") else None
        self._last_active_count = self.model_config.num_gaussians
        self._support_views = self._build_support_views()
        self.best_eval_score = (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))

        self._initialize_gaussians_from_case(self.case_data)

    @staticmethod
    def _resolve_device(configured_device: str) -> str:
        if configured_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured_device

    @staticmethod
    def _projection_matrix_from_view(view: Mapping[str, object], device: str) -> torch.Tensor:
        matrix = view["projection_matrix"]
        return torch.tensor(matrix, dtype=torch.float32, device=device)

    @staticmethod
    def _view_rotation(lao: float, cran: float) -> np.ndarray:
        lao_rad = np.radians(lao)
        cran_rad = np.radians(cran)
        ry = np.array(
            [
                [np.cos(lao_rad), 0.0, np.sin(lao_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(lao_rad), 0.0, np.cos(lao_rad)],
            ],
            dtype=np.float32,
        )
        rz = np.array(
            [
                [np.cos(cran_rad), -np.sin(cran_rad), 0.0],
                [np.sin(cran_rad), np.cos(cran_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rz @ ry

    def _initialize_gaussians_from_case(self, case_data: Mapping[str, object]) -> None:
        bootstrap_points = self._bootstrap_points_from_case(case_data)
        if bootstrap_points is None or len(bootstrap_points) == 0:
            return

        with torch.no_grad():
            self.model.gs.initialize_from_points(
                bootstrap_points,
                scale_value=self.config.scale_mean_target,
                opacity_value=self.config.opacity_mean_target,
            )

    def _bootstrap_points_from_case(self, case_data: Mapping[str, object]) -> torch.Tensor | None:
        views = list(case_data["views"])[: self.config.max_init_views]
        if not views:
            return None

        points_per_view = max(self.model_config.num_gaussians // len(views), 1)
        world_points: list[np.ndarray] = []

        for view in views:
            vessel_mask = np.asarray(view["vessel_mask"]) > 0.5
            vessel_pixels = np.argwhere(skeletonize(vessel_mask))
            if len(vessel_pixels) == 0:
                vessel_pixels = np.argwhere(vessel_mask)
            if len(vessel_pixels) == 0:
                continue

            sample_count = min(points_per_view, len(vessel_pixels))
            rng = np.random.default_rng(int(np.sum(view["image"]) % (2**32 - 1)))
            sample_indices = rng.choice(len(vessel_pixels), size=sample_count, replace=len(vessel_pixels) < sample_count)
            sampled_pixels = vessel_pixels[sample_indices]

            projection_matrix = np.asarray(view["projection_matrix"], dtype=np.float32)
            focal_x = projection_matrix[0, 0]
            focal_y = projection_matrix[1, 1]
            center_x = projection_matrix[0, 2]
            center_y = projection_matrix[1, 2]

            x_cam = rng.normal(loc=0.0, scale=self.config.init_depth_mm, size=sample_count).astype(np.float32)
            x_dist = np.clip(600.0 + x_cam, 1.0, None)
            pixel_x = sampled_pixels[:, 1].astype(np.float32)
            pixel_y = sampled_pixels[:, 0].astype(np.float32)
            y_cam = ((pixel_x - center_x) / focal_x) * x_dist
            z_cam = -((pixel_y - center_y) / focal_y) * x_dist

            jitter = rng.normal(loc=0.0, scale=self.config.init_jitter_mm, size=(sample_count, 3)).astype(np.float32)
            cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1) + jitter

            lao, cran = view["angles"]
            rotation = self._view_rotation(float(lao), float(cran))
            world_points.append(cam_points @ rotation)

        if not world_points:
            return None

        points_np = np.concatenate(world_points, axis=0)
        if len(points_np) < self.model_config.num_gaussians:
            repeat_count = (self.model_config.num_gaussians + len(points_np) - 1) // len(points_np)
            points_np = np.tile(points_np, (repeat_count, 1))

        return torch.from_numpy(points_np[: self.model_config.num_gaussians])

    def _silhouette_loss(
        self,
        rendered: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy(rendered.clamp(1e-5, 1.0 - 1e-5), target_mask)
        intersection = torch.sum(rendered * target_mask)
        dice = 1.0 - (2.0 * intersection + 1e-5) / (torch.sum(rendered) + torch.sum(target_mask) + 1e-5)
        outside_mass = torch.mean(rendered * (1.0 - target_mask))
        target_mass = torch.mean(target_mask)
        rendered_mass = torch.mean(rendered)
        mass_match = torch.abs(rendered_mass - target_mass)
        return (
            self.config.mask_bce_weight * bce
            + self.config.mask_dice_weight * dice
            + self.config.outside_mask_weight * outside_mass
            + self.config.mass_match_weight * mass_match
        )

    def _build_support_views(self) -> list[dict[str, torch.Tensor | Mapping[str, object]]]:
        support_views: list[dict[str, torch.Tensor | Mapping[str, object]]] = []
        skeleton_radius = max(int(self.config.point_skeleton_dilation_radius_px), 0)
        skeleton_kernel = 2 * skeleton_radius + 1

        for view in self.case_data["views"]:
            vessel_mask = torch.from_numpy(np.asarray(view["vessel_mask"], dtype=np.float32)).to(self.device)
            skeleton_mask_np = skeletonize(np.asarray(view["vessel_mask"], dtype=bool)).astype(np.float32)
            skeleton_mask = torch.from_numpy(skeleton_mask_np).to(self.device)
            if skeleton_radius > 0:
                skeleton_mask = F.max_pool2d(
                    skeleton_mask.unsqueeze(0).unsqueeze(0),
                    kernel_size=skeleton_kernel,
                    stride=1,
                    padding=skeleton_radius,
                ).squeeze(0).squeeze(0)
            support_views.append(
                {
                    "view": view,
                    "vessel_mask": vessel_mask,
                    "skeleton_mask": skeleton_mask.clamp(0.0, 1.0),
                }
            )
        return support_views

    def _skeleton_loss(
        self,
        rendered: torch.Tensor,
        target_mask: torch.Tensor,
        skeleton_mask: torch.Tensor,
    ) -> torch.Tensor:
        skeleton_focus = -(torch.log(rendered.clamp_min(1e-5)) * skeleton_mask).sum() / (skeleton_mask.sum() + 1e-5)
        vessel_shell = torch.clamp(target_mask - skeleton_mask, min=0.0)
        thickness_penalty = torch.mean(rendered * vessel_shell)
        return (
            self.config.skeleton_focus_weight * skeleton_focus
            + self.config.skeleton_thickness_weight * thickness_penalty
        )

    def _active_gaussian_count(self, iteration: int) -> int:
        return active_gaussian_count_from_schedule(
            total_count=self.model_config.num_gaussians,
            active_gaussian_schedule=self.config.active_gaussian_schedule,
            iteration=iteration,
        )

    def _active_gaussian_indices(self, iteration: int) -> torch.Tensor:
        active_count = self._active_gaussian_count(iteration)
        opacity = self.model.gs.get_opacity.squeeze(-1).detach()
        return torch.topk(opacity, k=active_count, largest=True).indices

    def _graph_sample_indices(self, active_indices: torch.Tensor) -> torch.Tensor:
        if len(active_indices) <= self.config.graph_sample_size:
            return active_indices
        opacity = self.model.gs.get_opacity.squeeze(-1).detach()[active_indices]
        sample_local = torch.topk(opacity, k=self.config.graph_sample_size, largest=True).indices
        return active_indices[sample_local]

    @staticmethod
    def _mst_edges_from_distances(pairwise_dist_detached: torch.Tensor) -> list[tuple[int, int]]:
        node_count = int(pairwise_dist_detached.shape[0])
        if node_count <= 1:
            return []
        visited = torch.zeros(node_count, dtype=torch.bool)
        visited[0] = True
        edges: list[tuple[int, int]] = []
        large_value = torch.tensor(float("inf"), dtype=pairwise_dist_detached.dtype)

        for _ in range(node_count - 1):
            masked = pairwise_dist_detached.clone()
            masked[~visited] = large_value
            masked[:, visited] = large_value
            flat_index = int(torch.argmin(masked).item())
            src = flat_index // node_count
            dst = flat_index % node_count
            if not torch.isfinite(masked[src, dst]):
                break
            edges.append((src, dst))
            visited[dst] = True
        return edges

    def _graph_connectivity_penalty(self, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        graph_indices = self._graph_sample_indices(active_indices)
        xyz = self.model.gs.get_xyz[graph_indices]
        if len(xyz) <= 1:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"graph_edge_mean": 0.0, "graph_edge_p90": 0.0, "graph_indices": graph_indices, "mst_edges": []}

        pairwise_dist = torch.cdist(xyz, xyz)
        detached = pairwise_dist.detach().cpu()
        mst_edges = self._mst_edges_from_distances(detached)
        if not mst_edges:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"graph_edge_mean": 0.0, "graph_edge_p90": 0.0, "graph_indices": graph_indices, "mst_edges": []}

        edge_lengths = torch.stack([pairwise_dist[src, dst] for src, dst in mst_edges])
        bridge_count = min(self.config.graph_bridge_edges, len(edge_lengths))
        bridge_lengths = torch.topk(edge_lengths, k=bridge_count, largest=True).values
        normalized_gap = torch.relu(bridge_lengths - self.config.graph_edge_target) / max(self.config.graph_edge_target, 1e-3)
        penalty = normalized_gap.pow(2).mean()
        edge_lengths_detached = edge_lengths.detach().cpu()
        return penalty, {
            "graph_edge_mean": float(edge_lengths_detached.mean().item()),
            "graph_edge_p90": float(torch.quantile(edge_lengths_detached, 0.9).item()),
            "graph_bridge_mean": float(bridge_lengths.detach().cpu().mean().item()),
            "graph_indices": graph_indices,
            "mst_edges": mst_edges,
        }

    def _component_purity_penalty(self, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if len(active_indices) <= 1 or self.config.component_purity_weight <= 0.0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {
                "sample_component_count": 0.0,
                "sample_largest_component_fraction": 0.0,
                "outside_component_opacity_fraction": 0.0,
            }

        sample_size = min(len(active_indices), self.config.component_sample_size)
        opacity = self.model.gs.get_opacity.squeeze(-1)[active_indices]
        sample_local = torch.topk(opacity.detach(), k=sample_size, largest=True).indices
        sampled_indices = active_indices[sample_local]
        sampled_xyz = self.model.gs.get_xyz[sampled_indices]
        sampled_opacity = self.model.gs.get_opacity.squeeze(-1)[sampled_indices]

        pairwise_dist = torch.cdist(sampled_xyz, sampled_xyz)
        knn_k = min(self.config.component_knn + 1, len(sampled_xyz))
        if knn_k <= 1:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {
                "sample_component_count": 1.0,
                "sample_largest_component_fraction": 1.0,
                "outside_component_opacity_fraction": 0.0,
            }

        knn_indices = torch.topk(pairwise_dist, k=knn_k, largest=False).indices[:, 1:]
        knn_distances = torch.topk(pairwise_dist, k=knn_k, largest=False).values[:, 1:]
        adjacency = torch.zeros((len(sampled_xyz), len(sampled_xyz)), dtype=torch.bool, device=self.device)
        neighbor_mask = knn_distances <= self.config.component_max_distance
        rows = torch.arange(len(sampled_xyz), device=self.device).unsqueeze(1).expand_as(knn_indices)
        adjacency[rows[neighbor_mask], knn_indices[neighbor_mask]] = True
        adjacency = adjacency | adjacency.T

        visited = torch.zeros(len(sampled_xyz), dtype=torch.bool, device=self.device)
        labels = torch.full((len(sampled_xyz),), -1, dtype=torch.long, device=self.device)
        component_sizes: list[int] = []
        component_id = 0

        for start in range(len(sampled_xyz)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            labels[start] = component_id
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                neighbors = torch.nonzero(adjacency[current], as_tuple=False).squeeze(-1)
                for neighbor in neighbors.tolist():
                    if visited[neighbor]:
                        continue
                    visited[neighbor] = True
                    labels[neighbor] = component_id
                    stack.append(neighbor)
            component_sizes.append(size)
            component_id += 1

        largest_component = int(np.argmax(component_sizes))
        largest_mask = labels == largest_component
        outside_mask = ~largest_mask
        opacity_mass = sampled_opacity.sum().clamp_min(1e-6)
        outside_opacity_fraction = sampled_opacity[outside_mask].sum() / opacity_mass
        return outside_opacity_fraction, {
            "sample_component_count": float(len(component_sizes)),
            "sample_largest_component_fraction": float(component_sizes[largest_component] / len(sampled_xyz)),
            "outside_component_opacity_fraction": float(outside_opacity_fraction.item()),
        }

    def _sample_projected_map(
        self,
        points: torch.Tensor,
        view: Mapping[str, object],
        map_tensor: torch.Tensor,
    ) -> torch.Tensor:
        projected = self._project_world_points(points, view)
        height, width = map_tensor.shape
        x_norm = (projected[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
        y_norm = (projected[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            map_tensor.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.view(-1)

    def _project_world_points(self, points: torch.Tensor, view: Mapping[str, object]) -> torch.Tensor:
        lao, cran = view["angles"]
        view_matrix = self.model.get_view_matrix(lao, cran, device=self.device)
        projection_matrix = self._projection_matrix_from_view(view, self.device)
        points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        xyz_cam = (points_h @ view_matrix.T)[:, :3]
        x_dist = torch.clamp(xyz_cam[:, 0] + 600.0, min=1.0)
        focal_x = projection_matrix[0, 0]
        focal_y = projection_matrix[1, 1]
        center_x = projection_matrix[0, 2]
        center_y = projection_matrix[1, 2]
        u_pix = (focal_x * xyz_cam[:, 1]) / x_dist + center_x
        v_pix = (focal_y * (-xyz_cam[:, 2])) / x_dist + center_y
        return torch.stack([u_pix, v_pix], dim=-1)

    def _edge_multiview_support(self, start_point: torch.Tensor, end_point: torch.Tensor) -> float:
        sample_count = max(self.config.densify_support_samples, 2)
        alphas = torch.linspace(0.0, 1.0, steps=sample_count, device=self.device, dtype=start_point.dtype).unsqueeze(-1)
        segment_points = torch.lerp(start_point.unsqueeze(0), end_point.unsqueeze(0), alphas)

        total_score = 0.0
        total_views = 0
        radius = max(int(self.config.densify_support_radius_px), 0)
        vessel_weight = float(self.config.densify_support_vessel_weight)
        skeleton_weight = float(self.config.densify_support_skeleton_weight)

        for support_view in self._support_views[: self.config.densify_support_views]:
            view = support_view["view"]
            vessel_mask = support_view["vessel_mask"]
            skeleton_mask = support_view["skeleton_mask"]

            projected = self._project_world_points(segment_points, view)
            width = vessel_mask.shape[1]
            height = vessel_mask.shape[0]
            sample_score = 0.0

            for coord in projected:
                x = int(round(float(coord[0].item())))
                y = int(round(float(coord[1].item())))
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                vessel_patch = vessel_mask[y0:y1, x0:x1]
                skeleton_patch = skeleton_mask[y0:y1, x0:x1]
                vessel_hit = 1.0 if torch.any(vessel_patch > 0.5) else 0.0
                skeleton_hit = 1.0 if torch.any(skeleton_patch > 0.5) else 0.0
                sample_score += vessel_weight * vessel_hit + skeleton_weight * skeleton_hit

            total_score += sample_score / sample_count
            total_views += 1

        if total_views == 0:
            return 0.0
        return float(total_score / total_views)

    def _point_multiview_support_loss(self, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if len(active_indices) == 0 or self.config.point_support_weight <= 0.0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"point_vessel_support": 0.0, "point_skeleton_support": 0.0}

        sample_size = min(len(active_indices), self.config.point_support_sample_size)
        opacity = self.model.gs.get_opacity.squeeze(-1)[active_indices]
        sample_local = torch.topk(opacity.detach(), k=sample_size, largest=True).indices
        sampled_indices = active_indices[sample_local]
        sampled_xyz = self.model.gs.get_xyz[sampled_indices]

        vessel_supports: list[torch.Tensor] = []
        skeleton_supports: list[torch.Tensor] = []
        for support_view in self._support_views[: self.config.point_support_views]:
            view = support_view["view"]
            vessel_mask = support_view["vessel_mask"]
            skeleton_mask = support_view["skeleton_mask"]
            vessel_supports.append(self._sample_projected_map(sampled_xyz, view, vessel_mask))
            skeleton_supports.append(self._sample_projected_map(sampled_xyz, view, skeleton_mask))

        if not vessel_supports:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"point_vessel_support": 0.0, "point_skeleton_support": 0.0}

        vessel_support = torch.stack(vessel_supports, dim=0).mean(dim=0)
        skeleton_support = torch.stack(skeleton_supports, dim=0).mean(dim=0)
        low_vessel_support = torch.relu(self.config.point_vessel_min_ratio - vessel_support).pow(2).mean()
        low_skeleton_support = (1.0 - skeleton_support).mean()
        loss = (
            self.config.point_support_weight * low_vessel_support
            + self.config.point_skeleton_weight * low_skeleton_support
        )
        return loss, {
            "point_vessel_support": float(vessel_support.mean().item()),
            "point_skeleton_support": float(skeleton_support.mean().item()),
        }

    @staticmethod
    def _inactive_gaussian_indices(total_count: int, active_indices: torch.Tensor) -> torch.Tensor:
        inactive_mask = torch.ones(total_count, dtype=torch.bool, device=active_indices.device)
        inactive_mask[active_indices] = False
        return torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)

    def _densify_to_count(self, previous_active_indices: torch.Tensor, target_count: int) -> None:
        inactive_indices = self._inactive_gaussian_indices(self.model_config.num_gaussians, previous_active_indices)
        growth = min(target_count - len(previous_active_indices), len(inactive_indices))
        if growth <= 0 or len(previous_active_indices) == 0:
            return

        with torch.no_grad():
            active_xyz = self.model.gs.get_xyz[previous_active_indices]
            active_scaling = self.model.gs.get_scaling[previous_active_indices]
            active_opacity = self.model.gs.get_opacity.squeeze(-1)[previous_active_indices]
            new_indices = inactive_indices[:growth]

            graph_penalty, graph_stats = self._graph_connectivity_penalty(previous_active_indices)
            del graph_penalty
            graph_indices = graph_stats["graph_indices"]
            mst_edges = graph_stats["mst_edges"]

            edge_pairs: list[tuple[int, int]] = []
            if mst_edges:
                graph_xyz = self.model.gs.get_xyz[graph_indices]
                edge_lengths = torch.tensor(
                    [torch.norm(graph_xyz[src] - graph_xyz[dst]).item() for src, dst in mst_edges],
                    device=self.device,
                )
                order = torch.argsort(edge_lengths, descending=True)
                top_edges = max(1, min(len(order), self.config.densify_edge_knn))
                scored_edges: list[tuple[float, tuple[int, int]]] = []
                for idx in order[:top_edges].tolist():
                    src_idx = int(graph_indices[mst_edges[idx][0]].item())
                    dst_idx = int(graph_indices[mst_edges[idx][1]].item())
                    support_score = self._edge_multiview_support(
                        self.model.gs.get_xyz[src_idx],
                        self.model.gs.get_xyz[dst_idx],
                    )
                    scored_edges.append((support_score, (src_idx, dst_idx)))

                supported_edges = [pair for score, pair in scored_edges if score >= self.config.densify_min_support_ratio]
                if supported_edges:
                    edge_pairs = supported_edges
                else:
                    edge_pairs = [pair for _, pair in sorted(scored_edges, key=lambda item: item[0], reverse=True)]

            if not edge_pairs:
                sample_weights = active_opacity / active_opacity.sum().clamp_min(1e-6)
                source_local_indices = torch.multinomial(
                    sample_weights,
                    num_samples=growth,
                    replacement=len(active_xyz) < growth,
                )
                edge_pairs = [
                    (
                        int(previous_active_indices[idx].item()),
                        int(previous_active_indices[idx].item()),
                    )
                    for idx in source_local_indices.tolist()
                ]

            source_indices = []
            neighbor_indices = []
            for slot in range(growth):
                src_idx, dst_idx = edge_pairs[slot % len(edge_pairs)]
                source_indices.append(src_idx)
                neighbor_indices.append(dst_idx)
            source_indices_t = torch.tensor(source_indices, device=self.device, dtype=torch.long)
            neighbor_indices_t = torch.tensor(neighbor_indices, device=self.device, dtype=torch.long)

            source_xyz = self.model.gs.get_xyz[source_indices_t]
            neighbor_xyz = self.model.gs.get_xyz[neighbor_indices_t]
            source_scaling = self.model.gs.get_scaling[source_indices_t]
            neighbor_scaling = self.model.gs.get_scaling[neighbor_indices_t]
            source_opacity = self.model.gs.get_opacity.squeeze(-1)[source_indices_t]
            neighbor_opacity = self.model.gs.get_opacity.squeeze(-1)[neighbor_indices_t]

            tangents = F.normalize(neighbor_xyz - source_xyz + 1e-6, dim=-1)
            blend = torch.rand((growth, 1), device=self.device, dtype=source_xyz.dtype) * 0.5 + 0.25
            backbone_xyz = torch.lerp(source_xyz, neighbor_xyz, blend)
            tangent_steps = (
                torch.randn((growth, 1), device=self.device, dtype=source_xyz.dtype)
                * source_scaling.mean(dim=-1, keepdim=True)
                * self.config.densify_spacing_scale
                * 0.15
            )
            tangent_offsets = tangents * tangent_steps

            noise = torch.randn_like(source_xyz)
            transverse_noise = noise - (noise * tangents).sum(dim=-1, keepdim=True) * tangents
            transverse_offsets = transverse_noise * (
                source_scaling.mean(dim=-1, keepdim=True) * self.config.densify_jitter_scale
            )
            new_xyz = backbone_xyz + tangent_offsets + transverse_offsets

            interp_scale = torch.lerp(source_scaling, neighbor_scaling, blend)
            interp_opacity = torch.lerp(source_opacity, neighbor_opacity, blend.squeeze(-1))
            new_scale = torch.clamp(interp_scale * self.config.densify_scale_shrink, min=0.04)
            new_opacity = torch.clamp(interp_opacity * self.config.densify_opacity_scale, min=1e-4, max=1.0 - 1e-4)

            self.model.gs._xyz.data[new_indices] = new_xyz
            self.model.gs._scaling.data[new_indices] = torch.log(new_scale)
            self.model.gs._opacity.data[new_indices] = torch.logit(new_opacity).unsqueeze(-1)
            self.model.gs._rotation.data[new_indices] = self.model.gs._rotation.data[source_indices_t]
            self.model.gs._features_dc.data[new_indices] = self.model.gs._features_dc.data[source_indices_t]
            self.model.gs._features_rest.data[new_indices] = self.model.gs._features_rest.data[source_indices_t]

    def _maybe_densify(self, iteration: int) -> None:
        target_count = self._active_gaussian_count(iteration)
        if target_count <= self._last_active_count:
            self._last_active_count = target_count
            return

        previous_active_indices = self._active_gaussian_indices(max(iteration - 1, 0))
        self._densify_to_count(previous_active_indices, target_count)
        self._last_active_count = target_count

    def _volume_thickness_loss(self, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        occupancy_grid, spans = self._occupancy_grid(active_indices)
        if occupancy_grid is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"volume_fill": 0.0, "volume_core_fill": 0.0}

        occupancy_5d = occupancy_grid.unsqueeze(0).unsqueeze(0)
        core_occupancy = -F.max_pool3d(-occupancy_5d, kernel_size=3, stride=1, padding=1)
        volume_fill = occupancy_grid.mean()
        core_fill = core_occupancy.mean()
        loss = volume_fill + self.config.volume_core_weight * core_fill
        return loss, {
            "volume_fill": float(volume_fill.item()),
            "volume_core_fill": float(core_fill.item()),
            "volume_extent_mean": float((spans.mean()).item()),
        }

    def _occupancy_grid(self, active_indices: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        xyz = self.model.gs.get_xyz[active_indices]
        scaling = self.model.gs.get_scaling[active_indices]
        opacity = self.model.gs.get_opacity.squeeze(-1)[active_indices]
        if len(xyz) == 0:
            return None, None

        if len(xyz) > self.config.volume_sample_size:
            top_idx = torch.topk(opacity.detach(), k=self.config.volume_sample_size, largest=True).indices
            xyz = xyz[top_idx]
            scaling = scaling[top_idx]
            opacity = opacity[top_idx]

        radii = scaling.max(dim=-1).values
        mins = torch.min(xyz - radii[:, None], dim=0).values
        maxs = torch.max(xyz + radii[:, None], dim=0).values
        spans = (maxs - mins).clamp_min(1.0)
        axes = [
            torch.linspace(mins[dim], maxs[dim], self.config.volume_grid_size, device=self.device, dtype=xyz.dtype)
            for dim in range(3)
        ]
        grid_x, grid_y, grid_z = torch.meshgrid(*axes, indexing="ij")
        flat_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
        density = torch.zeros(len(flat_grid), device=self.device, dtype=xyz.dtype)

        for start in range(0, len(xyz), self.config.volume_chunk_size):
            end = min(start + self.config.volume_chunk_size, len(xyz))
            chunk_xyz = xyz[start:end]
            chunk_scaling = scaling[start:end].clamp_min(1e-3)
            chunk_opacity = opacity[start:end]
            delta = flat_grid.unsqueeze(1) - chunk_xyz.unsqueeze(0)
            normalized = delta / chunk_scaling.unsqueeze(0)
            squared_distance = normalized.pow(2).sum(dim=-1)
            chunk_density = torch.exp(-0.5 * squared_distance) * chunk_opacity.unsqueeze(0)
            density = density + chunk_density.sum(dim=1)

        occupancy = 1.0 - torch.exp(-density)
        occupancy_grid = occupancy.view(
            self.config.volume_grid_size,
            self.config.volume_grid_size,
            self.config.volume_grid_size,
        )
        return occupancy_grid, spans

    def _voxel_topology_loss(self, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.config.voxel_topology_weight <= 0.0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {
                "voxel_largest_component_fraction": 0.0,
                "voxel_component_count": 0.0,
                "occupancy_fill_ratio": 0.0,
                "occupancy_compactness": 0.0,
            }

        occupancy_grid, _spans = self._occupancy_grid(active_indices)
        if occupancy_grid is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {
                "voxel_largest_component_fraction": 0.0,
                "voxel_component_count": 0.0,
                "occupancy_fill_ratio": 0.0,
                "occupancy_compactness": 0.0,
            }

        threshold = torch.quantile(occupancy_grid.detach().reshape(-1), self.config.voxel_density_quantile)
        occupancy_mask_np = (occupancy_grid.detach().cpu().numpy() >= float(threshold.item()))
        if not np.any(occupancy_mask_np):
            fill_ratio = occupancy_grid.mean()
            return fill_ratio, {
                "voxel_largest_component_fraction": 0.0,
                "voxel_component_count": 0.0,
                "occupancy_fill_ratio": float(fill_ratio.item()),
                "occupancy_compactness": 0.0,
            }

        labels, component_count = label(occupancy_mask_np)
        component_sizes = np.bincount(labels.reshape(-1))[1:]
        largest_label = int(np.argmax(component_sizes)) + 1
        largest_component_mask = torch.from_numpy(labels == largest_label).to(self.device)
        occupancy_mass = occupancy_grid.sum().clamp_min(1e-6)
        outside_mass = occupancy_grid[~largest_component_mask].sum() / occupancy_mass

        surface_mask = occupancy_mask_np & ~binary_erosion(occupancy_mask_np)
        surface_count = max(int(surface_mask.sum()), 1)
        largest_fraction = float(component_sizes.max() / max(int(occupancy_mask_np.sum()), 1))
        compactness = float(component_sizes.max() / surface_count)
        compactness_penalty = torch.tensor(max(0.0, 0.02 - compactness), device=self.device, dtype=occupancy_grid.dtype)
        loss = (
            self.config.voxel_connectivity_weight * outside_mass
            + self.config.voxel_compactness_weight * compactness_penalty
        )
        return loss, {
            "voxel_largest_component_fraction": largest_fraction,
            "voxel_component_count": float(component_count),
            "occupancy_fill_ratio": float(occupancy_grid.mean().item()),
            "occupancy_compactness": compactness,
        }

    def _geometry_regularization(self, iteration: int, active_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        xyz = self.model.gs.get_xyz[active_indices]
        opacity = self.model.gs.get_opacity.squeeze(-1)[active_indices]
        scaling = self.model.gs.get_scaling[active_indices]

        sample_count = min(self.config.repulsion_num_samples, len(xyz))
        sample_idx = torch.randperm(len(xyz), device=xyz.device)[:sample_count]
        sampled_xyz = xyz[sample_idx]
        pairwise_dist = torch.cdist(sampled_xyz, sampled_xyz)
        valid_mask = ~torch.eye(sample_count, dtype=torch.bool, device=xyz.device)
        repulsion = torch.relu(self.config.min_gaussian_separation - pairwise_dist[valid_mask]).pow(2).mean()

        axis_std = torch.std(xyz, dim=0)
        std_floor = torch.relu(self.config.axis_std_floor - axis_std).pow(2).mean()

        continuity = torch.tensor(0.0, device=xyz.device)
        knn_k = min(self.config.continuity_knn + 1, sample_count)
        if knn_k > 1:
            knn_distances = torch.topk(pairwise_dist, k=knn_k, largest=False).values[:, 1:]
            continuity = torch.relu(knn_distances - self.config.continuity_max_distance).pow(2).mean()

        line_structure = torch.tensor(0.0, device=xyz.device)
        line_knn_k = min(self.config.line_structure_knn + 1, sample_count)
        if line_knn_k > 2:
            knn_indices = torch.topk(pairwise_dist, k=line_knn_k, largest=False).indices[:, 1:]
            neighbors = sampled_xyz[knn_indices]
            offsets = neighbors - sampled_xyz.unsqueeze(1)
            covariance = offsets.transpose(1, 2) @ offsets / max(line_knn_k - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance)
            transverse_energy = eigenvalues[:, 0] + eigenvalues[:, 1]
            total_energy = eigenvalues.sum(dim=-1) + 1e-6
            line_structure = (transverse_energy / total_energy).mean()

        graph_connectivity, graph_stats = self._graph_connectivity_penalty(active_indices)
        component_purity, component_stats = self._component_purity_penalty(active_indices)
        voxel_topology, voxel_stats = self._voxel_topology_loss(active_indices)
        point_support, point_support_stats = self._point_multiview_support_loss(active_indices)

        opacity_mean = opacity.mean()
        opacity_reg = (opacity_mean - self.config.opacity_mean_target).pow(2)

        scaling_mean = scaling.mean()
        scale_reg = (scaling_mean - self.config.scale_mean_target).pow(2)

        total_reg = (
            self.config.repulsion_weight * repulsion
            + self.config.std_floor_weight * std_floor
            + self.config.continuity_weight * continuity
            + self.config.graph_connectivity_weight * graph_connectivity
            + self.config.component_purity_weight * component_purity
            + self.config.voxel_topology_weight * voxel_topology
            + self.config.line_structure_weight * line_structure
            + point_support
            + self.config.opacity_weight * opacity_reg
            + self.config.scale_weight * scale_reg
        )
        stats = {
            "active_gaussians": float(len(active_indices)),
            "repulsion": float(repulsion.item()),
            "std_floor": float(std_floor.item()),
            "continuity": float(continuity.item()),
            "graph_connectivity": float(graph_connectivity.item()),
            "graph_edge_mean": float(graph_stats["graph_edge_mean"]),
            "graph_edge_p90": float(graph_stats["graph_edge_p90"]),
            "graph_bridge_mean": float(graph_stats["graph_bridge_mean"]),
            "sample_component_count": float(component_stats["sample_component_count"]),
            "sample_largest_component_fraction": float(component_stats["sample_largest_component_fraction"]),
            "outside_component_opacity_fraction": float(component_stats["outside_component_opacity_fraction"]),
            "voxel_largest_component_fraction": float(voxel_stats["voxel_largest_component_fraction"]),
            "voxel_component_count": float(voxel_stats["voxel_component_count"]),
            "occupancy_fill_ratio": float(voxel_stats["occupancy_fill_ratio"]),
            "occupancy_compactness": float(voxel_stats["occupancy_compactness"]),
            "line_structure": float(line_structure.item()),
            "point_vessel_support": float(point_support_stats["point_vessel_support"]),
            "point_skeleton_support": float(point_support_stats["point_skeleton_support"]),
            "opacity_mean": float(opacity_mean.item()),
            "scale_mean": float(scaling_mean.item()),
            "xyz_std_mean": float(axis_std.mean().item()),
        }
        return total_reg, stats

    def _save_debug_projection(
        self,
        iteration: int,
        rendered_views: list[torch.Tensor],
        target_views: list[torch.Tensor],
    ) -> None:
        if iteration % self.config.debug_projection_interval != 0:
            return

        overlays = []
        for rendered_view, target_view in zip(rendered_views, target_views, strict=False):
            rendered_np = (rendered_view.detach().cpu().numpy() * 255.0).astype(np.uint8)
            target_np = (target_view.detach().cpu().numpy() * 255.0).astype(np.uint8)
            overlays.append(np.stack([rendered_np, target_np, np.zeros_like(rendered_np)], axis=-1))

        if not overlays:
            return

        tile_height, tile_width, _ = overlays[0].shape
        columns = min(3, len(overlays))
        rows = ceil(len(overlays) / columns)
        canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

        for index, overlay in enumerate(overlays):
            row = index // columns
            col = index % columns
            y0 = row * tile_height
            x0 = col * tile_width
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = overlay

        Image.fromarray(canvas).save(self.debug_projection_dir / f"iter_{iteration:06d}_{self.case_id}.png")

    def train_step(self, iteration: int) -> tuple[float, float, float, float, dict[str, float]]:
        self.gs_optimizer.zero_grad()
        self.pinn_optimizer.zero_grad()

        total_silhouette_loss = torch.tensor(0.0, device=self.device)
        total_skeleton_loss = torch.tensor(0.0, device=self.device)
        rendered_views: list[torch.Tensor] = []
        target_views: list[torch.Tensor] = []
        active_indices = self._active_gaussian_indices(iteration)

        for view_index, view in enumerate(self.case_data["views"]):
            lao, cran = view["angles"]
            view_matrix = self.model.get_view_matrix(lao, cran, device=self.device)
            projection_matrix = self._projection_matrix_from_view(view, self.device)
            vessel_mask = torch.from_numpy(np.asarray(view["vessel_mask"], dtype=np.float32)).to(self.device)
            target_mask = downsample_mask(vessel_mask, self.config.render_image_size)
            skeleton_mask_np = skeletonize(np.asarray(view["vessel_mask"], dtype=bool)).astype(np.float32)
            skeleton_mask = downsample_mask(torch.from_numpy(skeleton_mask_np).to(self.device), self.config.render_image_size)

            rendered = render_gaussian_silhouette(
                model=self.model,
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
                source_image_size=vessel_mask.shape,
                render_size=self.config.render_image_size,
                active_indices=active_indices,
                chunk_size=self.config.gaussian_chunk_size,
                min_sigma=self.config.render_min_sigma,
                max_sigma=self.config.render_max_sigma,
            )
            if view_index < 6:
                rendered_views.append(rendered)
                target_views.append(target_mask)

            total_silhouette_loss += self._silhouette_loss(rendered, target_mask)
            total_skeleton_loss += self._skeleton_loss(rendered, target_mask, skeleton_mask)

        loss_image = total_silhouette_loss / len(self.case_data["views"])
        loss_skeleton = total_skeleton_loss / len(self.case_data["views"])
        loss_reg, reg_stats = self._geometry_regularization(iteration, active_indices)
        loss_volume, volume_stats = self._volume_thickness_loss(active_indices)

        if iteration >= self.config.physics_warmup_iterations:
            raw_coords = torch.rand(1024, 4, device=self.device, requires_grad=True)
            coords_xyz = (raw_coords[:, :3] - 0.5) * 120.0
            coords_t = raw_coords[:, 3:4]
            coords = torch.cat([coords_xyz, coords_t], dim=-1)
            pinn_out = self.model(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4])
            loss_physics = navier_stokes_loss(pinn_out, coords)
        else:
            loss_physics = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.config.silhouette_loss_weight * loss_image
            + self.config.skeleton_loss_weight * loss_skeleton
            + self.config.volume_thickness_weight * loss_volume
            + self.config.physics_loss_weight * loss_physics
            + loss_reg
        )
        total_loss.backward()

        self.gs_optimizer.step()
        if iteration >= self.config.physics_warmup_iterations:
            self.pinn_optimizer.step()

        if not rendered_views or not target_views:
            raise RuntimeError("No rendered projection was produced.")
        self._save_debug_projection(iteration, rendered_views, target_views)

        reg_stats["skeleton_loss"] = float(loss_skeleton.item())
        reg_stats.update(volume_stats)
        return total_loss.item(), loss_image.item(), loss_physics.item(), loss_reg.item(), reg_stats

    def evaluate_current_reconstruction(self, iteration: int) -> dict[str, float | int | list[float] | bool]:
        geometry = ReconstructionGeometry(
            xyz=self.model.gs.get_xyz.detach().cpu().numpy(),
            scales=self.model.gs.get_scaling.detach().cpu().numpy(),
            opacities=self.model.gs.get_opacity.detach().cpu().numpy().squeeze(-1),
        )
        active_geometry = select_active_geometry(
            geometry,
            active_count=self._active_gaussian_count(iteration),
        )
        metrics = evaluate_reconstruction(
            active_geometry,
            gt_mesh_path=self.gt_mesh_path,
            voxel_grid_size=max(self.config.volume_grid_size * 4, 64),
            density_quantile=self.config.voxel_density_quantile,
        )
        gate_pass = (
            metrics["largest_component_fraction"] >= self.config.gate_min_graph_largest_component_fraction
            and metrics["component_count"] <= self.config.gate_max_graph_component_count
            and metrics["voxel_largest_component_fraction"] >= self.config.gate_min_voxel_largest_component_fraction
            and metrics["voxel_component_count"] <= self.config.gate_max_voxel_component_count
            and metrics["occupancy_fill_ratio"] <= self.config.gate_max_occupancy_fill_ratio
            and (
                self.config.gate_max_mesh_vertex_chamfer_p95 < 0.0
                or metrics["mesh_vertex_chamfer_p95"] <= self.config.gate_max_mesh_vertex_chamfer_p95
            )
        )
        score = (
            -float(metrics["voxel_largest_component_fraction"]),
            float(metrics["voxel_component_count"]),
            float(metrics["mesh_vertex_chamfer_p95"]) if float(metrics["mesh_vertex_chamfer_p95"]) >= 0.0 else float("inf"),
            float(metrics["occupancy_fill_ratio"]),
            float(metrics["mst_p95"]),
        )
        metrics.update(
            {
                "iteration": iteration,
                "gate_pass": bool(gate_pass),
                "score": list(score),
                "active_gaussians": int(len(active_geometry.xyz)),
            }
        )
        return metrics

    def _write_eval_metrics(self, metrics: Mapping[str, object]) -> None:
        eval_dir = Path(self.config.log_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / f"iter_{int(metrics['iteration']):06d}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _maybe_run_eval(self, iteration: int) -> None:
        if iteration <= 0 or iteration % self.config.eval_interval != 0:
            return
        metrics = self.evaluate_current_reconstruction(iteration)
        self._write_eval_metrics(metrics)
        gate_text = "PASS" if metrics["gate_pass"] else "FAIL"
        print(
            "Eval "
            f"{gate_text} @ {iteration}: "
            f"voxel_lcc={metrics['voxel_largest_component_fraction']:.3f}, "
            f"voxel_cc={int(metrics['voxel_component_count'])}, "
            f"fill={metrics['occupancy_fill_ratio']:.5f}, "
            f"mesh_p95={metrics['mesh_vertex_chamfer_p95']:.3f}"
        )
        if metrics["gate_pass"]:
            score_tuple = tuple(float(value) for value in metrics["score"])
            if score_tuple < self.best_eval_score:
                self.best_eval_score = score_tuple
                self.save_checkpoint(iteration, filename="best_gate_checkpoint.pt", eval_metrics=metrics)

    def train(self) -> None:
        self.train_from_iteration(0)

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.gs_optimizer.load_state_dict(checkpoint["gs_optimizer_state_dict"])
        self.pinn_optimizer.load_state_dict(checkpoint["pinn_optimizer_state_dict"])

        loaded_case_index = int(checkpoint.get("case_index", self.case_index))
        if loaded_case_index != self.case_index:
            raise ValueError(
                "Checkpoint case index does not match the active trainer case index: "
                f"{loaded_case_index} != {self.case_index}"
            )

        return int(checkpoint["iteration"])

    def train_from_iteration(self, start_iteration: int) -> None:
        previous_iteration = max(start_iteration - 1, 0)
        self._last_active_count = self._active_gaussian_count(previous_iteration)
        print(
            f"Starting training for {self.config.iterations} iterations on case "
            f"{self.case_index} ({self.case_id}) from iteration {start_iteration}..."
        )

        pbar = tqdm(range(start_iteration, self.config.iterations))
        for i in pbar:
            try:
                self._maybe_densify(i)
                loss, l_img, l_phys, l_reg, reg_stats = self.train_step(i)
                self.failure_count = 0
                if i % 10 == 0:
                    pbar.set_description(
                        "Loss: "
                        f"{loss:.4f} | Sil: {l_img:.4f} | Skel: {reg_stats['skeleton_loss']:.4f} "
                        f"| Vol: {reg_stats['volume_core_fill']:.4f} | Phys: {l_phys:.4f} | Reg: {l_reg:.4f} "
                        f"| Active: {int(reg_stats['active_gaussians'])} "
                        f"| CompFrac: {reg_stats['sample_largest_component_fraction']:.2f} "
                        f"| XYZstd: {reg_stats['xyz_std_mean']:.2f}"
                    )
            except Exception as exc:
                self.failure_count += 1
                print(f"Training failed at iteration {i}: {exc}")
                if self.failure_count >= self.config.max_failures:
                    raise RuntimeError(
                        f"Training aborted after {self.failure_count} consecutive failures."
                    ) from exc

            self._maybe_run_eval(i)
            if i > 0 and i % self.config.save_interval == 0:
                self.save_checkpoint(i)

        self.save_checkpoint(self.config.iterations)

    def save_checkpoint(
        self,
        iteration: int,
        filename: str | None = None,
        eval_metrics: Mapping[str, object] | None = None,
    ) -> None:
        path = Path(self.config.checkpoint_dir) / (filename or f"checkpoint_{iteration}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "case_index": self.case_index,
                "case_id": self.case_id,
                "model_state_dict": self.model.state_dict(),
                "gs_optimizer_state_dict": self.gs_optimizer.state_dict(),
                "pinn_optimizer_state_dict": self.pinn_optimizer.state_dict(),
                "training_config": self.config.to_dict(),
                "model_config": self.model_config.to_dict(),
                "eval_metrics": dict(eval_metrics) if eval_metrics is not None else None,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")
