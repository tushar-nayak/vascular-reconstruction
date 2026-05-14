"""Batch-score reconstruction checkpoints and rank experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from visualize_reconstruction import visualize_checkpoint


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        iteration = int(stem.split("_")[-1])
    except ValueError:
        iteration = -1
    return iteration, path.name


def _rank_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        -float(metrics["voxel_largest_component_fraction"]),
        float(metrics["voxel_component_count"]),
        float(metrics["mesh_vertex_chamfer_p95"]) if float(metrics["mesh_vertex_chamfer_p95"]) >= 0.0 else float("inf"),
        -float(metrics["largest_component_fraction"]),
        float(metrics["occupancy_fill_ratio"]),
        float(metrics["mst_p95"]),
        -float(metrics["line_score_mean"]),
    )


def _resolve_mesh_path(case_id: str, mesh_root: Path) -> Path | None:
    candidate = mesh_root / f"{case_id}.stl"
    return candidate if candidate.exists() else None


def _collect_metrics(checkpoint_dirs: list[Path], output_root: Path, mesh_root: Path | None) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key)
        for checkpoint_path in checkpoints:
            experiment_name = checkpoint_dir.name
            output_dir = output_root / experiment_name
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            mesh_path = None
            if mesh_root is not None and checkpoint.get("case_id") is not None:
                mesh_path = _resolve_mesh_path(str(checkpoint["case_id"]), mesh_root)
            metrics = visualize_checkpoint(checkpoint_path, output_dir, mesh_path=mesh_path, save_figure=False)
            metrics["experiment"] = experiment_name
            results.append(metrics)
    return results


def _write_summary(results: list[dict[str, object]], output_root: Path) -> Path:
    ranked = sorted(results, key=_rank_key)
    summary = {
        "ranking_rule": [
            "voxel_largest_component_fraction desc",
            "voxel_component_count asc",
            "mesh_vertex_chamfer_p95 asc",
            "largest_component_fraction desc",
            "occupancy_fill_ratio asc",
            "mst_p95 asc",
            "line_score_mean desc",
        ],
        "best_baseline": ranked[0] if ranked else None,
        "results": ranked,
    }
    summary_path = output_root / "scoreboard.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score reconstruction checkpoints and rank experiments.")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing experiment checkpoint subdirectories.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment subdirectory names to score, e.g. single_case_v10 single_case_v11.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("metrics"),
        help="Directory for per-checkpoint JSON metrics and the overall scoreboard.",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path("data/processed/imagecas/meshes_split"),
        help="Directory containing ground-truth STL meshes keyed by case_id.",
    )
    args = parser.parse_args()

    checkpoint_dirs = [args.checkpoints_root / experiment for experiment in args.experiments]
    args.output_root.mkdir(parents=True, exist_ok=True)

    mesh_root = args.mesh_root if args.mesh_root.exists() else None
    results = _collect_metrics(checkpoint_dirs, args.output_root, mesh_root)
    summary_path = _write_summary(results, args.output_root)
    print(f"Scored {len(results)} checkpoints across {len(checkpoint_dirs)} experiments")
    print(f"Scoreboard saved to {summary_path}")
    if results:
        best = sorted(results, key=_rank_key)[0]
        print(
            "Best baseline: "
            f"{best['experiment']} @ iter {best['iteration']} "
            f"(voxel_largest_component_fraction={best['voxel_largest_component_fraction']:.3f}, "
            f"voxel_component_count={int(best['voxel_component_count'])}, "
            f"mesh_vertex_chamfer_p95={best['mesh_vertex_chamfer_p95']:.3f}, "
            f"largest_component_fraction={best['largest_component_fraction']:.3f}, "
            f"mst_p95={best['mst_p95']:.3f}, "
            f"line_score_mean={best['line_score_mean']:.3f})"
        )


if __name__ == "__main__":
    main()
