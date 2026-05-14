"""Batch-score reconstruction checkpoints and rank experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        -float(metrics["largest_component_fraction"]),
        float(metrics["component_count"]),
        float(metrics["mst_p95"]),
        -float(metrics["line_score_mean"]),
    )


def _collect_metrics(checkpoint_dirs: list[Path], output_root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key)
        for checkpoint_path in checkpoints:
            experiment_name = checkpoint_dir.name
            output_dir = output_root / experiment_name
            metrics = visualize_checkpoint(checkpoint_path, output_dir, save_figure=False)
            metrics["experiment"] = experiment_name
            results.append(metrics)
    return results


def _write_summary(results: list[dict[str, object]], output_root: Path) -> Path:
    ranked = sorted(results, key=_rank_key)
    summary = {
        "ranking_rule": [
            "largest_component_fraction desc",
            "component_count asc",
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
    args = parser.parse_args()

    checkpoint_dirs = [args.checkpoints_root / experiment for experiment in args.experiments]
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = _collect_metrics(checkpoint_dirs, args.output_root)
    summary_path = _write_summary(results, args.output_root)
    print(f"Scored {len(results)} checkpoints across {len(checkpoint_dirs)} experiments")
    print(f"Scoreboard saved to {summary_path}")
    if results:
        best = sorted(results, key=_rank_key)[0]
        print(
            "Best baseline: "
            f"{best['experiment']} @ iter {best['iteration']} "
            f"(largest_component_fraction={best['largest_component_fraction']:.3f}, "
            f"component_count={best['component_count']}, "
            f"mst_p95={best['mst_p95']:.3f}, "
            f"line_score_mean={best['line_score_mean']:.3f})"
        )


if __name__ == "__main__":
    main()
