"""Entry point for synthetic X-ray and depth-map dataset generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vascular_reconstruction.data.dataset_generation import DatasetGenerationConfig, generate_dataset, load_config


def _parse_gpu_ids(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(int(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic X-ray/depth dataset from meshes.")
    parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    parser.add_argument("--input-dir", type=str, help="Directory containing *_mesh.stl files.")
    parser.add_argument("--output-dir", type=str, help="Directory for generated images and manifests.")
    parser.add_argument("--image-size", type=int, help="Output X-ray image size in pixels.")
    parser.add_argument("--depth-size", type=int, help="Intermediate depth-map size in pixels.")
    parser.add_argument("--gpu-ids", type=str, help="Comma-separated GPU ids, for example '0,1'.")
    parser.add_argument("--mesh-pattern", type=str, help="Glob pattern for input meshes.")
    parser.add_argument("--no-resume", action="store_true", help="Disable skip logic for already rendered meshes.")
    return parser


def _apply_overrides(config: DatasetGenerationConfig, args: argparse.Namespace) -> DatasetGenerationConfig:
    data = config.to_json_dict()

    if args.input_dir is not None:
        data["input_dir"] = args.input_dir
    if args.output_dir is not None:
        data["output_dir"] = args.output_dir
    if args.image_size is not None:
        data["image_size"] = args.image_size
    if args.depth_size is not None:
        data["depth_size"] = args.depth_size
    if args.gpu_ids is not None:
        data["gpu_ids"] = list(_parse_gpu_ids(args.gpu_ids))
    if args.mesh_pattern is not None:
        data["mesh_pattern"] = args.mesh_pattern
    if args.no_resume:
        data["resume"] = False

    return DatasetGenerationConfig.from_mapping(data)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = DatasetGenerationConfig(
            input_dir="",
            output_dir="",
        )

    config = _apply_overrides(config, args)
    if not config.input_dir or not config.output_dir:
        parser.error("Both --input-dir and --output-dir must be set, either in the config file or via CLI overrides.")

    generate_dataset(config)


if __name__ == "__main__":
    main()
