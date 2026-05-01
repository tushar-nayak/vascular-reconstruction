"""Training entry point for vascular reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.config import ModelConfig, TrainingConfig
from vascular_reconstruction.data.dataset import ProjectionDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the vascular reconstruction model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--config", type=Path, help="Path to a training config JSON.")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name.")
    return parser


def train(args: argparse.Namespace) -> None:
    # Load config
    if args.config:
        train_config = TrainingConfig.load(args.config)
    else:
        train_config = TrainingConfig()
        
    if args.experiment_name:
        train_config.experiment_name = args.experiment_name
        
    print(f"Starting experiment: {train_config.experiment_name}")
    
    # Load dataset
    dataset = ProjectionDataset(args.data_dir)
    print(f"Loaded dataset with {len(dataset)} cases.")
    
    # For now, just print status since models are not implemented
    print("Models and training loop implementation pending...")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
