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
from vascular_reconstruction.models.pinn_gs import PINN_GS
from vascular_reconstruction.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the vascular reconstruction model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--config", type=Path, help="Path to a training config JSON.")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name.")
    return parser


def train(args: argparse.Namespace) -> None:
    # 1. Load config
    if args.config:
        train_config = TrainingConfig.load(args.config)
    else:
        train_config = TrainingConfig()
        
    model_config = ModelConfig()
        
    if args.experiment_name:
        train_config.experiment_name = args.experiment_name
        
    print(f"Starting experiment: {train_config.experiment_name}")
    
    # 2. Load dataset
    dataset = ProjectionDataset(args.data_dir)
    print(f"Loaded dataset with {len(dataset)} cases.")
    
    # 3. Initialize model
    pinn_config = {
        "hidden_dim": model_config.pinn_hidden_dim,
        "num_layers": model_config.pinn_num_layers
    }
    model = PINN_GS(num_gaussians=model_config.num_gaussians, pinn_config=pinn_config)
    
    # 4. Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        train_config=train_config,
        model_config=model_config
    )
    
    # 5. Start training
    trainer.train()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
