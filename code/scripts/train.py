"""Training entry point for vascular reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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
    parser.add_argument("--model-config", type=Path, help="Path to a model config JSON.")
    parser.add_argument("--case-index", type=int, help="Dataset case index to reconstruct.")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name.")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume training from an existing checkpoint.")
    return parser


def train(args: argparse.Namespace) -> None:
    # 1. Load config
    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")

    if args.config:
        train_config = TrainingConfig.load(args.config)
    elif resume_checkpoint is not None and "training_config" in resume_checkpoint:
        train_config = TrainingConfig.from_dict(resume_checkpoint["training_config"])
    else:
        train_config = TrainingConfig()
        
    if args.model_config:
        model_config = ModelConfig.load(args.model_config)
    elif resume_checkpoint is not None and "model_config" in resume_checkpoint:
        model_config = ModelConfig.from_dict(resume_checkpoint["model_config"])
    else:
        model_config = ModelConfig()
        
    if args.experiment_name:
        train_config.experiment_name = args.experiment_name
    if args.case_index is not None:
        train_config.train_case_index = args.case_index
    elif resume_checkpoint is not None:
        train_config.train_case_index = int(resume_checkpoint.get("case_index", train_config.train_case_index))
        
    print(f"Starting experiment: {train_config.experiment_name}")
    
    # 2. Load dataset
    dataset = ProjectionDataset(args.data_dir)
    print(f"Loaded dataset with {len(dataset)} cases. Reconstructing case index {train_config.train_case_index}.")
    
    # 3. Initialize model
    pinn_config = {
        "hidden_dim": model_config.pinn_hidden_dim,
        "num_layers": model_config.pinn_num_layers,
    }
    model = PINN_GS(
        num_gaussians=model_config.num_gaussians,
        pinn_config=pinn_config,
        sh_degree=model_config.sh_degree,
    )
    
    # 4. Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        train_config=train_config,
        model_config=model_config,
    )
    
    # 5. Start training
    if args.resume_checkpoint:
        start_iteration = trainer.load_checkpoint(args.resume_checkpoint)
        trainer.train_from_iteration(start_iteration)
    else:
        trainer.train()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
