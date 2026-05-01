"""Visualization tools for vascular reconstruction results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.models.pinn_gs import PINN_GS


def visualize_checkpoint(checkpoint_path: Path, output_dir: Path):
    """Load a checkpoint and visualize the 3D geometry and flow field."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. Initialize Model
    # Note: These configs should ideally match training
    num_gaussians = 100000
    pinn_config = {"hidden_dim": 128, "num_layers": 4}
    model = PINN_GS(num_gaussians=num_gaussians, pinn_config=pinn_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Visualize Geometry (Gaussians)
    xyz = model.gs.get_xyz.detach().numpy()
    opacity = model.gs.get_opacity.detach().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    # Use opacity for alpha if possible, or just plot a subset
    idx = np.random.choice(len(xyz), min(5000, len(xyz)), replace=False)
    ax1.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c='red', s=1, alpha=0.5)
    ax1.set_title(f"Reconstructed Geometry (Iteration {checkpoint['iteration']})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # 4. Visualize Flow Field (PINN)
    # Create a grid to sample flow
    x = np.linspace(-100, 100, 10)
    y = np.linspace(-100, 100, 10)
    z = np.linspace(-100, 100, 10)
    xx, yy, zz = np.meshgrid(x, y, z)
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    t = torch.zeros((len(coords), 1))
    coords_torch = torch.from_numpy(coords).float()
    
    with torch.no_grad():
        # pinn output: (u, v, w, p)
        flow_out = model.pinn(coords_torch[:, 0:1], coords_torch[:, 1:2], coords_torch[:, 2:3], t)
        u, v, w = flow_out[:, 0].numpy(), flow_out[:, 1].numpy(), flow_out[:, 2].numpy()
        p = flow_out[:, 3].numpy()
        
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(coords[:, 0], coords[:, 1], coords[:, 2], u, v, w, length=10, normalize=True, cmap='jet')
    ax2.set_title("Neural Flow Field (PINN)")
    
    plt.savefig(output_dir / f"reconstruction_iter_{checkpoint['iteration']}.png")
    print(f"Visualization saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reconstruction checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"), help="Output directory.")
    args = parser.parse_args()
    
    visualize_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
