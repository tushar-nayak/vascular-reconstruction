"""Visualization tools for vascular reconstruction with Ground Truth comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vascular_reconstruction.models.pinn_gs import PINN_GS


def visualize_checkpoint(checkpoint_path: Path, output_dir: Path, mesh_path: Path | None = None):
    """Load a checkpoint and visualize the 3D geometry, flow field, and optional Ground Truth mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    
    # 2. Initialize Model
    num_gaussians = state_dict['gs._xyz'].shape[0]
    pinn_config = {"hidden_dim": 128, "num_layers": 4}
    model = PINN_GS(num_gaussians=num_gaussians, pinn_config=pinn_config)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Prepare Figure
    fig = plt.figure(figsize=(18, 8))
    
    # --- Plot 1: Reconstructed Geometry vs Ground Truth ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Reconstructed Points (Gaussians)
    xyz = model.gs.get_xyz.detach().numpy()
    idx = np.random.choice(len(xyz), min(10000, len(xyz)), replace=False)
    ax1.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c='red', s=0.5, alpha=0.4, label='Reconstruction')
    
    # Ground Truth Mesh
    if mesh_path and mesh_path.exists():
        gt_mesh = trimesh.load(mesh_path, process=False)
        
        # Apply the SAME centering/scaling used during dataset generation
        box_center = gt_mesh.bounding_box.centroid
        gt_mesh.apply_translation(-box_center)
        if np.max(gt_mesh.bounding_box.extents) < 50.0:
            gt_mesh.apply_scale(10.0)
            
        # Plot mesh as wireframe or edges
        # We plot a subset of faces or edges for speed/clarity
        ax1.plot_trisurf(gt_mesh.vertices[:, 0], gt_mesh.vertices[:, 1], gt_mesh.vertices[:, 2], 
                         triangles=gt_mesh.faces, color='blue', alpha=0.1, shade=False)
        ax1.set_title(f"Reconstruction (Red) vs Ground Truth (Blue)\nIteration {checkpoint['iteration']}")
    else:
        ax1.set_title(f"Reconstructed Geometry (Iteration {checkpoint['iteration']})")
        
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim([-100, 100])
    ax1.set_ylim([-100, 100])
    ax1.set_zlim([-100, 100])
    
    # --- Plot 2: Neural Flow Field (PINN) ---
    ax2 = fig.add_subplot(122, projection='3d')
    x_grid = np.linspace(-60, 60, 10)
    y_grid = np.linspace(-60, 60, 10)
    z_grid = np.linspace(-60, 60, 10)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    t = torch.zeros((len(coords), 1))
    coords_torch = torch.from_numpy(coords).float()
    
    with torch.no_grad():
        flow_out = model.pinn(coords_torch[:, 0:1], coords_torch[:, 1:2], coords_torch[:, 2:3], t)
        u, v, w = flow_out[:, 0].numpy(), flow_out[:, 1].numpy(), flow_out[:, 2].numpy()
        
    speed = np.sqrt(u**2 + v**2 + w**2)
    mask = speed > 0.01
    
    ax2.quiver(coords[mask, 0], coords[mask, 1], coords[mask, 2], 
               u[mask], v[mask], w[mask], length=5, normalize=True, cmap='jet')
    ax2.set_title("Neural Flow Field (PINN)")
    ax2.set_xlim([-100, 100])
    ax2.set_ylim([-100, 100])
    ax2.set_zlim([-100, 100])
    
    plt.tight_layout()
    iter_name = checkpoint['iteration']
    plt.savefig(output_dir / f"reconstruction_comparison_iter_{iter_name}.png")
    print(f"Visualization saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reconstruction checkpoints with GT mesh.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--mesh", type=Path, help="Path to ground truth STL mesh.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"), help="Output directory.")
    args = parser.parse_args()
    
    visualize_checkpoint(args.checkpoint, args.output_dir, args.mesh)


if __name__ == "__main__":
    main()
