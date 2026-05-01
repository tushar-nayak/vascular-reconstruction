# Physics-Informed Gaussian Splatting for Vascular Reconstruction

A mesh-free framework for real-time 3D hemodynamic reconstruction from sparse 2D angiography. By embedding Physics-Informed Neural Networks (PINNs) into 3D Gaussian Splatting, it explicitly reconstructs vascular geometry and solves blood flow dynamics (Navier-Stokes) simultaneously.

## Features

- **Smooth Mesh Extraction**: Optimized Marching Cubes with Gaussian smoothing for high-fidelity vasculature from CT segmentations.
- **Tree Splitting & Denoising**: Automated separation of Left and Right coronary trees with connected-component analysis.
- **High-Speed Projection Pipeline**: GPU-accelerated X-ray projection rendering using `gVXR`, capable of generating thousands of training views in minutes.
- **Hybrid PINN-GS Model**: Combined differentiable Gaussian Splatting for geometry and neural networks for flow velocity (u, v, w) and pressure (p).
- **Physics-Informed Training**: Integrated Navier-Stokes residuals for fluid dynamic consistency.

## Project Structure

- `code/src/vascular_reconstruction/`: Core package
    - `models/`: Hybrid PINN-GS architecture.
    - `simulation/`: Navier-Stokes loss functions and physics constraints.
    - `data/`: Dataset loaders for multi-view projections.
    - `training/`: Optimization loops and trainer logic.
- `code/scripts/`: 
    - `convert_imagecas_to_mesh.py`: Smooth mesh generation.
    - `split_meshes.py`: Left/Right coronary tree separation.
    - `generate_dataset.py`: Multi-worker gVXR X-ray rendering.
    - `train.py`: Main training entry point.
- `data/`: (Gitignored) Large binary assets including STL meshes and rendered projections.

## Getting Started

### 1. Environment Setup
```bash
# Recommended Python 3.10
pip install numpy pillow trimesh tqdm gvxr torch torchvision nibabel scikit-image scipy
```

### 2. Data Preparation
```bash
# Convert NIfTI labels to smooth meshes
python code/scripts/convert_imagecas_to_mesh.py --input-dir data/raw/imagecas --output-dir data/processed/imagecas/meshes --sigma 1.5

# Split coronary trees
python code/scripts/split_meshes.py --input-dir data/processed/imagecas/meshes --output-dir data/processed/imagecas/meshes_split

# Generate synthetic X-ray projections
python code/scripts/generate_dataset.py --config code/configs/imagecas_generation.json
```

### 3. Training
```bash
# Launch hybrid optimization
python code/scripts/train.py --data-dir data/processed/imagecas/projections --experiment-name coronary_reconstruction_v1
```

## Current Status

- **Dataset**: ~2,000 coronary trees processed; ~11,000 X-ray projections generated.
- **Training**: Active. Optimizing for 3D geometry matching and fluid dynamic residuals.
- **Next Steps**: Differentiable X-ray rasterization optimization and 3D flow visualization.
