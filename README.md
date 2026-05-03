# Physics-Informed Gaussian Splatting for Vascular Reconstruction

A framework for vascular reconstruction from sparse 2D angiography that combines learned geometry with physics-informed flow modeling. The current codebase uses a per-case differentiable silhouette reconstruction stage for geometry and a PINN-based Navier-Stokes loss for flow regularization after geometry has stabilized.

## Features

- **Smooth Mesh Extraction**: Optimized Marching Cubes with Gaussian smoothing for high-fidelity vasculature from CT segmentations.
- **Tree Splitting & Denoising**: Automated separation of Left and Right coronary trees with connected-component analysis.
- **High-Speed Projection Pipeline**: GPU-accelerated X-ray projection rendering using `gVXR`, capable of generating thousands of training views in minutes.
- **Per-Case Silhouette Reconstruction**: Differentiable low-resolution Gaussian silhouette rendering supervised against vessel masks from synthetic angiography.
- **Hybrid PINN-GS Model**: Gaussian geometry parameters plus a neural field for flow velocity `(u, v, w)` and pressure `p`.
- **Physics-Informed Training**: Integrated Navier-Stokes residuals for incompressible flow consistency, enabled after a geometry warmup period.

## Project Structure

- `code/src/vascular_reconstruction/`: Core package
    - `models/`: Hybrid PINN-GS architecture and Gaussian parameterization.
    - `rendering/`: Differentiable silhouette rendering utilities.
    - `simulation/`: Navier-Stokes residuals and physics constraints.
    - `data/`: Dataset loaders for multi-view projections and masks.
    - `training/`: Per-case reconstruction loops, initialization, and diagnostics.
- `code/scripts/`: 
    - `convert_imagecas_to_mesh.py`: Smooth mesh generation.
    - `split_meshes.py`: Left/Right coronary tree separation.
    - `generate_dataset.py`: Multi-worker gVXR X-ray rendering.
    - `train.py`: Main per-case reconstruction entry point.
- `data/`: (Gitignored) Large binary assets including STL meshes and rendered projections.

## Training Strategy

The training path is intentionally staged:

1. **Geometry first**
   - A single case is selected from the dataset.
   - Gaussian centers are initialized from backprojected vessel-mask pixels rather than a uniform random cube.
   - A differentiable silhouette renderer projects the current Gaussian set into each angiographic view.
   - The rendered soft silhouette is compared against the target vessel mask using BCE and Dice losses.

2. **Physics second**
   - The PINN predicts `(u, v, w, p)` as functions of `(x, y, z, t)`.
   - Navier-Stokes residuals are not used immediately.
   - A warmup period lets geometry lock onto the observed views before the flow field is asked to satisfy PDE constraints.

3. **Anti-collapse regularization**
   - Pairwise Gaussian repulsion discourages point collapse.
   - Minimum axis spread penalties discourage the geometry from shrinking into a few modes.
   - Opacity and scale targets keep the rendered silhouette numerically usable.
   - Debug overlays are written during training so projection collapse can be seen early.

This sequencing matters. Applying fluid constraints before the projection geometry has stabilized is an easy way to get degenerate local minima.

## Navier-Stokes Formulation

The PINN flow model is implemented in [equations.py](/home/sofa/host_dir/hub/vascular-reconstruction/code/src/vascular_reconstruction/simulation/equations.py). It treats the flow field as a continuous function

```text
(x, y, z, t) -> (u, v, w, p)
```

where:

- `u, v, w` are the velocity components
- `p` is pressure
- `rho` is fluid density
- `mu` is dynamic viscosity

### Governing Equations

The code enforces the incompressible Navier-Stokes equations in residual form.

**Continuity**

```text
du/dx + dv/dy + dw/dz = 0
```

This is the incompressibility constraint. In practical terms, it discourages the network from inventing sources or sinks inside the reconstructed vessel volume.

**Momentum**

For each velocity component, the code forms a PDE residual:

```text
f_u = rho * (du/dt + u du/dx + v du/dy + w du/dz) + dp/dx - mu * (d2u/dx2 + d2u/dy2 + d2u/dz2)
f_v = rho * (dv/dt + u dv/dx + v dv/dy + w dv/dz) + dp/dy - mu * (d2v/dx2 + d2v/dy2 + d2v/dz2)
f_w = rho * (dw/dt + u dw/dx + v dw/dy + w dw/dz) + dp/dz - mu * (d2w/dx2 + d2w/dy2 + d2w/dz2)
```

These terms decompose into:

- **Temporal acceleration**: `du/dt`, `dv/dt`, `dw/dt`
- **Convective acceleration**: `u du/dx + v du/dy + w du/dz`, and analogs for `v` and `w`
- **Pressure gradient**: `dp/dx`, `dp/dy`, `dp/dz`
- **Viscous diffusion**: the Laplacian terms scaled by `mu`

### How the Residual Is Computed

The implementation uses PyTorch autograd to differentiate the PINN outputs with respect to the input coordinates:

- First derivatives are used for continuity, advection, pressure gradients, and time derivatives.
- Second derivatives are used for viscous terms.
- The final loss is the mean squared residual:

```text
mean(continuity^2 + f_u^2 + f_v^2 + f_w^2)
```

That means the PINN is not solving the equations by explicit time stepping. Instead, it is being optimized so that its predicted field is as close as possible to a function whose derivatives satisfy the PDE everywhere it is sampled.

### Why This Is Useful Here

Sparse angiographic views constrain vessel geometry well enough to estimate silhouette and gross topology, but they do not directly specify a full 3D hemodynamic field. The Navier-Stokes residual provides a strong prior:

- flow should be divergence-free
- local accelerations should balance pressure and viscosity
- predicted fields should vary smoothly in physically plausible ways

In other words, the PINN is not replacing measurement. It is narrowing the space of admissible flow fields once geometry is sufficiently good.

### What the Current Code Does Not Yet Model

The current physics layer is intentionally minimal. It does **not** yet encode:

- inlet and outlet boundary conditions
- wall no-slip constraints
- patient-specific Reynolds scaling
- pulsatile waveform supervision
- vessel-lumen masking of PINN sample points

Those are the next important upgrades if the goal is quantitatively credible coronary hemodynamics rather than just geometry-aware regularization.

### Practical Implication

For this repository, the correct use of Navier-Stokes is:

- first obtain a stable silhouette-consistent geometry reconstruction
- then turn on PDE residual regularization
- then inspect whether the flow field remains smooth and physically coherent without destabilizing geometry

If geometry is still collapsing, the right move is not to strengthen the physics term. It is to improve the renderer, geometry prior, or mask supervision first.

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
# Launch per-case silhouette reconstruction
python code/scripts/train.py \
  --data-dir data/processed/imagecas/projections \
  --config path/to/train_config.json \
  --model-config path/to/model_config.json \
  --case-index 0
```

### 4. Visualization
```bash
# Generate geometry diagnostics from a checkpoint
python code/scripts/visualize_reconstruction.py --checkpoint checkpoints/checkpoint_10000.pt
```

## Current Status

- **Dataset**: ~2,000 coronary trees processed; ~11,000 X-ray projections generated.
- **Training Direction**: The training path has been refactored toward per-case differentiable silhouette reconstruction rather than global point-center matching.
- **Physics Usage**: Navier-Stokes residuals are now treated as a late-stage regularizer instead of an immediate co-objective.
- **Open Problem**: Good reconstruction quality still depends on validating the new single-case training path with fresh runs from scratch.
