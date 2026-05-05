# Vascular Reconstruction

This repository is an experiment in reconstructing 3D coronary vessel geometry from a small set of 2D angiographic views.

The project has two main parts:

1. A data pipeline that turns coronary segmentations into meshes and synthetic X-ray projections.
2. A reconstruction pipeline that tries to recover a 3D vessel representation from those projections using Gaussians plus a physics-informed neural network.

At the moment, the mesh and projection pipeline is reliable. The learned reconstruction pipeline is improving, but it is still an active work in progress.

## What the code does

The repo starts from coronary labels, usually from CT-derived segmentations, and builds a synthetic training set:

- convert label volumes to smooth meshes
- split the coronary trees into cleaner left and right structures
- render multiple angiographic views with `gVXR`
- train a per-case reconstruction model against those rendered vessel masks

The reconstruction model currently uses a Gaussian-based geometry representation. During training, the model renders soft vessel silhouettes into each view and compares them against the target masks. A PINN is included as a later-stage regularizer for flow, not as the primary source of geometric supervision.

## What is working today

- mesh extraction from segmentation volumes
- coronary tree splitting and cleanup
- synthetic multi-view projection generation
- per-case training from angiographic masks
- checkpointing, resume support, and debug overlays
- geometry diagnostics and coarse post-hoc centerline extraction

## What is not solved yet

The hard part is still the model itself.

Right now the training code can usually do one of these two things:

- fit the 2D silhouettes fairly well but stay too thick in 3D
- become thinner and more line-like locally but fragment into too many disconnected 3D pieces

That tradeoff is the main problem in the repo right now. The recent work has been about pushing the model toward thinner, more vessel-like geometry without destroying connectivity.

## Repository layout

- `code/src/vascular_reconstruction/`
  - `models/`: Gaussian and PINN model definitions
  - `rendering/`: differentiable silhouette rendering utilities
  - `simulation/`: Navier-Stokes residuals
  - `data/`: dataset generation and loading
  - `training/`: training loop, initialization, regularization, diagnostics
- `code/scripts/`
  - `convert_imagecas_to_mesh.py`
  - `split_meshes.py`
  - `generate_dataset.py`
  - `train.py`
  - `visualize_reconstruction.py`
  - `extract_centerline.py`
- `code/configs/`
  - training and model configs for different experiments
- `data/`
  - large binary inputs and generated assets

## How training is set up

The training loop is staged on purpose.

### 1. Geometry first

Training is done one case at a time.

The Gaussian cloud is initialized from backprojected vessel pixels rather than from a uniform random cube. For each angiographic view, the model renders a soft silhouette and compares it against the target vessel mask.

The image loss is mostly:

- binary cross-entropy
- Dice-style overlap
- penalties for leaking mass outside the target mask

There are also extra terms that try to keep the geometry useful:

- skeleton-focused mask supervision
- anti-collapse regularization
- scale and opacity targets
- staged activation of Gaussians
- densification rules for adding new Gaussians during training

### 2. Physics later

The PINN predicts:

```text
(x, y, z, t) -> (u, v, w, p)
```

where `u, v, w` are velocity and `p` is pressure.

The important point is that the Navier-Stokes loss is not used from the start. Geometry has to settle first. If the PDE term is turned on too early, it tends to make optimization harder rather than better.

### 3. Diagnostics throughout

The code writes debug projection panels during training and can also produce post-training geometry diagnostics:

- projection overlays across multiple views
- kNN graph plots
- connected-component counts
- local line-likeness scores
- coarse centerline extraction from voxelized occupancy

Those diagnostics matter because a model can look fine in one 2D view and still be wrong in 3D.

## Navier-Stokes, in plain terms

The PINN side of the model is implemented in [equations.py](/home/sofa/host_dir/hub/vascular-reconstruction/code/src/vascular_reconstruction/simulation/equations.py).

It uses the incompressible Navier-Stokes equations as a soft regularizer. The network is asked to produce a flow field whose derivatives make physical sense.

### Continuity

```text
du/dx + dv/dy + dw/dz = 0
```

This is the incompressibility constraint. It discourages the network from inventing flow sources or sinks inside the vessel.

### Momentum

For each velocity component, the code builds a residual that includes:

- time derivatives
- convective terms
- pressure gradients
- viscous diffusion

Written out, the residuals look like:

```text
f_u = rho * (du/dt + u du/dx + v du/dy + w du/dz) + dp/dx - mu * (d2u/dx2 + d2u/dy2 + d2u/dz2)
f_v = rho * (dv/dt + u dv/dx + v dv/dy + w dv/dz) + dp/dy - mu * (d2v/dx2 + d2v/dy2 + d2v/dz2)
f_w = rho * (dw/dt + u dw/dx + v dw/dy + w dw/dz) + dp/dz - mu * (d2w/dx2 + d2w/dy2 + d2w/dz2)
```

PyTorch autograd is used to get the first and second derivatives needed for those terms.

The PINN is not solving the PDE by marching forward in time. It is being optimized so that its outputs look like a field that satisfies the PDE residuals at sampled points.

### Why the physics term is there

Sparse angiographic projections tell you something about vessel shape, but they do not tell you a full 3D flow field. The Navier-Stokes residual gives the model a physical bias:

- flow should be divergence-free
- fields should vary smoothly
- pressure and viscous effects should balance local accelerations in a plausible way

That said, the current physics layer is still minimal. It does not yet include:

- inlet or outlet boundary conditions
- no-slip wall constraints
- patient-specific scaling
- waveform supervision
- a strict vessel-lumen sampling mask for the PINN points

So the physics term should be thought of as regularization, not as a clinically valid hemodynamic model.

## Setup

### 1. Install dependencies

Python `3.10` is a good target.

```bash
pip install numpy pillow trimesh tqdm gvxr torch torchvision nibabel scikit-image scipy
```

`pytest` is useful as well, but it is not included above.

### 2. Build meshes and projections

```bash
python code/scripts/convert_imagecas_to_mesh.py \
  --input-dir data/raw/imagecas \
  --output-dir data/processed/imagecas/meshes \
  --sigma 1.5

python code/scripts/split_meshes.py \
  --input-dir data/processed/imagecas/meshes \
  --output-dir data/processed/imagecas/meshes_split

python code/scripts/generate_dataset.py \
  --config code/configs/imagecas_generation.json
```

### 3. Train one case

```bash
python code/scripts/train.py \
  --data-dir data/processed/imagecas/projections \
  --config path/to/train_config.json \
  --model-config path/to/model_config.json \
  --case-index 0
```

### 4. Inspect a checkpoint

```bash
python code/scripts/visualize_reconstruction.py \
  --checkpoint checkpoints/checkpoint_10000.pt

python code/scripts/extract_centerline.py \
  --checkpoint checkpoints/checkpoint_10000.pt
```

## Current status

This is the honest version:

- the preprocessing and synthetic data pipeline are in good shape
- the reconstruction code is much better instrumented than it was originally
- the model can now be pushed toward either better silhouette fit or thinner geometry
- it still does not consistently produce one clean connected vessel tree

The repo is useful if you want to work on that problem. It is not at the point where I would present the learned reconstruction results as solved.

## If you are picking this up next

The highest-value work is not another plotting tweak. It is training-objective design.

The next likely wins are:

1. better global connectivity objectives
2. densification driven by multi-view support, not just local geometry
3. a stronger 3D representation than a raw Gaussian cloud for final topology extraction
