# vascular-reconstruction
A mesh-free framework for real-time 3D hemodynamic reconstruction from sparse 2D angiography. By embedding Physics-Informed Neural Networks (PINNs) into 3D Gaussian Splatting, it explicitly reconstructs vascular geometry while simultaneously solving Navier-Stokes equations to render blood flow velocity and pressure in real time.

## What Changed

The repository is now structured around a `code/` workspace so the project root stays clean and the implementation lives in one place.

Current layout:

- `code/src/vascular_reconstruction/` - core Python package
- `code/configs/` - configuration files for experiments and models
- `code/scripts/` - runnable entry points, including `download_datasets.py`
- `data/` - raw and processed dataset assets
- `code/experiments/` - experiment notes and result artifacts
- `code/tests/` - test scaffolding

The package skeleton inside `code/src/vascular_reconstruction/` is split into:

- `data` for angiography loading and preprocessing
- `data.adapters.imagecas` for ImageCAS volume and mesh discovery
- `models` for PINN, Gaussian Splatting, and hybrid components
- `simulation` for flow equations and solvers
- `rendering` for visualization and inference output
- `training` for optimization and experiment loops
- `evaluation` for metrics and benchmarking
- `utils` for shared helpers

## Notes

- The previous top-level source folders were moved under `code/`.
- Generated outputs, checkpoints, and logs should live under `code/`, while raw dataset assets live under `data/`.
- `code/scripts/download_datasets.py` downloads ImageCAS through `kagglehub` and stages it under `data/raw/imagecas/`.
- `LICENSE` remains at the repository root.

## Next Step

If you want, I can turn this scaffold into a working package next by adding:

1. a `pyproject.toml` with dependencies and tooling
2. a training/evaluation entry point
3. a minimal dataset and config loader
