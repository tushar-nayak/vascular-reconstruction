"""Convert ImageCAS segmentation volumes to STL meshes with smoothing."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure
from tqdm import tqdm


def convert_volume_to_mesh(nii_path: Path, output_path: Path, label_value: int = 1, sigma: float = 1.0) -> None:
    """Extract a smooth mesh from a specific label in a NIfTI volume."""
    
    # Load the NIfTI file
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine
    
    # Threshold to get the desired label
    binary_mask = (data == label_value).astype(np.float32)
    
    if np.sum(binary_mask) == 0:
        print(f"Warning: No voxels found for label {label_value} in {nii_path}")
        return

    # Apply Gaussian blur to the mask to create a smooth transition
    if sigma > 0:
        smoothed_data = gaussian_filter(binary_mask, sigma=sigma)
    else:
        smoothed_data = binary_mask

    # Marching cubes to extract the surface
    verts, faces, normals, values = measure.marching_cubes(smoothed_data, level=0.5)
    
    # Transform vertices to world coordinates using the affine matrix
    verts_homogeneous = np.c_[verts, np.ones(verts.shape[0])]
    verts_world = verts_homogeneous @ affine.T
    verts_world = verts_world[:, :3]
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
    
    # Save the mesh
    mesh.export(output_path)


def process_case(label_file: Path, output_dir: Path, label_value: int, sigma: float) -> None:
    case_id = label_file.name.split(".")[0]
    output_file = output_dir / f"{case_id}_mesh.stl"
    
    if output_file.exists():
        return
        
    try:
        convert_volume_to_mesh(label_file, output_file, label_value, sigma)
    except Exception as e:
        print(f"Error processing {label_file}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ImageCAS labels to smooth meshes.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .label.nii.gz files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save .stl files.")
    parser.add_argument("--label", type=int, default=1, help="Label value to extract (default: 1).")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing sigma (default: 1.5).")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="Number of worker processes.")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(args.input_dir.rglob("*.label.nii.gz"))
    if not label_files:
        label_files = list(args.input_dir.rglob("*label*.nii.gz"))
        
    print(f"Found {len(label_files)} label files. Using {args.num_workers} workers.")
    
    # Use multiprocessing pool
    worker_fn = partial(process_case, output_dir=args.output_dir, label_value=args.label, sigma=args.sigma)
    
    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_fn, label_files), total=len(label_files)))


if __name__ == "__main__":
    main()
