"""Split disconnected vascular meshes into separate components (Left/Right Coronary Trees)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import trimesh
from tqdm import tqdm


def split_mesh(mesh_path: Path, output_dir: Path, min_faces: int = 1000) -> None:
    """
    Split mesh into components and save the top 2 largest ones.
    Filters out small noise components with fewer than min_faces.
    """
    base_name = mesh_path.name.replace(".stl", "")
    
    try:
        # We must process the mesh (merge vertices) for split() to work correctly on STLs
        mesh = trimesh.load(mesh_path, process=True)
        
        # Split into connected components
        components = mesh.split(only_watertight=False)
        
        # Filter and sort by number of faces (largest first)
        significant_components = [c for c in components if len(c.faces) >= min_faces]
        significant_components.sort(key=lambda m: len(m.faces), reverse=True)
        
        if not significant_components:
            print(f"Warning: No significant components (>= {min_faces} faces) found in {mesh_path.name}")
            # Fallback: just take the largest if any
            if components:
                components.sort(key=lambda m: len(m.faces), reverse=True)
                significant_components = [components[0]]
            else:
                return

        # Save components
        # We usually expect 2 for coronary trees (LCA and RCA)
        for i, comp in enumerate(significant_components[:2]):
            out_path = output_dir / f"{base_name}_part{i+1}.stl"
            comp.export(out_path)
            
        if len(significant_components) < 2:
             print(f"Note: Only {len(significant_components)} component(s) found in {mesh_path.name}")
        
    except Exception as e:
        print(f"Error processing {mesh_path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split disconnected halves of vascular meshes.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .stl files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save split .stl files.")
    parser.add_argument("--limit", type=int, help="Limit number of meshes to process (for smoke runs).")
    parser.add_argument("--min-faces", type=int, default=1000, help="Minimum face count to consider a component significant.")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="Number of worker processes.")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_files = sorted(list(args.input_dir.glob("*.stl")))
    if args.limit:
        mesh_files = mesh_files[:args.limit]
        print(f"Smoke run: Limiting to first {args.limit} meshes.")
        
    print(f"Found {len(mesh_files)} mesh files to process. Using {args.num_workers} workers.")
    
    worker_fn = partial(split_mesh, output_dir=args.output_dir, min_faces=args.min_faces)
    
    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_fn, mesh_files), total=len(mesh_files)))


if __name__ == "__main__":
    main()
